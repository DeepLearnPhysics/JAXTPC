"""
Estimate max_keys without running the full simulation.

Computes group IDs, wire/pixel indices, and time indices using numpy,
then estimates unique (spatial_key, time, group) triplets per plane via
vectorized group-footprint bounding boxes.

Effective kernel extents are derived from the actual kernel data:
  - Wire: CDF diffusion kernel thresholded at 1% of peak per s level
  - Pixel: DKernel table thresholded at 1% of peak per s level
This gives a lookup table s → (K_spatial_eff, K_time_eff) that varies
with drift distance, matching the actual pipeline's pruning behavior.

The final max_keys suggestion extrapolates the observed keys/deps ratio
to total_pad, providing a safe bound without arbitrary headroom.

Usage:
    python3 -m profiler.estimate_max_keys --data events.h5 --config config.yaml
    python3 -m profiler.estimate_max_keys --data events.h5 --config config.yaml --total-pad 900000
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from scipy.special import erf

from tools.geometry import generate_detector
from tools.config import create_sim_config
from tools.loader import compute_group_ids

THRESH_FRAC = 0.01


def _cdf_1d(mu, sigma, n):
    offsets = np.arange(-(n // 2), n // 2 + 1)
    if sigma < 1e-6:
        r = np.zeros(n)
        r[n // 2] = 1.0
        return r
    lo = offsets - 0.5 - mu
    hi = offsets + 0.5 - mu
    return 0.5 * (erf(hi / (sigma * np.sqrt(2))) - erf(lo / (sigma * np.sqrt(2))))


def build_wire_extent_table(diffusion, num_s=16):
    """Build per-s effective kernel extents from CDF diffusion.

    Returns (kw_table, kt_table) each shape (num_s,) int32.
    """
    max_sigma_w = diffusion.max_sigma_trans_unitless
    max_sigma_t = diffusion.max_sigma_long_unitless
    K_wire = diffusion.K_wire
    K_time = diffusion.K_time

    kw_table = np.zeros(num_s, dtype=np.int32)
    kt_table = np.zeros(num_s, dtype=np.int32)

    for i in range(num_s):
        s = i / max(num_s - 1, 1)
        sw = max(max_sigma_w * np.sqrt(s), 1e-3)
        st = max(max_sigma_t * np.sqrt(s), 1e-3)
        wf = _cdf_1d(0, sw, 2 * K_wire + 1)
        tf = _cdf_1d(0, st, 2 * K_time + 1)
        k2d = wf[:, None] * tf[None, :]
        peak = k2d.max()
        mask = k2d > THRESH_FRAC * peak
        wa = np.where(mask.any(axis=1))[0]
        ta = np.where(mask.any(axis=0))[0]
        kw_table[i] = max(abs(wa.min() - K_wire), abs(wa.max() - K_wire))
        kt_table[i] = max(abs(ta.min() - K_time), abs(ta.max() - K_time))

    return kw_table, kt_table


def build_pixel_extent_table(pixel_kernel_path, pixel_pitch_cm,
                             max_sigma_trans, max_sigma_long, num_s=16):
    """Build per-s effective kernel extents from pixel DKernel.

    Returns (kpy_table, kpz_table, kt_table) each shape (num_s,) int32,
    in output pixel/time units.
    """
    from tools.kernels import load_pixel_response_kernel

    pk = load_pixel_response_kernel(
        pixel_kernel_path, num_s=num_s,
        time_spacing=1.0,  # placeholder, extents are relative
        pixel_pitch_cm=pixel_pitch_cm,
        max_sigma_trans_unitless=max_sigma_trans,
        max_sigma_long_unitless=max_sigma_long)

    dk = np.array(pk.DKernel)
    bins_per_pixel = int(round(pixel_pitch_cm / (pixel_pitch_cm * pk.pixel_spacing)))

    kpy_table = np.zeros(num_s, dtype=np.int32)
    kpz_table = np.zeros(num_s, dtype=np.int32)
    kt_table = np.zeros(num_s, dtype=np.int32)

    cy = dk.shape[1] // 2
    cz = dk.shape[2] // 2

    for i in range(num_s):
        k3d = dk[i]
        peak = np.abs(k3d).max()
        if peak < 1e-12:
            continue
        mask = np.abs(k3d) > THRESH_FRAC * peak
        py_a = np.where(mask.any(axis=(1, 2)))[0]
        pz_a = np.where(mask.any(axis=(0, 2)))[0]
        t_a = np.where(mask.any(axis=(0, 1)))[0]

        kpy_table[i] = int(np.ceil(
            max(abs(py_a.min() - cy), abs(py_a.max() - cy)) / bins_per_pixel))
        kpz_table[i] = int(np.ceil(
            max(abs(pz_a.min() - cz), abs(pz_a.max() - cz)) / bins_per_pixel))
        kt_table[i] = int(np.ceil((t_a.max() - t_a.min() + 1) / pk.rebin_factor))

    return kpy_table, kpz_table, kt_table


def _estimate_plane_keys(wire_idx, time_idx, group_ids, kw_eff, kt_eff,
                         keep, num_wires, num_time, n_groups):
    """Estimate unique keys for one wire plane via group bounding boxes."""
    gids = group_ids.copy()
    gids[~keep] = 0

    w_lo = wire_idx - kw_eff
    w_hi = wire_idx + kw_eff
    t_lo = time_idx - kt_eff
    t_hi = time_idx + kt_eff

    g_w_lo = np.full(n_groups, num_wires, dtype=np.int32)
    g_w_hi = np.full(n_groups, -1, dtype=np.int32)
    g_t_lo = np.full(n_groups, num_time, dtype=np.int32)
    g_t_hi = np.full(n_groups, -1, dtype=np.int32)

    active = gids > 0
    ag = gids[active]
    np.minimum.at(g_w_lo, ag, w_lo[active])
    np.maximum.at(g_w_hi, ag, w_hi[active])
    np.minimum.at(g_t_lo, ag, t_lo[active])
    np.maximum.at(g_t_hi, ag, t_hi[active])

    g_w_lo = np.maximum(g_w_lo, 0)
    g_w_hi = np.minimum(g_w_hi, num_wires - 1)
    g_t_lo = np.maximum(g_t_lo, 0)
    g_t_hi = np.minimum(g_t_hi, num_time - 1)

    has_deposits = g_w_hi >= g_w_lo
    footprints = np.where(
        has_deposits,
        (g_w_hi - g_w_lo + 1).astype(np.int64) * (g_t_hi - g_t_lo + 1).astype(np.int64),
        0)
    return int(footprints.sum())


def _estimate_pixel_plane_keys(py_idx, pz_idx, time_idx, group_ids,
                               kpy_eff, kpz_eff, kt_eff,
                               keep, num_py, num_pz, num_time, n_groups):
    """Estimate unique keys for one pixel plane via group bounding boxes."""
    gids = group_ids.copy()
    gids[~keep] = 0

    py_lo = py_idx - kpy_eff
    py_hi = py_idx + kpy_eff
    pz_lo = pz_idx - kpz_eff
    pz_hi = pz_idx + kpz_eff
    t_lo = time_idx - kt_eff
    t_hi = time_idx + kt_eff

    g_py_lo = np.full(n_groups, num_py, dtype=np.int32)
    g_py_hi = np.full(n_groups, -1, dtype=np.int32)
    g_pz_lo = np.full(n_groups, num_pz, dtype=np.int32)
    g_pz_hi = np.full(n_groups, -1, dtype=np.int32)
    g_t_lo = np.full(n_groups, num_time, dtype=np.int32)
    g_t_hi = np.full(n_groups, -1, dtype=np.int32)

    active = gids > 0
    ag = gids[active]
    np.minimum.at(g_py_lo, ag, py_lo[active])
    np.maximum.at(g_py_hi, ag, py_hi[active])
    np.minimum.at(g_pz_lo, ag, pz_lo[active])
    np.maximum.at(g_pz_hi, ag, pz_hi[active])
    np.minimum.at(g_t_lo, ag, t_lo[active])
    np.maximum.at(g_t_hi, ag, t_hi[active])

    g_py_lo = np.maximum(g_py_lo, 0)
    g_py_hi = np.minimum(g_py_hi, num_py - 1)
    g_pz_lo = np.maximum(g_pz_lo, 0)
    g_pz_hi = np.minimum(g_pz_hi, num_pz - 1)
    g_t_lo = np.maximum(g_t_lo, 0)
    g_t_hi = np.minimum(g_t_hi, num_time - 1)

    has = (g_py_hi >= g_py_lo) & (g_pz_hi >= g_pz_lo) & (g_t_hi >= g_t_lo)
    footprints = np.where(
        has,
        (g_py_hi - g_py_lo + 1).astype(np.int64)
        * (g_pz_hi - g_pz_lo + 1).astype(np.int64)
        * (g_t_hi - g_t_lo + 1).astype(np.int64),
        0)
    return int(footprints.sum())


def estimate_keys_for_event(pstep_data, sim_config, extent_tables,
                            group_size=5, gap_threshold_mm=5.0):
    """Estimate max_keys for one event across all volumes and planes.

    Parameters
    ----------
    extent_tables : dict
        Per-volume extent tables. For wire volumes:
            {vol_idx: (kw_table, kt_table)}
        For pixel volumes:
            {vol_idx: (kpy_table, kpz_table, kt_table)}

    Returns dict of {(vol_idx, plane_idx): n_unique_keys}
    and dict of {vol_idx: n_deposits_in_volume}.
    """
    positions_mm = np.column_stack([
        pstep_data['x'].astype(np.float32),
        pstep_data['y'].astype(np.float32),
        pstep_data['z'].astype(np.float32),
    ])
    de = pstep_data['de'].astype(np.float32)
    track_ids = pstep_data['track_id'].astype(np.int32)
    n = len(de)

    if 't' in pstep_data.dtype.names:
        t0_us = pstep_data['t'].astype(np.float32) / 1000.0
    else:
        t0_us = np.zeros(n, dtype=np.float32)

    pos_cm = positions_mm / 10.0
    results = {}
    vol_deps = {}

    for v, vol_geom in enumerate(sim_config.volumes):
        ranges = vol_geom.ranges_cm
        x_range, y_range, z_range = ranges

        mask = (
            (pos_cm[:, 0] >= x_range[0]) & (pos_cm[:, 0] < x_range[1]) &
            (pos_cm[:, 1] >= y_range[0]) & (pos_cm[:, 1] < y_range[1]) &
            (pos_cm[:, 2] >= z_range[0]) & (pos_cm[:, 2] < z_range[1])
        )
        vol_idx = np.where(mask)[0]
        n_planes = vol_geom.n_planes if vol_geom.readout_type == 'wire' else 1
        if len(vol_idx) == 0:
            vol_deps[v] = 0
            for p in range(n_planes):
                results[(v, p)] = 0
            continue

        vol_pos_mm = positions_mm[vol_idx]
        vol_pos_cm = pos_cm[vol_idx]
        vol_de = de[vol_idx]
        vol_tids = track_ids[vol_idx]
        vol_t0 = t0_us[vol_idx]

        valid = vol_de > 0
        vol_deps[v] = int(valid.sum())
        if valid.sum() == 0:
            for p in range(n_planes):
                results[(v, p)] = 0
            continue

        group_ids, _, n_groups = compute_group_ids(
            vol_pos_mm, vol_tids, valid,
            group_size=group_size, gap_threshold_mm=gap_threshold_mm)

        drift_dist_cm = np.abs(vol_pos_cm[:, 0] - vol_geom.x_anode_cm)
        velocity = vol_geom.diffusion.velocity_cm_us
        num_time = sim_config.num_time_steps

        # Per-deposit s index for extent table lookup
        s_vals = np.clip(np.sqrt(drift_dist_cm / vol_geom.max_drift_cm), 0, 1)
        num_s = len(extent_tables[v][0])
        s_idx = np.clip((s_vals * (num_s - 1)).astype(int), 0, num_s - 1)

        if vol_geom.readout_type == 'pixel':
            kpy_table, kpz_table, kt_table = extent_tables[v]
            kpy_eff = kpy_table[s_idx]
            kpz_eff = kpz_table[s_idx]
            kt_eff = kt_table[s_idx]

            # Pixel indices
            origins = np.array(vol_geom.pixel_origins_cm, dtype=np.float32)
            pitch = vol_geom.pixel_pitch_cm
            d_yz = vol_pos_cm[:, 1:3] - origins
            centers = np.floor(d_yz / pitch).astype(np.int32)
            py_idx = centers[:, 0]
            pz_idx = centers[:, 1]
            num_py, num_pz = vol_geom.pixel_shape

            plane_drift_dist = drift_dist_cm
            drift_time = np.where(velocity > 1e-9, plane_drift_dist / velocity, 0.0)
            tick_us = drift_time + vol_t0 + sim_config.pre_window_us
            time_idx = np.floor(tick_us / sim_config.time_step_us).astype(np.int32)

            keep = valid & (group_ids > 0) & (plane_drift_dist > 0)
            keep &= (time_idx >= 0) & (time_idx < num_time)
            keep &= (py_idx >= 0) & (py_idx < num_py)
            keep &= (pz_idx >= 0) & (pz_idx < num_pz)

            n_unique = _estimate_pixel_plane_keys(
                py_idx, pz_idx, time_idx, group_ids,
                kpy_eff, kpz_eff, kt_eff,
                keep, num_py, num_pz, num_time, n_groups)
            results[(v, 0)] = n_unique

        else:
            kw_table, kt_table = extent_tables[v]
            kw_eff = kw_table[s_idx]
            kt_eff_arr = kt_table[s_idx]

            yz_center = np.array(vol_geom.yz_center_cm, dtype=np.float32)
            yz_cm = vol_pos_cm[:, 1:3] - yz_center

            for p in range(vol_geom.n_planes):
                angle_rad = vol_geom.angles_rad[p]
                wire_spacing = vol_geom.wire_spacings_cm[p]
                index_offset = vol_geom.index_offsets[p]
                num_wires = vol_geom.num_wires[p]
                plane_dist = vol_geom.plane_distances_cm[p]

                plane_drift_dist = drift_dist_cm - plane_dist
                plane_drift_time = np.where(velocity > 1e-9,
                                            plane_drift_dist / velocity, 0.0)

                r_prime = yz_cm[:, 0] * np.sin(angle_rad) + yz_cm[:, 1] * np.cos(angle_rad)
                wire_idx = np.round(r_prime / wire_spacing).astype(np.int32) + index_offset

                tick_us = plane_drift_time + vol_t0 + sim_config.pre_window_us
                time_idx = np.floor(tick_us / sim_config.time_step_us).astype(np.int32)

                keep = valid & (group_ids > 0)
                keep &= (plane_drift_dist > 0)
                keep &= (time_idx >= 0) & (time_idx < num_time)
                keep &= (wire_idx >= 0) & (wire_idx < num_wires)

                n_unique = _estimate_plane_keys(
                    wire_idx, time_idx, group_ids, kw_eff, kt_eff_arr,
                    keep, num_wires, num_time, n_groups)
                results[(v, p)] = n_unique

    return results, vol_deps


def estimate_max_keys(data_path, config_path, events=None,
                      total_pad=None, group_size=5, gap_threshold=5.0,
                      round_to=100_000, pixel_kernel_path=None):
    """Estimate max_keys from deposit data.

    Scans all events, computes per-volume keys and deposit counts, then
    extrapolates to total_pad using the upper-envelope keys/deps ratio.

    Returns (suggestion, details_dict).
    """
    detector_config = generate_detector(config_path)
    sim_config = create_sim_config(detector_config)

    from profiler.find_optimal_pad import get_volume_ranges, count_deposits_per_volume
    volume_ranges = get_volume_ranges(detector_config)
    num_s = 16

    # Build extent tables per volume
    extent_tables = {}
    for v, vol_geom in enumerate(sim_config.volumes):
        if vol_geom.readout_type == 'pixel':
            if pixel_kernel_path is None:
                pixel_kernel_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'config', 'pixel_response.npz')
            extent_tables[v] = build_pixel_extent_table(
                pixel_kernel_path, vol_geom.pixel_pitch_cm,
                vol_geom.diffusion.max_sigma_trans_unitless,
                vol_geom.diffusion.max_sigma_long_unitless,
                num_s=num_s)
        else:
            extent_tables[v] = build_wire_extent_table(
                vol_geom.diffusion, num_s=num_s)

    with h5py.File(data_path, 'r') as f:
        ds = f['pstep/lar_vol']
        n_events = ds.shape[0]
        if events is not None:
            n_events = min(n_events, events)

        all_deps = []
        all_keys = []
        all_event_maxes = []

        for i in range(n_events):
            pstep = ds[i]
            event_keys, event_vol_deps = estimate_keys_for_event(
                pstep, sim_config, extent_tables,
                group_size=group_size, gap_threshold_mm=gap_threshold)

            event_max = 0
            for v_idx, n_dep in event_vol_deps.items():
                if n_dep == 0:
                    continue
                vol = sim_config.volumes[v_idx]
                n_planes = vol.n_planes if vol.readout_type == 'wire' else 1
                vol_max = max(
                    (event_keys.get((v_idx, p), 0) for p in range(n_planes)),
                    default=0)
                all_deps.append(n_dep)
                all_keys.append(vol_max)
                event_max = max(event_max, vol_max)
            all_event_maxes.append(event_max)

    deps = np.array(all_deps)
    keys = np.array(all_keys)
    ratio = keys / np.maximum(deps, 1)

    median_deps = np.median(deps)
    upper_mask = deps >= median_deps
    upper_max_ratio = float(ratio[upper_mask].max())

    if total_pad is None:
        total_pad = int(deps.max())

    extrapolated = int(upper_max_ratio * total_pad)
    suggestion = int(math.ceil(extrapolated / round_to) * round_to)

    return suggestion, {
        'n_events': n_events,
        'max_observed_keys': int(keys.max()),
        'max_observed_deps': int(deps.max()),
        'total_pad': total_pad,
        'upper_max_ratio': upper_max_ratio,
        'extrapolated': extrapolated,
        'all_event_maxes': np.array(all_event_maxes),
        'all_deps': deps,
        'all_keys': keys,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Estimate max_keys from deposit geometry (no simulation)')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--events', type=int, default=None,
                        help='Max events to scan (default: all)')
    parser.add_argument('--total-pad', type=int, default=None,
                        help='Total pad to extrapolate to (default: from data)')
    parser.add_argument('--group-size', type=int, default=5)
    parser.add_argument('--gap-threshold', type=float, default=5.0)
    parser.add_argument('--round-to', type=int, default=100_000)
    parser.add_argument('--pixel-kernel', default=None,
                        help='Path to pixel response NPZ (pixel readout only)')
    parser.add_argument('--save-config', default=None,
                        help='Save max_keys to production config YAML')

    args = parser.parse_args()

    print('=' * 70)
    print(' JAXTPC — Estimate max_keys (no simulation)')
    print('=' * 70)
    print(f'  Data:      {args.data}')
    print(f'  Config:    {args.config}')

    suggestion, info = estimate_max_keys(
        args.data, args.config,
        events=args.events,
        total_pad=args.total_pad,
        group_size=args.group_size,
        gap_threshold=args.gap_threshold,
        round_to=args.round_to,
        pixel_kernel_path=args.pixel_kernel)

    maxes = info['all_event_maxes']
    pcts = np.percentile(maxes, [50, 90, 99, 99.9, 100])
    deps = info['all_deps']
    keys = info['all_keys']
    ratio = keys / np.maximum(deps, 1)

    print(f'  Events:    {info["n_events"]}')
    print(f'  Total pad: {info["total_pad"]:,}')
    print()

    print(f'  Per-event max keys distribution:')
    print(f'    P50   = {int(pcts[0]):>10,}')
    print(f'    P90   = {int(pcts[1]):>10,}')
    print(f'    P99   = {int(pcts[2]):>10,}')
    print(f'    P99.9 = {int(pcts[3]):>10,}')
    print(f'    Max   = {int(pcts[4]):>10,}')

    print(f'\n  Keys/deps ratio (per volume):')
    print(f'    Median = {np.median(ratio):.3f}')
    print(f'    P95    = {np.percentile(ratio, 95):.3f}')
    print(f'    Max    = {ratio.max():.3f}')
    print(f'    Upper-half max = {info["upper_max_ratio"]:.3f}  '
          f'(deposits >= {int(np.median(deps)):,})')

    print(f'\n  Extrapolation to total_pad={info["total_pad"]:,}:')
    print(f'    upper_max_ratio x total_pad = {info["extrapolated"]:,}')
    print(f'    Rounded: {suggestion:,}')
    print(f'    --max-keys {suggestion}')

    if args.save_config:
        from profiler.production_config import update_config
        update_config(args.save_config, {'max_keys': suggestion},
                      detector_config_path=args.config)
        print(f'\n  Saved to {args.save_config}')

    print()


if __name__ == '__main__':
    main()
