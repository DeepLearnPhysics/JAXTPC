"""
Estimate max_keys without running the full simulation.

Computes group IDs, wire indices, and time indices using numpy only,
then estimates unique (wire, time, group) triplets per plane via
vectorized group-footprint bounding boxes.

Each group's footprint is (wire_span + 2*Kw_eff) × (time_span + 2*Kt_eff)
where Kw_eff/Kt_eff are per-deposit effective kernel half-widths at 2sigma
of the deposit's drift-distance-dependent diffusion.  Group min/max are
computed in one pass via np.minimum.at / np.maximum.at (no per-group loops).

Calibrated against actual merge-state counts: overestimates by ~8-17%
(safe for sizing max_keys with headroom).

Usage:
    python3 -m profiler.estimate_max_keys --data events.h5 --config config.yaml
    python3 -m profiler.estimate_max_keys --data events.h5 --config config.yaml --events 50
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from tools.geometry import generate_detector
from tools.config import create_sim_config
from tools.loader import compute_group_ids

N_SIGMA = 2.0


def _estimate_plane_keys(wire_idx, time_idx, group_ids, kw_eff, kt_eff,
                         keep, num_wires, num_time, n_groups):
    """Estimate unique keys for one plane via group bounding boxes."""
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


def estimate_keys_for_event(pstep_data, sim_config, group_size=5,
                            gap_threshold_mm=5.0):
    """Estimate max_keys for one event across all volumes and planes.

    Returns dict: {(vol_idx, plane_idx): n_unique_keys}.
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

    for v, vol_geom in enumerate(sim_config.volumes):
        ranges = vol_geom.ranges_cm
        x_range, y_range, z_range = ranges

        mask = (
            (pos_cm[:, 0] >= x_range[0]) & (pos_cm[:, 0] < x_range[1]) &
            (pos_cm[:, 1] >= y_range[0]) & (pos_cm[:, 1] < y_range[1]) &
            (pos_cm[:, 2] >= z_range[0]) & (pos_cm[:, 2] < z_range[1])
        )
        vol_idx = np.where(mask)[0]
        if len(vol_idx) == 0:
            for p in range(vol_geom.n_planes):
                results[(v, p)] = 0
            continue

        vol_pos_mm = positions_mm[vol_idx]
        vol_pos_cm = pos_cm[vol_idx]
        vol_de = de[vol_idx]
        vol_tids = track_ids[vol_idx]
        vol_t0 = t0_us[vol_idx]

        valid = vol_de > 0
        if valid.sum() == 0:
            for p in range(vol_geom.n_planes):
                results[(v, p)] = 0
            continue

        group_ids, _, n_groups = compute_group_ids(
            vol_pos_mm, vol_tids, valid,
            group_size=group_size, gap_threshold_mm=gap_threshold_mm)

        x_anode = vol_geom.x_anode_cm
        drift_dist_cm = np.abs(vol_pos_cm[:, 0] - x_anode)

        diff = vol_geom.diffusion
        velocity = diff.velocity_cm_us
        D_trans = diff.trans_cm2_us
        D_long = diff.long_cm2_us
        K_wire_max = diff.K_wire
        K_time_max = diff.K_time

        drift_time_us = np.where(velocity > 1e-9,
                                 drift_dist_cm / velocity, 0.0)

        yz_center = np.array(vol_geom.yz_center_cm, dtype=np.float32)
        yz_cm = vol_pos_cm[:, 1:3] - yz_center

        sigma_w = np.sqrt(np.maximum(2 * D_trans * drift_time_us, 0))
        sigma_t = np.sqrt(np.maximum(2 * D_long / velocity**2 * drift_time_us, 0))

        num_time = sim_config.num_time_steps

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

            sigma_w_unitless = sigma_w / wire_spacing
            sigma_t_unitless = sigma_t / sim_config.time_step_us
            kw_eff = np.minimum(K_wire_max,
                                np.maximum(1, np.ceil(N_SIGMA * sigma_w_unitless).astype(np.int32)))
            kt_eff = np.minimum(K_time_max,
                                np.maximum(1, np.ceil(N_SIGMA * sigma_t_unitless).astype(np.int32)))

            keep = valid & (group_ids > 0)
            keep &= (plane_drift_dist > 0)
            keep &= (time_idx >= 0) & (time_idx < num_time)
            keep &= (wire_idx >= 0) & (wire_idx < num_wires)

            n_unique = _estimate_plane_keys(
                wire_idx, time_idx, group_ids, kw_eff, kt_eff,
                keep, num_wires, num_time, n_groups)
            results[(v, p)] = n_unique

    return results


def estimate_max_keys(data_path, config_path, events=None,
                      total_pad=None, group_size=5, gap_threshold=5.0,
                      round_to=100_000):
    """Estimate max_keys from deposit data.

    Scans all events, computes per-volume keys and deposit counts, then
    extrapolates to total_pad using the upper-envelope keys/deps ratio
    from the largest events.

    Returns (suggestion, details_dict).
    """
    detector_config = generate_detector(config_path)
    sim_config = create_sim_config(detector_config)

    from profiler.find_optimal_pad import get_volume_ranges, count_deposits_per_volume
    volume_ranges = get_volume_ranges(detector_config)

    with h5py.File(data_path, 'r') as f:
        ds = f['pstep/lar_vol']
        n_events = ds.shape[0]
        if events is not None:
            n_events = min(n_events, events)

        all_deps = []   # per (event, volume) deposit counts
        all_keys = []   # per (event, volume) max keys across planes
        all_event_maxes = []

        for i in range(n_events):
            pstep = ds[i]
            positions_mm = np.column_stack([
                pstep['x'].astype(np.float32),
                pstep['y'].astype(np.float32),
                pstep['z'].astype(np.float32),
            ])
            counts = count_deposits_per_volume(positions_mm, volume_ranges)
            event_keys = estimate_keys_for_event(
                pstep, sim_config,
                group_size=group_size, gap_threshold_mm=gap_threshold)

            event_max = 0
            for v in range(len(counts)):
                if counts[v] == 0:
                    continue
                vol_max = max(
                    (event_keys.get((v, p), 0)
                     for p in range(sim_config.volumes[v].n_planes)),
                    default=0)
                all_deps.append(counts[v])
                all_keys.append(vol_max)
                event_max = max(event_max, vol_max)
            all_event_maxes.append(event_max)

    deps = np.array(all_deps)
    keys = np.array(all_keys)
    ratio = keys / np.maximum(deps, 1)

    # Extrapolate: use max ratio from the upper half of deposit counts
    # (large events have lower ratios due to track overlap; using the
    # upper half avoids inflating the bound with small-event outliers
    # while still capturing realistic worst-case topology)
    median_deps = np.median(deps)
    upper_mask = deps >= median_deps
    upper_max_ratio = ratio[upper_mask].max()

    if total_pad is None:
        total_pad = int(deps.max())

    extrapolated = int(upper_max_ratio * total_pad)
    suggestion = int(math.ceil(extrapolated / round_to) * round_to)

    return suggestion, {
        'n_events': n_events,
        'n_volumes': len(all_deps),
        'max_observed_keys': int(keys.max()),
        'max_observed_deps': int(deps.max()),
        'total_pad': total_pad,
        'upper_max_ratio': float(upper_max_ratio),
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
        round_to=args.round_to)

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
