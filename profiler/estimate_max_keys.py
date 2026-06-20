"""
Estimate max_keys without running the full simulation.

Computes group IDs and drift distances using numpy, then estimates
the number of merge-state entries per group from the actual kernel
element counts at each diffusion level.

For each s level, counts how many kernel output elements exceed a
threshold fraction of peak (default 0.5%).  Each group's footprint
is approximately one kernel's worth of elements (deposits within a
group overlap heavily).  Summing across groups gives the total keys.

Wire: CDF diffusion kernel evaluated at output resolution.
Pixel: actual response kernel evaluated via apply_pixel_diffusion_response.

Usage:
    python3 -m profiler.estimate_max_keys --data events.h5 --config config.yaml
    python3 -m profiler.estimate_max_keys --data f1.h5 f2.h5 --config config.yaml --total-pad 900000
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from scipy.special import erf

from tools.geometry import generate_detector
from tools.config import create_sim_config
from tools.loader import compute_group_ids

THRESH_FRAC = 0.005


def _cdf_1d(mu, sigma, n):
    offsets = np.arange(-(n // 2), n // 2 + 1)
    if sigma < 1e-6:
        r = np.zeros(n)
        r[n // 2] = 1.0
        return r
    lo = offsets - 0.5 - mu
    hi = offsets + 0.5 - mu
    return 0.5 * (erf(hi / (sigma * np.sqrt(2))) - erf(lo / (sigma * np.sqrt(2))))


def build_wire_element_table(diffusion, num_s=16, thresh_frac=THRESH_FRAC):
    """Build per-s element count from CDF diffusion kernel.

    Returns element_table shape (num_s,) int32: number of kernel
    elements above thresh_frac * peak at each diffusion level.
    """
    max_sigma_w = diffusion.max_sigma_trans_unitless
    max_sigma_t = diffusion.max_sigma_long_unitless
    K_wire = diffusion.K_wire
    K_time = diffusion.K_time

    table = np.zeros(num_s, dtype=np.int32)
    for i in range(num_s):
        s = i / max(num_s - 1, 1)
        sw = max(max_sigma_w * np.sqrt(s), 1e-3)
        st = max(max_sigma_t * np.sqrt(s), 1e-3)
        wf = _cdf_1d(0, sw, 2 * K_wire + 1)
        tf = _cdf_1d(0, st, 2 * K_time + 1)
        k2d = wf[:, None] * tf[None, :]
        peak = k2d.max()
        table[i] = int(np.sum(k2d > thresh_frac * peak))
    return table


def build_pixel_element_table(sim_config, vol_geom, num_s=16,
                              thresh_frac=THRESH_FRAC,
                              pixel_kernel_path=None):
    """Build per-s element count from the actual pixel response kernel.

    Evaluates apply_pixel_diffusion_response at each s level with zero
    offsets and counts output elements above threshold.

    Returns element_table shape (num_s,) int32.
    """
    import jax.numpy as jnp
    from tools.kernels import (load_pixel_response_kernel,
                               apply_pixel_diffusion_response)

    diff = vol_geom.diffusion
    if pixel_kernel_path is None:
        pixel_kernel_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'pixel_response.npz')

    pk = load_pixel_response_kernel(
        pixel_kernel_path, num_s=num_s,
        time_spacing=sim_config.time_step_us,
        pixel_pitch_cm=vol_geom.pixel_pitch_cm,
        max_sigma_trans_unitless=diff.max_sigma_trans_unitless,
        max_sigma_long_unitless=diff.max_sigma_long_unitless)

    s_batch = jnp.linspace(0.0, 1.0, num_s)
    zero = jnp.zeros(num_s, dtype=jnp.float32)

    output = apply_pixel_diffusion_response(
        pk.DKernel, s_batch, zero, zero, zero,
        pk.pixel_spacing, pk.kernel_py, pk.kernel_pz, pk.rebin_factor)

    output_np = np.array(output)
    table = np.zeros(num_s, dtype=np.int32)
    for i in range(num_s):
        k = output_np[i]
        peak = np.abs(k).max()
        if peak > 1e-12:
            table[i] = int(np.sum(np.abs(k) > thresh_frac * peak))
    return table


def _estimate_keys_element_count(group_ids, s_idx, keep, element_table,
                                 n_groups):
    """Estimate keys by summing each deposit's kernel footprint.

    Ground truth (track_hits.py box path): a "key" is a per-group box cell
    whose *accumulated* |signal| clears inter_thresh — i.e. the UNION of each
    group's deposits' kernel footprints.

    The earlier model summed ONE footprint per group (element count at the
    group's max-s level), implicitly assuming all deposits in a group land on
    the same cells. That perfect-overlap assumption under-counts badly when a
    group's deposits are spatially spread (coarse Geant4 steps): validated at
    ~3x median under-count, up to ~8x, on the 300um dataset.

    Summing the per-DEPOSIT footprint instead is the no-within-group-overlap
    bound: exact across groups (different groups never share a key), and only
    slightly over *within* a group (~group_size=5 deposits, partial overlap).
    That converts a large, unsafe under-count into a small, data-adaptive
    over-estimate that automatically tracks step size / diffusion. Same cost as
    before (one masked fancy-index sum).
    """
    mask = keep & (group_ids > 0)
    return int(element_table[s_idx[mask]].sum())


# ── charge-aware footprint (matches box path: cells where |signal|>inter_thresh)

def build_pixel_value_table(sim_config, vol_geom, num_s=16,
                            pixel_kernel_path=None):
    """Sorted-DESC |kernel value| per s-level (ADC per electron) for pixel.

    Same kernel as build_pixel_element_table, but keeps the full value
    distribution so the estimate can count, per deposit, how many cells clear
    the *absolute* inter_thresh given that deposit's intensity. Returns a list
    of length num_s of 1-D float arrays sorted descending.
    """
    import jax.numpy as jnp
    from tools.kernels import (load_pixel_response_kernel,
                               apply_pixel_diffusion_response)
    diff = vol_geom.diffusion
    if pixel_kernel_path is None:
        pixel_kernel_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'pixel_response.npz')
    pk = load_pixel_response_kernel(
        pixel_kernel_path, num_s=num_s, time_spacing=sim_config.time_step_us,
        pixel_pitch_cm=vol_geom.pixel_pitch_cm,
        max_sigma_trans_unitless=diff.max_sigma_trans_unitless,
        max_sigma_long_unitless=diff.max_sigma_long_unitless)
    s_batch = jnp.linspace(0.0, 1.0, num_s)
    zero = jnp.zeros(num_s, dtype=jnp.float32)
    output = apply_pixel_diffusion_response(
        pk.DKernel, s_batch, zero, zero, zero,
        pk.pixel_spacing, pk.kernel_py, pk.kernel_pz, pk.rebin_factor)
    out_np = np.array(output)
    return [np.sort(np.abs(out_np[i]).ravel())[::-1] for i in range(num_s)]


def build_wire_value_table(diffusion, num_s=16):
    """Sorted-DESC |kernel value| per s-level for wire (CDF diffusion kernel)."""
    K_wire, K_time = diffusion.K_wire, diffusion.K_time
    msw, mst = diffusion.max_sigma_trans_unitless, diffusion.max_sigma_long_unitless
    out = []
    for i in range(num_s):
        s = i / max(num_s - 1, 1)
        wf = _cdf_1d(0, max(msw * np.sqrt(s), 1e-3), 2 * K_wire + 1)
        tf = _cdf_1d(0, max(mst * np.sqrt(s), 1e-3), 2 * K_time + 1)
        out.append(np.sort(np.abs(wf[:, None] * tf[None, :]).ravel())[::-1])
    return out


def build_charge_model(detector_config):
    """Charge model: de -> surviving ionization electrons (matches the sim).

    Replicates tools/recombination (Modified Box / EMB) so the estimate's
    per-deposit intensity tracks the real charge that drives how far down the
    kernel tail clears the absolute inter_thresh. detector_config is the parsed
    YAML dict. NOTE: pstep dx is in mm (like positions); converted to cm here.
    """
    sim = detector_config.get('simulation', {})
    med = detector_config.get('medium', {}).get('properties', {})
    recc = sim.get('charge_recombination', {})
    rec = recc.get('recomb_parameters', {})
    model = recc.get('model', 'emb')
    return {
        'model': model,
        # EMB uses alpha_emb/beta_90/R_anisotropy; Modified Box uses alpha/beta.
        'alpha': float(rec.get('alpha_emb', 0.904) if model == 'emb'
                       else rec.get('alpha', 0.93)),
        'beta90': float(rec.get('beta_90', 0.204)),
        'beta': float(rec.get('beta', 0.212)),
        'R_aniso': float(rec.get('R_anisotropy', 1.25)),
        'density': float(med.get('density', 1.396)),
        'w_ion_mev': float(med.get('ionization_energy', 23.6)) * 1e-6,
        'field_kVcm': float(detector_config.get('electric_field', {})
                            .get('field_strength', 500.0)) / 1000.0,
        'lifetime_us': float(sim.get('drift', {})
                             .get('electron_lifetime', 10.0)) * 1000.0,
    }


def _deposit_intensity(de, dx_mm, drift_time_us, phi_drift, cm):
    """Per-deposit intensity = surviving electrons x drift attenuation.

    Mirrors tools/recombination.compute_quanta (validated: Q p50 1634 vs the
    sim's 1633). dx is mm -> cm; EMB adds the angular beta_eff(phi_drift).
    """
    dx_cm = dx_mm / 10.0
    de_dx = np.where(dx_cm > 0, de / np.maximum(dx_cm, 1e-10), 0.0)
    if cm['model'] == 'emb':
        ang = np.sqrt(np.sin(phi_drift) ** 2
                      + np.cos(phi_drift) ** 2 / cm['R_aniso'] ** 2)
        eff_beta = cm['beta90'] / np.maximum(ang, 1e-10)
    else:
        eff_beta = cm['beta']
    xi = (eff_beta / cm['density']) * de_dx / max(cm['field_kVcm'], 1e-10)
    R = np.where(xi > 1e-10,
                 np.log(np.maximum(cm['alpha'] + xi, 1.0)) / np.maximum(xi, 1e-10),
                 0.0)
    Q = (np.maximum(de, 0.0) / cm['w_ion_mev']) * R
    atten = np.exp(-np.maximum(drift_time_us, 0.0) / cm['lifetime_us'])
    return Q * atten


def _estimate_keys_charge_aware(s_idx, intensity, keep, value_table, inter_thresh):
    """Σ over kept deposits of #{kernel cells : intensity_i*|value| > inter_thresh}.

    Per deposit, the footprint is the number of kernel cells whose signal clears
    the *absolute* box threshold for that deposit's intensity (value_table[s] is
    sorted DESC, so the count above inter_thresh/intensity is a searchsorted on
    the reversed array). A bright deposit's kernel tail clears the threshold,
    which a geometry-only (0.5%-of-peak) count misses.

    This sum over-counts the within-group UNION by a per-readout overlap factor
    (pixel ~1.5x in the kernel tails, wire ~3.8x in the cores from the 1-D wire
    projection). The estimate compensates with a calibrated threshold multiplier
    c* (pixel: caller passes inter_thresh = 2.5 x box_inter_thresh -> ~1.0x) or a
    division factor (wire), set in setup_production / scan_values.
    """
    m = keep & (intensity > 0)
    s_m, inten_m = s_idx[m], intensity[m]
    total = 0
    for s in np.unique(s_m):
        vals_asc = value_table[s][::-1]
        if vals_asc.size == 0:
            continue
        thr = inter_thresh / inten_m[s_m == s]
        total += int((vals_asc.size
                      - np.searchsorted(vals_asc, thr, side='right')).sum())
    return total


def estimate_keys_for_event(pstep_data, sim_config, element_tables,
                            group_size=5, gap_threshold_mm=5.0,
                            value_tables=None, charge_model=None,
                            inter_thresh=1.0):
    """Estimate max_keys for one event across all volumes and planes.

    Parameters
    ----------
    element_tables : dict
        {vol_idx: element_table} where element_table is (num_s,) int32.

    Returns (results, vol_deps) where results = {(vol, plane): n_keys}
    and vol_deps = {vol: n_deposits}.
    """
    positions_mm = np.column_stack([
        pstep_data['x'].astype(np.float32),
        pstep_data['y'].astype(np.float32),
        pstep_data['z'].astype(np.float32),
    ])
    de = pstep_data['de'].astype(np.float32)
    track_ids = pstep_data['track_id'].astype(np.int32)
    n = len(de)
    dx = (pstep_data['dx'].astype(np.float32) if 'dx' in pstep_data.dtype.names
          else np.ones(n, np.float32))
    # phi_drift = angle between track and the E-field (||x, no SCE here):
    # arccos(|sin(theta)cos(phi)|). Used by the EMB recombination charge model.
    if {'theta', 'phi'} <= set(pstep_data.dtype.names):
        _th = pstep_data['theta'].astype(np.float32)
        _ph = pstep_data['phi'].astype(np.float32)
        phi_drift_all = np.arccos(
            np.clip(np.abs(np.sin(_th) * np.cos(_ph)), 0.0, 1.0))
    else:
        phi_drift_all = np.zeros(n, np.float32)

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
            positions_mm[vol_idx], vol_tids, valid,
            group_size=group_size, gap_threshold_mm=gap_threshold_mm)

        drift_dist_cm = np.abs(vol_pos_cm[:, 0] - vol_geom.x_anode_cm)
        velocity = vol_geom.diffusion.velocity_cm_us
        num_time = sim_config.num_time_steps
        num_s = (len(value_tables[v]) if value_tables is not None
                 else len(element_tables[v]))

        s_vals = np.clip(np.sqrt(drift_dist_cm / vol_geom.max_drift_cm), 0, 1)
        s_idx = np.clip((s_vals * (num_s - 1)).astype(int), 0, num_s - 1)

        # Time index for readout window filter
        drift_time = np.where(velocity > 1e-9, drift_dist_cm / velocity, 0.0)
        intensity = (_deposit_intensity(vol_de, dx[vol_idx], drift_time,
                                        phi_drift_all[vol_idx], charge_model)
                     if charge_model is not None else None)
        tick_us = drift_time + vol_t0 + sim_config.pre_window_us
        time_idx = np.floor(tick_us / sim_config.time_step_us).astype(np.int32)

        # Base keep: valid, has group, positive drift, in readout window
        keep = valid & (group_ids > 0) & (drift_dist_cm > 0)
        keep &= (time_idx >= 0) & (time_idx < num_time)

        if vol_geom.readout_type == 'pixel':
            origins = np.array(vol_geom.pixel_origins_cm, dtype=np.float32)
            # vol_pos_cm is GLOBAL here, but pixel_origins_cm is in the volume-LOCAL
            # frame (= -half_extent), matching the sim. Localize the positions
            # (subtract yz_center) before indexing, exactly like the wire branch
            # below. Without this, off-center pixel volumes mark deposits
            # out-of-bounds and undercount max_keys.
            yz_center = np.array(vol_geom.yz_center_cm, dtype=np.float32)
            pitch = vol_geom.pixel_pitch_cm
            py_idx = np.floor((vol_pos_cm[:, 1] - yz_center[0] - origins[0]) / pitch).astype(np.int32)
            pz_idx = np.floor((vol_pos_cm[:, 2] - yz_center[1] - origins[1]) / pitch).astype(np.int32)
            num_py, num_pz = vol_geom.pixel_shape
            keep &= (py_idx >= 0) & (py_idx < num_py)
            keep &= (pz_idx >= 0) & (pz_idx < num_pz)

            if value_tables is not None and intensity is not None:
                # per-deposit charge-aware footprint sum = safe over-bound
                # (~1.5-1.8x; the within-group union dedup is 3D/anisotropic
                # and not captured by a simple tube).
                n_keys = _estimate_keys_charge_aware(
                    s_idx, intensity, keep, value_tables[v], inter_thresh)
            else:
                n_keys = _estimate_keys_element_count(
                    group_ids, s_idx, keep, element_tables[v], n_groups)
            results[(v, 0)] = n_keys
        else:
            yz_center = np.array(vol_geom.yz_center_cm, dtype=np.float32)
            yz_cm = vol_pos_cm[:, 1:3] - yz_center

            for p in range(vol_geom.n_planes):
                wire_spacing = vol_geom.wire_spacings_cm[p]
                angle_rad = vol_geom.angles_rad[p]
                index_offset = vol_geom.index_offsets[p]
                num_wires = vol_geom.num_wires[p]
                plane_dist = vol_geom.plane_distances_cm[p]

                plane_drift_dist = drift_dist_cm - plane_dist

                r_prime = (yz_cm[:, 0] * np.sin(angle_rad)
                           + yz_cm[:, 1] * np.cos(angle_rad))
                wire_idx = np.round(r_prime / wire_spacing).astype(np.int32) + index_offset

                plane_keep = keep & (plane_drift_dist > 0)
                plane_keep &= (wire_idx >= 0) & (wire_idx < num_wires)

                if value_tables is not None and intensity is not None:
                    n_keys = _estimate_keys_charge_aware(
                        s_idx, intensity, plane_keep, value_tables[v], inter_thresh)
                else:
                    n_keys = _estimate_keys_element_count(
                        group_ids, s_idx, plane_keep, element_tables[v], n_groups)
                results[(v, p)] = n_keys

    return results, vol_deps


def _resolve_data_paths(data_arg):
    """Resolve a single path, list of paths, or directory to HDF5 files."""
    if isinstance(data_arg, str):
        data_arg = [data_arg]
    paths = []
    for p in data_arg:
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, '*.h5'))))
        else:
            paths.append(p)
    return paths


_worker_state = {}


def _init_keys_worker(config_path, element_tables_dict,
                      group_size, gap_threshold_mm):
    """Worker process initializer: build sim_config from the YAML path
    in this process so we don't pickle JAX-bearing NamedTuples."""
    detector_config = generate_detector(config_path)
    _worker_state['sim_config'] = create_sim_config(detector_config)
    _worker_state['element_tables'] = element_tables_dict
    _worker_state['group_size'] = group_size
    _worker_state['gap_threshold_mm'] = gap_threshold_mm


def _scan_file_for_keys(args):
    """Worker: scan one HDF5 file, return per-event (deps, keys, event_max)."""
    fpath, max_events = args
    sim_config = _worker_state['sim_config']
    element_tables = _worker_state['element_tables']
    group_size = _worker_state['group_size']
    gap_threshold_mm = _worker_state['gap_threshold_mm']

    file_deps = []
    file_keys = []
    file_event_maxes = []
    with h5py.File(fpath, 'r') as f:
        ds = f['pstep/lar_vol']
        n_events = ds.shape[0]
        if max_events is not None:
            n_events = min(n_events, max_events)
        for i in range(n_events):
            pstep = ds[i]
            event_keys, event_vol_deps = estimate_keys_for_event(
                pstep, sim_config, element_tables,
                group_size=group_size, gap_threshold_mm=gap_threshold_mm)

            event_max = 0
            for v_idx, n_dep in event_vol_deps.items():
                if n_dep == 0:
                    continue
                vol = sim_config.volumes[v_idx]
                n_planes = vol.n_planes if vol.readout_type == 'wire' else 1
                vol_max = max(
                    (event_keys.get((v_idx, p), 0) for p in range(n_planes)),
                    default=0)
                file_deps.append(n_dep)
                file_keys.append(vol_max)
                event_max = max(event_max, vol_max)
            file_event_maxes.append(event_max)
    return fpath, n_events, file_deps, file_keys, file_event_maxes


def estimate_max_keys(data_paths, config_path, events_per_file=None,
                      total_pad=None, group_size=5, gap_threshold=5.0,
                      round_to=100_000, pixel_kernel_path=None,
                      thresh_frac=THRESH_FRAC, n_workers=1, headroom=1.5):
    """Estimate max_keys from deposit data across one or more files.

    The suggestion is sized off the largest observed per-(event, volume)
    key requirement times a headroom factor -- i.e. max(keys_i) * headroom.

    An earlier version multiplied the worst-case keys/deposit ratio by
    total_pad (`upper_max_ratio * total_pad`). That overshoots ~2.5x: the
    peak ratio comes from sparse, low-deposit volumes while total_pad comes
    from the densest, and keys grow sublinearly with deposits, so the two
    maxima never co-occur in one event. The extrapolation is still computed
    and returned in the details dict as a conservative ceiling for reference.

    Returns (suggestion, details_dict).
    """
    h5_files = _resolve_data_paths(data_paths)

    detector_config = generate_detector(config_path)
    sim_config = create_sim_config(detector_config)
    num_s = 16

    # Build element count tables per volume (once, on main process)
    element_tables = {}
    for v, vol_geom in enumerate(sim_config.volumes):
        if vol_geom.readout_type == 'pixel':
            element_tables[v] = build_pixel_element_table(
                sim_config, vol_geom, num_s=num_s,
                thresh_frac=thresh_frac,
                pixel_kernel_path=pixel_kernel_path)
        else:
            element_tables[v] = build_wire_element_table(
                vol_geom.diffusion, num_s=num_s, thresh_frac=thresh_frac)

    for v, tbl in element_tables.items():
        rtype = sim_config.volumes[v].readout_type
        print(f'  Vol {v} ({rtype}) element table: {tbl.tolist()}', flush=True)

    all_deps = []
    all_keys = []
    all_event_maxes = []
    total_scanned = 0

    args_list = [(fp, events_per_file) for fp in h5_files]

    if n_workers <= 1 or len(h5_files) <= 1:
        # Serial path: still go through the worker function for consistency.
        _init_keys_worker(config_path, element_tables, group_size, gap_threshold)
        for i, a in enumerate(args_list, 1):
            fpath, n_events, fd, fk, fm = _scan_file_for_keys(a)
            all_deps.extend(fd)
            all_keys.extend(fk)
            all_event_maxes.extend(fm)
            total_scanned += n_events
            if len(h5_files) > 1:
                print(f'  [{i}/{len(h5_files)}] {os.path.basename(fpath)}: '
                      f'{n_events} events scanned', flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx,
            initializer=_init_keys_worker,
            initargs=(config_path, element_tables, group_size, gap_threshold),
        ) as ex:
            futures = [ex.submit(_scan_file_for_keys, a) for a in args_list]
            for done, fut in enumerate(as_completed(futures), 1):
                fpath, n_events, fd, fk, fm = fut.result()
                all_deps.extend(fd)
                all_keys.extend(fk)
                all_event_maxes.extend(fm)
                total_scanned += n_events
                print(f'  [{done}/{len(h5_files)}] {os.path.basename(fpath)}: '
                      f'{n_events} events scanned', flush=True)

    deps = np.array(all_deps)
    keys = np.array(all_keys)
    ratio = keys / np.maximum(deps, 1)

    median_deps = np.median(deps)
    upper_mask = deps >= median_deps
    upper_max_ratio = float(ratio[upper_mask].max())

    if total_pad is None:
        total_pad = int(deps.max())

    # Reference-only legacy extrapolation (conservative ceiling).
    extrapolated = int(upper_max_ratio * total_pad)

    # Primary suggestion: largest observed per-event key requirement, with
    # headroom for events larger than those scanned.
    max_observed_keys = int(keys.max())
    raw_suggestion = int(max_observed_keys * headroom)
    suggestion = int(math.ceil(raw_suggestion / round_to) * round_to)

    return suggestion, {
        'n_events': total_scanned,
        'n_files': len(h5_files),
        'max_observed_keys': max_observed_keys,
        'max_observed_deps': int(deps.max()),
        'total_pad': total_pad,
        'headroom': headroom,
        'headroom_suggestion': suggestion,
        'upper_max_ratio': upper_max_ratio,
        'extrapolated': extrapolated,
        'all_event_maxes': np.array(all_event_maxes),
        'all_deps': deps,
        'all_keys': keys,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Estimate max_keys from deposit geometry (no simulation)')
    parser.add_argument('--data', required=True, nargs='+',
                        help='Input HDF5 file(s) or directory')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--events', type=int, default=None,
                        help='Max events per file (default: all)')
    parser.add_argument('--total-pad', type=int, default=None,
                        help='Total pad to extrapolate to (default: from data)')
    parser.add_argument('--group-size', type=int, default=5)
    parser.add_argument('--gap-threshold', type=float, default=5.0)
    parser.add_argument('--round-to', type=int, default=100_000)
    parser.add_argument('--headroom', type=float, default=1.5,
                        help='Multiply max observed keys by this factor '
                             '(default: 1.5)')
    parser.add_argument('--pixel-kernel', default=None,
                        help='Path to pixel response NPZ (pixel readout only)')
    parser.add_argument('--thresh-frac', type=float, default=THRESH_FRAC,
                        help=f'Kernel threshold as fraction of peak (default: {THRESH_FRAC})')
    parser.add_argument('--save-config', default=None,
                        help='Save max_keys to production config YAML')
    parser.add_argument('--tag', default=None,
                        help='Tag for figure filenames (default: config name)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker processes for file scanning '
                             '(default: 1 = serial)')

    args = parser.parse_args()

    h5_files = _resolve_data_paths(args.data)
    print('=' * 70)
    print(' JAXTPC — Estimate max_keys (no simulation)')
    print('=' * 70)
    print(f'  Files:     {len(h5_files)}')
    for p in h5_files:
        print(f'    {p}')
    print(f'  Config:    {args.config}')
    print(f'  Threshold: {args.thresh_frac*100:.1f}% of peak')

    suggestion, info = estimate_max_keys(
        args.data, args.config,
        events_per_file=args.events,
        total_pad=args.total_pad,
        group_size=args.group_size,
        gap_threshold=args.gap_threshold,
        round_to=args.round_to,
        pixel_kernel_path=args.pixel_kernel,
        thresh_frac=args.thresh_frac,
        n_workers=args.workers,
        headroom=args.headroom)

    maxes = info['all_event_maxes']
    pcts = np.percentile(maxes, [50, 90, 99, 99.9, 100])
    deps = info['all_deps']
    keys = info['all_keys']
    ratio = keys / np.maximum(deps, 1)

    print(f'  Events:    {info["n_events"]} across {info["n_files"]} file(s)')
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

    print(f'\n  Suggestion (max observed keys x {args.headroom} headroom):')
    print(f'    Max observed keys = {info["max_observed_keys"]:,}')
    print(f'    Rounded:            {suggestion:,}')
    print(f'    --max-keys {suggestion}')
    print(f'\n  (Reference) legacy upper_max_ratio x total_pad'
          f'={info["total_pad"]:,}: {info["extrapolated"]:,}')

    if args.save_config:
        from profiler.production_config import update_config
        update_config(args.save_config, {'max_keys': suggestion},
                      detector_config_path=args.config)
        print(f'\n  Saved to {args.save_config}')

    # Figures
    from profiler.plots import (plot_keys_vs_deposits, plot_keys_ratio,
                                plot_keys_distribution)
    tag = args.tag or os.path.splitext(os.path.basename(args.config))[0]
    print()
    plot_keys_vs_deposits(deps, keys, info['total_pad'], suggestion,
                          info['upper_max_ratio'], tag=tag)
    plot_keys_ratio(deps, keys, tag=tag)
    plot_keys_distribution(maxes, suggestion, tag=tag)
    print()


if __name__ == '__main__':
    main()
