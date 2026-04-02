"""
Production HDF5 save functions.

Writes simulation output to three file types:
    resp — sparse thresholded wire signals (delta-encoded + lzf)
    seg  — 3D truth deposits (uint16 positions + float16 physics)
    corr — 3D-to-2D correspondence (CSR + delta + uint16/peak)

See DATA_FORMAT.md for the full schema.
"""

import numpy as np


_PLANE_LABELS = {0: 'U', 1: 'V', 2: 'Y'}


def _plane_label(plane_idx):
    """Plane index → short label."""
    return _PLANE_LABELS.get(plane_idx, str(plane_idx))


# =============================================================================
# File-level config writers (once per file)
# =============================================================================

def write_config_resp(f, cfg, params, recomb_model, dataset_name, file_index,
                      source_file, n_events, global_offset, threshold_adc,
                      digitization_config=None):
    """Write config group for response file."""
    if 'config' in f:
        return
    g = f.create_group('config')
    g.attrs['dataset_name'] = dataset_name
    g.attrs['file_index'] = file_index
    g.attrs['source_file'] = source_file
    g.attrs['n_events'] = n_events
    g.attrs['global_event_offset'] = global_offset
    g.attrs['num_time_steps'] = cfg.num_time_steps
    g.attrs['time_step_us'] = cfg.time_step_us
    g.attrs['pre_window_us'] = cfg.pre_window_us
    g.attrs['post_window_us'] = cfg.post_window_us
    g.attrs['electrons_per_adc'] = cfg.electrons_per_adc
    g.attrs['velocity_cm_us'] = float(params.velocity_cm_us)
    g.attrs['lifetime_us'] = float(params.lifetime_us)
    g.attrs['recombination_model'] = recomb_model
    g.attrs['include_noise'] = cfg.include_noise
    g.attrs['include_electronics'] = cfg.include_electronics
    g.attrs['include_digitize'] = cfg.include_digitize
    g.attrs['threshold_adc'] = threshold_adc
    g.attrs['n_volumes'] = cfg.n_volumes
    # Store num_wires per (volume, plane)
    max_planes = max(v.n_planes for v in cfg.volumes)
    num_wires = np.zeros((cfg.n_volumes, max_planes), dtype=np.int32)
    for v in range(cfg.n_volumes):
        for p in range(cfg.volumes[v].n_planes):
            num_wires[v, p] = cfg.volumes[v].num_wires[p]
    g.create_dataset('num_wires', data=num_wires)
    # Store volume ranges in mm: (n_volumes, 3, 2) — [vol][axis][min/max]
    vol_ranges = np.zeros((cfg.n_volumes, 3, 2), dtype=np.float32)
    for v in range(cfg.n_volumes):
        for ax in range(3):
            vol_ranges[v, ax, 0] = cfg.volumes[v].ranges_cm[ax][0] * 10  # cm→mm
            vol_ranges[v, ax, 1] = cfg.volumes[v].ranges_cm[ax][1] * 10
    g.create_dataset('volume_ranges', data=vol_ranges)
    # Store per-plane pedestals for uint16 decoding when digitized
    if digitization_config is not None:
        pedestals = np.zeros((cfg.n_volumes, max_planes), dtype=np.int32)
        for v in range(cfg.n_volumes):
            for p in range(cfg.volumes[v].n_planes):
                ptype = cfg.plane_names[v][p]
                if ptype == 'Y':
                    pedestals[v, p] = digitization_config.pedestal_collection
                else:
                    pedestals[v, p] = digitization_config.pedestal_induction
        g.create_dataset('pedestals', data=pedestals)
        g.attrs['n_bits'] = digitization_config.n_bits


def write_config_seg(f, cfg, dataset_name, file_index, source_file,
                     n_events, global_offset, group_size, gap_threshold_mm):
    """Write config group for segments file."""
    if 'config' in f:
        return
    g = f.create_group('config')
    g.attrs['dataset_name'] = dataset_name
    g.attrs['file_index'] = file_index
    g.attrs['source_file'] = source_file
    g.attrs['n_events'] = n_events
    g.attrs['global_event_offset'] = global_offset
    g.attrs['group_size'] = group_size
    g.attrs['gap_threshold_mm'] = gap_threshold_mm
    g.attrs['n_volumes'] = cfg.n_volumes
    max_planes = max(v.n_planes for v in cfg.volumes)
    num_wires = np.zeros((cfg.n_volumes, max_planes), dtype=np.int32)
    for v in range(cfg.n_volumes):
        for p in range(cfg.volumes[v].n_planes):
            num_wires[v, p] = cfg.volumes[v].num_wires[p]
    g.create_dataset('num_wires', data=num_wires)
    vol_ranges = np.zeros((cfg.n_volumes, 3, 2), dtype=np.float32)
    for v in range(cfg.n_volumes):
        for ax in range(3):
            vol_ranges[v, ax, 0] = cfg.volumes[v].ranges_cm[ax][0] * 10
            vol_ranges[v, ax, 1] = cfg.volumes[v].ranges_cm[ax][1] * 10
    g.create_dataset('volume_ranges', data=vol_ranges)


def write_config_corr(f, cfg, dataset_name, file_index, source_file,
                      n_events, global_offset, group_size, gap_threshold_mm):
    """Write config group for correspondence file."""
    if 'config' in f:
        return
    g = f.create_group('config')
    g.attrs['dataset_name'] = dataset_name
    g.attrs['file_index'] = file_index
    g.attrs['source_file'] = source_file
    g.attrs['n_events'] = n_events
    g.attrs['global_event_offset'] = global_offset
    g.attrs['group_size'] = group_size
    g.attrs['gap_threshold_mm'] = gap_threshold_mm
    g.attrs['num_time_steps'] = cfg.num_time_steps
    g.attrs['pre_window_us'] = cfg.pre_window_us
    g.attrs['post_window_us'] = cfg.post_window_us
    g.attrs['n_volumes'] = cfg.n_volumes
    max_planes = max(v.n_planes for v in cfg.volumes)
    num_wires = np.zeros((cfg.n_volumes, max_planes), dtype=np.int32)
    for v in range(cfg.n_volumes):
        for p in range(cfg.volumes[v].n_planes):
            num_wires[v, p] = cfg.volumes[v].num_wires[p]
    g.create_dataset('num_wires', data=num_wires)
    vol_ranges = np.zeros((cfg.n_volumes, 3, 2), dtype=np.float32)
    for v in range(cfg.n_volumes):
        for ax in range(3):
            vol_ranges[v, ax, 0] = cfg.volumes[v].ranges_cm[ax][0] * 10
            vol_ranges[v, ax, 1] = cfg.volumes[v].ranges_cm[ax][1] * 10
    g.create_dataset('volume_ranges', data=vol_ranges)


# =============================================================================
# Per-event save functions
# =============================================================================

def save_event_resp(f, event_key, response_signals, threshold_adc,
                    source_event_idx, deposits,
                    digitized=False):
    """Save one event's response signals (sparse, delta-encoded, gzip).

    When digitized=True, values are packed as uint16 (pedestal added back
    so values are in [0, 2^n_bits - 1]). The per-plane pedestal is read
    from /config/pedestals to decode on load.
    """
    evt = f.create_group(event_key)
    evt.attrs['source_event_idx'] = source_event_idx
    evt.attrs['n_volumes'] = len(deposits.volumes)
    for v in range(len(deposits.volumes)):
        evt.attrs[f'n_vol{v}'] = deposits.volumes[v].n_actual

    # Read per-plane pedestals if digitized
    pedestals = None
    if digitized and 'config' in f and 'pedestals' in f['config']:
        pedestals = f['config']['pedestals'][:]

    for (vol_idx, plane_idx), signal in response_signals.items():
        arr = np.asarray(signal)
        mask = np.abs(arr) >= threshold_adc
        wire_idx, time_idx = np.where(mask)
        values = arr[mask]

        vol_grp = evt.require_group(f'volume_{vol_idx}')
        g = vol_grp.create_group(_plane_label(plane_idx))

        if len(wire_idx) == 0:
            continue

        # Sort by wire then time for delta encoding
        order = np.lexsort((time_idx, wire_idx))
        wire_s = wire_idx[order].astype(np.int32)
        time_s = time_idx[order].astype(np.int32)
        values_s = values[order]

        # Delta encode (int16 — fits any detector geometry)
        delta_wire = np.diff(wire_s, prepend=wire_s[0]).astype(np.int16)
        delta_time = np.diff(time_s, prepend=time_s[0]).astype(np.int16)

        g.create_dataset('delta_wire', data=delta_wire, compression='gzip')
        g.create_dataset('delta_time', data=delta_time, compression='gzip')

        if digitized and pedestals is not None:
            ped = int(pedestals[vol_idx, plane_idx])
            values_unsigned = np.round(values_s + ped).clip(0, 65535).astype(np.uint16)
            g.create_dataset('values', data=values_unsigned, compression='gzip')
            g.attrs['pedestal'] = ped
        else:
            g.create_dataset('values', data=values_s.astype(np.float32), compression='gzip')

        g.attrs['wire_start'] = int(wire_s[0])
        g.attrs['time_start'] = int(time_s[0])
        g.attrs['n_pixels'] = len(wire_s)


def save_event_seg(f, event_key, deposits, source_event_idx, pos_step_mm=0.3):
    """Save one event's 3D truth deposits in compact format.

    Saves per-volume: positions, physics, charge, photons, qs_fractions,
    track/group IDs, group_to_track.
    """
    evt = f.create_group(event_key)
    evt.attrs['source_event_idx'] = source_event_idx
    evt.attrs['n_volumes'] = len(deposits.volumes)

    for v in range(len(deposits.volumes)):
        vol = deposits.volumes[v]
        n = vol.n_actual
        vg = evt.create_group(f'volume_{v}')
        vg.attrs['n_actual'] = n

        if n == 0:
            continue

        pos = np.asarray(vol.positions_mm[:n])

        # Positions: uint16 voxelized
        origin = pos.min(axis=0).astype(np.float32)
        pos_u16 = np.round((pos - origin) / pos_step_mm).clip(0, 65535).astype(np.uint16)
        vg.create_dataset('positions', data=pos_u16, compression='gzip')
        vg.attrs['pos_origin_x'] = float(origin[0])
        vg.attrs['pos_origin_y'] = float(origin[1])
        vg.attrs['pos_origin_z'] = float(origin[2])
        vg.attrs['pos_step_mm'] = pos_step_mm

        # Physics: float16
        vg.create_dataset('de', data=np.asarray(vol.de[:n]).astype(np.float16), compression='gzip')
        vg.create_dataset('dx', data=np.asarray(vol.dx[:n]).astype(np.float16), compression='gzip')
        vg.create_dataset('theta', data=np.asarray(vol.theta[:n]).astype(np.float16), compression='gzip')
        vg.create_dataset('phi', data=np.asarray(vol.phi[:n]).astype(np.float16), compression='gzip')
        vg.create_dataset('t0_us', data=np.asarray(vol.t0_us[:n]).astype(np.float16), compression='gzip')

        # IDs
        vg.create_dataset('track_ids', data=np.asarray(vol.track_ids[:n]), compression='gzip')
        vg.create_dataset('group_ids', data=np.asarray(vol.group_ids[:n]), compression='gzip')

        # Per-volume group_to_track
        g2t = deposits.group_to_track[v]
        if g2t is not None:
            vg.create_dataset('group_to_track', data=g2t, compression='gzip')
            vg.attrs['n_groups'] = len(g2t)

        # Simulation outputs (charge, photons as float32 — can exceed float16 range; qs as float16)
        vg.create_dataset('charge', data=np.asarray(vol.charge[:n]).astype(np.float32), compression='gzip')
        vg.create_dataset('photons', data=np.asarray(vol.photons[:n]).astype(np.float32), compression='gzip')
        vg.create_dataset('qs_fractions', data=np.asarray(vol.qs_fractions[:n]).astype(np.float16), compression='gzip')

        # Original indices (mapping back to input deposit order)
        oi = deposits.original_indices[v]
        if oi is not None:
            vg.create_dataset('original_indices', data=oi, compression='gzip')


def encode_correspondence_csr(gp_pk, gp_gid, gp_ch, gp_count, num_time_steps,
                              threshold=0.0):
    """Convert flat correspondence arrays to CSR + delta + uint16/peak format.

    Fully vectorized — no Python loop over groups.
    """
    P = int(gp_count)
    pks = np.asarray(gp_pk[:P])
    gids = np.asarray(gp_gid[:P])
    chs = np.asarray(gp_ch[:P])

    if threshold > 0:
        keep = chs > threshold
        pks, gids, chs = pks[keep], gids[keep], chs[keep]

    order = np.argsort(gids, kind='stable')
    s_gids, s_pks, s_chs = gids[order], pks[order], chs[order]
    s_wires = (s_pks // num_time_steps).astype(np.int32)
    s_times = (s_pks % num_time_steps).astype(np.int32)

    unique_gids, group_starts, group_counts = np.unique(
        s_gids, return_index=True, return_counts=True)
    G = len(unique_gids)

    # Peak charge per group via reduceat
    peak_vals = np.maximum.reduceat(s_chs, group_starts)

    # Broadcast group index to each entry
    group_labels = np.repeat(np.arange(G), group_counts)
    peak_per_entry = peak_vals[group_labels]

    # Find first peak index per group
    is_peak = s_chs == peak_per_entry
    peak_positions = np.where(is_peak)[0]
    first_peak_in_group = np.searchsorted(peak_positions, group_starts)
    peak_indices = peak_positions[first_peak_in_group]

    # Per-group arrays
    center_wires = s_wires[peak_indices].astype(np.int16)
    center_times = s_times[peak_indices].astype(np.int16)
    peak_charges = s_chs[peak_indices].astype(np.float32)

    # Per-entry: broadcast centers and compute deltas
    cw_per_entry = center_wires[group_labels]
    ct_per_entry = center_times[group_labels]
    pc_per_entry = peak_charges[group_labels]

    delta_wires = (s_wires - cw_per_entry).astype(np.int8)
    delta_times = (s_times - ct_per_entry).astype(np.int8)

    safe_pc = np.where(pc_per_entry > 0, pc_per_entry, 1.0)
    charges_u16 = np.round(s_chs / safe_pc * 65535).clip(0, 65535).astype(np.uint16)
    charges_u16 = np.where(pc_per_entry > 0, charges_u16, 0)

    return {
        'group_ids': unique_gids.astype(np.int32),
        'group_sizes': group_counts.astype(np.uint8),
        'center_wires': center_wires, 'center_times': center_times,
        'peak_charges': peak_charges,
        'delta_wires': delta_wires, 'delta_times': delta_times,
        'charges_u16': charges_u16,
    }


def save_event_corr(f, event_key, raw_track_hits, deposits,
                    source_event_idx, num_time_steps,
                    corr_threshold=0.0):
    """Save one event's correspondence in CSR format."""
    evt = f.create_group(event_key)
    evt.attrs['source_event_idx'] = source_event_idx
    evt.attrs['n_volumes'] = len(deposits.volumes)
    evt.attrs['threshold'] = corr_threshold

    for v in range(len(deposits.volumes)):
        vol_grp = evt.create_group(f'volume_{v}')

        # Per-volume group_to_track
        g2t = deposits.group_to_track[v]
        if g2t is not None:
            vol_grp.create_dataset('group_to_track', data=g2t, compression='gzip')

        # Per-plane correspondence
        for key, raw in raw_track_hits.items():
            if not isinstance(key, tuple):
                continue
            vol_idx, plane_idx = key
            if vol_idx != v:
                continue
            pk, gid, ch, count, _row_sums = raw

            csr = encode_correspondence_csr(pk, gid, ch, count, num_time_steps,
                                            threshold=corr_threshold)

            g = vol_grp.create_group(_plane_label(plane_idx))
            for k, arr in csr.items():
                g.create_dataset(k, data=arr, compression='gzip')
            g.attrs['n_groups_plane'] = len(csr['group_ids'])
            g.attrs['n_entries'] = len(csr['delta_wires'])
