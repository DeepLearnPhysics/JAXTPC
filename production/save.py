"""
Production HDF5 save functions.

Writes simulation output to three file types:
    resp — sparse thresholded wire signals (delta-encoded + lzf)
    seg  — 3D truth deposits (uint16 positions + float16 physics)
    corr — 3D-to-2D correspondence (CSR + delta + uint16/peak)

See DATA_FORMAT.md for the full schema.
"""

import numpy as np

PLANE_NAMES = {
    (0, 0): 'east_U', (0, 1): 'east_V', (0, 2): 'east_Y',
    (1, 0): 'west_U', (1, 1): 'west_V', (1, 2): 'west_Y',
}


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
    g.attrs['electrons_per_adc'] = cfg.electrons_per_adc
    g.attrs['velocity_cm_us'] = float(params.velocity_cm_us)
    g.attrs['lifetime_us'] = float(params.lifetime_us)
    g.attrs['recombination_model'] = recomb_model
    g.attrs['include_noise'] = cfg.include_noise
    g.attrs['include_electronics'] = cfg.include_electronics
    g.attrs['include_digitize'] = cfg.include_digitize
    g.attrs['threshold_adc'] = threshold_adc
    num_wires = np.array([[sg.num_wires[p] for p in range(3)]
                          for sg in cfg.side_geom], dtype=np.int32)
    g.create_dataset('num_wires', data=num_wires)
    # Store per-plane pedestals for uint16 decoding when digitized
    if digitization_config is not None:
        plane_types = cfg.plane_names  # (('U','V','Y'), ('U','V','Y'))
        pedestals = np.zeros((2, 3), dtype=np.int32)
        for s in range(2):
            for p in range(3):
                if plane_types[s][p] == 'Y':
                    pedestals[s, p] = digitization_config.pedestal_collection
                else:
                    pedestals[s, p] = digitization_config.pedestal_induction
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
    num_wires = np.array([[sg.num_wires[p] for p in range(3)]
                          for sg in cfg.side_geom], dtype=np.int32)
    g.create_dataset('num_wires', data=num_wires)


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
    num_wires = np.array([[sg.num_wires[p] for p in range(3)]
                          for sg in cfg.side_geom], dtype=np.int32)
    g.create_dataset('num_wires', data=num_wires)


# =============================================================================
# Per-event save functions
# =============================================================================

def save_event_resp(f, event_key, response_signals, threshold_adc,
                    source_event_idx, n_deposits, n_east, n_west,
                    digitized=False):
    """Save one event's response signals (sparse, delta-encoded, gzip).

    When digitized=True, values are packed as uint16 (pedestal added back
    so values are in [0, 2^n_bits - 1]). The per-plane pedestal is read
    from /config/pedestals to decode on load.
    """
    evt = f.create_group(event_key)
    evt.attrs['source_event_idx'] = source_event_idx
    evt.attrs['n_deposits'] = n_deposits
    evt.attrs['n_east'] = n_east
    evt.attrs['n_west'] = n_west

    # Read per-plane pedestals if digitized
    pedestals = None
    if digitized and 'config' in f and 'pedestals' in f['config']:
        pedestals = f['config']['pedestals'][:]

    for (side_idx, plane_idx), signal in response_signals.items():
        arr = np.asarray(signal)
        mask = np.abs(arr) >= threshold_adc
        wire_idx, time_idx = np.where(mask)
        values = arr[mask]

        name = PLANE_NAMES[(side_idx, plane_idx)]
        g = evt.create_group(name)

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
            # Add pedestal back: signed ADC → unsigned [0, 4095] → uint16
            ped = int(pedestals[side_idx, plane_idx])
            values_unsigned = np.round(values_s + ped).clip(0, 65535).astype(np.uint16)
            g.create_dataset('values', data=values_unsigned, compression='gzip')
            g.attrs['pedestal'] = ped
        else:
            g.create_dataset('values', data=values_s.astype(np.float32), compression='gzip')

        g.attrs['wire_start'] = int(wire_s[0])
        g.attrs['time_start'] = int(time_s[0])
        g.attrs['n_pixels'] = len(wire_s)


def save_event_seg(f, event_key, deposit_data, group_to_track,
                   source_event_idx, n_east, n_west, qs_fractions=None,
                   pos_step_mm=0.3):
    """Save one event's 3D truth deposits in compact format."""
    evt = f.create_group(event_key)
    pos = np.asarray(deposit_data.positions_mm)
    n = pos.shape[0]
    evt.attrs['source_event_idx'] = source_event_idx
    evt.attrs['n_deposits'] = n
    evt.attrs['n_east'] = n_east
    evt.attrs['n_west'] = n_west
    evt.attrs['n_groups'] = len(group_to_track)

    # Positions: uint16 voxelized
    origin = pos.min(axis=0).astype(np.float32)
    pos_u16 = np.round((pos - origin) / pos_step_mm).clip(0, 65535).astype(np.uint16)
    evt.create_dataset('positions', data=pos_u16, compression='gzip')
    evt.attrs['pos_origin_x'] = float(origin[0])
    evt.attrs['pos_origin_y'] = float(origin[1])
    evt.attrs['pos_origin_z'] = float(origin[2])
    evt.attrs['pos_step_mm'] = pos_step_mm

    # Physics: float16
    evt.create_dataset('de', data=np.asarray(deposit_data.de).astype(np.float16), compression='gzip')
    evt.create_dataset('dx', data=np.asarray(deposit_data.dx).astype(np.float16), compression='gzip')
    evt.create_dataset('theta', data=np.asarray(deposit_data.theta).astype(np.float16), compression='gzip')
    evt.create_dataset('phi', data=np.asarray(deposit_data.phi).astype(np.float16), compression='gzip')

    # IDs: int32
    evt.create_dataset('track_ids', data=np.asarray(deposit_data.track_ids), compression='gzip')
    evt.create_dataset('group_ids', data=np.asarray(deposit_data.group_ids), compression='gzip')
    evt.create_dataset('group_to_track', data=group_to_track, compression='gzip')

    if qs_fractions is not None:
        evt.create_dataset('qs_fractions', data=qs_fractions, compression='gzip')


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


def save_event_corr(f, event_key, raw_track_hits, group_to_track,
                    source_event_idx, n_deposits, n_groups, num_time_steps,
                    corr_threshold=0.0):
    """Save one event's correspondence in CSR format."""
    evt = f.create_group(event_key)
    evt.attrs['source_event_idx'] = source_event_idx
    evt.attrs['n_deposits'] = n_deposits
    evt.attrs['n_groups'] = n_groups
    evt.attrs['threshold'] = corr_threshold

    evt.create_dataset('group_to_track', data=group_to_track, compression='gzip')

    for key, raw in raw_track_hits.items():
        if not isinstance(key, tuple):
            continue
        side_idx, plane_idx = key
        pk, gid, ch, count, _row_sums = raw

        csr = encode_correspondence_csr(pk, gid, ch, count, num_time_steps,
                                        threshold=corr_threshold)

        name = PLANE_NAMES[(side_idx, plane_idx)]
        g = evt.create_group(name)
        for k, arr in csr.items():
            g.create_dataset(k, data=arr, compression='gzip')
        g.attrs['n_groups_plane'] = len(csr['group_ids'])
        g.attrs['n_entries'] = len(csr['delta_wires'])
