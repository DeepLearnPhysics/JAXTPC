"""
HDF5 I/O for truth-level segment correspondence data.

Saves the 3D→2D correspondence map from the group-based merge pipeline:
  - Per-event: group assignments and deposit metadata
  - Per-plane: M_reduced (group correspondence) and per-deposit row sums

The deposit 3D information (positions, dE, dx, angles) is stored in a
separate segment file, referenced by array index.

Schema:
    /config/
        attrs: group_size, gap_threshold_mm, num_time_steps,
               max_time, n_groups, n_deposits
    /event_{idx}/
        groups/
            group_ids          (N,) int32     deposit → group
            group_to_track     (G,) int32     group → track
        correspondence/{plane}/
            pixel_keys         (M,) int32     encoded wire*max_time+time
            group_ids          (M,) int32     group ID per entry
            charges            (M,) float32   charge per (pixel, group)
            row_sum_indices    (K,) int32     deposit indices with signal
            row_sum_values     (K,) float32   per-deposit total diffused charge
            attrs: count, n_row_sums

Derivation utilities:
    derive_track_hits()   — dominant track per pixel from correspondence
    derive_pixel_groups() — all groups contributing to a specific pixel
    derive_group_pixels() — all pixels a specific group contributes to

Segment 3D file (separate):
    /event_{idx}/
        positions_mm       (N, 3) float32
        de                 (N,) float32
        dx                 (N,) float32
        track_ids          (N,) int32
        theta              (N,) float32
        phi                (N,) float32
        attrs: n_deposits
"""

import h5py
import numpy as np
import os

_PLANE_NAMES = {
    (0, 0): 'east_U', (0, 1): 'east_V', (0, 2): 'east_Y',
    (1, 0): 'west_U', (1, 1): 'west_V', (1, 2): 'west_Y',
}
_NAME_TO_KEY = {v: k for k, v in _PLANE_NAMES.items()}


# =========================================================================
# Save / Load
# =========================================================================

def save_truth_event(filepath, event_idx, track_hits, group_data,
                     detector_config, group_size=5, gap_threshold_mm=5.0,
                     threshold=0.0):
    """
    Save one event's truth/correspondence data to HDF5.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file (created or appended).
    event_idx : int
        Event index.
    track_hits : dict
        Finalized track hits from DetectorSimulator.finalize_track_hits().
        Each value has 'group_correspondence' and 'row_sums'.
    group_data : dict
        From DetectorSimulator._last_group_data after process_event().
        Contains 'east_group_ids', 'west_group_ids', 'group_to_track', 'n_groups'.
    detector_config : dict
        Detector configuration from generate_detector().
    group_size : int
        Group size used (stored as metadata).
    gap_threshold_mm : float
        Gap threshold used (stored as metadata).
    threshold : float
        If > 0, prune correspondence entries below this charge before saving.
        Reduces file size by dropping entries below the detection threshold.
        Default 0.0 (save all entries that passed inter_thresh during merge).
    """
    mode = 'a' if os.path.exists(filepath) else 'w'

    # Reconstruct full group_ids from east/west padded arrays + counts
    east_gids = np.asarray(group_data['east_group_ids'])
    west_gids = np.asarray(group_data['west_group_ids'])
    g2t = np.asarray(group_data['group_to_track'])

    with h5py.File(filepath, mode) as f:
        # Write config once
        if 'config' not in f:
            cfg = f.create_group('config')
            cfg.attrs['group_size'] = group_size
            cfg.attrs['gap_threshold_mm'] = gap_threshold_mm
            cfg.attrs['threshold'] = float(threshold)
            cfg.attrs['num_time_steps'] = int(detector_config['num_time_steps'])
            cfg.attrs['max_time'] = int(detector_config['num_time_steps'])
            cfg.create_dataset('num_wires_actual',
                               data=np.array(detector_config['num_wires_actual'], dtype=np.int32))
            cfg.create_dataset('min_wire_indices',
                               data=np.array(detector_config['min_wire_indices_abs'], dtype=np.int32))

        event_key = f'event_{event_idx}'
        if event_key in f:
            del f[event_key]
        evt = f.create_group(event_key)

        # --- Group assignments ---
        grp = evt.create_group('groups')
        grp.create_dataset('east_group_ids', data=east_gids, compression='gzip')
        grp.create_dataset('west_group_ids', data=west_gids, compression='gzip')
        grp.create_dataset('group_to_track', data=g2t, compression='gzip')
        grp.attrs['n_groups'] = int(group_data['n_groups'])

        # --- Per-plane correspondence ---
        prov_grp = evt.create_group('correspondence')
        for plane_key, result in track_hits.items():
            gp = result.get('group_correspondence')
            if gp is None:
                continue

            gp_pk, gp_gid, gp_ch, gp_count = gp
            count = int(gp_count)
            pk_np = np.asarray(gp_pk[:count])
            gid_np = np.asarray(gp_gid[:count])
            ch_np = np.asarray(gp_ch[:count])

            # Apply threshold to correspondence entries before saving
            if threshold > 0:
                keep = ch_np > threshold
                pk_np = pk_np[keep]
                gid_np = gid_np[keep]
                ch_np = ch_np[keep]
                count = len(ch_np)

            # Row sums: extract sparse (nonzero entries only)
            row_sums_full = np.asarray(result['row_sums'])
            nz_mask = row_sums_full > 0
            rs_indices = np.where(nz_mask)[0].astype(np.int32)
            rs_values = row_sums_full[nz_mask].astype(np.float32)

            name = _PLANE_NAMES[plane_key]
            g = prov_grp.create_group(name)
            g.create_dataset('pixel_keys', data=pk_np, compression='gzip')
            g.create_dataset('group_ids', data=gid_np, compression='gzip')
            g.create_dataset('charges', data=ch_np, compression='gzip')
            g.create_dataset('row_sum_indices', data=rs_indices, compression='gzip')
            g.create_dataset('row_sum_values', data=rs_values, compression='gzip')
            g.attrs['count'] = count
            g.attrs['n_row_sums'] = len(rs_indices)


def load_truth_event(filepath, event_idx):
    """
    Load one event's truth/correspondence data from HDF5.

    Returns
    -------
    group_data : dict
        'east_group_ids', 'west_group_ids', 'group_to_track', 'n_groups'
    correspondence : dict
        Keyed by (side_idx, plane_idx), each containing:
        'pixel_keys', 'group_ids', 'charges', 'count',
        'row_sum_indices', 'row_sum_values'
    config : dict
        group_size, gap_threshold_mm, max_time, num_wires_actual, min_wire_indices
    """
    with h5py.File(filepath, 'r') as f:
        cfg = f['config']
        config = {
            'group_size': int(cfg.attrs['group_size']),
            'gap_threshold_mm': float(cfg.attrs['gap_threshold_mm']),
            'threshold': float(cfg.attrs.get('threshold', 0.0)),
            'max_time': int(cfg.attrs['max_time']),
            'num_time_steps': int(cfg.attrs['num_time_steps']),
            'num_wires_actual': np.array(cfg['num_wires_actual']),
            'min_wire_indices_abs': np.array(cfg['min_wire_indices']),
        }

        evt = f[f'event_{event_idx}']

        # Groups
        grp = evt['groups']
        group_data = {
            'east_group_ids': np.array(grp['east_group_ids']),
            'west_group_ids': np.array(grp['west_group_ids']),
            'group_to_track': np.array(grp['group_to_track']),
            'n_groups': int(grp.attrs['n_groups']),
        }

        # Provenance per plane
        correspondence = {}
        for name, key in _NAME_TO_KEY.items():
            if name in evt['correspondence']:
                g = evt['correspondence'][name]
                correspondence[key] = {
                    'pixel_keys': np.array(g['pixel_keys']),
                    'group_ids': np.array(g['group_ids']),
                    'charges': np.array(g['charges']),
                    'count': int(g.attrs['count']),
                    'row_sum_indices': np.array(g['row_sum_indices']),
                    'row_sum_values': np.array(g['row_sum_values']),
                    'n_row_sums': int(g.attrs['n_row_sums']),
                }

    return group_data, correspondence, config


def list_truth_events(filepath):
    """Return sorted list of event indices in the truth file."""
    with h5py.File(filepath, 'r') as f:
        return sorted(
            int(k.split('_')[1]) for k in f.keys() if k.startswith('event_')
        )


# =========================================================================
# Derivation utilities
# =========================================================================

def derive_track_hits_from_correspondence(correspondence_plane, group_to_track, max_time):
    """
    Derive dominant track per pixel from group correspondence.

    All entries already passed inter_thresh during the merge — no additional
    threshold is applied. Filter the output by charge downstream if needed.

    Parameters
    ----------
    correspondence_plane : dict
        Single plane's correspondence data.
    group_to_track : np.ndarray
        Group → track lookup.
    max_time : int
        Time dimension for decoding pixel_key = wire * max_time + time.

    Returns
    -------
    dict with labeled_track_ids, labeled_wires, labeled_times,
    labeled_charges, num_labeled.
    """
    pks = correspondence_plane['pixel_keys']
    gids = correspondence_plane['group_ids']
    chs = correspondence_plane['charges']

    if len(pks) == 0:
        return {
            'labeled_track_ids': np.zeros(0, dtype=np.int32),
            'labeled_wires': np.zeros(0, dtype=np.int32),
            'labeled_times': np.zeros(0, dtype=np.int32),
            'labeled_charges': np.zeros(0, dtype=np.float32),
            'num_labeled': 0,
        }

    # Map group → track
    tids = group_to_track[gids]
    wires = pks // max_time
    times = pks % max_time

    # Sort by (pixel, track) using two-pass stable sort
    # (no group-level threshold — entries already passed inter_thresh during merge)
    order1 = np.argsort(tids, kind='stable')
    order2 = np.argsort(pks[order1], kind='stable')
    order = order1[order2]
    s_pks, s_tids, s_chs = pks[order], tids[order], chs[order]
    s_wires, s_times = wires[order], times[order]

    # (pixel, track) groups → sum charges
    pt_boundary = np.ones(len(s_pks), dtype=bool)
    pt_boundary[1:] = (s_pks[1:] != s_pks[:-1]) | (s_tids[1:] != s_tids[:-1])
    pt_starts = np.where(pt_boundary)[0]
    pt_charges = np.add.reduceat(s_chs, pt_starts)
    pt_pks = s_pks[pt_starts]
    pt_tids = s_tids[pt_starts]
    pt_wires = s_wires[pt_starts]
    pt_times = s_times[pt_starts]

    # No additional threshold — entries already passed inter_thresh during merge.

    # Pixel groups → dominant track (max charge)
    px_boundary = np.ones(len(pt_pks), dtype=bool)
    px_boundary[1:] = pt_pks[1:] != pt_pks[:-1]
    px_starts = np.where(px_boundary)[0]
    max_charges = np.maximum.reduceat(pt_charges, px_starts)

    # Find first max per pixel
    px_ids = np.zeros(len(pt_pks), dtype=np.int64)
    px_ids[px_starts] = 1
    px_ids = np.cumsum(px_ids) - 1
    is_max = pt_charges >= max_charges[px_ids]

    n_pixels = len(px_starts)
    winner_idx = np.zeros(n_pixels, dtype=np.int64)
    seen = np.zeros(n_pixels, dtype=bool)
    for pos in np.where(is_max)[0]:
        pid = px_ids[pos]
        if not seen[pid]:
            winner_idx[pid] = pos
            seen[pid] = True

    return {
        'labeled_track_ids': pt_tids[winner_idx].astype(np.int32),
        'labeled_wires': pt_wires[winner_idx].astype(np.int32),
        'labeled_times': pt_times[winner_idx].astype(np.int32),
        'labeled_charges': pt_charges[winner_idx].astype(np.float32),
        'num_labeled': n_pixels,
    }


def derive_pixel_groups(correspondence_plane, pixel_wire, pixel_time, max_time):
    """
    Find all groups contributing to a specific pixel.

    Returns
    -------
    dict with group_ids, charges (sorted by descending charge).
    """
    pk_target = pixel_wire * max_time + pixel_time
    pks = correspondence_plane['pixel_keys']
    mask = pks == pk_target

    gids = correspondence_plane['group_ids'][mask]
    chs = correspondence_plane['charges'][mask]

    order = np.argsort(-chs)
    return {
        'group_ids': gids[order],
        'charges': chs[order],
    }


def derive_group_pixels(correspondence_plane, group_id, max_time):
    """
    Find all pixels a specific group contributes to.

    Returns
    -------
    dict with wires, times, charges (sorted by descending charge).
    """
    gids = correspondence_plane['group_ids']
    mask = gids == group_id

    pks = correspondence_plane['pixel_keys'][mask]
    chs = correspondence_plane['charges'][mask]

    order = np.argsort(-chs)
    pks_sorted = pks[order]

    return {
        'wires': (pks_sorted // max_time).astype(np.int32),
        'times': (pks_sorted % max_time).astype(np.int32),
        'charges': chs[order],
    }


def derive_deposit_row_sum(correspondence_plane, deposit_index):
    """
    Get a specific deposit's total diffused charge on this plane.

    Parameters
    ----------
    correspondence_plane : dict
        Single plane correspondence data.
    deposit_index : int
        Index into the padded side array.

    Returns
    -------
    float
        Total diffused charge, or 0 if deposit has no signal on this plane.
    """
    indices = correspondence_plane['row_sum_indices']
    values = correspondence_plane['row_sum_values']
    match = np.where(indices == deposit_index)[0]
    if len(match) > 0:
        return float(values[match[0]])
    return 0.0


# =========================================================================
# Segment 3D file (separate from correspondence)
# =========================================================================

def save_segments_event(filepath, event_idx, deposit_data):
    """
    Save one event's 3D segment (deposit) data to HDF5.

    The segment index in this file is the same index used by group_ids
    and row_sums in the correspondence file.

    Parameters
    ----------
    filepath : str
        Path to the segment HDF5 file (created or appended).
    event_idx : int
        Event index.
    deposit_data : DepositData
        Input deposit data (positions, dE, dx, angles, track_ids).
    """
    mode = 'a' if os.path.exists(filepath) else 'w'

    positions_mm = np.asarray(deposit_data.positions_mm)
    de = np.asarray(deposit_data.de)
    dx = np.asarray(deposit_data.dx)
    track_ids = np.asarray(deposit_data.track_ids)
    theta = np.asarray(deposit_data.theta)
    phi = np.asarray(deposit_data.phi)
    valid_mask = np.asarray(deposit_data.valid_mask)

    # Store only valid deposits
    valid_idx = np.where(valid_mask)[0]
    n = len(valid_idx)

    with h5py.File(filepath, mode) as f:
        event_key = f'event_{event_idx}'
        if event_key in f:
            del f[event_key]
        evt = f.create_group(event_key)

        evt.create_dataset('positions_mm', data=positions_mm[valid_idx].astype(np.float32),
                           compression='gzip')
        evt.create_dataset('de', data=de[valid_idx].astype(np.float32),
                           compression='gzip')
        evt.create_dataset('dx', data=dx[valid_idx].astype(np.float32),
                           compression='gzip')
        evt.create_dataset('track_ids', data=track_ids[valid_idx].astype(np.int32),
                           compression='gzip')
        evt.create_dataset('theta', data=theta[valid_idx].astype(np.float32),
                           compression='gzip')
        evt.create_dataset('phi', data=phi[valid_idx].astype(np.float32),
                           compression='gzip')
        evt.attrs['n_deposits'] = n


def load_segments_event(filepath, event_idx):
    """
    Load one event's 3D segment data from HDF5.

    Returns
    -------
    dict with positions_mm, de, dx, track_ids, theta, phi, n_deposits.
    Array index = segment_id (same index used by group_ids in correspondence file).
    """
    with h5py.File(filepath, 'r') as f:
        evt = f[f'event_{event_idx}']
        return {
            'positions_mm': np.array(evt['positions_mm']),
            'de': np.array(evt['de']),
            'dx': np.array(evt['dx']),
            'track_ids': np.array(evt['track_ids']),
            'theta': np.array(evt['theta']),
            'phi': np.array(evt['phi']),
            'n_deposits': int(evt.attrs['n_deposits']),
        }


def list_segment_events(filepath):
    """Return sorted list of event indices in the segment file."""
    with h5py.File(filepath, 'r') as f:
        return sorted(
            int(k.split('_')[1]) for k in f.keys() if k.startswith('event_')
        )
