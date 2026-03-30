"""
Production HDF5 load functions.

Reads simulation output from the three file types produced by run_batch.py:
    resp — sparse thresholded wire signals
    seg  — 3D truth deposits
    corr — 3D-to-2D correspondence

See DATA_FORMAT.md for the full schema.
"""

import os
import numpy as np
import h5py
from collections import namedtuple

PLANE_KEYS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
PLANE_NAMES = {
    (0, 0): 'east_U', (0, 1): 'east_V', (0, 2): 'east_Y',
    (1, 0): 'west_U', (1, 1): 'west_V', (1, 2): 'west_Y',
}


# =============================================================================
# File paths
# =============================================================================

def get_file_paths(production_dir, dataset, file_index):
    """Return (resp_path, seg_path, corr_path) for a given batch file."""
    tag = f'{dataset}_{{}}{file_index:04d}.h5'
    return (
        os.path.join(production_dir, 'resp', tag.format('resp_')),
        os.path.join(production_dir, 'seg', tag.format('seg_')),
        os.path.join(production_dir, 'corr', tag.format('corr_')),
    )


# =============================================================================
# Config / metadata
# =============================================================================

# Minimal config object that satisfies visualization functions
_SideGeomMin = namedtuple('_SideGeomMin', ['num_wires'])
_ConfigMin = namedtuple('_ConfigMin', [
    'side_geom', 'num_time_steps', 'time_step_us', 'electrons_per_adc'])


def load_config(resp_path):
    """Load production metadata from a response file.

    Returns a dict with all config attributes plus 'num_wires_arr' (2,3).
    """
    with h5py.File(resp_path, 'r') as f:
        cfg = f['config']
        meta = dict(cfg.attrs)
        meta['num_wires_arr'] = cfg['num_wires'][:]
    return meta


def build_viz_config(resp_path):
    """Build a minimal config object for visualization functions.

    Only requires the response file — no YAML or generate_detector needed.
    """
    meta = load_config(resp_path)
    nw = meta['num_wires_arr']
    return _ConfigMin(
        side_geom=tuple(
            _SideGeomMin(num_wires=tuple(int(nw[s, p]) for p in range(3)))
            for s in range(2)),
        num_time_steps=int(meta['num_time_steps']),
        time_step_us=float(meta['time_step_us']),
        electrons_per_adc=float(meta['electrons_per_adc']),
    )


# =============================================================================
# Response loading
# =============================================================================

def load_event_resp(resp_path, event_idx):
    """Load one event's response signals as dense arrays.

    Parameters
    ----------
    resp_path : str
        Path to response HDF5 file.
    event_idx : int
        Event index within the file.

    Returns
    -------
    dense_signals : dict
        {(side, plane): (num_wires, num_time_steps) ndarray}
    event_attrs : dict
        Event-level attributes (n_deposits, n_east, n_west, source_event_idx).
    """
    event_key = f'event_{event_idx:03d}'

    with h5py.File(resp_path, 'r') as f:
        cfg = f['config']
        num_time_steps = int(cfg.attrs['num_time_steps'])
        num_wires_arr = cfg['num_wires'][:]

        evt = f[event_key]
        event_attrs = dict(evt.attrs)

        dense_signals = {}
        for (s, p) in PLANE_KEYS:
            name = PLANE_NAMES[(s, p)]
            nw = int(num_wires_arr[s, p])
            dense = np.zeros((nw, num_time_steps), dtype=np.float32)

            if name in evt and 'delta_wire' in evt[name]:
                g = evt[name]
                wire_start = int(g.attrs['wire_start'])
                time_start = int(g.attrs['time_start'])

                wires = wire_start + np.cumsum(g['delta_wire'][:]).astype(np.int32)
                times = time_start + np.cumsum(g['delta_time'][:]).astype(np.int32)
                values = g['values'][:]

                valid = ((wires >= 0) & (wires < nw) &
                         (times >= 0) & (times < num_time_steps))
                dense[wires[valid], times[valid]] = values[valid]

            dense_signals[(s, p)] = dense

    return dense_signals, event_attrs


# =============================================================================
# Segment loading
# =============================================================================

def load_event_seg(seg_path, event_idx):
    """Load one event's 3D truth deposits.

    Returns
    -------
    seg : dict with keys:
        positions_mm (N, 3), de (N,), dx (N,), theta (N,), phi (N,),
        track_ids (N,), group_ids (N,), group_to_track (G,),
        qs_fractions (N,) or None, n_groups (int).
    """
    event_key = f'event_{event_idx:03d}'

    with h5py.File(seg_path, 'r') as f:
        evt = f[event_key]

        pos_step = float(evt.attrs['pos_step_mm'])
        origin = np.array([evt.attrs['pos_origin_x'],
                           evt.attrs['pos_origin_y'],
                           evt.attrs['pos_origin_z']])
        positions_mm = evt['positions'][:].astype(np.float32) * pos_step + origin

        seg = {
            'positions_mm': positions_mm,
            'de': evt['de'][:].astype(np.float32),
            'dx': evt['dx'][:].astype(np.float32),
            'theta': evt['theta'][:].astype(np.float32),
            'phi': evt['phi'][:].astype(np.float32),
            'track_ids': evt['track_ids'][:],
            'group_ids': evt['group_ids'][:],
            'group_to_track': evt['group_to_track'][:],
            'n_groups': int(evt.attrs['n_groups']),
            'qs_fractions': (evt['qs_fractions'][:].astype(np.float32)
                             if 'qs_fractions' in evt else None),
        }

    return seg


# =============================================================================
# Correspondence loading
# =============================================================================

def _decode_plane_corr(g, num_time_steps):
    """Decode one plane's CSR correspondence into flat arrays."""
    grp_ids = g['group_ids'][:]
    grp_sizes = g['group_sizes'][:]
    center_wires = g['center_wires'][:]
    center_times = g['center_times'][:]
    peak_charges = g['peak_charges'][:]
    delta_wires = g['delta_wires'][:]
    delta_times = g['delta_times'][:]
    charges_u16 = g['charges_u16'][:]

    group_starts = np.cumsum(grp_sizes) - grp_sizes
    n_entries = int(grp_sizes.sum())

    pk_flat = np.empty(n_entries, dtype=np.int32)
    gid_flat = np.empty(n_entries, dtype=np.int32)
    ch_flat = np.empty(n_entries, dtype=np.float32)

    for i in range(len(grp_ids)):
        s = int(group_starts[i])
        sz = int(grp_sizes[i])
        w = int(center_wires[i]) + delta_wires[s:s + sz].astype(np.int32)
        t = int(center_times[i]) + delta_times[s:s + sz].astype(np.int32)
        ch = float(peak_charges[i]) * charges_u16[s:s + sz].astype(np.float32) / 65535.0

        pk_flat[s:s + sz] = w * num_time_steps + t
        gid_flat[s:s + sz] = grp_ids[i]
        ch_flat[s:s + sz] = ch

    return pk_flat, gid_flat, ch_flat, n_entries


def load_event_corr(corr_path, event_idx, num_time_steps):
    """Load correspondence and derive track labels + diffused charge.

    Parameters
    ----------
    corr_path : str
    event_idx : int
    num_time_steps : int
        From config (needed to decode pixel keys and build dense arrays).

    Returns
    -------
    track_hits : dict
        {(side, plane): result from label_from_groups} with keys
        'labeled_hits', 'labeled_track_ids', 'num_labeled'.
    truth_dense : dict
        {(side, plane): (num_wires, num_time) ndarray} total diffused charge.
    group_to_track : (G,) int32 array.
    """
    from tools.track_hits import label_from_groups

    event_key = f'event_{event_idx:03d}'
    track_hits = {}
    truth_dense = {}

    with h5py.File(corr_path, 'r') as f:
        evt = f[event_key]
        g2t = evt['group_to_track'][:]

        # Need num_wires from corr config or caller — read from file
        nw_arr = f['config']['num_wires'][:]

        for (s, p) in PLANE_KEYS:
            name = PLANE_NAMES[(s, p)]
            nw = int(nw_arr[s, p])

            if name not in evt:
                track_hits[(s, p)] = {
                    'labeled_hits': np.zeros((0, 3), dtype=np.float32),
                    'labeled_track_ids': np.array([], dtype=np.int32),
                    'num_labeled': 0,
                }
                truth_dense[(s, p)] = np.zeros((nw, num_time_steps), dtype=np.float32)
                continue

            pk, gid, ch, n_entries = _decode_plane_corr(evt[name], num_time_steps)

            # Diffused charge: sum all entries into dense array
            dense = np.zeros((nw, num_time_steps), dtype=np.float32)
            all_w = pk // num_time_steps
            all_t = pk % num_time_steps
            valid = (all_w >= 0) & (all_w < nw) & (all_t >= 0) & (all_t < num_time_steps)
            np.add.at(dense, (all_w[valid], all_t[valid]), ch[valid])
            truth_dense[(s, p)] = dense

            # Track labels
            result = label_from_groups(pk, gid, ch, n_entries, g2t, num_time_steps)
            track_hits[(s, p)] = result

    return track_hits, truth_dense, g2t
