"""
Production HDF5 load functions.

Reads simulation output from the three file types produced by run_batch.py:
    resp — sparse thresholded wire signals
    seg  — 3D truth deposits (per-volume)
    corr — 3D-to-2D correspondence

See DATA_FORMAT.md for the full schema.
"""

import os
import numpy as np
import h5py
from collections import namedtuple


_PLANE_LABELS = {0: 'U', 1: 'V', 2: 'Y'}


def _plane_label(plane_idx):
    """Plane index → short label."""
    return _PLANE_LABELS.get(plane_idx, str(plane_idx))


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

# Minimal config objects that satisfy visualization functions
_VolGeomMin = namedtuple('_VolGeomMin', ['num_wires', 'n_planes'])
_ConfigMin = namedtuple('_ConfigMin', [
    'volumes', 'n_volumes', 'num_time_steps', 'time_step_us', 'electrons_per_adc',
    'plane_names'])


def load_config(resp_path):
    """Load production metadata from a response file.

    Returns a dict with all config attributes plus 'num_wires_arr'.
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
    n_vol = nw.shape[0]
    n_planes = nw.shape[1]
    volumes = tuple(
        _VolGeomMin(
            num_wires=tuple(int(nw[v, p]) for p in range(n_planes)),
            n_planes=int(np.sum(nw[v] > 0)),
        )
        for v in range(n_vol)
    )
    _PLANE_LABELS = ('U', 'V', 'Y')
    plane_names = tuple(
        tuple(_PLANE_LABELS[:vol.n_planes]) for vol in volumes)
    return _ConfigMin(
        volumes=volumes,
        n_volumes=n_vol,
        num_time_steps=int(meta['num_time_steps']),
        time_step_us=float(meta['time_step_us']),
        electrons_per_adc=float(meta['electrons_per_adc']),
        plane_names=plane_names,
    )


# =============================================================================
# Response loading
# =============================================================================

def load_event_resp(resp_path, event_idx):
    """Load one event's response signals as dense arrays.

    Automatically detects uint16 (digitized) vs float32 format.

    Returns
    -------
    dense_signals : dict
        {(vol, plane): (num_wires, num_time_steps) ndarray}
    event_attrs : dict
    pedestals : dict or None
        {(vol, plane): int} if digitized, None otherwise.
    """
    event_key = f'event_{event_idx:03d}'

    with h5py.File(resp_path, 'r') as f:
        cfg = f['config']
        num_time_steps = int(cfg.attrs['num_time_steps'])
        num_wires_arr = cfg['num_wires'][:]
        n_vol = num_wires_arr.shape[0]
        n_planes = num_wires_arr.shape[1]

        evt = f[event_key]
        event_attrs = dict(evt.attrs)

        dense_signals = {}
        pedestals = {}
        digitized = False

        for v in range(n_vol):
            vol_key = f'volume_{v}'
            for p in range(n_planes):
                nw = int(num_wires_arr[v, p])
                if nw == 0:
                    continue
                plabel = _plane_label(p)

                if vol_key not in evt or plabel not in evt[vol_key] or \
                   'delta_wire' not in evt[vol_key][plabel]:
                    dense_signals[(v, p)] = np.zeros((nw, num_time_steps),
                                                      dtype=np.uint16)
                    continue

                g = evt[vol_key][plabel]
                wire_start = int(g.attrs['wire_start'])
                time_start = int(g.attrs['time_start'])

                wires = wire_start + np.cumsum(g['delta_wire'][:]).astype(np.int32)
                times = time_start + np.cumsum(g['delta_time'][:]).astype(np.int32)
                valid = ((wires >= 0) & (wires < nw) &
                         (times >= 0) & (times < num_time_steps))

                raw_values = g['values'][:]
                if raw_values.dtype == np.uint16:
                    digitized = True
                    ped = int(g.attrs['pedestal'])
                    pedestals[(v, p)] = ped
                    dense = np.zeros((nw, num_time_steps), dtype=np.uint16)
                    dense[wires[valid], times[valid]] = raw_values[valid]
                else:
                    dense = np.zeros((nw, num_time_steps), dtype=np.float32)
                    dense[wires[valid], times[valid]] = raw_values[valid]

                dense_signals[(v, p)] = dense

    return dense_signals, event_attrs, pedestals if digitized else None


# =============================================================================
# Segment loading
# =============================================================================

def load_event_seg(seg_path, event_idx):
    """Load one event's 3D truth deposits (per-volume).

    Returns
    -------
    volumes : list of dict, one per volume. Each has:
        positions_mm (N, 3), de (N,), dx (N,), theta (N,), phi (N,),
        track_ids (N,), group_ids (N,), group_to_track (G,) or None,
        qs_fractions (N,) or None, original_indices (N,) or None,
        n_actual (int).
    """
    event_key = f'event_{event_idx:03d}'

    with h5py.File(seg_path, 'r') as f:
        evt = f[event_key]
        n_volumes = int(evt.attrs.get('n_volumes', 2))

        volumes = []
        for v in range(n_volumes):
            vg_key = f'volume_{v}'
            if vg_key not in evt:
                volumes.append({'n_actual': 0})
                continue

            vg = evt[vg_key]
            n = int(vg.attrs['n_actual'])

            if n == 0:
                volumes.append({'n_actual': 0})
                continue

            pos_step = float(vg.attrs['pos_step_mm'])
            origin = np.array([vg.attrs['pos_origin_x'],
                               vg.attrs['pos_origin_y'],
                               vg.attrs['pos_origin_z']])
            positions_mm = vg['positions'][:].astype(np.float32) * pos_step + origin

            vol = {
                'positions_mm': positions_mm,
                'de': vg['de'][:].astype(np.float32),
                'dx': vg['dx'][:].astype(np.float32),
                'theta': vg['theta'][:].astype(np.float32),
                'phi': vg['phi'][:].astype(np.float32),
                'track_ids': vg['track_ids'][:],
                'group_ids': vg['group_ids'][:],
                'group_to_track': (vg['group_to_track'][:] if 'group_to_track' in vg
                                   else None),
                't0_us': (vg['t0_us'][:].astype(np.float32) if 't0_us' in vg
                          else None),
                'n_actual': n,
                'n_groups': int(vg.attrs.get('n_groups', 0)),
                'charge': (vg['charge'][:].astype(np.float32)
                           if 'charge' in vg else None),
                'photons': (vg['photons'][:].astype(np.float32)
                             if 'photons' in vg else None),
                'qs_fractions': (vg['qs_fractions'][:].astype(np.float32)
                                  if 'qs_fractions' in vg else None),
                'original_indices': (vg['original_indices'][:]
                                      if 'original_indices' in vg else None),
            }
            volumes.append(vol)

    return volumes


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


def load_event_corr(corr_path, event_idx, num_time_steps, n_volumes=2, max_planes=3):
    """Load correspondence and derive track labels + diffused charge.

    Parameters
    ----------
    corr_path : str
    event_idx : int
    num_time_steps : int
    n_volumes : int
    max_planes : int

    Returns
    -------
    track_hits : dict
        {(vol, plane): result from label_from_groups}
    truth_dense : dict
        {(vol, plane): (num_wires, num_time) ndarray}
    group_to_track : list of arrays, one per volume
    """
    from tools.track_hits import label_from_groups

    event_key = f'event_{event_idx:03d}'
    track_hits = {}
    truth_dense = {}

    with h5py.File(corr_path, 'r') as f:
        evt = f[event_key]
        nw_arr = f['config']['num_wires'][:]

        # Load per-volume group_to_track
        g2t_per_vol = []
        for v in range(n_volumes):
            vol_key = f'volume_{v}'
            if vol_key in evt and 'group_to_track' in evt[vol_key]:
                g2t_per_vol.append(evt[vol_key]['group_to_track'][:])
            else:
                g2t_per_vol.append(np.array([0], dtype=np.int32))

        for v in range(n_volumes):
            vol_key = f'volume_{v}'
            for p in range(max_planes):
                nw = int(nw_arr[v, p])
                if nw == 0:
                    continue
                plabel = _plane_label(p)

                if vol_key not in evt or plabel not in evt[vol_key]:
                    track_hits[(v, p)] = {
                        'labeled_hits': np.zeros((0, 3), dtype=np.float32),
                        'labeled_track_ids': np.array([], dtype=np.int32),
                        'num_labeled': 0,
                    }
                    truth_dense[(v, p)] = np.zeros((nw, num_time_steps), dtype=np.float32)
                    continue

                pk, gid, ch, n_entries = _decode_plane_corr(evt[vol_key][plabel], num_time_steps)

                # Diffused charge: sum all entries into dense array
                dense = np.zeros((nw, num_time_steps), dtype=np.float32)
                all_w = pk // num_time_steps
                all_t = pk % num_time_steps
                valid = (all_w >= 0) & (all_w < nw) & (all_t >= 0) & (all_t < num_time_steps)
                np.add.at(dense, (all_w[valid], all_t[valid]), ch[valid])
                truth_dense[(v, p)] = dense

                # Track labels — use this volume's group_to_track
                result = label_from_groups(pk, gid, ch, n_entries,
                                           g2t_per_vol[v], num_time_steps)
                track_hits[(v, p)] = result

    return track_hits, truth_dense, g2t_per_vol
