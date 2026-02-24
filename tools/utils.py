"""
HDF5 I/O for saving and loading simulation events.

Saves both output paths (response + hits) per event into a single file.
Events are stored under /event_{idx}/ groups. Detector config is written
once at the root level on first event.

Schema:
    /config/
        num_wires_actual   (2,3) int32
        min_wire_indices   (2,3) int32
        attrs: num_time_steps, electrons_per_adc
    /event_{idx}/
        response/{plane}/
            indices  (N,2) int32
            values   (N,)  float32
            signal   (N_s,) float32
            attrs: n_signal, threshold_adc
        hits/{plane}/
            hits_by_track    (H,3) float32
            track_boundaries (T,)  int32
            track_ids        (T,)  int32
            attrs: num_hits, num_tracks
"""

import h5py
import numpy as np
import os

_PLANE_NAMES = {
    (0, 0): 'east_U', (0, 1): 'east_V', (0, 2): 'east_Y',
    (1, 0): 'west_U', (1, 1): 'west_V', (1, 2): 'west_Y',
}
_NAME_TO_KEY = {v: k for k, v in _PLANE_NAMES.items()}


def save_event(filepath, event_idx, sparse_output, track_hits, detector_config):
    """
    Save one event's simulation output to HDF5 (creates or appends).

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    event_idx : int
        Event index (used as group name: event_0, event_1, ...).
    sparse_output : dict
        Output from process_response(), keyed by (side_idx, plane_idx).
    track_hits : dict
        Output from DetectorSimulator(), keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    """
    mode = 'a' if os.path.exists(filepath) else 'w'

    with h5py.File(filepath, mode) as f:
        # Write config once
        if 'config' not in f:
            cfg = f.create_group('config')
            cfg.create_dataset('num_wires_actual',
                               data=np.array(detector_config['num_wires_actual'], dtype=np.int32))
            cfg.create_dataset('min_wire_indices',
                               data=np.array(detector_config['min_wire_indices_abs'], dtype=np.int32))
            cfg.attrs['num_time_steps'] = int(detector_config['num_time_steps'])
            cfg.attrs['electrons_per_adc'] = float(detector_config['electrons_per_adc'])

        event_key = f'event_{event_idx}'
        if event_key in f:
            del f[event_key]
        evt = f.create_group(event_key)

        # --- Response path ---
        resp_grp = evt.create_group('response')
        for (si, pi), data in sparse_output.items():
            name = _PLANE_NAMES[(si, pi)]
            g = resp_grp.create_group(name)
            g.create_dataset('indices', data=np.asarray(data['indices']), compression='gzip')
            g.create_dataset('values', data=np.asarray(data['values']), compression='gzip')
            g.create_dataset('signal', data=np.asarray(data['signal']), compression='gzip')
            g.attrs['n_signal'] = int(data['n_signal'])
            g.attrs['threshold_adc'] = float(data['threshold_adc'])

        # --- Hits path ---
        hits_grp = evt.create_group('hits')
        for (si, pi), data in track_hits.items():
            name = _PLANE_NAMES[(si, pi)]
            g = hits_grp.create_group(name)
            # All arrays already sliced to valid length by process_event
            g.create_dataset('hits_by_track',
                             data=np.asarray(data['hits_by_track']),
                             compression='gzip')
            g.create_dataset('track_boundaries',
                             data=np.asarray(data['track_boundaries']),
                             compression='gzip')
            g.create_dataset('track_ids',
                             data=np.asarray(data['track_ids']),
                             compression='gzip')
            g.attrs['num_hits'] = int(data['num_hits'])
            g.attrs['num_tracks'] = int(data['num_tracks'])


def load_event(filepath, event_idx):
    """
    Load one event from HDF5.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    event_idx : int
        Event index to load.

    Returns
    -------
    sparse_output : dict
        Keyed by (side_idx, plane_idx), matching process_response() output.
    track_hits : dict
        Keyed by (side_idx, plane_idx), matching DetectorSimulator() output.
    config : dict
        Detector config subset needed for downstream use.
    """
    with h5py.File(filepath, 'r') as f:
        # Config
        cfg = f['config']
        config = {
            'num_wires_actual': np.array(cfg['num_wires_actual']),
            'min_wire_indices_abs': np.array(cfg['min_wire_indices']),
            'num_time_steps': int(cfg.attrs['num_time_steps']),
            'electrons_per_adc': float(cfg.attrs['electrons_per_adc']),
        }

        evt = f[f'event_{event_idx}']

        # Response
        sparse_output = {}
        for name, key in _NAME_TO_KEY.items():
            if name in evt['response']:
                g = evt['response'][name]
                sparse_output[key] = {
                    'indices': np.array(g['indices']),
                    'values': np.array(g['values']),
                    'signal': np.array(g['signal']),
                    'n_signal': int(g.attrs['n_signal']),
                    'threshold_adc': float(g.attrs['threshold_adc']),
                }

        # Hits
        track_hits = {}
        for name, key in _NAME_TO_KEY.items():
            if name in evt['hits']:
                g = evt['hits'][name]
                track_hits[key] = {
                    'hits_by_track': np.array(g['hits_by_track']),
                    'track_boundaries': np.array(g['track_boundaries']),
                    'track_ids': np.array(g['track_ids']),
                    'num_hits': int(g.attrs['num_hits']),
                    'num_tracks': int(g.attrs['num_tracks']),
                }

    return sparse_output, track_hits, config


def list_events(filepath):
    """Return sorted list of event indices stored in the file."""
    with h5py.File(filepath, 'r') as f:
        return sorted(
            int(k.split('_')[1]) for k in f.keys() if k.startswith('event_')
        )


# =========================================================================
# Space Charge Effect (SCE) maps I/O
# =========================================================================

def _save_side(group, efield_map, drift_correction_map, origin_cm, spacing_cm):
    """Write one TPC side's SCE data into an HDF5 group."""
    group.create_dataset('efield_map', data=np.asarray(efield_map, dtype=np.float32))
    group.create_dataset('drift_correction_map',
                         data=np.asarray(drift_correction_map, dtype=np.float32))
    group.create_dataset('origin_cm', data=np.asarray(origin_cm, dtype=np.float64))
    group.create_dataset('spacing_cm', data=np.asarray(spacing_cm, dtype=np.float64))


def _load_side(group):
    """Read one TPC side's SCE data from an HDF5 group."""
    return {
        'efield_map': np.array(group['efield_map']),
        'drift_correction_map': np.array(group['drift_correction_map']),
        'origin_cm': np.array(group['origin_cm']),
        'spacing_cm': np.array(group['spacing_cm']),
    }


def save_sce_data(filepath, east_efield, east_corrections, east_origin, east_spacing,
                  west_efield, west_corrections, west_origin, west_spacing,
                  metadata=None):
    """
    Save per-side space charge effect maps to HDF5.

    File layout::

        east/efield_map              (Nx, Ny, Nz, 3) float32
        east/drift_correction_map    (Nx, Ny, Nz, 3) float32
        east/origin_cm               (3,) float64
        east/spacing_cm              (3,) float64
        west/efield_map              (Nx, Ny, Nz, 3) float32
        west/drift_correction_map    (Nx, Ny, Nz, 3) float32
        west/origin_cm               (3,) float64
        west/spacing_cm              (3,) float64

    Parameters
    ----------
    filepath : str
        Output HDF5 path.
    east_efield, west_efield : np.ndarray, shape (Nx, Ny, Nz, 3)
        E-field maps per side.
    east_corrections, west_corrections : np.ndarray, shape (Nx, Ny, Nz, 3)
        Drift correction maps per side.
    east_origin, west_origin : array-like, shape (3,)
    east_spacing, west_spacing : array-like, shape (3,)
    metadata : dict, optional
        Extra key/value pairs stored as file-level HDF5 attributes.
    """
    with h5py.File(filepath, 'w') as f:
        _save_side(f.create_group('east'),
                   east_efield, east_corrections, east_origin, east_spacing)
        _save_side(f.create_group('west'),
                   west_efield, west_corrections, west_origin, west_spacing)

        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


def load_sce_data(filepath):
    """
    Load per-side space charge effect maps from HDF5.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file written by ``save_sce_data``.

    Returns
    -------
    dict
        ``{'east': {...}, 'west': {...}}`` where each side dict contains
        ``'efield_map'``, ``'drift_correction_map'``, ``'origin_cm'``,
        ``'spacing_cm'``.
    """
    with h5py.File(filepath, 'r') as f:
        return {
            'east': _load_side(f['east']),
            'west': _load_side(f['west']),
        }
