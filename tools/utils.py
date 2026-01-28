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
