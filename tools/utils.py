"""
HDF5 I/O for space-charge-effect (SCE) maps.

Event I/O lives in production/save.py and production/load.py (any number of
volumes, wire or pixel). The former single-file save_event/load_event here
expected keys that no current output produces and only knew the 2-volume U/V/Y
detector, so it was removed.
"""

import h5py
import numpy as np


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


def save_sce_data(filepath, volume_data, metadata=None):
    """
    Save per-volume space charge effect maps to HDF5.

    File layout::

        volume_0/efield_map              (Nx, Ny, Nz, 3) float32
        volume_0/drift_correction_map    (Nx, Ny, Nz, 3) float32
        volume_0/origin_cm               (3,) float64
        volume_0/spacing_cm              (3,) float64
        volume_1/...

    Parameters
    ----------
    filepath : str
        Output HDF5 path.
    volume_data : list of dicts
        Each dict has 'efield_map', 'drift_correction_map', 'origin_cm', 'spacing_cm'.
    metadata : dict, optional
        Extra key/value pairs stored as file-level HDF5 attributes.
    """
    with h5py.File(filepath, 'w') as f:
        for i, vol in enumerate(volume_data):
            _save_side(f.create_group(f'volume_{i}'),
                       vol['efield_map'], vol['drift_correction_map'],
                       vol['origin_cm'], vol['spacing_cm'])

        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


def load_sce_data(filepath):
    """
    Load per-volume space charge effect maps from HDF5.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file written by ``save_sce_data``.

    Returns
    -------
    list of dict
        Each dict contains 'efield_map', 'drift_correction_map',
        'origin_cm', 'spacing_cm'.
    """
    with h5py.File(filepath, 'r') as f:
        keys = sorted((k for k in f.keys() if k.startswith('volume_')),
                      key=lambda k: int(k.split('_')[1]))   # volume_10 after volume_9
        return [_load_side(f[k]) for k in keys]
