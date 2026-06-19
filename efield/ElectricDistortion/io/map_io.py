"""Save and load SCE distortion maps."""

import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

_PARAM_KEYS = ("Lx", "Ly", "Lz", "E0", "Q_charge_production",
               "epsilon_r", "mu_ion", "temperature")


def _pack_params(params):
    """Extract serialisable scalar params from the full dict."""
    if params is None:
        return {}
    return {k: params[k] for k in _PARAM_KEYS if k in params}


def save_maps_npz(path, x_grid, y_grid, z_grid,
                  delta_x, delta_y, delta_z,
                  Ex, Ey, Ez, E_mag, E_ratio,
                  params=None):
    """Save distortion maps to a compressed .npz file."""
    data = dict(
        x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
        delta_x=delta_x, delta_y=delta_y, delta_z=delta_z,
        Ex=Ex, Ey=Ey, Ez=Ez,
        E_mag=E_mag, E_ratio=E_ratio,
    )
    for k, v in _pack_params(params).items():
        data[f"param_{k}"] = np.array(v)
    np.savez_compressed(path, **data)


def load_maps_npz(path):
    """Load distortion maps from a .npz file.

    Returns a dict with array data and a nested ``'params'`` dict.
    """
    raw = dict(np.load(path))
    out = {}
    params = {}
    for k, v in raw.items():
        if k.startswith("param_"):
            params[k[6:]] = float(v)
        else:
            out[k] = v
    if params:
        out["params"] = params
    return out


def save_maps_hdf5(path, x_grid, y_grid, z_grid,
                   delta_x, delta_y, delta_z,
                   Ex, Ey, Ez, E_mag, E_ratio,
                   params=None):
    """Save distortion maps to an HDF5 file."""
    if not _HAS_H5PY:
        raise ImportError("h5py is required for HDF5 output")

    with h5py.File(path, "w") as f:
        for name, arr in [("x_grid", x_grid), ("y_grid", y_grid),
                          ("z_grid", z_grid),
                          ("delta_x", delta_x), ("delta_y", delta_y),
                          ("delta_z", delta_z),
                          ("Ex", Ex), ("Ey", Ey), ("Ez", Ez),
                          ("E_mag", E_mag), ("E_ratio", E_ratio)]:
            f.create_dataset(name, data=arr)

        for k, v in _pack_params(params).items():
            f.attrs[f"param_{k}"] = v


def load_maps_hdf5(path):
    """Load distortion maps from an HDF5 file.

    Returns a dict with array data and a nested ``'params'`` dict,
    matching the structure of ``load_maps_npz``.
    """
    if not _HAS_H5PY:
        raise ImportError("h5py is required for HDF5 input")

    out = {}
    params = {}
    with h5py.File(path, "r") as f:
        for k in f:
            out[k] = f[k][()]
        for k, v in f.attrs.items():
            if k.startswith("param_"):
                params[k[6:]] = float(v)
    if params:
        out["params"] = params
    return out
