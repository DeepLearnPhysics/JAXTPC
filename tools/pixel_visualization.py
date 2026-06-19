"""
Visualization utilities for LArTPC pixel readout signals.

All functions accept SimConfig (from tools.config) for detector geometry.

Pixel data formats:
    - Dense: {(vol, 0): (num_py, num_pz, num_time) ndarray}
    - Sparse: {(vol, 0): {'pixel_y': (N,), 'pixel_z': (N,), 'time': (N,), 'values': (N,)}}
      Produced by tools.output.to_sparse().

Bucketed: {(vol, 0): (buckets, num_active, ctk, B1, B2, B3)} — 6-tuple.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================================================================
# Helpers
# =========================================================================

def _vol_name(vol_idx):
    return f'Volume {vol_idx}'


def _add_colorbar(fig, ax, mappable, label='Signal', label_size=12, tick_size=10):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label(label, fontsize=label_size)
    return cbar


def _extract_sparse(data):
    """Extract numpy arrays from a sparse pixel dict."""
    return (np.asarray(data['pixel_y']),
            np.asarray(data['pixel_z']),
            np.asarray(data['time']),
            np.asarray(data['values']))


def _to_sparse_if_dense(data, threshold=0.0):
    """Convert dense (py, pz, T) to sparse dict. Pass-through if already sparse."""
    if isinstance(data, dict):
        return data
    arr = np.asarray(data)
    thresh = max(threshold, 1e-30)
    mask = np.abs(arr) >= thresh
    py, pz, t = np.where(mask)
    return {
        'pixel_y': py.astype(np.int32),
        'pixel_z': pz.astype(np.int32),
        'time': t.astype(np.int32),
        'values': arr[mask].astype(np.float32),
    }


def _project_sparse(py, pz, t, vals, axis, num_bins_a, num_bins_b, reduce='sum'):
    """Project sparse 3D data onto a 2D plane by summing along one axis.

    Parameters
    ----------
    axis : str
        'yz' (sum over time), 'yt' (sum over z), 'zt' (sum over y)
    num_bins_a, num_bins_b : int
        Shape of the output 2D array.
    reduce : str
        'sum' or 'max'.

    Returns
    -------
    2D ndarray of shape (num_bins_a, num_bins_b).
    """
    out = np.zeros((num_bins_a, num_bins_b), dtype=np.float64)
    if len(vals) == 0:
        return out.astype(np.float32)

    if axis == 'yz':
        a, b = py, pz
    elif axis == 'yt':
        a, b = py, t
    elif axis == 'zt':
        a, b = pz, t
    else:
        raise ValueError(f"Unknown axis '{axis}'")

    valid = (a >= 0) & (a < num_bins_a) & (b >= 0) & (b < num_bins_b)
    a, b, v = a[valid], b[valid], vals[valid]

    if reduce == 'max':
        np.maximum.at(out, (a, b), v)
    else:
        np.add.at(out, (a, b), v)

    return out.astype(np.float32)


def _make_norm(arr, log_norm=False, threshold=0.0):
    """Build a Normalize or LogNorm from data."""
    valid = arr[np.abs(arr) > threshold] if threshold > 0 else arr[arr != 0]
    if len(valid) == 0:
        return Normalize(vmin=-1, vmax=1)

    if log_norm:
        pos = valid[valid > 0]
        if len(pos) == 0:
            return Normalize(vmin=-1, vmax=1)
        p1, p99 = np.percentile(pos, [1, 99])
        return LogNorm(vmin=max(p1, 1), vmax=p99)

    p1, p99 = np.percentile(valid, [1, 99])
    return Normalize(vmin=p1, vmax=p99)


# =========================================================================
# 1. Three orthogonal projections (Y-Z, Y-Time, Z-Time)
# =========================================================================

def visualize_pixel_projections(pixel_signals, config, vol_idx=0,
                                figsize=(18, 5), cmap='inferno',
                                log_norm=False, threshold=0.0,
                                reduce='sum'):
    """Three orthogonal projections of the 3D pixel signal.

    Parameters
    ----------
    pixel_signals : dict
        Keyed by (vol_idx, 0). Dense (py, pz, T) or sparse dict.
    config : SimConfig
    vol_idx : int
    reduce : str
        'sum' (integrate along projected axis) or 'max' (max projection).

    Returns
    -------
    matplotlib Figure
    """
    vol = config.volumes[vol_idx]
    num_py, num_pz = vol.pixel_shape
    num_time = config.num_time_steps
    time_step = config.time_step_us
    pitch = vol.pixel_pitch_cm

    key = (vol_idx, 0)
    if key not in pixel_signals:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax in axes:
            ax.text(0.5, 0.5, '(No data)', ha='center', va='center',
                    transform=ax.transAxes, color='grey')
        return fig

    sparse = _to_sparse_if_dense(pixel_signals[key], threshold)
    py, pz, t, vals = _extract_sparse(sparse)

    # Project
    yz = _project_sparse(py, pz, t, vals, 'yz', num_py, num_pz, reduce)
    yt = _project_sparse(py, pz, t, vals, 'yt', num_py, num_time, reduce)
    zt = _project_sparse(py, pz, t, vals, 'zt', num_pz, num_time, reduce)

    # Physical extents
    y_origin, z_origin = vol.pixel_origins_cm
    y_max = y_origin + num_py * pitch
    z_max = z_origin + num_pz * pitch
    max_time = num_time * time_step

    projections = [
        (yz, 'Y-Z (sum over time)',
         [z_origin, z_max, y_origin, y_max], 'Z (cm)', 'Y (cm)'),
        (yt, 'Y-Time (sum over Z)',
         [0, max_time, y_origin, y_max], 'Time (us)', 'Y (cm)'),
        (zt, 'Z-Time (sum over Y)',
         [0, max_time, z_origin, z_max], 'Time (us)', 'Z (cm)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor='white')

    for ax, (arr, title, extent, xlabel, ylabel) in zip(axes, projections):
        norm = _make_norm(arr, log_norm=log_norm, threshold=threshold)
        im = ax.imshow(arr, aspect='auto', origin='lower', extent=extent,
                       cmap=cmap, norm=norm, interpolation='nearest')
        _add_colorbar(fig, ax, im, label=reduce.capitalize())
        ax.set_title(f'{_vol_name(vol_idx)} {title}', fontsize=12, pad=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

    fig.suptitle(f'Pixel Projections ({reduce})', fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# 2. Anode heatmap (pixel plane summed/max over time or at a time slice)
# =========================================================================

def visualize_pixel_anode(pixel_signals, config, vol_idx=0,
                          time_range=None, figsize=(8, 8), cmap='inferno',
                          log_norm=False, threshold=0.0, reduce='sum'):
    """2D heatmap of the pixel anode plane.

    Parameters
    ----------
    pixel_signals : dict
        Keyed by (vol_idx, 0).
    config : SimConfig
    vol_idx : int
    time_range : tuple (t_min_us, t_max_us) or None
        If given, only include signal within this time window.
        If None, integrate over all time.
    reduce : str
        'sum' or 'max'.

    Returns
    -------
    matplotlib Figure
    """
    vol = config.volumes[vol_idx]
    num_py, num_pz = vol.pixel_shape
    num_time = config.num_time_steps
    time_step = config.time_step_us
    pitch = vol.pixel_pitch_cm

    key = (vol_idx, 0)
    if key not in pixel_signals:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, '(No data)', ha='center', va='center',
                transform=ax.transAxes, color='grey')
        return fig

    sparse = _to_sparse_if_dense(pixel_signals[key], threshold)
    py, pz, t, vals = _extract_sparse(sparse)

    # Time window filter
    if time_range is not None:
        t_us = t * time_step
        mask = (t_us >= time_range[0]) & (t_us < time_range[1])
        py, pz, t, vals = py[mask], pz[mask], t[mask], vals[mask]

    anode = _project_sparse(py, pz, t, vals, 'yz', num_py, num_pz, reduce)

    y_origin, z_origin = vol.pixel_origins_cm
    extent = [z_origin, z_origin + num_pz * pitch,
              y_origin, y_origin + num_py * pitch]

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    norm = _make_norm(anode, log_norm=log_norm, threshold=threshold)
    im = ax.imshow(anode, aspect='equal', origin='lower', extent=extent,
                   cmap=cmap, norm=norm, interpolation='nearest')
    _add_colorbar(fig, ax, im, label=f'{reduce.capitalize()} signal')

    title = f'{_vol_name(vol_idx)} Pixel Anode'
    if time_range is not None:
        title += f' [{time_range[0]:.0f}-{time_range[1]:.0f} us]'
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel('Z (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)

    fig.tight_layout()
    return fig


# =========================================================================
# 3. Single-pixel waveform (time series at given pixel coordinates)
# =========================================================================

def visualize_pixel_waveforms(pixel_signals, config, pixel_coords,
                              vol_idx=0, figsize=(12, 5)):
    """Plot signal vs time for specific pixel coordinates.

    Parameters
    ----------
    pixel_signals : dict
        Keyed by (vol_idx, 0). Dense or sparse format.
    config : SimConfig
    pixel_coords : list of (py, pz) tuples
        Pixel indices to plot.
    vol_idx : int

    Returns
    -------
    matplotlib Figure
    """
    num_time = config.num_time_steps
    time_step = config.time_step_us
    time_axis = np.arange(num_time) * time_step

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    key = (vol_idx, 0)

    if key not in pixel_signals:
        ax.text(0.5, 0.5, '(No data)', ha='center', va='center',
                transform=ax.transAxes, color='grey')
        return fig

    data = pixel_signals[key]

    if isinstance(data, dict):
        # Sparse format
        py_all, pz_all, t_all, v_all = _extract_sparse(data)
        for (py_q, pz_q) in pixel_coords:
            mask = (py_all == py_q) & (pz_all == pz_q)
            if np.any(mask):
                t_us = t_all[mask] * time_step
                order = np.argsort(t_us)
                ax.plot(t_us[order], v_all[mask][order],
                        label=f'({py_q}, {pz_q})', alpha=0.8,
                        marker='.', markersize=2, linestyle='-')
            else:
                ax.plot([], [], label=f'({py_q}, {pz_q}) [empty]')
    else:
        # Dense (py, pz, T)
        arr = np.asarray(data)
        for (py_q, pz_q) in pixel_coords:
            if 0 <= py_q < arr.shape[0] and 0 <= pz_q < arr.shape[1]:
                ax.plot(time_axis, arr[py_q, pz_q, :],
                        label=f'({py_q}, {pz_q})', alpha=0.8)
            else:
                ax.plot([], [], label=f'({py_q}, {pz_q}) [out of range]')

    ax.set_xlabel('Time (us)', fontsize=12)
    ax.set_ylabel('Signal', fontsize=12)
    ax.set_title(f'{_vol_name(vol_idx)} Pixel Waveforms', fontsize=13)
    ax.grid(True, alpha=0.3)
    if len(pixel_coords) <= 12:
        ax.legend(fontsize=9, title='(py, pz)')
    fig.tight_layout()
    return fig


# =========================================================================
# 4. 3D scatter plot of pixel voxels
# =========================================================================

def visualize_pixel_3d(pixel_signals, config, vol_idx=0,
                       figsize=(10, 8), cmap='inferno',
                       threshold=0.0, log_norm=True,
                       max_points=500_000, elev=25, azim=-60,
                       point_size=1.0, alpha=0.6):
    """3D scatter of pixel voxels colored by signal value.

    Parameters
    ----------
    pixel_signals : dict
        Keyed by (vol_idx, 0). Dense or sparse format.
    config : SimConfig
    vol_idx : int
    threshold : float
        Minimum |value| to display.
    max_points : int
        Downsample if more points than this (random subsample).
    elev, azim : float
        Initial 3D view angles.
    point_size : float
        Marker size.
    alpha : float
        Marker alpha.

    Returns
    -------
    matplotlib Figure
    """
    vol = config.volumes[vol_idx]
    num_py, num_pz = vol.pixel_shape
    num_time = config.num_time_steps
    time_step = config.time_step_us
    pitch = vol.pixel_pitch_cm
    y_origin, z_origin = vol.pixel_origins_cm

    key = (vol_idx, 0)
    if key not in pixel_signals:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.text2D(0.5, 0.5, '(No data)', ha='center', va='center',
                  transform=ax.transAxes, color='grey')
        return fig

    sparse = _to_sparse_if_dense(pixel_signals[key], threshold)
    py, pz, t, vals = _extract_sparse(sparse)

    # Threshold filter
    thresh = max(threshold, 1e-30)
    mask = np.abs(vals) >= thresh
    py, pz, t, vals = py[mask], pz[mask], t[mask], vals[mask]

    if len(vals) == 0:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.text2D(0.5, 0.5, '(No signal above threshold)', ha='center',
                  va='center', transform=ax.transAxes, color='grey')
        return fig

    # Downsample if needed
    if len(vals) > max_points:
        idx = np.random.choice(len(vals), max_points, replace=False)
        py, pz, t, vals = py[idx], pz[idx], t[idx], vals[idx]

    # Physical coordinates
    y_cm = y_origin + (py + 0.5) * pitch
    z_cm = z_origin + (pz + 0.5) * pitch
    t_us = t * time_step

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    abs_vals = np.abs(vals)
    if log_norm:
        norm = LogNorm(vmin=max(abs_vals.min(), 1), vmax=abs_vals.max())
    else:
        norm = Normalize(vmin=abs_vals.min(), vmax=abs_vals.max())

    sc = ax.scatter(z_cm, y_cm, t_us, c=abs_vals, cmap=cmap, norm=norm,
                    s=point_size, alpha=alpha, edgecolors='none',
                    rasterized=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('|Signal|', fontsize=11)

    ax.set_xlabel('Z (cm)', fontsize=11)
    ax.set_ylabel('Y (cm)', fontsize=11)
    ax.set_zlabel('Time (us)', fontsize=11)
    ax.set_title(f'{_vol_name(vol_idx)} 3D Pixel Display '
                 f'({len(vals):,} voxels)', fontsize=13)
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    return fig


# =========================================================================
# 5. Active bucket visualization (pixel 6-tuple)
# =========================================================================

def visualize_pixel_buckets(response_signals, config, vol_idx=0,
                            figsize=(18, 5), cmap='hot'):
    """Visualize active 3D pixel buckets via three projections.

    Shows the bucket tile grid projected onto Y-Z, Y-Time, and Z-Time,
    colored by total signal energy per tile.

    Parameters
    ----------
    response_signals : dict
        Raw bucketed output from process_event(). Values are 6-tuples
        (buckets, num_active, compact_to_key, B1, B2, B3).
    config : SimConfig
    vol_idx : int

    Returns
    -------
    matplotlib Figure
    """
    vol = config.volumes[vol_idx]
    num_py, num_pz = vol.pixel_shape
    num_time = config.num_time_steps
    time_step = config.time_step_us
    pitch = vol.pixel_pitch_cm
    y_origin, z_origin = vol.pixel_origins_cm

    key = (vol_idx, 0)
    if key not in response_signals:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax in axes:
            ax.text(0.5, 0.5, '(No data)', ha='center', va='center',
                    transform=ax.transAxes, color='grey')
        return fig

    signal = response_signals[key]
    if not isinstance(signal, tuple) or len(signal) != 6:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax in axes:
            ax.text(0.5, 0.5, '(Not pixel bucketed)', ha='center',
                    va='center', transform=ax.transAxes, color='grey')
        return fig

    buckets, num_active, compact_to_key, B1, B2, B3 = signal
    na = int(num_active)
    B1, B2, B3 = int(B1), int(B2), int(B3)

    NUM_BPZ = (num_pz + B2 - 1) // B2
    NUM_BT = (num_time + B3 - 1) // B3

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor='white')

    if na == 0:
        for ax in axes:
            ax.text(0.5, 0.5, '(No active buckets)', ha='center',
                    va='center', transform=ax.transAxes, color='grey')
        fig.suptitle(f'{_vol_name(vol_idx)} Pixel Buckets (0 active)',
                     fontsize=13)
        return fig

    ctk = np.asarray(compact_to_key[:na])
    bucket_data = np.asarray(buckets[:na])

    # Decode tile origins (pixel indices)
    bpy = ctk // (NUM_BPZ * NUM_BT)
    remainder = ctk % (NUM_BPZ * NUM_BT)
    bpz = remainder // NUM_BT
    bt = remainder % NUM_BT

    py_start = bpy * B1
    pz_start = bpz * B2
    t_start = bt * B3

    # Energy per tile
    energies = np.sum(np.abs(bucket_data), axis=(1, 2, 3))
    energies_log = np.log1p(energies)
    e_max = max(energies_log.max(), 1e-10)
    cmap_obj = plt.cm.get_cmap(cmap)

    # Physical coordinates
    py_start_cm = y_origin + py_start * pitch
    pz_start_cm = z_origin + pz_start * pitch
    t_start_us = t_start * time_step  # t_start is in ticks (bt * B3)

    tile_h_py = B1 * pitch
    tile_h_pz = B2 * pitch
    tile_h_t = B3 * time_step

    y_max = y_origin + num_py * pitch
    z_max = z_origin + num_pz * pitch
    max_time = num_time * time_step

    panels = [
        (axes[0], pz_start_cm, py_start_cm, tile_h_pz, tile_h_py,
         'Y-Z Buckets', 'Z (cm)', 'Y (cm)',
         (z_origin, z_max), (y_origin, y_max)),
        (axes[1], t_start_us, py_start_cm, tile_h_t, tile_h_py,
         'Y-Time Buckets', 'Time (us)', 'Y (cm)',
         (0, max_time), (y_origin, y_max)),
        (axes[2], t_start_us, pz_start_cm, tile_h_t, tile_h_pz,
         'Z-Time Buckets', 'Time (us)', 'Z (cm)',
         (0, max_time), (z_origin, z_max)),
    ]

    for ax, x_starts, y_starts, w, h, title, xlabel, ylabel, xlim, ylim in panels:
        ax.set_facecolor('#1a1a1a')
        for i in range(na):
            color = cmap_obj(energies_log[i] / e_max)
            rect = plt.Rectangle(
                (x_starts[i], y_starts[i]), w, h,
                linewidth=0.3, edgecolor='white', alpha=0.7,
                facecolor=color)
            ax.add_patch(rect)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f'{_vol_name(vol_idx)} {title}', fontsize=12, pad=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

    fig.suptitle(f'Pixel Buckets: {na} active  (B1={B1}, B2={B2}, B3={B3})',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# Multi-volume convenience: auto-detect pixel volumes
# =========================================================================

def visualize_all_pixel_volumes(pixel_signals, config, figsize_per_vol=(18, 5),
                                cmap='inferno', log_norm=False, threshold=0.0,
                                reduce='sum'):
    """Run visualize_pixel_projections for every pixel volume.

    Returns a list of Figures, one per pixel volume.
    """
    figs = []
    for vol_idx in range(config.n_volumes):
        if config.volumes[vol_idx].readout_type != 'pixel':
            continue
        fig = visualize_pixel_projections(
            pixel_signals, config, vol_idx=vol_idx,
            figsize=figsize_per_vol, cmap=cmap,
            log_norm=log_norm, threshold=threshold, reduce=reduce)
        figs.append(fig)
    return figs
