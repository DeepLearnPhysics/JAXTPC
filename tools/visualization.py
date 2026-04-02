"""
Visualization utilities for LArTPC wire signals.

All functions accept SimConfig (from sim.config) for detector geometry.

Data formats:
    - Dense: {(side, plane): (num_wires, num_time) ndarray}
    - Sparse: {(side, plane): {'wire': (N,), 'time': (N,), 'values': (N,)}}
      Produced by sim.to_sparse() or tools.output.to_sparse().

Pass sparse=True/False to indicate which format the data is in.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================================================================
# DeadbandNorm
# =========================================================================

class DeadbandNorm(Normalize):
    """Power-law compression with optional dead zone around zero.

    With gamma < 1: expands weak signals near the deadband, compresses
    strong signals near vmin/vmax. deadband=0 gives pure symmetric
    power-law compression.
    """

    def __init__(self, vmin, vmax, deadband, gamma=0.2, dead_frac=0.08):
        super().__init__(vmin=vmin, vmax=vmax)
        self.deadband = abs(deadband)
        self.gamma = gamma
        self.dead_frac = dead_frac if self.deadband > 0 else 0.0

    def __call__(self, value, clip=None):
        x = np.ma.asarray(value, dtype=float)
        mask = np.ma.getmask(x)
        x_filled = np.ma.filled(x, 0.0)

        result = np.full_like(x_filled, 0.5, dtype=float)
        half = self.dead_frac / 2.0
        signal_range = 0.5 - half

        if self.deadband > 0:
            neg = x_filled < -self.deadband
            if np.any(neg):
                denom = -self.deadband - self.vmin
                if abs(denom) > 1e-30:
                    t = (-self.deadband - x_filled[neg]) / denom
                    result[neg] = (0.5 - half) - signal_range * np.clip(t, 0, 1) ** self.gamma

            pos = x_filled > self.deadband
            if np.any(pos):
                denom = self.vmax - self.deadband
                if abs(denom) > 1e-30:
                    t = (x_filled[pos] - self.deadband) / denom
                    result[pos] = (0.5 + half) + signal_range * np.clip(t, 0, 1) ** self.gamma
        else:
            neg = x_filled < 0
            if np.any(neg) and abs(self.vmin) > 1e-30:
                t = -x_filled[neg] / (-self.vmin)
                result[neg] = 0.5 - 0.5 * np.clip(t, 0, 1) ** self.gamma

            pos = x_filled > 0
            if np.any(pos) and abs(self.vmax) > 1e-30:
                t = x_filled[pos] / self.vmax
                result[pos] = 0.5 + 0.5 * np.clip(t, 0, 1) ** self.gamma

        return np.ma.array(np.clip(result, 0, 1), mask=mask)

    def inverse(self, value):
        y = np.asarray(value, dtype=float)
        result = np.zeros_like(y)
        half = self.dead_frac / 2.0
        signal_range = 0.5 - half

        if self.deadband > 0:
            neg = y < (0.5 - half)
            if np.any(neg):
                t_g = np.clip(((0.5 - half) - y[neg]) / signal_range, 0, 1)
                t = t_g ** (1.0 / self.gamma)
                result[neg] = -self.deadband - t * (-self.deadband - self.vmin)

            dead = (y >= (0.5 - half)) & (y <= (0.5 + half))
            if np.any(dead):
                t = (y[dead] - (0.5 - half)) / self.dead_frac
                result[dead] = t * 2.0 * self.deadband - self.deadband

            pos = y > (0.5 + half)
            if np.any(pos):
                t_g = np.clip((y[pos] - (0.5 + half)) / signal_range, 0, 1)
                t = t_g ** (1.0 / self.gamma)
                result[pos] = self.deadband + t * (self.vmax - self.deadband)
        else:
            neg = y < 0.5
            if np.any(neg):
                t_g = np.clip((0.5 - y[neg]) / 0.5, 0, 1)
                t = t_g ** (1.0 / self.gamma)
                result[neg] = -t * (-self.vmin)

            pos = y > 0.5
            if np.any(pos):
                t_g = np.clip((y[pos] - 0.5) / 0.5, 0, 1)
                t = t_g ** (1.0 / self.gamma)
                result[pos] = t * self.vmax

            result[y == 0.5] = 0.0

        return result


# =========================================================================
# Colormap and helpers
# =========================================================================

_OBSIDIAN = LinearSegmentedColormap.from_list('obsidian', [
    (0.0,  '#E0FFFF'), (0.2,  '#00E5FF'), (0.35, '#0088AA'),
    (0.5,  '#0A0A0A'),
    (0.65, '#AA5500'), (0.8,  '#FF8800'), (1.0,  '#FFEECC'),
])

_PLANE_TYPE_LABELS = {
    0: '1st Induction (U)',
    1: '2nd Induction (V)',
    2: 'Collection (Y)',
}
_PLANE_NORM_KEYS = {0: 'U-plane', 1: 'V-plane', 2: 'Y-plane'}


def _vol_name(vol_idx):
    return f'Volume {vol_idx}'


def _plane_type(plane_idx):
    return _PLANE_TYPE_LABELS.get(plane_idx, f'Plane {plane_idx}')


def _plane_norm_key(plane_idx):
    return _PLANE_NORM_KEYS.get(plane_idx, f'plane_{plane_idx}')


def _resolve_cmap(cmap):
    if cmap == 'obsidian':
        return _OBSIDIAN
    elif isinstance(cmap, str):
        return plt.cm.get_cmap(cmap)
    return cmap


def _add_colorbar(fig, ax, mappable, norm, label_size=12, tick_size=10):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.0)
    cbar = fig.colorbar(mappable, cax=cax)
    if isinstance(norm, DeadbandNorm):
        # Evenly spaced in norm-space, converted to data values
        tick_norm = np.linspace(0, 1, 7)
        tick_values = norm.inverse(tick_norm)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{v:.0f}' for v in tick_values])
    cbar.ax.tick_params(labelsize=tick_size, colors='black')
    cbar.set_label('Signal (ADC)', fontsize=label_size, color='black')
    return cbar


def _compute_marker_size(figsize, n_rows, n_cols, num_wires, max_time):
    """Compute scatter marker size (points²) to fill ~1 data bin.

    Estimates axes size from figure layout and computes a square marker
    that covers approximately one (wire, time_step) cell. Returns at
    least 0.01 so markers are always visible.

    Parameters
    ----------
    figsize : tuple (width, height) in inches
    n_rows, n_cols : grid dimensions
    num_wires : int, data range in wire direction
    max_time : float, data range in time direction (us)
    """
    # Approximate axes size (accounting for spacing/labels ~70% of cell)
    ax_w_inches = figsize[0] / n_cols * 0.7
    ax_h_inches = figsize[1] / n_rows * 0.7
    # With set_box_aspect(1), axes are square in display
    ax_inches = min(ax_w_inches, ax_h_inches)

    # Points per data unit (72 points per inch)
    pts_per_wire = (ax_inches * 72) / max(num_wires, 1)
    pts_per_tick = (ax_inches * 72) / max(max_time, 1e-6)
    side = min(pts_per_wire, pts_per_tick)
    return max(side ** 2, 0.01)


def _extract_viz_params(config):
    """Extract visualization parameters from SimConfig."""
    num_wires = {
        (v, p): config.volumes[v].num_wires[p]
        for v in range(config.n_volumes)
        for p in range(config.volumes[v].n_planes)
    }
    return {
        'num_time_steps': config.num_time_steps,
        'time_step_us': config.time_step_us,
        'num_wires': num_wires,
        'electrons_per_adc': config.electrons_per_adc,
    }


def _get_values(data, sparse):
    """Extract values array from dense or sparse format."""
    if sparse:
        if not isinstance(data, dict):
            raise ValueError("sparse=True but data is not a dict. Use sim.to_sparse().")
        return np.asarray(data['values'])
    if isinstance(data, dict):
        raise ValueError("sparse=False but data is a dict. Use sim.to_dense() or pass sparse=True.")
    return np.asarray(data)


def _build_colorize_fn(top_tracks_by_charge, max_tracks):
    """Build a colorize function and color map for track visualization."""
    distinct_colors = [
        '#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF',
        '#FFD700', '#FF8C00', '#8B008B', '#228B22', '#4B0082',
        '#FF1493', '#00CED1', '#FF4500', '#9400D3', '#32CD32',
        '#8B4513', '#20B2AA', '#FF69B4', '#4169E1', '#DC143C',
    ]
    distinct_rgba = [mcolors.to_rgba(c) for c in distinct_colors]
    top_tracks = [tid for tid, _ in top_tracks_by_charge[:max_tracks]]
    top_color_map = {tid: distinct_rgba[i]
                     for i, tid in enumerate(top_tracks[:len(distinct_colors)])}
    hash_cmap = plt.cm.hsv

    def colorize(track_ids):
        tids = np.asarray(track_ids, dtype=np.int64)
        colors = hash_cmap(((tids * 2654435761) % 2**32) / (2**32 - 1))
        for tid, color in top_color_map.items():
            colors[tids == tid] = color
        return colors

    return colorize, top_tracks, top_color_map, hash_cmap


def _diffused_charge_setup(valid_values, threshold, log_norm):
    """Compute norm, cmap, and background for diffused charge plots.

    Parameters
    ----------
    valid_values : ndarray
        Concatenated values above threshold across all planes.
    threshold : float
    log_norm : bool

    Returns
    -------
    norm, cmap_obj, background
    """
    if len(valid_values) > 0:
        p1, p99 = np.percentile(valid_values, [1, 99])
        vmin_plot = max(threshold, p1)
        vmax_plot = p99
    else:
        vmin_plot, vmax_plot = threshold, threshold + 1

    if log_norm:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(vmin_plot, 1), vmax=vmax_plot)
    else:
        norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)

    background = '#1a1a1a'
    cmap_obj = plt.cm.YlOrRd.copy()
    cmap_obj.set_bad(color=background)
    return norm, cmap_obj, background


# =========================================================================
# Response signal visualization
# =========================================================================

def visualize_wire_signals(wire_signals, config, figsize=None,
                           threshold_enc=0, gamma=0.2, cmap='obsidian',
                           sparse=False, point_size=None):
    """Visualize wire signals for all planes.

    Parameters
    ----------
    wire_signals : dict
        Keyed by (side, plane). Dense: (W, T) arrays. Sparse: dicts with
        'wire', 'time', 'values' keys.
    config : SimConfig
    sparse : bool
        True if data is in sparse dict format.
    """
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    num_wires = vp['num_wires']
    deadband_adc = threshold_enc / vp['electrons_per_adc']
    resolved_cmap = _resolve_cmap(cmap)
    max_time = num_time * time_step

    # Per-plane-type min/max
    plane_ranges = {'U-plane': [np.inf, -np.inf],
                    'V-plane': [np.inf, -np.inf],
                    'Y-plane': [np.inf, -np.inf]}

    for s in range(config.n_volumes):
        for p in range(config.volumes[s].n_planes):
            if (s, p) in wire_signals and num_wires[(s, p)] > 0:
                vals = _get_values(wire_signals[(s, p)], sparse)
                if len(vals) > 0:
                    pname = _plane_norm_key(p)
                    if pname in plane_ranges:
                        plane_ranges[pname][0] = min(plane_ranges[pname][0], vals.min())
                        plane_ranges[pname][1] = max(plane_ranges[pname][1], vals.max())

    for pname, (vmin, vmax) in plane_ranges.items():
        if vmin == np.inf:
            plane_ranges[pname] = [-25, 25]
        elif pname == 'Y-plane':
            m = max(abs(vmin), abs(vmax))
            plane_ranges[pname] = [-m, m]

    n_vol = config.n_volumes
    max_planes = max(v.n_planes for v in config.volumes)
    if figsize is None:
        figsize = (20, 5 * n_vol)
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(n_vol, max_planes, figure=fig, hspace=0.35, wspace=0.30)
    zero_color = resolved_cmap(0.5)

    for vol_idx in range(config.n_volumes):
        for plane_idx in range(config.volumes[vol_idx].n_planes):
            ax = fig.add_subplot(gs[vol_idx, plane_idx])
            ax.set_facecolor(zero_color)
            ax.grid(False)
            nw = int(num_wires[(vol_idx, plane_idx)])
            title = f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}"
            pname = _plane_norm_key(plane_idx)

            if (vol_idx, plane_idx) not in wire_signals or nw == 0:
                ax.text(0.5, 0.5, "(No data)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)
            else:
                vmin, vmax = plane_ranges[pname]
                norm = DeadbandNorm(vmin, vmax, deadband_adc, gamma)

                if sparse:
                    d = wire_signals[(vol_idx, plane_idx)]
                    w = np.asarray(d['wire'])
                    t = np.asarray(d['time']) * time_step
                    v = np.asarray(d['values'])
                    if len(v) > 0:
                        ps = point_size if point_size is not None else \
                            _compute_marker_size(figsize, n_vol, max_planes, nw, max_time)
                        sc = ax.scatter(w, t, c=v, cmap=resolved_cmap, norm=norm,
                                        s=ps, marker='s', linewidths=0,
                                        edgecolors='none', rasterized=True)
                        _add_colorbar(fig, ax, sc, norm)
                else:
                    arr = np.asarray(wire_signals[(vol_idx, plane_idx)])
                    im = ax.imshow(arr.T, aspect='auto', origin='lower',
                                   extent=[0, nw, 0, max_time],
                                   cmap=resolved_cmap, norm=norm,
                                   interpolation='nearest')
                    _add_colorbar(fig, ax, im, norm)

            ax.set_xlim(0, nw)
            ax.set_ylim(0, max_time)
            ax.set_box_aspect(1)
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel('Wire Index', fontsize=12)
            ax.set_ylabel('Time (us)', fontsize=12)

    return fig


def visualize_single_plane(wire_signals, config, vol_idx=0, plane_idx=0,
                           figsize=(10, 10), threshold_enc=0, gamma=0.2,
                           cmap='obsidian', sparse=False, point_size=0.5):
    """Visualize a single plane."""
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    nw = int(vp['num_wires'][(vol_idx, plane_idx)])
    deadband_adc = threshold_enc / vp['electrons_per_adc']
    resolved_cmap = _resolve_cmap(cmap)
    max_time = num_time * time_step

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(resolved_cmap(0.5))
    ax.grid(False)

    key = (vol_idx, plane_idx)
    if key in wire_signals and nw > 0:
        vals = _get_values(wire_signals[key], sparse)
        if len(vals) > 0:
            vmin, vmax = vals.min(), vals.max()
            if plane_idx == 2:  # Y-plane symmetric
                m = max(abs(vmin), abs(vmax))
                vmin, vmax = -m, m
            norm = DeadbandNorm(vmin, vmax, deadband_adc, gamma)

            if sparse:
                d = wire_signals[key]
                sc = ax.scatter(np.asarray(d['wire']),
                                np.asarray(d['time']) * time_step,
                                c=np.asarray(d['values']),
                                cmap=resolved_cmap, norm=norm,
                                s=point_size, marker='s', linewidths=0)
                _add_colorbar(fig, ax, sc, norm)
            else:
                arr = np.asarray(wire_signals[key])
                im = ax.imshow(arr.T, aspect='auto', origin='lower',
                               extent=[0, nw, 0, max_time],
                               cmap=resolved_cmap, norm=norm,
                               interpolation='nearest')
                _add_colorbar(fig, ax, im, norm)

    ax.set_xlim(0, nw)
    ax.set_ylim(0, max_time)
    ax.set_box_aspect(1)
    ax.set_title(f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}", fontsize=14)
    ax.set_xlabel('Wire Index', fontsize=12)
    ax.set_ylabel('Time (us)', fontsize=12)

    return fig


# =========================================================================
# Truth hits (diffused charge) visualization
# =========================================================================

def visualize_diffused_charge(wire_signals, config, figsize=None,
                              log_norm=False, threshold=100,
                              sparse=False, point_size=None):
    """Visualize diffused charge (truth hits) for all planes.

    Parameters
    ----------
    wire_signals : dict
        Keyed by (vol, plane). Dense: (W, T) arrays. Sparse: dict with
        'wire', 'time', 'values' arrays.
    config : SimConfig
    log_norm : bool
        Use log normalization.
    threshold : float
        Values below this are masked (dense) or filtered (sparse).
    sparse : bool
        If True, data is sparse format. Uses scatter instead of imshow.
    point_size : float
        Marker size for sparse scatter plot.
    """
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    num_wires = vp['num_wires']
    max_time = num_time * time_step

    # Collect all valid values for global percentile clipping
    all_valid = []
    for s in range(config.n_volumes):
        for p in range(config.volumes[s].n_planes):
            if (s, p) not in wire_signals or num_wires[(s, p)] == 0:
                continue
            if sparse:
                d = wire_signals[(s, p)]
                v = np.asarray(d['values'])
                valid = v[v > threshold]
            else:
                arr = np.asarray(wire_signals[(s, p)])
                valid = arr[arr > threshold]
            if len(valid) > 0:
                all_valid.append(valid)

    concat = np.concatenate(all_valid) if all_valid else np.array([])
    norm, cmap_obj, background = _diffused_charge_setup(concat, threshold, log_norm)

    n_vol = config.n_volumes
    max_planes = max(v.n_planes for v in config.volumes)
    if figsize is None:
        figsize = (20, 5 * n_vol)
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(n_vol, max_planes, figure=fig, hspace=0.35, wspace=0.30)

    for vol_idx in range(config.n_volumes):
        for plane_idx in range(config.volumes[vol_idx].n_planes):
            ax = fig.add_subplot(gs[vol_idx, plane_idx])
            ax.set_facecolor(background)
            ax.grid(True, alpha=0.3, color='#505050', linestyle='--', linewidth=0.5)
            nw = int(num_wires[(vol_idx, plane_idx)])
            title = f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}"

            if (vol_idx, plane_idx) not in wire_signals or nw == 0:
                ax.text(0.5, 0.5, "(No data)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)
            elif sparse:
                d = wire_signals[(vol_idx, plane_idx)]
                w = np.asarray(d['wire'])
                t = np.asarray(d['time']) * time_step
                v = np.asarray(d['values'])
                mask = v > threshold
                if mask.any():
                    ps = point_size if point_size is not None else \
                        _compute_marker_size(figsize, n_vol, max_planes, nw, max_time)
                    sc = ax.scatter(w[mask], t[mask], c=v[mask],
                                    cmap=cmap_obj, norm=norm,
                                    s=ps, marker='s', linewidths=0,
                                    edgecolors='none', rasterized=True)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.0)
                    fig.colorbar(sc, cax=cax)
            else:
                arr = np.asarray(wire_signals[(vol_idx, plane_idx)])
                masked = np.ma.masked_where(arr <= threshold, arr)
                im = ax.imshow(masked.T, aspect='auto', origin='lower',
                               extent=[0, nw, 0, max_time],
                               cmap=cmap_obj, norm=norm, interpolation='nearest')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.0)
                fig.colorbar(im, cax=cax)

            ax.set_xlim(0, nw)
            ax.set_ylim(0, max_time)
            ax.set_box_aspect(1)
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel('Wire Index', fontsize=12)
            ax.set_ylabel('Time (us)', fontsize=12)

    return fig


# =========================================================================
# Track label visualization
# =========================================================================

def get_top_tracks_by_charge(track_hits, top_n=20):
    """Get top N tracks sorted by total charge.

    Parameters
    ----------
    track_hits : dict
        Finalized track hits from sim.finalize_track_hits().

    Returns
    -------
    list of (track_id, total_charge) tuples.
    """
    all_tids = []
    all_charges = []

    for plane_key, data in track_hits.items():
        nl = int(data['num_labeled'])
        if nl > 0:
            all_tids.append(np.asarray(data['labeled_track_ids'][:nl], dtype=np.int32))
            all_charges.append(np.asarray(data['labeled_hits'][:nl, 2]))

    if not all_tids:
        return []

    tids = np.concatenate(all_tids)
    charges = np.concatenate(all_charges)

    order = np.argsort(tids)
    s_tids = tids[order]
    s_charges = charges[order]

    boundaries = np.ones(len(s_tids), dtype=bool)
    boundaries[1:] = s_tids[1:] != s_tids[:-1]
    starts = np.where(boundaries)[0]

    unique_tids = s_tids[starts]
    total_charges = np.add.reduceat(s_charges, starts)

    top_order = np.argsort(-total_charges)[:top_n]
    return [(int(unique_tids[i]), float(total_charges[i])) for i in top_order]


def visualize_track_labels(track_hits, config, top_tracks_by_charge,
                           max_tracks=15, figsize=None):
    """Visualize track labels with distinct colors for top tracks.

    Parameters
    ----------
    track_hits : dict
        Finalized track hits from sim.finalize_track_hits().
    config : SimConfig
    top_tracks_by_charge : list
        From get_top_tracks_by_charge().
    """
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    num_wires = vp['num_wires']
    max_time = num_time * time_step

    colorize, top_tracks, top_color_map, hash_cmap = _build_colorize_fn(
        top_tracks_by_charge, max_tracks)

    n_vol = config.n_volumes
    max_planes = max(v.n_planes for v in config.volumes)
    if figsize is None:
        figsize = (20, 6 * n_vol)
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(n_vol, max_planes + 1, figure=fig, hspace=0.35, wspace=0.30,
                           width_ratios=[1] * max_planes + [0.12])

    for vol_idx in range(config.n_volumes):
        for plane_idx in range(config.volumes[vol_idx].n_planes):
            ax = fig.add_subplot(gs[vol_idx, plane_idx])
            ax.set_facecolor('black')
            nw = int(num_wires[(vol_idx, plane_idx)])

            key = (vol_idx, plane_idx)
            data = track_hits[key]
            nl = int(data['num_labeled'])

            if nl > 0:
                labeled = np.asarray(data['labeled_hits'][:nl])
                tids = np.asarray(data['labeled_track_ids'][:nl])
                wires = labeled[:, 0]
                times = labeled[:, 1] * time_step
                colors = colorize(tids)
                ax.scatter(wires, times, c=colors, s=0.5, alpha=0.8)
                ax.text(0.02, 0.98, f"{nl:,} hits\n{len(np.unique(tids))} tracks",
                        transform=ax.transAxes, va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
            else:
                ax.text(0.5, 0.5, "(No labeled hits)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)

            ax.set_xlim(0, nw)
            ax.set_ylim(0, max_time)
            ax.set_box_aspect(1)
            ax.set_title(f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}",
                         fontsize=14, pad=10)
            ax.set_xlabel('Wire Index', fontsize=12)
            ax.set_ylabel('Time (us)', fontsize=12)

    # Legend
    if top_tracks:
        cbar_ax = fig.add_subplot(gs[:, max_planes])
        n_show = min(len(top_tracks), max_tracks)
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, n_show)
        for i, tid in enumerate(top_tracks[:n_show]):
            color = top_color_map.get(tid, hash_cmap((tid * 2654435761 % 2**32) / (2**32 - 1)))
            y = n_show - 1 - i
            cbar_ax.add_patch(plt.Rectangle((0, y), 0.4, 0.9,
                              facecolor=color, edgecolor='black', linewidth=0.5))
            cbar_ax.text(0.5, y + 0.45, f'Track {tid}', ha='left', va='center', fontsize=8)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        cbar_ax.set_title(f'Top {n_show} Tracks\n(by charge)', fontsize=11, pad=10)
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    return fig


# =========================================================================
# Wire waveform visualization
# =========================================================================

def visualize_waveforms(wire_signals, config, wire_indices, vol_idx=0,
                        plane_idx=0, figsize=(12, 6), sparse=False):
    """Plot wire signal waveforms at specific wire indices.

    Parameters
    ----------
    wire_signals : dict
        Keyed by (side, plane). Dense or sparse format.
    config : SimConfig
    wire_indices : list of int
        Wire indices to plot.
    sparse : bool
        True if data is sparse dict format.
    """
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    time_axis = np.arange(num_time) * time_step

    fig, ax = plt.subplots(figsize=figsize)
    key = (vol_idx, plane_idx)

    if key not in wire_signals:
        ax.text(0.5, 0.5, "(No data)", ha='center', va='center', transform=ax.transAxes)
    elif sparse:
        d = wire_signals[key]
        w = np.asarray(d['wire'])
        t = np.asarray(d['time'])
        v = np.asarray(d['values'])
        for wi in wire_indices:
            mask = w == wi
            if np.any(mask):
                t_us = t[mask] * time_step
                order = np.argsort(t_us)
                ax.plot(t_us[order], v[mask][order], label=f'Wire {wi}', alpha=0.8,
                        marker='.', markersize=1, linestyle='-')
    else:
        arr = np.asarray(wire_signals[key])
        for wi in wire_indices:
            if 0 <= wi < arr.shape[0]:
                ax.plot(time_axis, arr[wi], label=f'Wire {wi}', alpha=0.8)

    ax.set_xlabel('Time (us)', fontsize=12)
    ax.set_ylabel('Signal (ADC)', fontsize=12)
    ax.set_title(f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}", fontsize=14)
    ax.grid(True, alpha=0.3)
    if len(wire_indices) <= 10:
        ax.legend(fontsize=9)

    return fig


# =========================================================================
# Single-plane variants
# =========================================================================

def visualize_diffused_charge_single_plane(wire_signals, config, vol_idx=0,
                                           plane_idx=0, figsize=(10, 10),
                                           log_norm=False, threshold=100):
    """Visualize diffused charge for a single plane. Dense only."""
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    nw = int(vp['num_wires'][(vol_idx, plane_idx)])
    max_time = num_time * time_step

    key = (vol_idx, plane_idx)
    arr = np.asarray(wire_signals[key]) if key in wire_signals else np.zeros((nw, num_time))
    valid = arr[arr > threshold]
    norm, cmap_obj, background = _diffused_charge_setup(valid, threshold, log_norm)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(background)
    ax.grid(True, alpha=0.3, color='#505050', linestyle='--', linewidth=0.5)

    if nw > 0 and key in wire_signals:
        masked = np.ma.masked_where(arr <= threshold, arr)
        im = ax.imshow(masked.T, aspect='auto', origin='lower',
                       extent=[0, nw, 0, max_time],
                       cmap=cmap_obj, norm=norm, interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        fig.colorbar(im, cax=cax)

    ax.set_xlim(0, nw)
    ax.set_ylim(0, max_time)
    ax.set_box_aspect(1)
    ax.set_title(f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}", fontsize=14)
    ax.set_xlabel('Wire Index', fontsize=12)
    ax.set_ylabel('Time (us)', fontsize=12)

    return fig


def visualize_track_labels_single_plane(track_hits, config, top_tracks_by_charge,
                                        vol_idx=0, plane_idx=0,
                                        max_tracks=15, figsize=(12, 10)):
    """Visualize track labels for a single plane."""
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    nw = int(vp['num_wires'][(vol_idx, plane_idx)])
    max_time = num_time * time_step

    colorize, top_tracks, top_color_map, hash_cmap = _build_colorize_fn(
        top_tracks_by_charge, max_tracks)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('black')

    key = (vol_idx, plane_idx)
    data = track_hits[key]
    nl = int(data['num_labeled'])

    if nl > 0:
        labeled = np.asarray(data['labeled_hits'][:nl])
        tids = np.asarray(data['labeled_track_ids'][:nl])
        wires = labeled[:, 0]
        times = labeled[:, 1] * time_step
        colors = colorize(tids)
        ax.scatter(wires, times, c=colors, s=1.0, alpha=0.8)
        ax.text(0.02, 0.98, f"{nl:,} hits\n{len(np.unique(tids))} tracks",
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8), fontsize=11)
    else:
        ax.text(0.5, 0.5, "(No labeled hits)", color='grey',
                ha='center', va='center', transform=ax.transAxes)

    # Legend
    n_show = min(len(top_tracks), max_tracks)
    if n_show > 0:
        for i, tid in enumerate(top_tracks[:n_show]):
            color = top_color_map.get(tid, hash_cmap((tid * 2654435761 % 2**32) / (2**32 - 1)))
            ax.plot([], [], 's', color=color, markersize=8, label=f'Track {tid}')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    ax.set_xlim(0, nw)
    ax.set_ylim(0, max_time)
    ax.set_box_aspect(1)
    ax.set_title(f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}", fontsize=14)
    ax.set_xlabel('Wire Index', fontsize=12)
    ax.set_ylabel('Time (us)', fontsize=12)

    return fig


# =========================================================================
# Bucket debug visualization
# =========================================================================

def visualize_active_buckets(response_signals, config, figsize=None):
    """Visualize active bucket occupancy from bucketed simulation output.

    Parameters
    ----------
    response_signals : dict
        Raw bucketed output from process_event(). Values are 5-tuples
        (buckets, num_active, compact_to_key, B1, B2).
    config : SimConfig
    """
    vp = _extract_viz_params(config)
    num_time = vp['num_time_steps']
    time_step = vp['time_step_us']
    num_wires = vp['num_wires']
    max_time = num_time * time_step

    n_vol = config.n_volumes
    max_planes = max(v.n_planes for v in config.volumes)
    if figsize is None:
        figsize = (20, 5 * n_vol)
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(n_vol, max_planes, figure=fig, hspace=0.35, wspace=0.30)

    for vol_idx in range(config.n_volumes):
        for plane_idx in range(config.volumes[vol_idx].n_planes):
            ax = fig.add_subplot(gs[vol_idx, plane_idx])
            ax.set_facecolor('#1a1a1a')
            nw = int(num_wires[(vol_idx, plane_idx)])
            key = (vol_idx, plane_idx)
            title = f"{_vol_name(vol_idx)} {_plane_type(plane_idx)}"

            if key not in response_signals:
                ax.text(0.5, 0.5, "(No data)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=14, pad=10)
                continue

            signal = response_signals[key]
            if not isinstance(signal, tuple) or len(signal) != 5:
                ax.text(0.5, 0.5, "(Not bucketed)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=14, pad=10)
                continue

            buckets, num_active, compact_to_key, B1, B2 = signal
            num_active_int = int(num_active)
            B1_int = int(B1)
            B2_int = int(B2)

            num_buckets_t = (num_time + B2_int - 1) // B2_int

            if num_active_int > 0:
                ctk = np.asarray(compact_to_key[:num_active_int])
                bw = ctk // num_buckets_t
                bt = ctk % num_buckets_t

                wire_starts = bw * B1_int
                time_starts = bt * B2_int * time_step

                # Color by bucket signal energy
                bucket_data = np.asarray(buckets[:num_active_int])
                energies = np.sum(np.abs(bucket_data), axis=(1, 2))
                energies_log = np.log1p(energies)

                for i in range(num_active_int):
                    rect = plt.Rectangle(
                        (wire_starts[i], time_starts[i]),
                        B1_int, B2_int * time_step,
                        linewidth=0.3, edgecolor='white', alpha=0.7,
                        facecolor=plt.cm.hot(energies_log[i] / max(energies_log.max(), 1)))
                    ax.add_patch(rect)

                ax.text(0.02, 0.98,
                        f"{num_active_int} active\nB1={B1_int}, B2={B2_int}",
                        transform=ax.transAxes, va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

            ax.set_xlim(0, nw)
            ax.set_ylim(0, max_time)
            ax.set_box_aspect(1)
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel('Wire Index', fontsize=12)
            ax.set_ylabel('Time (us)', fontsize=12)

    return fig


# =========================================================================
# Wire geometry visualization (uses raw detector_config, not SimConfig)
# =========================================================================

def visualize_wire_planes(detector_config, figsize=None):
    """Visualize wire plane geometry colored by wire index.

    This function uses the raw detector_config dict (from generate_detector),
    not SimConfig, because it needs wire plane angle/spacing/distance info.
    """
    volumes = detector_config['volumes']
    n_vol = len(volumes)
    max_planes = max(len(v['planes']) for v in volumes)
    # Use first volume's y/z ranges for display dimensions
    ranges = volumes[0]['geometry']['ranges']
    det_y = ranges[1][1] - ranges[1][0]
    det_z = ranges[2][1] - ranges[2][0]

    if figsize is None:
        figsize = (15, 5 * n_vol)
    fig, axes = plt.subplots(n_vol, max_planes, figsize=figsize)
    if n_vol == 1:
        axes = axes[np.newaxis, :]
    cmap = plt.cm.viridis

    for vol_idx, vol_cfg in enumerate(volumes):
        for plane_idx, plane in enumerate(vol_cfg['planes']):
            ax = axes[vol_idx, plane_idx]
            angle_rad = np.radians(plane['angle'])
            spacing = plane['wire_spacing']

            ax.add_patch(plt.Rectangle((0, 0), det_z, det_y,
                         fill=False, color='black', linestyle='--'))

            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            corners = [(0, 0), (det_y, 0), (0, det_z), (det_y, det_z)]
            r_vals = [z * cos_a + y * sin_a for y, z in corners]
            r_min, r_max = min(r_vals), max(r_vals)

            offset = int(np.abs(np.floor(r_min / spacing))) + 1 if r_min < 0 else 0
            idx_min = int(np.floor(r_min / spacing)) + offset
            idx_max = int(np.ceil(r_max / spacing)) + offset
            n_wires = idx_max - idx_min + 1

            for wi in range(idx_min, idx_max + 1):
                r = (wi - offset) * spacing
                pts = []
                if abs(cos_a) > 1e-10:
                    z = r / cos_a
                    if 0 <= z <= det_z: pts.append((0, z))
                    z = (r - det_y * sin_a) / cos_a
                    if 0 <= z <= det_z: pts.append((det_y, z))
                if abs(sin_a) > 1e-10:
                    y = r / sin_a
                    if 0 <= y <= det_y: pts.append((y, 0))
                    y = (r - det_z * cos_a) / sin_a
                    if 0 <= y <= det_y: pts.append((y, det_z))

                # Deduplicate
                unique = []
                for pt in pts:
                    if not any(abs(pt[0]-u[0]) < 1e-6 and abs(pt[1]-u[1]) < 1e-6 for u in unique):
                        unique.append(pt)

                if len(unique) >= 2:
                    unique.sort(key=lambda pt: (pt[1], pt[0]))
                    p1, p2 = unique[0], unique[-1]
                    color = cmap((wi - idx_min) / max(1, n_wires - 1))
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color,
                            linewidth=0.8, alpha=0.7)

            ax.set_title(f"Vol {vol_idx} {_plane_type(plane_idx)}")
            ax.set_xlabel('Z (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_xlim(0, det_z)
            ax.set_ylim(0, det_y)
            ax.grid(alpha=0.3)
            ax.text(0.05, 0.95, f"Angle: {plane['angle']}°\nSpacing: {spacing} cm",
                    transform=ax.transAxes, va='top',
                    bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    return fig
