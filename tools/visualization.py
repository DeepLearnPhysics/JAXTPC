"""
Visualization utilities for LArTPC wire signals.

This module provides functions for visualizing wire signals from TPC simulations,
including multi-plane displays with power-law compression, track coloring, and
customizable color schemes for different plane types (U, V, Y).

Supports both dense and sparse data formats:
    - Dense: (num_wires, num_time_steps) arrays
    - Sparse: tuples of (indices, values) where
        - indices: (N, 2) int32 array with [wire_idx, time_idx] per row (relative)
        - values: (N,) float32 array with signal values

Use `sparse_data=True` parameter to enable sparse visualization mode.
"""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------------
# DeadbandNorm: power-law compression with optional dead zone
# ---------------------------------------------------------------------------

class DeadbandNorm(Normalize):
    """
    Matplotlib normalization with gamma power-law compression and optional deadband.

    Values inside ``[-deadband, +deadband]`` are linearly interpolated across a
    narrow band centered at 0.5 in the colormap.  Values outside are
    gamma-compressed: ``t^gamma`` where ``t`` measures fractional distance FROM
    the deadband boundary TOWARD the data extreme.  With ``gamma < 1`` this
    **expands** weak signals near the deadband (giving them more colour
    contrast) and **compresses** strong signals near vmin/vmax.

    When ``deadband=0`` this reduces to pure symmetric power-law compression
    around zero.

    Parameters
    ----------
    vmin, vmax : float
        Data limits.
    deadband : float
        Half-width of the dead zone in data units (>=0).
    gamma : float
        Power-law exponent (0 < gamma <= 1). Smaller = more compression
        of extreme values / more expansion of weak signals.
    dead_frac : float
        Fraction of the [0,1] colormap range reserved for the dead zone.
        Ignored when deadband is 0.
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
        signal_range = 0.5 - half  # colormap space available per side

        if self.deadband > 0:
            # Negative signal region: [vmin, -deadband]
            neg = x_filled < -self.deadband
            if np.any(neg):
                denom = -self.deadband - self.vmin
                if abs(denom) > 1e-30:
                    t = (-self.deadband - x_filled[neg]) / denom
                    result[neg] = (0.5 - half) - signal_range * np.clip(t, 0, 1) ** self.gamma

            # Dead band: [-deadband, +deadband] → hard threshold to 0.5
            # (default result is already 0.5, no action needed)

            # Positive signal region: [+deadband, vmax]
            pos = x_filled > self.deadband
            if np.any(pos):
                denom = self.vmax - self.deadband
                if abs(denom) > 1e-30:
                    t = (x_filled[pos] - self.deadband) / denom
                    result[pos] = (0.5 + half) + signal_range * np.clip(t, 0, 1) ** self.gamma
        else:
            # No deadband: pure power-law compression around zero
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
        """Map [0, 1] norm-space back to data-space for colorbar rendering."""
        y = np.asarray(value, dtype=float)
        result = np.zeros_like(y)
        half = self.dead_frac / 2.0
        signal_range = 0.5 - half

        if self.deadband > 0:
            # Negative signal region: y < (0.5 - half)
            neg = y < (0.5 - half)
            if np.any(neg):
                t_g = np.clip(((0.5 - half) - y[neg]) / signal_range, 0, 1)
                t = t_g ** (1.0 / self.gamma)
                result[neg] = -self.deadband - t * (-self.deadband - self.vmin)

            # Dead band: (0.5-half) <= y <= (0.5+half)
            dead = (y >= (0.5 - half)) & (y <= (0.5 + half))
            if np.any(dead):
                t = (y[dead] - (0.5 - half)) / self.dead_frac
                result[dead] = t * 2.0 * self.deadband - self.deadband

            # Positive signal region: y > (0.5 + half)
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


# ---------------------------------------------------------------------------
# Obsidian colormap: cool-toned negatives, dark center, warm-toned positives
# ---------------------------------------------------------------------------

_OBSIDIAN = LinearSegmentedColormap.from_list('obsidian', [
    (0.0,  '#E0FFFF'),
    (0.2,  '#00E5FF'),
    (0.35, '#0088AA'),
    (0.5,  '#0A0A0A'),
    (0.65, '#AA5500'),
    (0.8,  '#FF8800'),
    (1.0,  '#FFEECC'),
])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_viz_params(detector_config):
    """Extract the five visualisation keys from a detector_config dict."""
    return {
        'num_time_steps': detector_config['num_time_steps'],
        'time_step_size_us': detector_config['time_step_size_us'],
        'num_wires_actual': detector_config['num_wires_actual'],
        'max_abs_indices': detector_config['max_wire_indices_abs'],
        'min_abs_indices': detector_config['min_wire_indices_abs'],
    }


def _resolve_cmap(cmap):
    """Return a matplotlib colormap object from a name or passthrough."""
    if cmap == 'obsidian':
        return _OBSIDIAN
    elif isinstance(cmap, str):
        return plt.cm.get_cmap(cmap)
    return cmap


def _add_colorbar(fig, ax, mappable, norm, label_size=12, tick_size=10):
    """Add a colorbar with DeadbandNorm-aware tick placement."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.0)
    cbar = fig.colorbar(mappable, cax=cax)

    if isinstance(norm, DeadbandNorm):
        n_ticks = 7
        tick_norm = np.linspace(0, 1, n_ticks)
        tick_values = norm.inverse(tick_norm)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{v:.0f}' for v in tick_values])

    cbar.ax.tick_params(labelsize=tick_size, colors='black')
    cbar.set_label('Signal (ADC)', fontsize=label_size, color='black')
    return cbar


# ---------------------------------------------------------------------------
# Main visualisation functions
# ---------------------------------------------------------------------------

def visualize_wire_signals(wire_signals_dict, detector_config, figsize=(20, 10),
                           threshold_enc=0, gamma=0.2, cmap='obsidian',
                           sparse_data=False, point_size=0.1):
    """
    Visualize wire signals for all 6 planes using DeadbandNorm and configurable colormap.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    figsize : tuple, optional
        Figure size in inches.
    threshold_enc : float, optional
        Deadband threshold in electrons. Converted to ADC internally.
    gamma : float, optional
        Power-law exponent for DeadbandNorm compression.
    cmap : str or Colormap, optional
        Colormap name ('obsidian', 'seismic', etc.) or matplotlib Colormap.
    sparse_data : bool, optional
        If True, expect sparse (indices, values) format.
    point_size : float, optional
        Scatter point size for sparse mode.

    Returns
    -------
    matplotlib.figure.Figure
    """
    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    num_wires_actual = vp['num_wires_actual']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    electrons_per_adc = float(detector_config['electrons_per_adc'])
    deadband_adc = threshold_enc / electrons_per_adc
    resolved_cmap = _resolve_cmap(cmap)

    side_names = ['West Side', 'East Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    plane_name_mapping = {
        (0, 0): 'U-plane', (0, 1): 'V-plane', (0, 2): 'Y-plane',
        (1, 0): 'U-plane', (1, 1): 'V-plane', (1, 2): 'Y-plane',
    }

    # Pre-convert arrays
    converted_signals = {}
    for key, signal_data in wire_signals_dict.items():
        if sparse_data:
            indices, values = signal_data
            converted_signals[key] = (np.asarray(indices), np.asarray(values))
        else:
            converted_signals[key] = np.asarray(signal_data)

    # Gather per-plane-type min/max
    plane_min_max = {
        'U-plane': {'min': float('inf'), 'max': -float('inf')},
        'V-plane': {'min': float('inf'), 'max': -float('inf')},
        'Y-plane': {'min': float('inf'), 'max': -float('inf')},
    }

    for s in range(2):
        for p in range(3):
            if (s, p) in converted_signals and num_wires_actual[s, p] > 0:
                pname = plane_name_mapping[(s, p)]
                if sparse_data:
                    _, vals = converted_signals[(s, p)]
                    if len(vals) > 0:
                        plane_min_max[pname]['min'] = min(plane_min_max[pname]['min'], vals.min())
                        plane_min_max[pname]['max'] = max(plane_min_max[pname]['max'], vals.max())
                else:
                    arr = converted_signals[(s, p)]
                    if arr.size > 0:
                        plane_min_max[pname]['min'] = min(plane_min_max[pname]['min'], arr.min())
                        plane_min_max[pname]['max'] = max(plane_min_max[pname]['max'], arr.max())

    # Set vmin/vmax: asymmetric for U/V, symmetric for Y
    for pname in plane_min_max:
        mm = plane_min_max[pname]
        if mm['min'] == float('inf'):
            mm['min'], mm['max'] = -25, 25
        elif pname == 'Y-plane':
            abs_max = max(abs(mm['min']), abs(mm['max']))
            mm['min'], mm['max'] = -abs_max, abs_max

    print("   Visualization Norms by Plane Type:")
    for pname, mm in plane_min_max.items():
        print(f"   - {pname}: min={mm['min']:.2e}, max={mm['max']:.2e}")

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10
    zero_color = resolved_cmap(0.5)  # colormap value at zero signal

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor(zero_color)
            ax.grid(False)
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
            plot_title = f"{side_names[side_idx]} {plane_types[plane_idx]}"
            pname = plane_name_mapping[(side_idx, plane_idx)]

            if (side_idx, plane_idx) not in converted_signals or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
                ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
                ax.set_ylabel('Time (us)', fontsize=label_size, color='black')
                ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue

            vmin = plane_min_max[pname]['min']
            vmax = plane_min_max[pname]['max']
            norm = DeadbandNorm(vmin, vmax, deadband_adc, gamma)

            if sparse_data:
                indices_np, values_np = converted_signals[(side_idx, plane_idx)]
                if len(values_np) == 0:
                    ax.text(0.5, 0.5, "(No data)", color='grey',
                            ha='center', va='center', transform=ax.transAxes)
                else:
                    wire_abs = indices_np[:, 0] + min_idx_abs
                    time_us = indices_np[:, 1] * time_step_size_us
                    sc = ax.scatter(wire_abs, time_us, c=values_np,
                                    cmap=resolved_cmap, norm=norm,
                                    s=point_size, marker='s', linewidths=0)
                    _add_colorbar(fig, ax, sc, norm, label_size, tick_size)
            else:
                signal_data_to_plot = converted_signals[(side_idx, plane_idx)]
                extent = [min_idx_abs, max_idx_abs + 1, 0, max_time_axis]
                im = ax.imshow(signal_data_to_plot.T, aspect='auto', origin='lower',
                               extent=extent, cmap=resolved_cmap, norm=norm,
                               interpolation='nearest')
                _add_colorbar(fig, ax, im, norm, label_size, tick_size)

            ax.set_ylim(0, max_time_axis)
            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_box_aspect(1)
            ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
            ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
            ax.set_ylabel('Time (us)', fontsize=label_size, color='black')
            ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')

    return fig


def visualize_single_plane(wire_signals_dict, detector_config, side_idx=0, plane_idx=0,
                            figsize=(10, 10), threshold_enc=0, gamma=0.2, cmap='obsidian',
                            sparse_data=False, point_size=0.5):
    """
    Visualize wire signals for a single side/plane using DeadbandNorm.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    side_idx, plane_idx : int
        Side and plane to plot.
    figsize : tuple, optional
        Figure size in inches.
    threshold_enc : float, optional
        Deadband threshold in electrons.
    gamma : float, optional
        Power-law exponent for DeadbandNorm.
    cmap : str or Colormap, optional
        Colormap name or object.
    sparse_data : bool, optional
        If True, expect sparse (indices, values) format.
    point_size : float, optional
        Scatter point size for sparse mode.

    Returns
    -------
    matplotlib.figure.Figure
    """
    print(f"--- Starting Visualization for Side {side_idx}, Plane {plane_idx} ---")

    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    num_wires_actual = vp['num_wires_actual']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    electrons_per_adc = float(detector_config['electrons_per_adc'])
    deadband_adc = threshold_enc / electrons_per_adc
    resolved_cmap = _resolve_cmap(cmap)

    side_names = ['West Side', 'East Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    plane_name_mapping = {
        (0, 0): 'U-plane', (0, 1): 'V-plane', (0, 2): 'Y-plane',
        (1, 0): 'U-plane', (1, 1): 'V-plane', (1, 2): 'Y-plane',
    }

    s, p = side_idx, plane_idx
    pname = plane_name_mapping[(s, p)]

    # Find min/max across both sides for the same plane type
    min_val, max_val = float('inf'), -float('inf')
    for check_side in range(2):
        check_key = (check_side, p)
        if check_key in wire_signals_dict and num_wires_actual[check_side, p] > 0:
            if sparse_data:
                indices, values = wire_signals_dict[check_key]
                if len(values) > 0:
                    values_np = np.asarray(values)
                    min_val = min(min_val, values_np.min())
                    max_val = max(max_val, values_np.max())
            else:
                arr = np.asarray(wire_signals_dict[check_key])
                if arr.size > 0:
                    min_val = min(min_val, arr.min())
                    max_val = max(max_val, arr.max())

    if min_val == float('inf'):
        min_val, max_val = -25, 25
    elif pname == 'Y-plane':
        abs_max = max(abs(min_val), abs(max_val))
        min_val, max_val = -abs_max, abs_max

    print(f"   Visualization Norm for {pname}: min={min_val:.2e}, max={max_val:.2e}")

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    zero_color = resolved_cmap(0.5)  # colormap value at zero signal
    ax.set_facecolor(zero_color)
    ax.grid(False)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    min_idx_abs = int(min_abs_indices[s, p])
    max_idx_abs = int(max_abs_indices[s, p])
    actual_wire_count = int(num_wires_actual[s, p])
    plot_title = f"{side_names[s]} {plane_types[p]}"

    if (s, p) not in wire_signals_dict or actual_wire_count == 0:
        ax.text(0.5, 0.5, "(0 wires active)", color='grey',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
        ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
        ax.set_ylabel('Time (us)', fontsize=label_size, color='black')
        ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
        ax.set_xlim(min_idx_abs, max_idx_abs + 1)
        ax.set_ylim(0, max_time_axis)
        ax.set_box_aspect(1)
        return fig

    vmin, vmax = min_val, max_val
    norm = DeadbandNorm(vmin, vmax, deadband_adc, gamma)

    if sparse_data:
        indices, values = wire_signals_dict[(s, p)]
        if len(values) == 0:
            ax.text(0.5, 0.5, "(No data)", color='grey',
                    ha='center', va='center', transform=ax.transAxes)
        else:
            indices_np = np.asarray(indices)
            values_np = np.asarray(values)
            wire_abs = indices_np[:, 0] + min_idx_abs
            time_us = indices_np[:, 1] * time_step_size_us
            sc = ax.scatter(wire_abs, time_us, c=values_np,
                            cmap=resolved_cmap, norm=norm,
                            s=point_size, marker='s', linewidths=0)
            _add_colorbar(fig, ax, sc, norm, label_size, tick_size)
    else:
        signal_data_to_plot = np.asarray(wire_signals_dict[(s, p)])
        extent = [min_idx_abs, max_idx_abs + 1, 0, max_time_axis]
        im = ax.imshow(signal_data_to_plot.T, aspect='auto', origin='lower',
                       extent=extent, cmap=resolved_cmap, norm=norm,
                       interpolation='nearest')
        _add_colorbar(fig, ax, im, norm, label_size, tick_size)

    ax.set_ylim(0, max_time_axis)
    ax.set_xlim(min_idx_abs, max_idx_abs + 1)
    ax.set_box_aspect(1)
    ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
    ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
    ax.set_ylabel('Time (us)', fontsize=label_size, color='black')
    ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')

    return fig


# ---------------------------------------------------------------------------
# Diffused charge visualisation (unchanged behaviour, detector_config API)
# ---------------------------------------------------------------------------

def visualize_diffused_charge(wire_signals_dict, detector_config, figsize=(20, 10),
                              log_norm=False, threshold=100, sparse_data=False, point_size=0.5):
    """
    Visualize diffused charge (hit signals) with proper scaling.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization for all plots, by default False.
    threshold : float, optional
        Values below this threshold are masked/hidden, by default 100.
    sparse_data : bool, optional
        If True, expect sparse format (indices, values).
    point_size : float, optional
        Size of scatter points when using sparse_data=True, by default 0.5.

    Returns
    -------
    matplotlib.figure.Figure
    """
    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    num_wires_actual = vp['num_wires_actual']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    side_names = ['West Side', 'East Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    global_min, global_max = float('inf'), -float('inf')
    all_values = []

    for s in range(2):
        for p in range(3):
            if (s, p) in wire_signals_dict and num_wires_actual[s, p] > 0:
                if sparse_data:
                    indices, values = wire_signals_dict[(s, p)]
                    if len(values) > 0:
                        values_np = np.array(values)
                        valid_data = values_np[values_np > threshold]
                        if len(valid_data) > 0:
                            global_min = min(global_min, valid_data.min())
                            global_max = max(global_max, valid_data.max())
                            all_values.append(valid_data)
                else:
                    signal_data = np.array(wire_signals_dict[(s, p)])
                    if signal_data.size > 0:
                        valid_data = signal_data[signal_data > threshold]
                        if valid_data.size > 0:
                            global_min = min(global_min, valid_data.min())
                            global_max = max(global_max, valid_data.max())
                            all_values.append(valid_data.flatten())

    if global_min == float('inf'):
        global_min, global_max = threshold, threshold * 10
    elif all_values:
        all_values_concat = np.concatenate(all_values)
        if len(all_values_concat) > 0:
            p1, p99 = np.percentile(all_values_concat, [1, 99])
            global_min = max(global_min, p1)
            global_max = min(global_max, p99)

    print(f"   Diffused Charge Range: min={global_min:.2e}, max={global_max:.2e}")
    background_color = '#1a1a1a'
    colormap_name = 'YlOrRd'

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor(background_color)
            ax.grid(True, alpha=0.3, color='#505050', linestyle='--', linewidth=0.5)

            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
            plot_title = f"{side_names[side_idx]} {plane_types[plane_idx]}"

            if (side_idx, plane_idx) not in wire_signals_dict or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", color='gray',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=14, pad=10)
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue

            cmap_obj = plt.cm.get_cmap(colormap_name).copy()
            vmin_plot = max(threshold, global_min)
            vmax_plot = global_max

            if sparse_data:
                indices, values = wire_signals_dict[(side_idx, plane_idx)]
                if len(values) == 0:
                    ax.text(0.5, 0.5, "(No data)", color='gray',
                            ha='center', va='center', transform=ax.transAxes)
                else:
                    indices_np = np.array(indices)
                    values_np = np.array(values)
                    mask = values_np > threshold
                    if np.sum(mask) == 0:
                        ax.text(0.5, 0.5, "(Below threshold)", color='gray',
                                ha='center', va='center', transform=ax.transAxes)
                    else:
                        wire_abs = indices_np[mask, 0] + min_idx_abs
                        time_us = indices_np[mask, 1] * time_step_size_us
                        filtered_values = values_np[mask]
                        if log_norm:
                            norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
                        else:
                            norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)
                        sc = ax.scatter(wire_abs, time_us, c=filtered_values, cmap=cmap_obj,
                                        norm=norm, s=point_size, marker='s', linewidths=0)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='3%', pad=0.0)
                        cbar = fig.colorbar(sc, cax=cax)
                        cbar.set_label('Diffused Charge', fontsize=12)
            else:
                signal_data_to_plot = np.array(wire_signals_dict[(side_idx, plane_idx)])
                extent = [min_idx_abs, max_idx_abs + 1, 0, max_time_axis]
                masked_data = np.ma.masked_where(signal_data_to_plot.T <= threshold, signal_data_to_plot.T)
                cmap_obj.set_bad(background_color)
                cmap_obj.set_under(background_color)
                if log_norm:
                    norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
                    im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                                   cmap=cmap_obj, norm=norm, interpolation='nearest')
                else:
                    im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                                   cmap=cmap_obj, vmin=vmin_plot, vmax=vmax_plot, interpolation='nearest')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.0)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Diffused Charge', fontsize=12)

            ax.set_ylim(0, max_time_axis)
            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_box_aspect(1)
            ax.set_title(plot_title, fontsize=14, pad=10)
            ax.set_xlabel('Absolute Wire Index', fontsize=12)
            ax.set_ylabel('Time (us)', fontsize=12)

    return fig


def visualize_diffused_charge_single_plane(wire_signals_dict, detector_config, side_idx=0, plane_idx=0,
                                           figsize=(10, 10), log_norm=False, threshold=100,
                                           sparse_data=False, point_size=1.0):
    """
    Visualize diffused charge for a single side/plane combination.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    side_idx : int, optional
        Index of the side to plot (0=West, 1=East), by default 0.
    plane_idx : int, optional
        Index of the plane to plot (0=U, 1=V, 2=Y), by default 0.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization, by default False.
    threshold : float, optional
        Values below this threshold are masked/hidden, by default 100.
    sparse_data : bool, optional
        If True, expect sparse format (indices, values).
    point_size : float, optional
        Size of scatter points when using sparse_data=True, by default 1.0.

    Returns
    -------
    matplotlib.figure.Figure
    """
    print(f"--- Visualizing Diffused Charge for Side {side_idx}, Plane {plane_idx} ---")

    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    num_wires_actual = vp['num_wires_actual']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    side_names = ['West Side', 'East Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    s, p = side_idx, plane_idx
    min_val, max_val = float('inf'), -float('inf')

    for check_side in range(2):
        check_key = (check_side, p)
        if check_key in wire_signals_dict and num_wires_actual[check_side, p] > 0:
            if sparse_data:
                indices, values = wire_signals_dict[check_key]
                if len(values) > 0:
                    values_np = np.array(values)
                    valid_data = values_np[values_np > threshold]
                    if len(valid_data) > 0:
                        min_val = min(min_val, valid_data.min())
                        max_val = max(max_val, valid_data.max())
            else:
                signal_data = np.array(wire_signals_dict[check_key])
                if signal_data.size > 0:
                    valid_data = signal_data[signal_data > threshold]
                    if valid_data.size > 0:
                        min_val = min(min_val, valid_data.min())
                        max_val = max(max_val, valid_data.max())

    if min_val == float('inf'):
        min_val, max_val = threshold, threshold * 10

    print(f"   Visualization Range: min={min_val:.2e}, max={max_val:.2e}")

    background_color = '#1a1a1a'
    colormap_name = 'YlOrRd'

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(background_color)
    ax.grid(True, alpha=0.3, color='#505050', linestyle='--', linewidth=0.5)
    max_time_axis = num_time_steps * time_step_size_us

    min_idx_abs = int(min_abs_indices[s, p])
    max_idx_abs = int(max_abs_indices[s, p])
    actual_wire_count = int(num_wires_actual[s, p])
    plot_title = f"{side_names[s]} {plane_types[p]}"

    if (s, p) not in wire_signals_dict or actual_wire_count == 0:
        ax.text(0.5, 0.5, "(0 wires active)", color='gray',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(plot_title, fontsize=14, pad=10)
        ax.set_xlabel('Absolute Wire Index', fontsize=12)
        ax.set_ylabel('Time (us)', fontsize=12)
        ax.set_xlim(min_idx_abs, max_idx_abs + 1)
        ax.set_ylim(0, max_time_axis)
        ax.set_box_aspect(1)
        return fig

    cmap_obj = plt.cm.get_cmap(colormap_name).copy()
    vmin_plot = max(threshold, min_val)
    vmax_plot = max_val

    if sparse_data:
        indices, values = wire_signals_dict[(s, p)]
        if len(values) == 0:
            ax.text(0.5, 0.5, "(No data)", color='gray',
                    ha='center', va='center', transform=ax.transAxes)
        else:
            indices_np = np.array(indices)
            values_np = np.array(values)
            mask = values_np > threshold
            if np.sum(mask) == 0:
                ax.text(0.5, 0.5, "(Below threshold)", color='gray',
                        ha='center', va='center', transform=ax.transAxes)
            else:
                wire_abs = indices_np[mask, 0] + min_idx_abs
                time_us = indices_np[mask, 1] * time_step_size_us
                filtered_values = values_np[mask]
                if log_norm:
                    norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
                else:
                    norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)
                sc = ax.scatter(wire_abs, time_us, c=filtered_values, cmap=cmap_obj,
                                norm=norm, s=point_size, marker='s', linewidths=0)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.0)
                cbar = fig.colorbar(sc, cax=cax)
                cbar.set_label('Diffused Charge', fontsize=12)
    else:
        signal_data_to_plot = np.array(wire_signals_dict[(s, p)])
        extent = [min_idx_abs, max_idx_abs + 1, 0, max_time_axis]
        masked_data = np.ma.masked_where(signal_data_to_plot.T <= threshold, signal_data_to_plot.T)
        cmap_obj.set_bad(background_color)
        cmap_obj.set_under(background_color)
        if log_norm:
            norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
            im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                           cmap=cmap_obj, norm=norm, interpolation='nearest')
        else:
            im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                           cmap=cmap_obj, vmin=vmin_plot, vmax=vmax_plot, interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.0)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Diffused Charge', fontsize=12)

    ax.set_ylim(0, max_time_axis)
    ax.set_xlim(min_idx_abs, max_idx_abs + 1)
    ax.set_box_aspect(1)
    ax.set_title(plot_title, fontsize=14, pad=10)
    ax.set_xlabel('Absolute Wire Index', fontsize=12)
    ax.set_ylabel('Time (us)', fontsize=12)

    return fig


# ---------------------------------------------------------------------------
# Track label visualisation
# ---------------------------------------------------------------------------

def get_top_tracks_by_charge(track_hits_dict, top_n=20):
    """
    Find top tracks by total charge across all planes.

    Parameters
    ----------
    track_hits_dict : dict
        Dictionary of track hits results, keyed by (side_idx, plane_idx).
        Each entry should contain 'num_labeled' and 'labeled_hits' arrays.
    top_n : int, optional
        Number of top tracks to return, by default 20.

    Returns
    -------
    list
        List of tuples (track_id, total_charge) sorted by charge descending.
    """
    all_track_ids = []
    all_charges = []

    for plane_key, results in track_hits_dict.items():
        num_labeled = int(results['num_labeled'])
        if num_labeled > 0:
            labeled = results['labeled_hits'][:num_labeled]
            all_track_ids.append(jnp.asarray(labeled[:, 0], dtype=jnp.int32))
            all_charges.append(jnp.asarray(labeled[:, 3]))

    if not all_track_ids:
        return []

    all_track_ids = jnp.concatenate(all_track_ids)
    all_charges = jnp.concatenate(all_charges)

    sort_idx = jnp.argsort(all_track_ids)
    sorted_ids = all_track_ids[sort_idx]
    sorted_charges = all_charges[sort_idx]

    is_new_track = jnp.concatenate([jnp.array([True]), sorted_ids[1:] != sorted_ids[:-1]])
    unique_indices = jnp.where(is_new_track)[0]
    unique_tracks = sorted_ids[unique_indices]

    segment_ids = jnp.cumsum(is_new_track) - 1
    track_totals = jax.ops.segment_sum(sorted_charges, segment_ids, num_segments=len(unique_indices))

    top_indices = jnp.argsort(track_totals)[-top_n:][::-1]
    return [(int(unique_tracks[i]), float(track_totals[i])) for i in top_indices]


def visualize_track_labels(track_hits_dict, detector_config, top_tracks_by_charge,
                           max_tracks=15, figsize=(20, 12)):
    """
    Visualize track labels with distinct colors for top tracks by charge.

    Parameters
    ----------
    track_hits_dict : dict
        Dictionary of track hits results, keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    top_tracks_by_charge : list
        List of tuples (track_id, total_charge) from get_top_tracks_by_charge.
    max_tracks : int, optional
        Maximum number of tracks to show in legend, by default 15.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 12).

    Returns
    -------
    matplotlib.figure.Figure
    """
    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    side_names = ['West Side', 'East Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30, width_ratios=[1, 1, 1, 0.12])
    max_time_axis = num_time_steps * time_step_size_us

    distinct_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF',
                       '#FFD700', '#FF8C00', '#8B008B', '#228B22', '#4B0082',
                       '#FF1493', '#00CED1', '#FF4500', '#9400D3', '#32CD32',
                       '#8B4513', '#20B2AA', '#FF69B4', '#4169E1', '#DC143C']
    distinct_colors_rgba = [mcolors.to_rgba(c) for c in distinct_colors]

    top_tracks = [tid for tid, _ in top_tracks_by_charge[:max_tracks]]
    top_track_to_color = {tid: distinct_colors_rgba[i] for i, tid in enumerate(top_tracks[:len(distinct_colors)])}
    hash_cmap = plt.cm.hsv

    def get_track_colors_vectorized(track_ids):
        track_ids = np.asarray(track_ids, dtype=np.int64)
        colors = np.zeros((len(track_ids), 4))
        hash_values = (track_ids * 2654435761) % 2**32
        colors[:] = hash_cmap(hash_values / (2**32 - 1))
        for tid, color in top_track_to_color.items():
            colors[track_ids == tid] = color
        return colors

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor('black')

            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])

            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_ylim(0, max_time_axis)
            ax.set_box_aspect(1)
            ax.set_title(f"{side_names[side_idx]} {plane_types[plane_idx]}", fontsize=14, pad=10)
            ax.set_xlabel('Absolute Wire Index', fontsize=12)
            ax.set_ylabel('Time (us)', fontsize=12)

            plane_key = (side_idx, plane_idx)
            results = track_hits_dict[plane_key]
            num_labeled = int(results['num_labeled'])

            if num_labeled > 0:
                labeled = np.array(results['labeled_hits'][:num_labeled])
                tracks = labeled[:, 0].astype(np.int64)
                wires = labeled[:, 1]
                times = labeled[:, 2] * time_step_size_us

                colors = get_track_colors_vectorized(tracks)
                ax.scatter(wires, times, c=colors, s=0.5, alpha=0.8)
                ax.text(0.02, 0.98, f"{num_labeled:,} hits\n{len(np.unique(tracks))} tracks",
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
            else:
                ax.text(0.5, 0.5, "(No labeled hits)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)

    if top_tracks:
        cbar_ax = fig.add_subplot(gs[:, 3])
        n_show = min(len(top_tracks), max_tracks)
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, n_show)
        for i, tid in enumerate(top_tracks[:n_show]):
            color = top_track_to_color.get(tid, hash_cmap((tid * 2654435761 % 2**32) / (2**32 - 1)))
            y_pos = n_show - 1 - i
            rect = plt.Rectangle((0, y_pos), 0.4, 0.9, facecolor=color, edgecolor='black', linewidth=0.5)
            cbar_ax.add_patch(rect)
            cbar_ax.text(0.5, y_pos + 0.45, f'Track {tid}', ha='left', va='center', fontsize=8)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        cbar_ax.set_title(f'Top {n_show} Tracks\n(by total charge)', fontsize=11, pad=10)
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


def visualize_track_labels_single_plane(track_hits_dict, detector_config, top_tracks_by_charge,
                                        side_idx=0, plane_idx=0, max_tracks=15, figsize=(12, 10)):
    """
    Visualize track labels for a single side/plane with distinct colors for top tracks.

    Parameters
    ----------
    track_hits_dict : dict
        Dictionary of track hits results, keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    top_tracks_by_charge : list
        List of tuples (track_id, total_charge) from get_top_tracks_by_charge.
    side_idx : int, optional
        Index of the side to plot (0=West, 1=East), by default 0.
    plane_idx : int, optional
        Index of the plane to plot (0=U, 1=V, 2=Y), by default 0.
    max_tracks : int, optional
        Maximum number of tracks to show in legend, by default 15.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (12, 10).

    Returns
    -------
    matplotlib.figure.Figure
    """
    print(f"--- Visualizing Track Labels for Side {side_idx}, Plane {plane_idx} ---")

    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    side_names = ['West Side', 'East Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, width_ratios=[1, 0.12])
    max_time_axis = num_time_steps * time_step_size_us

    distinct_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF',
                       '#FFD700', '#FF8C00', '#8B008B', '#228B22', '#4B0082',
                       '#FF1493', '#00CED1', '#FF4500', '#9400D3', '#32CD32',
                       '#8B4513', '#20B2AA', '#FF69B4', '#4169E1', '#DC143C']
    distinct_colors_rgba = [mcolors.to_rgba(c) for c in distinct_colors]

    top_tracks = [tid for tid, _ in top_tracks_by_charge[:max_tracks]]
    top_track_to_color = {tid: distinct_colors_rgba[i] for i, tid in enumerate(top_tracks[:len(distinct_colors)])}
    hash_cmap = plt.cm.hsv

    def get_track_colors_vectorized(track_ids):
        track_ids = np.asarray(track_ids, dtype=np.int64)
        colors = np.zeros((len(track_ids), 4))
        hash_values = (track_ids * 2654435761) % 2**32
        colors[:] = hash_cmap(hash_values / (2**32 - 1))
        for tid, color in top_track_to_color.items():
            colors[track_ids == tid] = color
        return colors

    s, p = side_idx, plane_idx
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor('black')

    min_idx_abs = int(min_abs_indices[s, p])
    max_idx_abs = int(max_abs_indices[s, p])

    ax.set_xlim(min_idx_abs, max_idx_abs + 1)
    ax.set_ylim(0, max_time_axis)
    ax.set_box_aspect(1)
    ax.set_title(f"{side_names[s]} {plane_types[p]}", fontsize=14, pad=10)
    ax.set_xlabel('Absolute Wire Index', fontsize=12)
    ax.set_ylabel('Time (us)', fontsize=12)

    plane_key = (s, p)
    results = track_hits_dict[plane_key]
    num_labeled = int(results['num_labeled'])

    if num_labeled > 0:
        labeled = np.array(results['labeled_hits'][:num_labeled])
        tracks = labeled[:, 0].astype(np.int64)
        wires = labeled[:, 1]
        times = labeled[:, 2] * time_step_size_us

        colors = get_track_colors_vectorized(tracks)
        ax.scatter(wires, times, c=colors, s=0.5, alpha=0.8)
        ax.text(0.02, 0.98, f"{num_labeled:,} hits\n{len(np.unique(tracks))} tracks",
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        print(f"   {num_labeled:,} labeled hits, {len(np.unique(tracks))} unique tracks")
    else:
        ax.text(0.5, 0.5, "(No labeled hits)", color='grey',
                ha='center', va='center', transform=ax.transAxes)
        print("   No labeled hits found")

    if top_tracks:
        cbar_ax = fig.add_subplot(gs[0, 1])
        n_show = min(len(top_tracks), max_tracks)
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, n_show)
        for i, tid in enumerate(top_tracks[:n_show]):
            color = top_track_to_color.get(tid, hash_cmap((tid * 2654435761 % 2**32) / (2**32 - 1)))
            y_pos = n_show - 1 - i
            rect = plt.Rectangle((0, y_pos), 0.4, 0.9, facecolor=color, edgecolor='black', linewidth=0.5)
            cbar_ax.add_patch(rect)
            cbar_ax.text(0.5, y_pos + 0.45, f'Track {tid}', ha='left', va='center', fontsize=8)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        cbar_ax.set_title(f'Top {n_show} Tracks\n(by total charge)', fontsize=11, pad=10)
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bucketed / index visualisation
# ---------------------------------------------------------------------------

def visualize_active_buckets(response_signals, detector_config, figsize=(20, 10)):
    """
    Visualize active buckets in the bucketed simulation output.

    Parameters
    ----------
    response_signals : dict
        Dictionary of response signals from bucketed simulation.
    detector_config : dict
        Detector configuration from generate_detector().
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    num_wires_actual = vp['num_wires_actual']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    side_names = ['East Side', 'West Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor('#1a1a1a')

            plane_key = (side_idx, plane_idx)
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            num_wires = int(num_wires_actual[side_idx, plane_idx])

            if plane_key not in response_signals:
                ax.text(0.5, 0.5, "(No data)", color='grey',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{side_names[side_idx]} {plane_types[plane_idx]}",
                             fontsize=title_size, pad=10)
                continue

            buckets, num_active, compact_to_key, B1, B2 = response_signals[plane_key]
            num_active_int = int(num_active)
            B1_int = int(B1)
            B2_int = int(B2)

            num_buckets_w = (num_wires + B1_int - 1) // B1_int
            num_buckets_t = (num_time_steps + B2_int - 1) // B2_int
            total_buckets = num_buckets_w * num_buckets_t
            coverage_pct = (num_active_int / total_buckets) * 100 if total_buckets > 0 else 0

            active_keys = np.array(compact_to_key[:num_active_int])
            bucket_w_indices = active_keys // num_buckets_t
            bucket_t_indices = active_keys % num_buckets_t

            bucket_grid = np.zeros((num_buckets_w, num_buckets_t))
            for bw, bt in zip(bucket_w_indices, bucket_t_indices):
                if 0 <= bw < num_buckets_w and 0 <= bt < num_buckets_t:
                    bucket_grid[bw, bt] = 1

            extent = [
                min_idx_abs,
                min_idx_abs + num_buckets_w * B1_int,
                0,
                num_buckets_t * B2_int * time_step_size_us
            ]

            cmap_obj = plt.cm.YlOrRd.copy()
            cmap_obj.set_under('#1a1a1a')

            im = ax.imshow(
                bucket_grid.T, aspect='auto', origin='lower', extent=extent,
                cmap=cmap_obj, vmin=0.5, vmax=1.0, interpolation='nearest'
            )

            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_ylim(0, max_time_axis)
            ax.set_box_aspect(1)

            plot_title = f"{side_names[side_idx]} {plane_types[plane_idx]}"
            ax.set_title(plot_title, fontsize=title_size, pad=10)
            ax.set_xlabel('Absolute Wire Index', fontsize=label_size)
            ax.set_ylabel('Time (us)', fontsize=label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size)

            info_text = (f"Active: {num_active_int:,}\nTotal: {total_buckets:,}\n"
                         f"Coverage: {coverage_pct:.1f}%\nBucket: {B1_int}x{B2_int}")
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va='top', ha='left',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray',
                              boxstyle='round,pad=0.3'))

    fig.suptitle('Active Buckets Visualization (Bucketed Mode)', fontsize=16, y=0.98)
    return fig


def visualize_by_index(wire_signals_dict, detector_config, indices_list, figsize=(10, 8),
                       sparse_data=False):
    """
    Visualize wire signals at specific wire indices across time.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    detector_config : dict
        Detector configuration from generate_detector().
    indices_list : list
        List of wire indices to plot.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 8).
    sparse_data : bool, optional
        If True, expect sparse format (indices, values).

    Returns
    -------
    matplotlib.figure.Figure
    """
    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    num_wires_actual = vp['num_wires_actual']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']

    side_names = ['West Side', 'East Side']
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

    time_axis = np.arange(num_time_steps) * time_step_size_us

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = axes[side_idx, plane_idx]

            plane_key = (side_idx, plane_idx)
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])

            plot_title = f"{side_names[side_idx]} {plane_types[plane_idx]}"
            ax.set_title(plot_title, fontsize=12)

            if plane_key not in wire_signals_dict or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_xlabel('Time (us)')
                ax.set_ylabel('Signal Strength')
                continue

            if sparse_data:
                indices, values = wire_signals_dict[plane_key]
                if len(values) == 0:
                    ax.text(0.5, 0.5, "(No data)", ha='center', va='center',
                            transform=ax.transAxes)
                    ax.set_xlabel('Time (us)')
                    ax.set_ylabel('Signal Strength')
                    continue

                indices_np = np.array(indices)
                values_np = np.array(values)

                for wire_idx in indices_list:
                    rel_idx = wire_idx - min_idx_abs
                    mask = indices_np[:, 0] == rel_idx
                    if np.any(mask):
                        wire_times = indices_np[mask, 1] * time_step_size_us
                        wire_values = values_np[mask]
                        sort_order = np.argsort(wire_times)
                        ax.plot(wire_times[sort_order], wire_values[sort_order],
                               label=f'Wire {wire_idx}', alpha=0.8, marker='.', markersize=1,
                               linestyle='-')
            else:
                signal_data = np.array(wire_signals_dict[plane_key])

                for wire_idx in indices_list:
                    rel_idx = wire_idx - min_idx_abs
                    if 0 <= rel_idx < signal_data.shape[0]:
                        wire_signal = signal_data[rel_idx, :]
                        ax.plot(time_axis, wire_signal, label=f'Wire {wire_idx}', alpha=0.8)

            ax.set_xlabel('Time (us)')
            ax.set_ylabel('Signal Strength')
            ax.grid(True, alpha=0.3)

            if len(indices_list) <= 10:
                ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wire geometry visualisation (already uses detector_config - unchanged)
# ---------------------------------------------------------------------------

def visualize_wire_planes_colored_by_index(detector_config, figsize=(15, 10)):
    """
    Visualize all 6 wire planes (3 on each side) of the LArTPC detector,
    coloring the wires based on their index.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary with pre-calculated parameters.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (15, 10).

    Returns
    -------
    matplotlib.figure.Figure
    """
    detector_dims = detector_config['detector']['dimensions']
    detector_y = detector_dims['y']
    detector_z = detector_dims['z']

    sides = detector_config['wire_planes']['sides']

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']
    wire_side_names = ['West Side', 'East Side']

    cmap = plt.cm.viridis

    for side_idx, side in enumerate(sides):

        for plane_idx, plane in enumerate(side['planes']):
            ax = axes[side_idx, plane_idx]

            angle_deg = plane['angle']
            angle_rad = np.radians(angle_deg)
            wire_spacing = plane['wire_spacing']
            distance_from_anode = plane['distance_from_anode']

            title = f"{wire_side_names[side_idx]} {plane_types[plane_idx]}"
            ax.set_title(title)

            ax.add_patch(plt.Rectangle((0, 0), detector_z, detector_y,
                                       fill=False, color='black', linestyle='--'))

            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)

            corners = [
                (0, 0), (detector_y, 0),
                (0, detector_z), (detector_y, detector_z)
            ]

            r_values = [z * cos_theta + y * sin_theta for y, z in corners]
            r_min = min(r_values)
            r_max = max(r_values)

            offset = 0
            if r_min < 0:
                offset = int(np.abs(np.floor(r_min / wire_spacing))) + 1

            idx_min = int(np.floor(r_min / wire_spacing)) + offset
            idx_max = int(np.ceil(r_max / wire_spacing)) + offset

            num_wires = idx_max - idx_min + 1

            for wire_idx in range(idx_min, idx_max + 1):
                r = (wire_idx - offset) * wire_spacing

                intersections = []

                if abs(cos_theta) > 1e-10:
                    z = r / cos_theta
                    if 0 <= z <= detector_z:
                        intersections.append((0, z))

                if abs(cos_theta) > 1e-10:
                    z = (r - detector_y * sin_theta) / cos_theta
                    if 0 <= z <= detector_z:
                        intersections.append((detector_y, z))

                if abs(sin_theta) > 1e-10:
                    y = r / sin_theta
                    if 0 <= y <= detector_y:
                        intersections.append((y, 0))

                if abs(sin_theta) > 1e-10:
                    y = (r - detector_z * cos_theta) / sin_theta
                    if 0 <= y <= detector_y:
                        intersections.append((y, detector_z))

                if len(intersections) >= 2:
                    if len(intersections) > 2:
                        unique_intersections = []
                        for pt in intersections:
                            is_duplicate = False
                            for existing in unique_intersections:
                                if abs(pt[0] - existing[0]) < 1e-6 and abs(pt[1] - existing[1]) < 1e-6:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                unique_intersections.append(pt)
                        intersections = unique_intersections

                    if len(intersections) >= 2:
                        p1, p2 = intersections[0], intersections[1]
                        dy = abs(p2[0] - p1[0])
                        dz = abs(p2[1] - p1[1])

                        if dz > dy:
                            intersections.sort(key=lambda pt: pt[1])
                        else:
                            intersections.sort(key=lambda pt: pt[0])

                        p1, p2 = intersections[0], intersections[-1]

                        norm_idx = (wire_idx - idx_min) / max(1, num_wires - 1)
                        color = cmap(norm_idx)

                        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color,
                                linewidth=0.8, alpha=0.7)

            ax.set_xlabel('Z Position (cm)')
            ax.set_ylabel('Y Position (cm)')
            ax.set_xlim(0, detector_z)
            ax.set_ylim(0, detector_y)
            ax.grid(alpha=0.3)

            info_text = (f"Angle: {angle_deg} deg\nSpacing: {wire_spacing} cm\n"
                         f"Distance from anode: {distance_from_anode} cm")
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, va='top',
                    bbox=dict(facecolor='white', alpha=0.7))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=idx_min, vmax=idx_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Wire Index')

    plt.tight_layout()
    return fig
