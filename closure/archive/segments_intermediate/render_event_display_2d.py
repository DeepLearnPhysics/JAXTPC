"""
Render 2D event display GIFs from precomputed signals.
Uses same imshow/colormap/norm as visualize_wire_signals.

Usage:
    python3 closure_analysis_full/sweeps/precompute_signals.py  # run first
    python3 closure_analysis_full/sweeps/render_event_display_2d.py
    python3 closure_analysis_full/sweeps/render_event_display_2d.py --preview  # single frames only
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from tools.geometry import generate_detector
from tools.visualization import _extract_viz_params, _resolve_cmap, DeadbandNorm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_PATH = os.path.join(OUT_DIR, 'precomputed_signals.npz')
GIF_FPS = 8
THRESHOLD_ENC = 800
GAMMA = 0.2

# Font sizes
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
CBAR_LABEL_SIZE = 12
CBAR_TICK_SIZE = 10
SUPTITLE_SIZE = 18


def fig_to_image(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    return Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).convert('RGB').copy()


def add_colorbar(fig, ax, mappable, norm):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.02)
    cbar = fig.colorbar(mappable, cax=cax)
    if isinstance(norm, DeadbandNorm):
        tick_norm = np.linspace(0, 1, 7)
        tick_values = norm.inverse(tick_norm)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{v:.0f}' for v in tick_values])
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, colors='black')
    cbar.set_label('Signal (ADC)', fontsize=CBAR_LABEL_SIZE, color='black')
    return cbar


def build_figure(side_planes, detector_config, truth, cmap, deadband_adc,
                 plane_min_max, ptype_map):
    """Create figure once with 3 rows × n_planes, return imshow refs."""
    vp = _extract_viz_params(detector_config)
    num_time_steps = vp['num_time_steps']
    time_step_size_us = vp['time_step_size_us']
    max_time_axis = num_time_steps * time_step_size_us
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']
    zero_color = cmap(0.5)

    n_planes = len(side_planes)
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']
    row_labels = ['Truth', 'Sim', 'Diff']

    fig = plt.figure(figsize=(6.5 * n_planes, 15), facecolor='white')
    gs = gridspec.GridSpec(3, n_planes, figure=fig, hspace=0.25, wspace=0.28)

    imshow_objects = {}
    title_objects = {}

    for row in range(3):
        for col, (s, p) in enumerate(side_planes):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(zero_color)
            ax.grid(False)

            min_idx = int(min_abs_indices[s, p])
            max_idx = int(max_abs_indices[s, p])
            extent = [min_idx, max_idx + 1, 0, max_time_axis]

            pt = ptype_map[p]
            vmin = plane_min_max[pt]['min']
            vmax = plane_min_max[pt]['max']
            norm = DeadbandNorm(vmin, vmax, deadband_adc, GAMMA)

            init_data = truth[(s, p)]
            im = ax.imshow(init_data.T, aspect='auto', origin='lower',
                           extent=extent, cmap=cmap, norm=norm,
                           interpolation='nearest')
            imshow_objects[(row, col)] = im

            cbar = add_colorbar(fig, ax, im, norm)
            # Only show colorbar label on rightmost column
            if col < n_planes - 1:
                cbar.set_label('')

            ttl = ax.set_title(f'{row_labels[row]} {plane_types[p]}',
                                fontsize=TITLE_SIZE, pad=8)
            title_objects[(row, col)] = ttl

            # Only xlabel on bottom row
            if row == 2:
                ax.set_xlabel('Absolute Wire Index', fontsize=LABEL_SIZE)
            else:
                ax.set_xlabel('')

            # Only ylabel on leftmost column
            if col == 0:
                ax.set_ylabel('Time (μs)', fontsize=LABEL_SIZE)
            else:
                ax.set_ylabel('')

            ax.tick_params(labelsize=TICK_SIZE)
            ax.set_xlim(min_idx, max_idx + 1)
            ax.set_ylim(0, max_time_axis)

    suptitle = fig.suptitle('', fontsize=SUPTITLE_SIZE, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.94)

    return fig, imshow_objects, title_objects, suptitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true',
                        help='Save single frames only (no GIF)')
    args = parser.parse_args()

    print("Loading precomputed signals...")
    data = np.load(SIGNALS_PATH, allow_pickle=True)
    n_frames = int(data['n_frames'])
    frame_steps = data['frame_steps']
    frame_losses = data['frame_losses']
    active_planes = list(data['active_planes'])

    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    electrons_per_adc = float(detector_config['electrons_per_adc'])
    deadband_adc = THRESHOLD_ENC / electrons_per_adc
    cmap = _resolve_cmap('obsidian')

    ptype_map = {0: 'U', 1: 'V', 2: 'Y'}
    plane_types_short = ['U', 'V', 'Y']

    # Load truth
    truth = {}
    for pidx in active_planes:
        s, p = pidx // 3, pidx % 3
        truth[(s, p)] = data[f'truth_{s}_{p}']

    # Compute norms from truth
    plane_min_max = {'U': {'min': np.inf, 'max': -np.inf},
                     'V': {'min': np.inf, 'max': -np.inf},
                     'Y': {'min': np.inf, 'max': -np.inf}}
    for pidx in active_planes:
        s, p = pidx // 3, pidx % 3
        arr = truth[(s, p)]
        pt = ptype_map[p]
        plane_min_max[pt]['min'] = min(plane_min_max[pt]['min'], arr.min())
        plane_min_max[pt]['max'] = max(plane_min_max[pt]['max'], arr.max())
    for pt in plane_min_max:
        mm = plane_min_max[pt]
        if mm['min'] == np.inf:
            mm['min'], mm['max'] = -25, 25
        elif pt == 'Y':
            abs_max = max(abs(mm['min']), abs(mm['max']))
            mm['min'], mm['max'] = -abs_max, abs_max

    for side_idx, side_name in [(0, 'east'), (1, 'west')]:
        side_planes = [(side_idx, p) for p in range(3)
                       if side_idx * 3 + p in active_planes]
        if not side_planes:
            continue

        n_p = len(side_planes)
        print(f"\n{'='*50}")
        print(f"  {side_name.upper()} ({n_p} planes)")
        print(f"{'='*50}")

        fig, ims, titles, suptitle = build_figure(
            side_planes, detector_config, truth, cmap, deadband_adc,
            plane_min_max, ptype_map)

        plane_types_full = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']

        if args.preview:
            # Just save first and last frame
            for fi in [0, n_frames - 1]:
                step = frame_steps[fi]
                loss = frame_losses[fi]
                for col, (s, p) in enumerate(side_planes):
                    r_sig = data[f'sim_{fi}_{s}_{p}']
                    d_sig = truth[(s, p)] - r_sig
                    ims[(1, col)].set_data(r_sig.T)
                    ims[(2, col)].set_data(d_sig.T)
                suptitle.set_text(f'{side_name.upper()} — Step {step} | Loss: {loss:.6f} | '
                                   f'Threshold: {THRESHOLD_ENC} e⁻')
                fname = os.path.join(OUT_DIR, f'preview_{side_name}_step{step}.png')
                fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"  Saved {fname}")
        else:
            frames = []
            t0 = time.time()
            for fi in range(n_frames):
                step = frame_steps[fi]
                loss = frame_losses[fi]
                for col, (s, p) in enumerate(side_planes):
                    r_sig = data[f'sim_{fi}_{s}_{p}']
                    d_sig = truth[(s, p)] - r_sig
                    ims[(1, col)].set_data(r_sig.T)
                    ims[(2, col)].set_data(d_sig.T)
                suptitle.set_text(f'{side_name.upper()} — Step {step} | Loss: {loss:.6f} | '
                                   f'Threshold: {THRESHOLD_ENC} e⁻')
                frames.append(fig_to_image(fig))
                if fi % 10 == 0 or fi == n_frames - 1:
                    print(f"  Frame {fi+1}/{n_frames} (step {step}) — {time.time()-t0:.0f}s")

            out_path = os.path.join(OUT_DIR, f'event_display_2d_{side_name}.gif')
            duration_ms = int(1000 / GIF_FPS)
            frames[0].save(out_path, save_all=True, append_images=frames[1:],
                            duration=duration_ms, loop=0)
            print(f"  Saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

        plt.close(fig)


if __name__ == '__main__':
    main()
