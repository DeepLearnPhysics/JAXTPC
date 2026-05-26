"""
Render 2D event display GIFs from sparse precomputed signals (v2).
Loads sparse, densifies per frame, reuses figure with imshow set_data.

Usage:
    python3 closure/segments/render_2d.py --signals closure_analysis_full/sweeps/precomputed_signals_v2.npz
    python3 closure/segments/render_2d.py --signals closure_analysis_full/sweeps/precomputed_signals_v2.npz --preview
"""

import sys, os, time, argparse

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
GIF_FPS = 25
FRAME_SKIP = 2
GAMMA = 0.2

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


def sparse_to_dense(indices, values, shape):
    arr = np.zeros(shape, dtype=np.float32)
    if len(indices) > 0:
        arr[indices[:, 0], indices[:, 1]] = values
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals', required=True,
                        help='Path to precomputed_signals_v2.npz')
    parser.add_argument('--config', default='config/cubic_wireplane_config.yaml',
                        help='Detector config YAML')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()

    print("Loading sparse signals...")
    data = np.load(args.signals, allow_pickle=True)
    n_frames = int(data['n_frames'])
    frame_steps = data['frame_steps']
    frame_losses = data['frame_losses']
    active_planes = list(data['active_planes'])
    deadband_enc = int(data['deadband_enc'])

    detector_config = generate_detector(args.config)
    electrons_per_adc = float(detector_config['electrons_per_adc'])
    deadband_adc = deadband_enc / electrons_per_adc
    cmap = _resolve_cmap('obsidian')
    vp = _extract_viz_params(detector_config)
    max_time_axis = vp['num_time_steps'] * vp['time_step_size_us']
    max_abs_indices = vp['max_abs_indices']
    min_abs_indices = vp['min_abs_indices']
    zero_color = cmap(0.5)

    ptype_map = {0: 'U', 1: 'V', 2: 'Y'}
    plane_types = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']
    row_labels = ['Truth', 'Sim', 'Diff']

    truth = {}
    truth_shapes = {}
    for pidx in active_planes:
        s, p = pidx // 3, pidx % 3
        shape = tuple(data[f'truth_{s}_{p}_shape'])
        indices = data[f'truth_{s}_{p}_indices']
        values = data[f'truth_{s}_{p}_values']
        truth[(s, p)] = sparse_to_dense(indices, values, shape)
        truth_shapes[(s, p)] = shape

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

    print(f"  {n_frames} frames, deadband={deadband_enc} e-")

    for side_idx, side_name in [(0, 'east'), (1, 'west')]:
        side_planes = [(side_idx, p) for p in range(3)
                       if side_idx * 3 + p in active_planes]
        if not side_planes:
            continue
        n_p = len(side_planes)

        print(f"\n  {side_name.upper()} ({n_p} planes)...")

        fig = plt.figure(figsize=(6.5 * n_p, 15), facecolor='white')
        gs = gridspec.GridSpec(3, n_p, figure=fig, hspace=0.25, wspace=0.28)
        ims = {}
        titles = {}

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

                im = ax.imshow(truth[(s, p)].T, aspect='auto', origin='lower',
                               extent=extent, cmap=cmap, norm=norm,
                               interpolation='nearest')
                ims[(row, col)] = im

                cbar = add_colorbar(fig, ax, im, norm)
                if col < n_p - 1:
                    cbar.set_label('')

                ttl = ax.set_title(f'{row_labels[row]} {plane_types[p]}',
                                    fontsize=TITLE_SIZE, pad=8)
                titles[(row, col)] = ttl

                if row == 2:
                    ax.set_xlabel('Absolute Wire Index', fontsize=LABEL_SIZE)
                else:
                    ax.set_xlabel('')
                if col == 0:
                    ax.set_ylabel('Time (us)', fontsize=LABEL_SIZE)
                else:
                    ax.set_ylabel('')
                ax.tick_params(labelsize=TICK_SIZE)
                ax.set_xlim(min_idx, max_idx + 1)
                ax.set_ylim(0, max_time_axis)

        suptitle = fig.suptitle('', fontsize=SUPTITLE_SIZE, fontweight='bold', y=0.995)
        fig.subplots_adjust(top=0.94)

        if args.preview:
            for fi in [0, n_frames - 1]:
                step = frame_steps[fi]
                loss = frame_losses[fi]
                for col, (s, p) in enumerate(side_planes):
                    shape = truth_shapes[(s, p)]
                    r_sig = sparse_to_dense(
                        data[f'sim_{fi}_{s}_{p}_indices'],
                        data[f'sim_{fi}_{s}_{p}_values'], shape)
                    d_sig = truth[(s, p)] - r_sig
                    ims[(1, col)].set_data(r_sig.T)
                    ims[(2, col)].set_data(d_sig.T)
                suptitle.set_text(f'{side_name.upper()} -- Step {step} | '
                                   f'Loss: {loss:.6f} | Threshold: {deadband_enc} e-')
                fname = os.path.join(OUT_DIR, f'preview_v2_{side_name}_step{step}.png')
                fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"    Saved {fname}")
        else:
            frames = []
            t0 = time.time()
            frame_list = list(range(0, n_frames, FRAME_SKIP))
            if frame_list[-1] != n_frames - 1:
                frame_list.append(n_frames - 1)
            for fi in frame_list:
                step = frame_steps[fi]
                loss = frame_losses[fi]
                for col, (s, p) in enumerate(side_planes):
                    shape = truth_shapes[(s, p)]
                    r_sig = sparse_to_dense(
                        data[f'sim_{fi}_{s}_{p}_indices'],
                        data[f'sim_{fi}_{s}_{p}_values'], shape)
                    d_sig = truth[(s, p)] - r_sig
                    ims[(1, col)].set_data(r_sig.T)
                    ims[(2, col)].set_data(d_sig.T)
                suptitle.set_text(f'{side_name.upper()} -- Step {step} | '
                                   f'Loss: {loss:.6f} | Threshold: {deadband_enc} e-')
                frames.append(fig_to_image(fig))
                if len(frames) % 25 == 0 or fi == frame_list[-1]:
                    print(f"    Frame {len(frames)}/{len(frame_list)} (step {step}) -- {time.time()-t0:.0f}s")

            out_path = os.path.join(OUT_DIR, f'event_display_2d_v2_{side_name}.gif')
            duration_ms = int(1000 / GIF_FPS)
            frames[0].save(out_path, save_all=True, append_images=frames[1:],
                            duration=duration_ms, loop=0)
            print(f"    Saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

        plt.close(fig)


if __name__ == '__main__':
    main()
