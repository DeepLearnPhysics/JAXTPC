#!/usr/bin/env python3
"""Generate a rotating 3D GIF/MP4 of JAXTPC segments cycling through color modes.

Reads production seg HDF5 files directly and renders with matplotlib.
One full 360° rotation synchronized with dE → Track → PDG → Ancestor → Interaction.

Usage:
    python3 viewer/export_gif.py path/to/sim_seg_0000.h5 --event 0
    python3 viewer/export_gif.py path/to/sim_seg_0000.h5 -e 0 -o rotate.mp4 --fps 30
    python3 viewer/export_gif.py path/to/sim_seg_0000.h5 -e 0 --volume 0 --max-points 50000
"""

import argparse
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter


# ── Colormaps (matching viewer) ──────────────────────────────────

WARM_STOPS = [
    (0.00, (0x08/255, 0x08/255, 0x08/255)),
    (0.12, (0x2a/255, 0x0a/255, 0x02/255)),
    (0.30, (0x7a/255, 0x22/255, 0x00/255)),
    (0.50, (0xb8/255, 0x58/255, 0x00/255)),
    (0.70, (0xee/255, 0x88/255, 0x00/255)),
    (0.88, (0xff/255, 0xcc/255, 0x55/255)),
    (1.00, (0xff/255, 0xfd/255, 0xe0/255)),
]

INFERNO_R_STOPS = [
    (0.0, (0xfc/255, 0xff/255, 0xa4/255)),
    (0.2, (0xfc/255, 0xa5/255, 0x0a/255)),
    (0.4, (0xdd/255, 0x51/255, 0x3a/255)),
    (0.6, (0x93/255, 0x26/255, 0x67/255)),
    (0.8, (0x42/255, 0x0a/255, 0x68/255)),
    (1.0, (0x0d/255, 0x08/255, 0x29/255)),
]


def _build_cmap(stops, name='custom'):
    positions = [s[0] for s in stops]
    colors = [s[1] for s in stops]
    return LinearSegmentedColormap.from_list(name, list(zip(positions, colors)), N=256)


WARM_CMAP = _build_cmap(WARM_STOPS, 'warm')
INFERNO_R_CMAP = _build_cmap(INFERNO_R_STOPS, 'inferno_r_custom')


# ── Golden ratio hue hash (same as viewer) ──────────────────────

PHI_FRAC = 0.618033988749895

PDG_NAMES = {
    11: 'e⁻', -11: 'e⁺', 13: 'μ⁻', -13: 'μ⁺',
    211: 'π⁺', -211: 'π⁻', 111: 'π⁰',
    2212: 'p', 2112: 'n', 22: 'γ',
    321: 'K⁺', -321: 'K⁻', 310: 'K⁰_S', 130: 'K⁰_L',
}


def golden_hash_colors(ids):
    """Map integer IDs to HSL colors via golden ratio hash."""
    unique = np.unique(ids)
    hues = (np.abs(unique).astype(np.float64) * PHI_FRAC) % 1.0
    colors = np.zeros((len(ids), 4))
    id_to_idx = {uid: i for i, uid in enumerate(unique)}
    for i, uid in enumerate(unique):
        h = hues[i]
        s, l = 0.75, 0.55
        mask = ids == uid
        rgb = plt.cm.hsv(h)[:3]
        # Increase saturation
        colors[mask] = (*rgb, 0.7)
    return colors


def hsl_to_rgb(h, s, l):
    """HSL to RGB conversion."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = l - c / 2
    i = int(h * 6) % 6
    if i == 0:   r, g, b = c, x, 0
    elif i == 1: r, g, b = x, c, 0
    elif i == 2: r, g, b = 0, c, x
    elif i == 3: r, g, b = 0, x, c
    elif i == 4: r, g, b = x, 0, c
    else:        r, g, b = c, 0, x
    return (r + m, g + m, b + m)


def categorical_colors(ids, alpha=0.7):
    """Assign colors to categorical IDs using golden ratio hue spacing."""
    hues = (np.abs(ids).astype(np.float64) * PHI_FRAC) % 1.0
    colors = np.zeros((len(ids), 4))
    for i in range(len(ids)):
        r, g, b = hsl_to_rgb(hues[i], 0.75, 0.55)
        colors[i] = (r, g, b, alpha)
    return colors


# ── Data loading ─────────────────────────────────────────────────

def load_seg_data(seg_path, event_idx):
    """Load segment data including pdg/ancestor/interaction fields."""
    event_key = f'event_{event_idx:03d}'

    with h5py.File(seg_path, 'r') as f:
        evt = f[event_key]
        n_volumes = int(evt.attrs.get('n_volumes', 2))

        volumes = []
        for v in range(n_volumes):
            vg_key = f'volume_{v}'
            if vg_key not in evt:
                volumes.append(None)
                continue

            vg = evt[vg_key]
            n = int(vg.attrs['n_actual'])
            if n == 0:
                volumes.append(None)
                continue

            pos_step = float(vg.attrs['pos_step_mm'])
            origin = np.array([vg.attrs['pos_origin_x'],
                               vg.attrs['pos_origin_y'],
                               vg.attrs['pos_origin_z']])
            positions = vg['positions'][:].astype(np.float32) * pos_step + origin

            vol = {
                'positions_mm': positions,
                'de': vg['de'][:].astype(np.float32),
                'track_ids': vg['track_ids'][:],
                'pdg': vg['pdg'][:] if 'pdg' in vg else np.zeros(n, dtype=np.int32),
                'ancestor_ids': (vg['ancestor_track_ids'][:]
                                 if 'ancestor_track_ids' in vg
                                 else np.zeros(n, dtype=np.int32)),
                'interaction_ids': (vg['interaction_ids'][:]
                                    if 'interaction_ids' in vg
                                    else np.zeros(n, dtype=np.int16)),
                'n': n,
            }
            volumes.append(vol)

    return volumes


def merge_volumes(volumes, vol_idx=None):
    """Concatenate selected volumes into single arrays."""
    if vol_idx is not None:
        vols = [volumes[vol_idx]] if volumes[vol_idx] is not None else []
    else:
        vols = [v for v in volumes if v is not None]

    if not vols:
        sys.exit("No deposits found in selected volumes.")

    return {k: np.concatenate([v[k] for v in vols])
            for k in ('positions_mm', 'de', 'track_ids', 'pdg',
                      'ancestor_ids', 'interaction_ids')}


def subsample(data, max_points):
    """Random subsample if needed, preserving high-dE deposits."""
    n = len(data['de'])
    if n <= max_points:
        return data

    # Keep top 20% by dE, random sample the rest
    n_top = max_points // 5
    n_rand = max_points - n_top
    order = np.argsort(data['de'])[::-1]
    top_idx = order[:n_top]
    rest_idx = order[n_top:]
    rng = np.random.default_rng(42)
    rand_idx = rng.choice(rest_idx, size=min(n_rand, len(rest_idx)), replace=False)
    keep = np.sort(np.concatenate([top_idx, rand_idx]))

    return {k: v[keep] for k, v in data.items()}


# ── Color mode definitions ───────────────────────────────────────

COLOR_MODES = [
    ('Energy Deposit', 'de'),
    ('Track ID', 'track_ids'),
    ('PDG', 'pdg'),
    ('Ancestor ID', 'ancestor_ids'),
    ('Interaction ID', 'interaction_ids'),
]


def de_norm(de):
    """Log-normalize dE to [0, 1]."""
    de_floor = np.maximum(de, 1e-4)
    log_de = np.log10(de_floor)
    vmin, vmax = log_de.min(), log_de.max()
    if vmax <= vmin:
        vmax = vmin + 1
    return (log_de - vmin) / (vmax - vmin)


def de_emphasis(norm, emph_pow=5.0, emph_amt=0.75):
    """Compute emphasis factor from normalized dE — matches viewer shader.

    emph = norm^emph_pow  (steep: only high-dE deposits get emphasis ~1)
    eFactor = mix(1.0, emph, emph_amt)  (blend between uniform and emphasized)
    """
    emph = np.clip(norm, 0.001, 1.0) ** emph_pow
    return (1.0 - emph_amt) + emph_amt * emph


def compute_de_colors(norm, emph_factor, light=False):
    """dE colormap colors with emphasis-modulated alpha."""
    cmap = INFERNO_R_CMAP if light else WARM_CMAP
    colors = cmap(norm)
    colors[:, 3] = 0.85 * np.maximum(emph_factor, 0.03)
    return colors


def compute_colors(data, mode_key, norm, emph_factor, light=False):
    """Compute RGBA colors for a given mode with dE emphasis on alpha."""
    if mode_key == 'de':
        return compute_de_colors(norm, emph_factor, light=light)
    colors = categorical_colors(data[mode_key])
    colors[:, 3] *= np.maximum(emph_factor, 0.03)
    return colors


def compute_sizes(emph_factor, base=1.5, scale=4.0):
    """Point sizes from dE emphasis factor — matches viewer shader."""
    return base + scale * np.maximum(emph_factor, 0.2)


# ── Rendering ────────────────────────────────────────────────────

def make_gif(data, output, fps=30, duration=12.0, rotations=1, dpi=200,
             size=(1080, 1080), light=False, emph_pow=5.0, emph_amt=0.75):
    """Render rotating 3D GIF cycling through color modes."""
    total_duration = duration * rotations
    n_frames = int(fps * total_duration)
    n_modes = len(COLOR_MODES)
    frames_per_mode = n_frames // n_modes
    n_frames = frames_per_mode * n_modes  # round to exact multiple

    pos = data['positions_mm']
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # dE emphasis — applied to size and alpha across all color modes
    norm = de_norm(data['de'])
    emph = de_emphasis(norm, emph_pow, emph_amt)
    sizes = compute_sizes(emph)

    # Precompute colors for each mode (alpha modulated by emphasis)
    all_colors = []
    for label, key in COLOR_MODES:
        all_colors.append(compute_colors(data, key, norm, emph, light=light))

    # Figure setup
    bg = '#f0f0f0' if light else '#080808'
    text_color = '#333333' if light else '#cccccc'
    accent = '#1a73e8' if light else '#ff9900'

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi,
                     facecolor=bg)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg)

    # Hide all axes — just the 3D points
    ax.set_axis_off()

    # Tight bounding box — fit edges exactly
    cx, cy, cz = (x.min()+x.max())/2, (y.min()+y.max())/2, (z.min()+z.max())/2
    hx, hy, hz = np.ptp(x)/2, np.ptp(y)/2, np.ptp(z)/2
    span = max(hx, hy, hz) * 1.02  # minimal padding
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(cz - span, cz + span)

    # Title text only — no JAXTPC prefix
    title = fig.text(0.5, 0.98, '', color=accent, fontsize=13, fontweight='bold',
                     fontfamily='monospace', ha='center', va='top')

    scatter = ax.scatter(x, y, z, c=all_colors[0], s=sizes, depthshade=True,
                         linewidths=0)

    fig.subplots_adjust(left=-0.08, right=1.08, top=1.02, bottom=-0.06)

    import time as _time
    _t0 = [_time.time()]
    _frame_times = []

    def update(frame):
        mode_idx = frame // frames_per_mode
        mode_idx = min(mode_idx, n_modes - 1)

        # Rotation: rotations × 360° over all frames
        frac = frame / n_frames
        azim = frac * 360 * rotations + 30
        elev = 20 + 8 * np.sin(frac * 2 * np.pi * rotations)
        ax.view_init(elev=elev, azim=azim)

        # Update colors on mode switch
        scatter.set_facecolors(all_colors[mode_idx])

        label, key = COLOR_MODES[mode_idx]
        title.set_text(label)

        # Progress bar
        now = _time.time()
        _frame_times.append(now)
        elapsed = now - _t0[0]
        pct = (frame + 1) / n_frames
        if frame > 0:
            eta = elapsed / pct - elapsed
            eta_str = f'{eta:.0f}s' if eta < 120 else f'{eta/60:.1f}m'
        else:
            eta_str = '...'
        bar_w = 30
        filled = int(bar_w * pct)
        bar = '█' * filled + '░' * (bar_w - filled)
        print(f'\r  {bar} {pct*100:5.1f}% | {frame+1}/{n_frames} | '
              f'elapsed {elapsed:.0f}s | ETA {eta_str}', end='', flush=True)

        return scatter, title

    ext = os.path.splitext(output)[1].lower()
    print(f"Rendering {n_frames} frames at {fps} fps "
          f"({total_duration:.1f}s, {rotations} rotation{'s' if rotations>1 else ''})...")
    print(f"  Modes: {' → '.join(l for l,_ in COLOR_MODES)}")
    print(f"  Points: {len(x):,}")

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/fps)

    if ext == '.gif':
        anim.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    elif ext in ('.mp4', '.webm'):
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(output, writer=writer, dpi=dpi)
        except Exception as e:
            print(f"\nFFMpeg failed ({e}), falling back to GIF...")
            output = output.rsplit('.', 1)[0] + '.gif'
            anim.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(output, writer=PillowWriter(fps=fps), dpi=dpi)

    print()  # newline after progress bar
    sz = os.path.getsize(output)
    label = f'{sz/1e6:.1f} MB' if sz > 1e5 else f'{sz/1e3:.0f} KB'
    total_time = _time.time() - _t0[0]
    t_str = f'{total_time:.0f}s' if total_time < 120 else f'{total_time/60:.1f}m'
    print(f"Saved {output} ({label}) in {t_str}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate rotating 3D GIF of JAXTPC segments')
    parser.add_argument('seg_file', help='Path to *_seg_*.h5 file')
    parser.add_argument('--event', '-e', type=int, default=0,
                        help='Event index (default: 0)')
    parser.add_argument('--volume', '-v', type=int, default=None,
                        help='Volume index (default: all)')
    parser.add_argument('--output', '-o', default='jaxtpc_3d.gif',
                        help='Output file (.gif, .mp4, .webm)')
    parser.add_argument('--max-points', type=int, default=100000,
                        help='Max points to render (default: 100000)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--duration', type=float, default=12.0,
                        help='Duration of one full rotation in seconds (default: 12)')
    parser.add_argument('--rotations', type=int, default=1,
                        help='Number of full 360° rotations (default: 1)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Resolution (default: 200)')
    parser.add_argument('--size', type=int, nargs=2, default=[1440, 1440],
                        metavar=('W', 'H'), help='Frame size in pixels')
    parser.add_argument('--emph-pow', type=float, default=5.0,
                        help='dE emphasis power (default: 5.0)')
    parser.add_argument('--emph-amt', type=float, default=0.75,
                        help='dE emphasis amount 0-1 (default: 0.75)')
    parser.add_argument('--light', action='store_true',
                        help='Use light background')
    args = parser.parse_args()

    if not os.path.isfile(args.seg_file):
        sys.exit(f"Error: {args.seg_file} not found")

    print(f"Loading {args.seg_file} event {args.event}...")
    volumes = load_seg_data(args.seg_file, args.event)
    data = merge_volumes(volumes, args.volume)
    n_orig = len(data['de'])
    data = subsample(data, args.max_points)
    if len(data['de']) < n_orig:
        print(f"  Subsampled {n_orig:,} → {len(data['de']):,} points")

    make_gif(data, args.output, fps=args.fps, duration=args.duration,
             rotations=args.rotations, dpi=args.dpi, size=tuple(args.size),
             emph_pow=args.emph_pow, emph_amt=args.emph_amt,
             light=args.light)


if __name__ == '__main__':
    main()
