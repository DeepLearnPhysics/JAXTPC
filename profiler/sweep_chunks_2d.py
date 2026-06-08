"""
2D timing sweep over (max_keys, hits_chunk_size).

For each pair, builds a fresh DetectorSimulator, JIT-compiles, warms up,
and times ``n_iter`` iterations of ``process_event`` on a single benchmark
event. Produces a heatmap of mean time per event with each cell annotated.

Cells where the chosen ``max_keys`` is too small for the event raise an
overflow ``RuntimeError`` after ``process_event`` returns; those are
caught and recorded as NaN (shown hatched on the plot).

Usage:
    python3 -m profiler.sweep_chunks_2d --data file.h5 \\
        --config config/cubic_wireplane_config.yaml \\
        --production-config config/production_cubic_wireplane_doraemon.yaml \\
        --event 102 --tag doraemon
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tools.geometry import generate_detector

from profiler.find_optimal_chunks import bench_one
from profiler.production_config import load_config


def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser(description='2D sweep over max_keys × hits_chunk')
    p.add_argument('--data', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--production-config', required=True)
    p.add_argument('--event', type=int, default=0)
    p.add_argument('--n-iter', type=int, default=5)
    p.add_argument('--max-keys-values', default=
                   '500000,1000000,2000000,4000000,6000000,9500000',
                   help='Comma-separated max_keys values')
    p.add_argument('--hits-chunk-values', default=
                   '33250,66500,95000,133000,190000,332500',
                   help='Comma-separated hits_chunk values (must divide total_pad)')
    p.add_argument('--bucketed', action='store_true')
    p.add_argument('--tag', default=None)
    args = p.parse_args()

    detector_config = generate_detector(args.config)
    prod = load_config(args.production_config)
    total_pad = int(prod['total_pad'])
    seed_response = int(prod['response_chunk'])
    tag_base = args.tag or os.path.splitext(os.path.basename(args.config))[0]

    max_keys_vals = _parse_int_list(args.max_keys_values)
    hits_chunk_vals = _parse_int_list(args.hits_chunk_values)

    # Validate hits_chunk divisibility (warn but proceed; bench_one will fail loudly)
    for h in hits_chunk_vals:
        if total_pad % h != 0:
            print(f'  WARNING: hits_chunk={h:,} does not divide total_pad={total_pad:,}',
                  flush=True)

    print('=' * 70, flush=True)
    print(' JAXTPC — 2D Chunk Sweep (max_keys × hits_chunk)', flush=True)
    print('=' * 70, flush=True)
    print(f'  Data:        {args.data}  (event {args.event})', flush=True)
    print(f'  total_pad:   {total_pad:,}', flush=True)
    print(f'  seed response_chunk: {seed_response:,}', flush=True)
    print(f'  n_iter:      {args.n_iter}', flush=True)
    print(f'  max_keys:    {max_keys_vals}', flush=True)
    print(f'  hits_chunk:  {hits_chunk_vals}', flush=True)
    print(f'  Cells:       {len(max_keys_vals)} × {len(hits_chunk_vals)} = '
          f'{len(max_keys_vals) * len(hits_chunk_vals)}', flush=True)
    print(f'  Device:      {jax.devices()[0]}', flush=True)
    print(flush=True)

    grid_mean = np.full((len(max_keys_vals), len(hits_chunk_vals)), np.nan)
    grid_std = np.full_like(grid_mean, np.nan)
    overflowed = np.zeros_like(grid_mean, dtype=bool)

    total_cells = len(max_keys_vals) * len(hits_chunk_vals)
    cell_idx = 0
    for i, mk in enumerate(max_keys_vals):
        for j, hc in enumerate(hits_chunk_vals):
            cell_idx += 1
            label = f'[{cell_idx:>2}/{total_cells}] mk={mk:>10,} hc={hc:>8,}'
            try:
                mean, std, _ = bench_one(
                    detector_config, args.data, args.event, total_pad,
                    seed_response, hc, include_track_hits=True,
                    max_keys=mk, bucketed=args.bucketed,
                    n_timed=args.n_iter, label=label)
                grid_mean[i, j] = mean
                grid_std[i, j] = std
            except RuntimeError as e:
                msg = str(e)
                if 'overflow' in msg.lower() or 'max_keys' in msg.lower():
                    overflowed[i, j] = True
                    print(f'    {label}  OVERFLOW: {msg[:120]}', flush=True)
                else:
                    print(f'    {label}  FAILED: {msg[:200]}', flush=True)

    # Save raw data alongside the figure for re-plotting later
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'profiler', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    npz_name = f'chunk_2d_{tag_base}.npz'
    np.savez(os.path.join(out_dir, npz_name),
             max_keys=np.array(max_keys_vals),
             hits_chunk=np.array(hits_chunk_vals),
             grid_mean=grid_mean, grid_std=grid_std, overflow=overflowed)

    # Identify best (skip NaN/overflow)
    if np.isfinite(grid_mean).any():
        best_flat = np.nanargmin(grid_mean)
        bi, bj = np.unravel_index(best_flat, grid_mean.shape)
        best_mk = max_keys_vals[bi]
        best_hc = hits_chunk_vals[bj]
        best_t = grid_mean[bi, bj]
        best_s = grid_std[bi, bj]
        print(f'\n  Best cell: max_keys={best_mk:,}, hits_chunk={best_hc:,}  '
              f'({best_t:.1f} ± {best_s:.1f} ms)', flush=True)
    else:
        bi = bj = None
        print('\n  No successful cells.', flush=True)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    masked = np.ma.masked_invalid(grid_mean)
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color='lightgray')
    im = ax.imshow(masked, origin='lower', aspect='auto', cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time per event (ms)')

    ax.set_xticks(range(len(hits_chunk_vals)))
    ax.set_xticklabels([f'{h:,}' for h in hits_chunk_vals], rotation=30, ha='right')
    ax.set_yticks(range(len(max_keys_vals)))
    ax.set_yticklabels([f'{m:,}' for m in max_keys_vals])
    ax.set_xlabel('hits_chunk_size')
    ax.set_ylabel('max_keys')

    # Annotate cells
    finite = grid_mean[np.isfinite(grid_mean)]
    if finite.size > 0:
        mid = np.median(finite)
        for i in range(len(max_keys_vals)):
            for j in range(len(hits_chunk_vals)):
                v = grid_mean[i, j]
                if np.isnan(v):
                    label = 'OVF' if overflowed[i, j] else 'X'
                    ax.text(j, i, label, ha='center', va='center',
                            color='red', fontsize=9, fontweight='bold')
                else:
                    color = 'white' if v > mid else 'black'
                    ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                            color=color, fontsize=8)

    # Star the best
    if bi is not None:
        ax.plot(bj, bi, marker='*', color='red', markersize=18,
                markeredgecolor='black', markeredgewidth=0.8, linestyle='None',
                label=f'best ({best_t:.0f} ms)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    ax.set_title(f'2D sweep: max_keys × hits_chunk  (event {args.event}, '
                 f'n_iter={args.n_iter})')
    plt.tight_layout()
    fname = os.path.join(out_dir, f'chunk_2d_{tag_base}.png')
    plt.savefig(fname, dpi=120)
    print(f'\n  Saved: {fname}', flush=True)
    print(f'  Saved: {os.path.join(out_dir, npz_name)}', flush=True)


if __name__ == '__main__':
    main()
