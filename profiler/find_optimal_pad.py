"""
Scan HDF5 event files and find the optimal total_pad for a given detector geometry.

Reads positions from pstep data, splits into volumes by geometry ranges, and
reports per-volume deposit count statistics. No simulation is run.

Usage:
    python3 -m profiler.find_optimal_pad --data events.h5
    python3 -m profiler.find_optimal_pad --data dir_of_h5s/ --config config/sbnd_config.yaml
    python3 -m profiler.find_optimal_pad --data events.h5 --events 500
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from tools.geometry import generate_detector


def get_volume_ranges(detector_config):
    """Extract per-volume (x, y, z) ranges in cm from detector config."""
    volumes = []
    for vol in detector_config['volumes']:
        ranges = vol['geometry']['ranges']
        volumes.append({
            'id': vol['id'],
            'x_range': (ranges[0][0], ranges[0][1]),
            'y_range': (ranges[1][0], ranges[1][1]),
            'z_range': (ranges[2][0], ranges[2][1]),
        })
    return volumes


def count_deposits_per_volume(positions_mm, volume_ranges):
    """Count deposits falling in each volume. Returns list of counts."""
    pos_cm = positions_mm / 10.0
    x, y, z = pos_cm[:, 0], pos_cm[:, 1], pos_cm[:, 2]

    counts = []
    for vol in volume_ranges:
        mask = (
            (x >= vol['x_range'][0]) & (x < vol['x_range'][1]) &
            (y >= vol['y_range'][0]) & (y < vol['y_range'][1]) &
            (z >= vol['z_range'][0]) & (z < vol['z_range'][1])
        )
        counts.append(int(np.sum(mask)))
    return counts


def round_up_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def _scan_file_for_pad(args):
    """Worker: scan one HDF5 file, return (path, per-event counts)."""
    fpath, volume_ranges, max_events = args
    counts = []
    with h5py.File(fpath, 'r') as f:
        ds = f['pstep/lar_vol']
        n_events = ds.shape[0]
        if max_events is not None:
            n_events = min(n_events, max_events)
        for i in range(n_events):
            row = ds[i]
            positions_mm = np.column_stack([
                row['x'].astype(np.float32),
                row['y'].astype(np.float32),
                row['z'].astype(np.float32),
            ])
            counts.append(count_deposits_per_volume(positions_mm, volume_ranges))
    return fpath, counts


def scan_files_for_pad(h5_files, volume_ranges, max_events=None,
                       n_workers=1, print_progress=True):
    """Scan files (optionally in parallel) and collect per-event volume counts.

    Returns (all_counts, total_events) where all_counts is a list of
    per-volume count lists (one entry per event).
    """
    args_list = [(fp, volume_ranges, max_events) for fp in h5_files]
    all_counts = []
    total = 0

    if n_workers <= 1 or len(h5_files) <= 1:
        for i, args in enumerate(args_list, 1):
            fpath, counts = _scan_file_for_pad(args)
            all_counts.extend(counts)
            total += len(counts)
            if print_progress and len(h5_files) > 1:
                print(f'  [{i}/{len(h5_files)}] {os.path.basename(fpath)}: '
                      f'{len(counts)} events', flush=True)
        return all_counts, total

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(_scan_file_for_pad, a) for a in args_list]
        for done, fut in enumerate(as_completed(futures), 1):
            fpath, counts = fut.result()
            all_counts.extend(counts)
            total += len(counts)
            if print_progress:
                print(f'  [{done}/{len(h5_files)}] {os.path.basename(fpath)}: '
                      f'{len(counts)} events', flush=True)
    return all_counts, total


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal total_pad from event data')
    parser.add_argument('--data', required=True, nargs='+',
                        help='HDF5 file(s) or directory')
    parser.add_argument('--config', required=True,
                        help='Detector geometry YAML')
    parser.add_argument('--events', type=int, default=None,
                        help='Max events to scan per file (default: all)')
    parser.add_argument('--response-chunk', type=int, default=50_000,
                        help='Response chunk size for divisibility (default: 50000)')
    parser.add_argument('--save-config', default=None,
                        help='Save total_pad to production config YAML')
    parser.add_argument('--use-max', action='store_true',
                        help='Save the max-based suggestion (default: p99.9)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker processes for file scanning '
                             '(default: 1 = serial)')

    args = parser.parse_args()

    detector_config = generate_detector(args.config)
    volume_ranges = get_volume_ranges(detector_config)
    n_volumes = len(volume_ranges)

    # Collect H5 files
    h5_files = []
    for p in args.data:
        if os.path.isdir(p):
            h5_files.extend(sorted(glob.glob(os.path.join(p, '*.h5'))))
        else:
            h5_files.append(p)
    if not h5_files:
        print("No .h5 files found!")
        return

    print('=' * 70)
    print(' JAXTPC — Find Optimal total_pad')
    print('=' * 70)
    print(f'  Config:    {args.config}')
    print(f'  Volumes:   {n_volumes}')
    for vol in volume_ranges:
        print(f'    Vol {vol["id"]}: x=[{vol["x_range"][0]:.1f}, {vol["x_range"][1]:.1f}] '
              f'y=[{vol["y_range"][0]:.1f}, {vol["y_range"][1]:.1f}] '
              f'z=[{vol["z_range"][0]:.1f}, {vol["z_range"][1]:.1f}] cm')
    print(f'  Files:     {len(h5_files)}')
    print()

    # Scan all events (optionally in parallel)
    all_counts, total_events = scan_files_for_pad(
        h5_files, volume_ranges, max_events=args.events,
        n_workers=args.workers, print_progress=True)

    if not all_counts:
        print('No events found!')
        return

    counts_array = np.array(all_counts)  # (n_events, n_volumes)

    # Per-volume statistics
    print(f'\n  Total events scanned: {total_events:,}')
    print()

    header = f'  {"Volume":>8} {"Min":>8} {"P50":>8} {"P95":>8} {"P99":>8} {"P99.9":>8} {"Max":>8}'
    print(header)
    print(f'  {"─" * (len(header) - 2)}')

    for v in range(n_volumes):
        col = counts_array[:, v]
        p = np.percentile(col, [0, 50, 95, 99, 99.9, 100])
        print(f'  {v:>8d} {int(p[0]):>8,} {int(p[1]):>8,} {int(p[2]):>8,} '
              f'{int(p[3]):>8,} {int(p[4]):>8,} {int(p[5]):>8,}')

    # Max across volumes per event (this is what total_pad must cover)
    max_per_event = counts_array.max(axis=1)
    pcts = np.percentile(max_per_event, [50, 95, 99, 99.9, 100])

    print()
    print(f'  Max-across-volumes per event:')
    print(f'    P50   = {int(pcts[0]):>10,}')
    print(f'    P95   = {int(pcts[1]):>10,}')
    print(f'    P99   = {int(pcts[2]):>10,}')
    print(f'    P99.9 = {int(pcts[3]):>10,}')
    print(f'    Max   = {int(pcts[4]):>10,}')

    # Suggestions: round max to 10k, round p99.9 to 10k, then align to chunk
    max_rounded = round_up_to_multiple(int(pcts[4]), 10_000)
    p999_rounded = round_up_to_multiple(int(pcts[3]), 10_000)
    max_aligned = round_up_to_multiple(max_rounded, args.response_chunk)
    p999_aligned = round_up_to_multiple(p999_rounded, args.response_chunk)

    n_over_p999 = int(np.sum(max_per_event > p999_aligned))
    pct_over = 100.0 * n_over_p999 / total_events

    print()
    print(f'  Suggestions (rounded to 10k, aligned to response_chunk={args.response_chunk:,}):')
    print(f'    Max   → --total-pad {max_aligned:>10,}  (covers 100%)')
    print(f'    P99.9 → --total-pad {p999_aligned:>10,}  '
          f'({n_over_p999} events truncated, {pct_over:.2f}%)')

    if args.save_config:
        from profiler.production_config import update_config
        chosen = max_aligned if args.use_max else p999_aligned
        update_config(args.save_config, {'total_pad': chosen},
                      detector_config_path=args.config)
        print(f'  Saved total_pad={chosen:,} to {args.save_config}')

    # Figures
    from profiler.plots import plot_deposit_distribution
    tag = os.path.splitext(os.path.basename(args.config))[0]
    print()
    plot_deposit_distribution(counts_array, max_aligned, tag=tag)
    print()


if __name__ == '__main__':
    main()
