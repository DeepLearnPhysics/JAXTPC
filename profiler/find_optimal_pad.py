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
import os
import sys

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


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal total_pad from event data')
    parser.add_argument('--data', required=True,
                        help='HDF5 file or directory of HDF5 files')
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

    args = parser.parse_args()

    detector_config = generate_detector(args.config)
    volume_ranges = get_volume_ranges(detector_config)
    n_volumes = len(volume_ranges)

    # Collect H5 files
    if os.path.isdir(args.data):
        h5_files = sorted(glob.glob(os.path.join(args.data, '*.h5')))
        if not h5_files:
            print(f"No .h5 files found in {args.data}")
            return
    else:
        h5_files = [args.data]

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

    # Scan all events
    all_counts = []  # list of lists: [event][volume] = count
    total_events = 0

    for file_path in h5_files:
        fname = os.path.basename(file_path)
        with h5py.File(file_path, 'r') as f:
            pstep_path = 'pstep/lar_vol'
            if pstep_path not in f:
                print(f"  WARNING: {pstep_path} not found in {fname}, skipping")
                continue

            ds = f[pstep_path]
            n_events = ds.shape[0]
            if args.events is not None:
                n_events = min(n_events, args.events)

            for i in range(n_events):
                steps = ds[i]
                positions_mm = np.column_stack([
                    steps['x'].astype(np.float32),
                    steps['y'].astype(np.float32),
                    steps['z'].astype(np.float32),
                ])
                counts = count_deposits_per_volume(positions_mm, volume_ranges)
                all_counts.append(counts)

            total_events += n_events
            print(f'  {fname}: {n_events} events scanned')

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

    print()


if __name__ == '__main__':
    main()
