"""
Find the smallest inter_thresh that doesn't lose meaningful charge.

inter_thresh is the intermediate pruning threshold inside the track_hits
fori_loop. After each chunk merge, entries with charge <= inter_thresh
are discarded. Too aggressive = lost physics. Too lax = overflow risk
and slower merges.

Approach:
  - Run with inter_thresh=0 as baseline (no pruning)
  - Run with increasing values, compare total charge and entry counts
  - Report the smallest value where charge loss is below tolerance

Each inter_thresh value requires a separate JIT compilation since it's
part of the static TrackHitsConfig.

Usage:
    python3 -m profiler.find_optimal_inter_thresh --data events.h5 --config config.yaml
    python3 -m profiler.find_optimal_inter_thresh --data events.h5 --config config.yaml --tolerance 0.001
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np

from tools.geometry import generate_detector
from tools.config import create_track_hits_config
from tools.simulation import DetectorSimulator
from tools.loader import load_event

from profiler.timing import sync_result


INTER_THRESH_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def run_with_inter_thresh(detector_config, data_path, event_idx, total_pad,
                          response_chunk, hits_chunk, max_keys, inter_thresh,
                          seed=42):
    """Run sim with a given inter_thresh, return charge stats per plane."""
    jax.clear_caches()
    gc.collect()

    track_config = create_track_hits_config(
        max_keys=max_keys, hits_chunk_size=hits_chunk,
        inter_thresh=inter_thresh)

    sim = DetectorSimulator(
        detector_config,
        total_pad=total_pad,
        response_chunk_size=response_chunk,
        include_track_hits=True,
        track_config=track_config,
    )
    sim.warm_up()

    deposits = load_event(data_path, sim.config, event_idx=event_idx)
    key = jax.random.PRNGKey(seed)
    _, track_hits_raw, _ = sim.process_event(deposits, key=key)
    sync_result(track_hits_raw)

    # Extract charge totals and entry counts per plane
    total_charge = 0.0
    total_entries = 0

    for (vol_idx, plane_idx), raw in track_hits_raw.items():
        if not isinstance((vol_idx, plane_idx), tuple):
            continue
        sk, tk, gk, ch, count, _ = raw
        P = int(count)
        if P == 0:
            continue
        chs = np.asarray(ch[:P], dtype=np.float32)
        total_charge += float(np.sum(chs))
        total_entries += P

    del sim
    return total_charge, total_entries


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal inter_thresh for track hits pruning')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--events', type=int, default=1,
                        help='Number of events to test (default: 1)')
    parser.add_argument('--total-pad', type=int, default=500_000)
    parser.add_argument('--response-chunk', type=int, default=50_000)
    parser.add_argument('--hits-chunk', type=int, default=25_000)
    parser.add_argument('--max-keys', type=int, default=4_000_000)
    parser.add_argument('--tolerance', type=float, default=0.001,
                        help='Max acceptable charge loss fraction (default: 0.001 = 0.1%%)')
    parser.add_argument('--values', type=float, nargs='+', default=None,
                        help='Custom inter_thresh values to test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-config', default=None,
                        help='Save inter_thresh to production config YAML')

    args = parser.parse_args()

    values = args.values if args.values else INTER_THRESH_VALUES
    detector_config = generate_detector(args.config)

    print('=' * 70)
    print(' JAXTPC — Find Optimal inter_thresh')
    print('=' * 70)
    print(f'  Data:      {args.data}')
    print(f'  Config:    {args.config}')
    print(f'  Events:    {args.events} (starting at {args.event})')
    print(f'  Tolerance: {args.tolerance*100:.2f}% charge loss')
    print(f'  Values:    {values}')
    print(f'  Device:    {jax.devices()[0]}')

    # Accumulate across events
    accumulated = {}
    for val in values:
        accumulated[val] = {'charge': 0.0, 'entries': 0}

    for i in range(args.events):
        evt_idx = args.event + i
        print(f'\n  Event {evt_idx}:')

        for val in values:
            charge, entries = run_with_inter_thresh(
                detector_config, args.data, evt_idx, args.total_pad,
                args.response_chunk, args.hits_chunk, args.max_keys,
                val, seed=args.seed)
            accumulated[val]['charge'] += charge
            accumulated[val]['entries'] += entries
            print(f'    inter_thresh={val:>6.2f}  '
                  f'charge={charge:>14,.0f}  entries={entries:>10,}')

    # Compute loss relative to baseline (inter_thresh=0)
    baseline_charge = accumulated[values[0]]['charge']
    baseline_entries = accumulated[values[0]]['entries']

    print(f'\n  Results (accumulated over {args.events} event(s)):')
    print(f'  {"inter_thresh":>14} {"Total Charge":>14} {"Charge Lost %":>14} '
          f'{"Entries":>10} {"Entries Lost %":>14}')
    print(f'  {"─" * 70}')

    best = None
    for val in values:
        ch = accumulated[val]['charge']
        en = accumulated[val]['entries']
        ch_loss = (baseline_charge - ch) / baseline_charge if baseline_charge > 0 else 0
        en_loss = (baseline_entries - en) / baseline_entries if baseline_entries > 0 else 0

        marker = ''
        if best is None and ch_loss <= args.tolerance and val > 0:
            best = val
            marker = ' << recommended'

        print(f'  {val:>14.2f} {ch:>14,.0f} {ch_loss*100:>13.4f}% '
              f'{en:>10,} {en_loss*100:>13.1f}%{marker}')

    if best is None and len(values) > 1:
        # All values within tolerance, pick the largest
        best = values[-1]

    print()
    if best is not None:
        print(f'  Recommended: --inter-thresh {best}')
        print(f'  (largest value with <= {args.tolerance*100:.2f}% charge loss)')

        if args.save_config:
            from profiler.production_config import update_config
            update_config(args.save_config, {'inter_thresh': best},
                          detector_config_path=args.config)
            print(f'  Saved to {args.save_config}')
    else:
        print(f'  All tested values exceed {args.tolerance*100:.2f}% tolerance.')
        print(f'  Consider using inter_thresh=0 or testing smaller values.')

    print()


if __name__ == '__main__':
    main()
