"""
Find the optimal max_keys for track hits by observing actual entry counts.

max_keys is the size of the state arrays carried through the track_hits
fori_loop. Too small = silent truncation (lost tracks). Too large = wasted
memory and slower sort operations in merge_chunk_sensor_hits.

Approach:
  - Run with a large max_keys on representative events
  - Record final_count from each (volume, plane)
  - Take max across all events/planes, add headroom
  - Suggest a value

Usage:
    python3 -m profiler.find_optimal_max_keys --data events.h5 --config config.yaml
    python3 -m profiler.find_optimal_max_keys --data events.h5 --config config.yaml --events 10
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal max_keys for track hits')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--event', type=int, default=0, help='Starting event index')
    parser.add_argument('--events', type=int, default=5,
                        help='Number of events to scan (default: 5)')
    parser.add_argument('--total-pad', type=int, default=500_000)
    parser.add_argument('--response-chunk', type=int, default=50_000)
    parser.add_argument('--hits-chunk', type=int, default=25_000)
    parser.add_argument('--probe-max-keys', type=int, default=8_000_000,
                        help='Large max_keys for probing (default: 8M)')
    parser.add_argument('--inter-thresh', type=float, default=1.0)
    parser.add_argument('--headroom', type=float, default=1.5,
                        help='Multiply observed max by this factor (default: 1.5)')
    parser.add_argument('--round-to', type=int, default=100_000,
                        help='Round suggestion up to nearest N (default: 100000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-config', default=None,
                        help='Save max_keys to production config YAML')

    args = parser.parse_args()

    detector_config = generate_detector(args.config)

    print('=' * 70)
    print(' JAXTPC — Find Optimal max_keys')
    print('=' * 70)
    print(f'  Data:           {args.data}')
    print(f'  Config:         {args.config}')
    print(f'  Events:         {args.events} (starting at {args.event})')
    print(f'  Probe max_keys: {args.probe_max_keys:,}')
    print(f'  inter_thresh:   {args.inter_thresh}')
    print(f'  Headroom:       {args.headroom}x')
    print(f'  Device:         {jax.devices()[0]}')

    track_config = create_track_hits_config(
        max_keys=args.probe_max_keys,
        hits_chunk_size=args.hits_chunk,
        inter_thresh=args.inter_thresh)

    sim = DetectorSimulator(
        detector_config,
        total_pad=args.total_pad,
        response_chunk_size=args.response_chunk,
        include_track_hits=True,
        track_config=track_config,
    )
    sim.warm_up()

    cfg = sim.config
    key = jax.random.PRNGKey(args.seed)

    # Collect counts across events
    all_counts = []  # list of dicts: {(vol, plane): count}
    max_count = 0
    max_count_event = -1

    for i in range(args.events):
        evt_idx = args.event + i
        key, subkey = jax.random.split(key)

        deposits = load_event(args.data, cfg, event_idx=evt_idx)
        n_deps = sum(v.n_actual for v in deposits.volumes)

        _, track_hits_raw, _ = sim.process_event(deposits, key=subkey)
        sync_result(track_hits_raw)

        event_counts = {}
        event_max = 0
        overflow = False

        for (vol_idx, plane_idx), raw in track_hits_raw.items():
            if not isinstance((vol_idx, plane_idx), tuple):
                continue
            _, _, _, _, count, _ = raw
            c = int(count)
            event_counts[(vol_idx, plane_idx)] = c
            event_max = max(event_max, c)
            if c >= args.probe_max_keys:
                overflow = True

        all_counts.append(event_counts)
        if event_max > max_count:
            max_count = event_max
            max_count_event = evt_idx

        warn = ' *** OVERFLOW ***' if overflow else ''
        print(f'  Event {evt_idx}: {n_deps:,} deps, '
              f'max entries/plane = {event_max:,}{warn}')

        # Per-plane detail
        for (v, p), c in sorted(event_counts.items()):
            plane_names = cfg.plane_names[v] if v < len(cfg.plane_names) else ('?',)
            pname = plane_names[p] if p < len(plane_names) else '?'
            print(f'    vol {v} plane {p} ({pname}): {c:,}')

    # Statistics across all events/planes
    all_plane_counts = []
    for ec in all_counts:
        all_plane_counts.extend(ec.values())

    if not all_plane_counts:
        print('\n  No track hits data collected!')
        return

    arr = np.array(all_plane_counts)
    pcts = np.percentile(arr, [50, 90, 95, 99, 100])

    print(f'\n  Entry count distribution ({len(all_plane_counts)} plane-events):')
    print(f'    P50  = {int(pcts[0]):>10,}')
    print(f'    P90  = {int(pcts[1]):>10,}')
    print(f'    P95  = {int(pcts[2]):>10,}')
    print(f'    P99  = {int(pcts[3]):>10,}')
    print(f'    Max  = {int(pcts[4]):>10,}  (event {max_count_event})')

    # Suggestion
    import math
    raw_suggestion = int(max_count * args.headroom)
    suggestion = int(math.ceil(raw_suggestion / args.round_to) * args.round_to)

    print(f'\n  Suggestion:')
    print(f'    Observed max:   {max_count:>10,}')
    print(f'    × {args.headroom} headroom: {raw_suggestion:>10,}')
    print(f'    Rounded:        {suggestion:>10,}')
    print(f'    --max-keys {suggestion}')

    if max_count >= args.probe_max_keys:
        print(f'\n  WARNING: Observed count hit probe limit ({args.probe_max_keys:,})!')
        print(f'  Re-run with a larger --probe-max-keys value.')

    if args.save_config:
        from profiler.production_config import update_config
        update_config(args.save_config, {'max_keys': suggestion},
                      detector_config_path=args.config)
        print(f'  Saved to {args.save_config}')

    print()


if __name__ == '__main__':
    main()
