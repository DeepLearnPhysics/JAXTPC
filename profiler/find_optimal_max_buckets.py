"""
Find the optimal max_buckets for bucketed accumulation by observing actual active tile counts.

Runs the simulation with a large probe max_buckets on representative events,
records num_active per volume/plane, and suggests a safe value with headroom.

Usage:
    python3 -m profiler.find_optimal_max_buckets --data events.h5 --config config/cubic_pixel_config.yaml
    python3 -m profiler.find_optimal_max_buckets --data events.h5 --config config/cubic_wireplane_config.yaml --bucketed
"""

import argparse
import math
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
        description='Find optimal max_buckets for bucketed accumulation')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--event', type=int, default=0, help='Starting event index')
    parser.add_argument('--events', type=int, default=5,
                        help='Number of events to scan (default: 5)')
    parser.add_argument('--total-pad', type=int, default=300_000)
    parser.add_argument('--response-chunk', type=int, default=50_000)
    parser.add_argument('--probe-max-buckets', type=int, default=100_000,
                        help='Large max_buckets for probing (default: 100k)')
    parser.add_argument('--bucketed', action='store_true',
                        help='Force bucketed mode (auto-enabled for pixel readout)')
    parser.add_argument('--headroom', type=float, default=1.5,
                        help='Multiply observed max by this factor (default: 1.5)')
    parser.add_argument('--round-to', type=int, default=5_000,
                        help='Round suggestion up to nearest N (default: 5000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-config', default=None,
                        help='Save max_buckets to production config YAML')

    args = parser.parse_args()

    detector_config = generate_detector(args.config)

    print('=' * 70)
    print(' JAXTPC — Find Optimal max_buckets')
    print('=' * 70)
    print(f'  Data:              {args.data}')
    print(f'  Config:            {args.config}')
    print(f'  Events:            {args.events} (starting at {args.event})')
    print(f'  Probe max_buckets: {args.probe_max_buckets:,}')
    print(f'  Headroom:          {args.headroom}x')
    print(f'  Device:            {jax.devices()[0]}')

    sim = DetectorSimulator(
        detector_config,
        total_pad=args.total_pad,
        response_chunk_size=args.response_chunk,
        use_bucketed=args.bucketed,
        max_active_buckets=args.probe_max_buckets,
        include_track_hits=False,
    )
    sim.warm_up()

    cfg = sim.config
    if not cfg.use_bucketed:
        print('\n  Bucketed mode not active (wire readout without --bucketed).')
        print('  Nothing to probe. Use --bucketed or a pixel config.')
        return

    key = jax.random.PRNGKey(args.seed)
    max_active = 0
    max_active_event = -1

    for i in range(args.events):
        evt_idx = args.event + i
        key, subkey = jax.random.split(key)

        deposits = load_event(args.data, cfg, event_idx=evt_idx)
        n_deps = sum(v.n_actual for v in deposits.volumes)

        response_signals, _, _ = sim.process_event(deposits, key=subkey)
        sync_result(response_signals)

        event_max = 0
        for (v, p), sig in response_signals.items():
            if not isinstance(sig, tuple) or len(sig) < 3:
                continue
            na = int(sig[1])
            event_max = max(event_max, na)
            print(f'    vol {v} plane {p}: num_active = {na:,}')

        if event_max > max_active:
            max_active = event_max
            max_active_event = evt_idx

        overflow = event_max >= args.probe_max_buckets
        warn = ' *** OVERFLOW ***' if overflow else ''
        print(f'  Event {evt_idx}: {n_deps:,} deps, max active = {event_max:,}{warn}')

    raw_suggestion = int(max_active * args.headroom)
    suggestion = int(math.ceil(raw_suggestion / args.round_to) * args.round_to)

    print(f'\n  Observed max:     {max_active:>10,}  (event {max_active_event})')
    print(f'  × {args.headroom} headroom:   {raw_suggestion:>10,}')
    print(f'  Rounded:          {suggestion:>10,}')
    print(f'  --max-buckets {suggestion}')

    if max_active >= args.probe_max_buckets:
        print(f'\n  WARNING: Hit probe limit! Re-run with larger --probe-max-buckets')

    if args.save_config:
        from profiler.production_config import update_config
        update_config(args.save_config, {'max_buckets': suggestion},
                      detector_config_path=args.config)
        print(f'  Saved to {args.save_config}')

    print()


if __name__ == '__main__':
    main()
