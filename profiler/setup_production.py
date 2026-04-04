"""
One-shot production setup: scan data for total_pad, probe max_keys,
find optimal chunks, save config.

Steps:
  1. Scan events → total_pad (no sim)
  2. Probe max_keys with large buffer → set safe value (one sim build)
  3. Find optimal response_chunk and hits_chunk (multiple sim builds)
  4. Save everything to a production config YAML

Uses established defaults for thresholds and inter_thresh.

Usage:
    python3 -m profiler.setup_production --data events.h5 --config config/sbnd_config.yaml
    python3 -m profiler.setup_production --data events.h5 --config config/sbnd_config.yaml -o config/production.yaml
    python3 -m profiler.setup_production --data events.h5 --config config/sbnd_config.yaml --bucketed
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np

from profiler.production_config import save_config


def round_up_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def main():
    parser = argparse.ArgumentParser(
        description='One-shot production config setup')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('-o', '--output', default=None,
                        help='Output config path (default: config/production_<name>.yaml)')
    parser.add_argument('--events-pad', type=int, default=None,
                        help='Events to scan for total_pad (default: all)')
    parser.add_argument('--event-bench', type=int, default=0,
                        help='Event index for chunk benchmarks (default: 0)')
    parser.add_argument('--use-max', action='store_true',
                        help='Use max deposit count instead of p99.9 for total_pad')
    parser.add_argument('--bucketed', action='store_true')
    parser.add_argument('--probe-max-keys', type=int, default=8_000_000,
                        help='Large max_keys for probing actual counts (default: 8M)')
    parser.add_argument('--probe-max-buckets', type=int, default=100_000,
                        help='Large max_buckets for probing active tiles (default: 100k)')
    parser.add_argument('--headroom', type=float, default=1.5,
                        help='Multiply observed max entries by this (default: 1.5)')
    parser.add_argument('--probe-events', type=int, default=5,
                        help='Events to probe for max_keys/max_buckets (default: 5)')
    parser.add_argument('--n-coarse', type=int, default=3)
    parser.add_argument('--n-fine', type=int, default=10)
    parser.add_argument('--lo', type=int, default=1_000)
    parser.add_argument('--hi', type=int, default=100_000)
    parser.add_argument('--skip-hits', action='store_true',
                        help='Skip hits_chunk optimization')

    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f'config/production_{base}.yaml'

    print('=' * 70)
    print(' JAXTPC — Production Setup')
    print('=' * 70)
    print(f'  Data:    {args.data}')
    print(f'  Config:  {args.config}')
    print(f'  Output:  {args.output}')
    print(f'  Device:  {jax.devices()[0]}')

    # ── Step 1: Find total_pad ──────────────────────────────────────────

    print('\n' + '─' * 70)
    print(' Step 1: Scanning events for optimal total_pad')
    print('─' * 70)

    from profiler.find_optimal_pad import get_volume_ranges, count_deposits_per_volume
    from tools.geometry import generate_detector
    import h5py

    detector_config = generate_detector(args.config)
    volume_ranges = get_volume_ranges(detector_config)

    all_counts = []
    with h5py.File(args.data, 'r') as f:
        ds = f['pstep/lar_vol']
        n_events = ds.shape[0]
        if args.events_pad is not None:
            n_events = min(n_events, args.events_pad)

        for i in range(n_events):
            steps = ds[i]
            positions_mm = np.column_stack([
                steps['x'].astype(np.float32),
                steps['y'].astype(np.float32),
                steps['z'].astype(np.float32),
            ])
            counts = count_deposits_per_volume(positions_mm, volume_ranges)
            all_counts.append(counts)

    counts_array = np.array(all_counts)
    max_per_event = counts_array.max(axis=1)
    pcts = np.percentile(max_per_event, [50, 99.9, 100])

    print(f'  Scanned {n_events} events')
    print(f'  Max-across-volumes: P50={int(pcts[0]):,}, P99.9={int(pcts[1]):,}, Max={int(pcts[2]):,}')

    raw_pad = int(pcts[2]) if args.use_max else int(pcts[1])
    total_pad_10k = round_up_to_multiple(raw_pad, 10_000)
    label = 'max' if args.use_max else 'p99.9'
    print(f'  Using {label}: {raw_pad:,} → rounded to 10k: {total_pad_10k:,}')

    # Ensure divisors exist in chunk search range, bump if needed
    from profiler.find_optimal_chunks import divisors_in_range
    candidates = divisors_in_range(total_pad_10k, args.lo, args.hi)
    if not candidates:
        total_pad = round_up_to_multiple(total_pad_10k, 50_000)
        print(f'  No divisors in [{args.lo:,}, {args.hi:,}] for {total_pad_10k:,}, '
              f'bumped to {total_pad:,}')
    else:
        total_pad = total_pad_10k

    # ── Step 2: Probe max_keys ──────────────────────────────────────────

    print('\n' + '─' * 70)
    print(' Step 2: Probing max_keys with representative events')
    print('─' * 70)

    from tools.config import create_track_hits_config
    from tools.simulation import DetectorSimulator
    from tools.loader import load_event
    from profiler.timing import sync_result

    # Use a temporary response_chunk that divides total_pad
    temp_response_chunk = candidates[len(candidates) // 2] if candidates else 50_000
    # Also need hits_chunk to divide total_pad
    hits_candidates = divisors_in_range(total_pad, args.lo, args.hi)
    temp_hits_chunk = hits_candidates[len(hits_candidates) // 2] if hits_candidates else 25_000

    probe_track_config = create_track_hits_config(
        max_keys=args.probe_max_keys,
        hits_chunk_size=temp_hits_chunk)

    probe_sim = DetectorSimulator(
        detector_config,
        total_pad=total_pad,
        response_chunk_size=temp_response_chunk,
        include_track_hits=True,
        track_config=probe_track_config,
    )
    probe_sim.warm_up()

    max_count = 0
    key = jax.random.PRNGKey(42)
    n_probe = min(args.probe_events, n_events)

    for i in range(n_probe):
        key, subkey = jax.random.split(key)
        deposits = load_event(args.data, probe_sim.config, event_idx=i)
        n_deps = sum(v.n_actual for v in deposits.volumes)

        _, track_hits_raw, _ = probe_sim.process_event(deposits, key=subkey)
        sync_result(track_hits_raw)

        event_max = 0
        for pk, raw in track_hits_raw.items():
            if not isinstance(pk, tuple):
                continue
            c = int(raw[4])
            event_max = max(event_max, c)

        max_count = max(max_count, event_max)
        overflow = event_max >= args.probe_max_keys
        warn = ' *** OVERFLOW ***' if overflow else ''
        print(f'  Event {i}: {n_deps:,} deps, max entries/plane = {event_max:,}{warn}')

    del probe_sim

    raw_max_keys = int(max_count * args.headroom)
    max_keys = round_up_to_multiple(raw_max_keys, 100_000)

    print(f'\n  Observed max: {max_count:,}')
    print(f'  × {args.headroom} headroom, rounded: {max_keys:,}')

    if max_count >= args.probe_max_keys:
        print(f'  WARNING: Hit probe limit! Re-run with larger --probe-max-keys')

    # ── Step 3: Probe max_buckets (if bucketed) ─────────────────────────

    # Detect if bucketed is needed: pixel readout forces it, or --bucketed
    readout_type = detector_config['volumes'][0].get('readout', {}).get('type', 'wire')
    needs_bucketed = args.bucketed or readout_type == 'pixel'
    max_buckets = 1000  # wire default

    if needs_bucketed:
        print('\n' + '─' * 70)
        print(' Step 3: Probing max_buckets for bucketed accumulation')
        print('─' * 70)

        import gc
        jax.clear_caches()
        gc.collect()

        probe_bucket_sim = DetectorSimulator(
            detector_config,
            total_pad=total_pad,
            response_chunk_size=temp_response_chunk,
            use_bucketed=True,
            max_active_buckets=args.probe_max_buckets,
            include_track_hits=False,
        )
        probe_bucket_sim.warm_up()

        max_active = 0
        key = jax.random.PRNGKey(42)
        for i in range(n_probe):
            key, subkey = jax.random.split(key)
            deposits = load_event(args.data, probe_bucket_sim.config, event_idx=i)
            n_deps = sum(v.n_actual for v in deposits.volumes)

            response_signals, _, _ = probe_bucket_sim.process_event(deposits, key=subkey)
            sync_result(response_signals)

            event_max = 0
            for (v, p), sig in response_signals.items():
                if isinstance(sig, tuple) and len(sig) >= 3:
                    na = int(sig[1])
                    event_max = max(event_max, na)

            max_active = max(max_active, event_max)
            overflow = event_max >= args.probe_max_buckets
            warn = ' *** OVERFLOW ***' if overflow else ''
            print(f'  Event {i}: {n_deps:,} deps, max active tiles = {event_max:,}{warn}')

        del probe_bucket_sim

        raw_max_buckets = int(max_active * args.headroom)
        max_buckets = round_up_to_multiple(raw_max_buckets, 5_000)

        print(f'\n  Observed max: {max_active:,}')
        print(f'  × {args.headroom} headroom, rounded: {max_buckets:,}')

        if max_active >= args.probe_max_buckets:
            print(f'  WARNING: Hit probe limit! Re-run with larger --probe-max-buckets')

    # ── Step 4: Find optimal chunks ─────────────────────────────────────

    print('\n' + '─' * 70)
    print(' Step 4: Finding optimal chunk sizes')
    print('─' * 70)
    print(f'  total_pad: {total_pad:,}, max_keys: {max_keys:,}')

    from profiler.find_optimal_chunks import auto_search

    # Phase 1: response_chunk (track_hits OFF — max_keys irrelevant)
    print('\n  Phase 1: response_chunk_size (track_hits OFF)')
    best_response = auto_search(
        detector_config, args.data, args.event_bench, total_pad,
        'response_chunk', args.lo, args.hi,
        include_track_hits=False, fixed_response_chunk=50_000,
        max_keys=max_keys, bucketed=args.bucketed,
        n_coarse=args.n_coarse, n_fine=args.n_fine)

    if not best_response:
        best_response = 50_000
        print(f'  No optimal found, using default: {best_response:,}')
    else:
        print(f'  Best response_chunk: {best_response:,}')

    # Re-align total_pad to response_chunk if needed
    if total_pad % best_response != 0:
        total_pad = round_up_to_multiple(total_pad, best_response)
        print(f'  Re-aligned total_pad to {total_pad:,}')

    # Phase 2: hits_chunk (track_hits ON, uses real max_keys)
    # Default: largest divisor of total_pad that's <= 25000
    hits_divs = divisors_in_range(total_pad, 1000, 25_000)
    best_hits = hits_divs[-1] if hits_divs else best_response
    if not args.skip_hits:
        print('\n  Phase 2: hits_chunk_size (track_hits ON)')
        found = auto_search(
            detector_config, args.data, args.event_bench, total_pad,
            'hits_chunk', args.lo, args.hi,
            include_track_hits=True, fixed_response_chunk=best_response,
            max_keys=max_keys, bucketed=args.bucketed,
            n_coarse=args.n_coarse, n_fine=args.n_fine)
        if found:
            best_hits = found
            print(f'  Best hits_chunk: {best_hits:,}')

    # ── Save ────────────────────────────────────────────────────────────

    config_values = {
        'total_pad': total_pad,
        'response_chunk': best_response,
        'hits_chunk': best_hits,
        'max_keys': max_keys,
        'inter_thresh': 1.0,
        'threshold_adc': 2.0,
        'corr_threshold': 25.0,
        'max_buckets': max_buckets,
    }

    save_config(args.output, config_values, detector_config_path=args.config)

    print('\n' + '=' * 70)
    print(' Production Config Saved')
    print('=' * 70)
    for k, v in config_values.items():
        print(f'  {k:<20} {v:>12,}' if isinstance(v, int) else f'  {k:<20} {v:>12}')
    print(f'\n  File: {args.output}')
    print(f'\n  Usage:')
    print(f'    python3 production/run_batch.py --data {args.data} '
          f'--config {args.config} --production-config {args.output}')
    print()


if __name__ == '__main__':
    main()
