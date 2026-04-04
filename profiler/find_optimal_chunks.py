"""
Find optimal response_chunk_size and hits_chunk_size via two-pass search.

Pass 1 (coarse): time each divisor of total_pad with few iterations.
Pass 2 (fine): re-time the top 3 with more iterations.

Sweeps response_chunk first (track_hits off), then hits_chunk (track_hits on,
using the best response_chunk from pass 1).

Usage:
    python3 -m profiler.find_optimal_chunks --data events.h5 --config config.yaml
    python3 -m profiler.find_optimal_chunks --data events.h5 --config config.yaml --total-pad 500000
    python3 -m profiler.find_optimal_chunks --data events.h5 --config config.yaml --lo 5000 --hi 100000
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np

from tools.geometry import generate_detector
from tools.config import create_track_hits_config
from tools.simulation import DetectorSimulator
from tools.loader import load_event

from profiler.timing import sync_result


def divisors_in_range(total, lo, hi):
    """Return sorted divisors of total in [lo, hi]."""
    return sorted(d for d in range(max(1, lo), hi + 1) if total % d == 0)


def _time_sim(sim, deposits, n_iter):
    """Run n_iter process_event calls, return list of wall times in ms."""
    key = jax.random.PRNGKey(42)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = sim.process_event(deposits, key=key)
        sync_result(result)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_one(detector_config, data_path, event_idx, total_pad,
              response_chunk, hits_chunk, include_track_hits,
              max_keys, bucketed, n_timed, label=''):
    """Benchmark a single chunk configuration. Returns (mean_ms, std_ms, times)."""
    jax.clear_caches()
    gc.collect()

    track_config = None
    if include_track_hits:
        track_config = create_track_hits_config(
            max_keys=max_keys, hits_chunk_size=hits_chunk)

    sim = DetectorSimulator(
        detector_config,
        total_pad=total_pad,
        response_chunk_size=response_chunk,
        include_track_hits=include_track_hits,
        track_config=track_config,
        use_bucketed=bucketed,
    )
    sim.warm_up()

    deposits = load_event(data_path, sim.config, event_idx=event_idx)
    n_deps = sum(v.n_actual for v in deposits.volumes)

    # Real-data warmup
    _time_sim(sim, deposits, 1)

    # Timed runs
    times = _time_sim(sim, deposits, n_timed)
    mean_t = np.mean(times)
    std_t = np.std(times)

    if label:
        print(f'    {label:<30} {mean_t:>8.1f} ± {std_t:>5.1f} ms  ({n_deps:,} deps)')

    del sim
    return mean_t, std_t, times


def auto_search(detector_config, data_path, event_idx, total_pad,
                chunk_label, lo, hi, include_track_hits,
                fixed_response_chunk, max_keys, bucketed,
                n_coarse=3, n_fine=10):
    """Two-pass search over divisors of total_pad in [lo, hi]."""
    candidates = divisors_in_range(total_pad, lo, hi)
    if not candidates:
        print(f'  No divisors of {total_pad:,} in [{lo:,}, {hi:,}]!')
        return None

    print(f'  Candidates ({len(candidates)}): {candidates}')

    # Coarse pass
    print(f'\n  Coarse pass ({n_coarse} iters each):')
    coarse = {}
    for val in candidates:
        if chunk_label == 'response_chunk':
            rc, hc = val, 25_000
        else:
            rc, hc = fixed_response_chunk, val

        mean, std, _ = bench_one(
            detector_config, data_path, event_idx, total_pad,
            rc, hc, include_track_hits, max_keys, bucketed,
            n_coarse, label=f'{val:>8,}')
        coarse[val] = mean

    # Top 3
    ranked = sorted(coarse, key=lambda k: coarse[k])
    top3 = ranked[:3]
    print(f'\n  Top 3: {[f"{v:,}" for v in top3]}')

    # Fine pass
    print(f'\n  Fine pass ({n_fine} iters each):')
    fine = {}
    for val in top3:
        if chunk_label == 'response_chunk':
            rc, hc = val, 25_000
        else:
            rc, hc = fixed_response_chunk, val

        mean, std, _ = bench_one(
            detector_config, data_path, event_idx, total_pad,
            rc, hc, include_track_hits, max_keys, bucketed,
            n_fine, label=f'{val:>8,}')
        fine[val] = mean

    best = min(fine, key=lambda k: fine[k])

    # Summary table
    print(f'\n  {"─" * 60}')
    print(f'  {chunk_label:<20} {"Coarse (ms)":>12} {"Fine (ms)":>12}')
    print(f'  {"─" * 60}')
    for val in sorted(coarse.keys()):
        c = coarse[val]
        if val in fine:
            f = fine[val]
            marker = ' << best' if val == best else ''
            print(f'  {val:>15,}   {c:>12.1f} {f:>12.1f}{marker}')
        else:
            print(f'  {val:>15,}   {c:>12.1f} {"--":>12}')
    print(f'  {"─" * 60}')

    return best


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal chunk sizes for JAXTPC simulation')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--event', type=int, default=0, help='Event index (default: 0)')
    parser.add_argument('--total-pad', type=int, default=500_000)
    parser.add_argument('--lo', type=int, default=1_000, help='Min chunk size (default: 1000)')
    parser.add_argument('--hi', type=int, default=100_000, help='Max chunk size (default: 100000)')
    parser.add_argument('--n-coarse', type=int, default=3)
    parser.add_argument('--n-fine', type=int, default=10)
    parser.add_argument('--max-keys', type=int, default=4_000_000)
    parser.add_argument('--bucketed', action='store_true')
    parser.add_argument('--skip-hits', action='store_true',
                        help='Skip hits_chunk optimization')
    parser.add_argument('--save-config', default=None,
                        help='Save results to production config YAML')

    args = parser.parse_args()

    detector_config = generate_detector(args.config)

    print('=' * 70)
    print(' JAXTPC — Find Optimal Chunk Sizes')
    print('=' * 70)
    print(f'  Data:      {args.data}')
    print(f'  Config:    {args.config}')
    print(f'  total_pad: {args.total_pad:,}')
    print(f'  Range:     [{args.lo:,}, {args.hi:,}]')
    print(f'  Device:    {jax.devices()[0]}')

    # Phase 1: response_chunk_size (track_hits OFF)
    print('\n  Phase 1: response_chunk_size (track_hits OFF)')
    print('  ' + '─' * 56)

    best_response = auto_search(
        detector_config, args.data, args.event, args.total_pad,
        'response_chunk', args.lo, args.hi,
        include_track_hits=False, fixed_response_chunk=50_000,
        max_keys=args.max_keys, bucketed=args.bucketed,
        n_coarse=args.n_coarse, n_fine=args.n_fine)

    if best_response:
        print(f'\n  Best response_chunk_size: {best_response:,}')

    # Phase 2: hits_chunk_size (track_hits ON)
    best_hits = None
    if not args.skip_hits and best_response:
        print('\n  Phase 2: hits_chunk_size (track_hits ON)')
        print('  ' + '─' * 56)

        best_hits = auto_search(
            detector_config, args.data, args.event, args.total_pad,
            'hits_chunk', args.lo, args.hi,
            include_track_hits=True, fixed_response_chunk=best_response,
            max_keys=args.max_keys, bucketed=args.bucketed,
            n_coarse=args.n_coarse, n_fine=args.n_fine)

        if best_hits:
            print(f'\n  Best hits_chunk_size: {best_hits:,}')

    # Summary
    print('\n' + '=' * 70)
    print('  RESULTS')
    print('=' * 70)
    if best_response:
        print(f'  --response-chunk {best_response}')
    if best_hits:
        print(f'  --hits-chunk {best_hits}')

    if args.save_config:
        from profiler.production_config import update_config
        updates = {}
        if best_response:
            updates['response_chunk'] = best_response
        if best_hits:
            updates['hits_chunk'] = best_hits
        if updates:
            update_config(args.save_config, updates,
                          detector_config_path=args.config)
            print(f'  Saved to {args.save_config}')

    print()


if __name__ == '__main__':
    main()
