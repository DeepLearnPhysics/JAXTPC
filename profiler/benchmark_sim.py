"""
Benchmark simulation timing across feature combinations and parameter sweeps.

Times the full process_event pipeline with different feature toggles
(noise, electronics, track_hits, digitization, bucketed) and optional
sweeps of numeric parameters (total_pad, response_chunk, etc.).

Reuses the profiler.timing infrastructure for proper GPU synchronization.

Usage:
    python3 -m profiler.benchmark_sim --data events.h5 --config config/sbnd_config.yaml
    python3 -m profiler.benchmark_sim --data events.h5 --runs 10 --bucketed
    python3 -m profiler.benchmark_sim --data events.h5 --sweep total_pad 100000 200000 500000
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

from profiler.timing import TimingResult, sync_result


# Feature combinations to test
FEATURE_MODES = [
    # (label, noise, electronics, track_hits, digitize)
    ('Baseline',              False, False, False, False),
    ('+ Track Hits',          False, False, True,  False),
    ('+ Digitize',            False, False, False, True),
    ('+ Noise',               True,  False, False, False),
    ('+ Electronics',         False, True,  False, False),
    ('+ Noise+Elec',          True,  True,  False, False),
    ('+ Noise+Elec+TH+Dig',  True,  True,  True,  True),
]


def _build_simulator(detector_config, total_pad, response_chunk,
                     bucketed, max_buckets, noise, electronics,
                     track_hits, digitize, hits_chunk, max_keys):
    """Build a simulator with the given configuration."""
    track_config = None
    if track_hits:
        track_config = create_track_hits_config(
            max_keys=max_keys, hits_chunk_size=hits_chunk)

    return DetectorSimulator(
        detector_config,
        total_pad=total_pad,
        response_chunk_size=response_chunk,
        use_bucketed=bucketed,
        max_active_buckets=max_buckets if bucketed else None,
        include_noise=noise,
        include_electronics=electronics,
        include_track_hits=track_hits,
        include_digitize=digitize,
        track_config=track_config,
    )


def benchmark_config(detector_config, data_path, event_idx,
                     total_pad, response_chunk, bucketed, max_buckets,
                     noise, electronics, track_hits, digitize,
                     hits_chunk, max_keys, num_runs, warmup, label=''):
    """Benchmark a single configuration. Returns TimingResult."""
    jax.clear_caches()
    gc.collect()

    sim = _build_simulator(
        detector_config, total_pad, response_chunk,
        bucketed, max_buckets, noise, electronics,
        track_hits, digitize, hits_chunk, max_keys)

    # JIT warmup
    t_jit = time.perf_counter()
    sim.warm_up()
    t_jit = time.perf_counter() - t_jit

    # Load real data
    deposits = load_event(data_path, sim.config, event_idx=event_idx)
    n_deps = sum(v.n_actual for v in deposits.volumes)

    key = jax.random.PRNGKey(42)

    # Real-data warmup
    for _ in range(warmup):
        result = sim.process_event(deposits, key=key)
        sync_result(result)

    # Timed runs
    times = []
    for _ in range(num_runs):
        gc.collect()
        t0 = time.perf_counter()
        result = sim.process_event(deposits, key=key)
        sync_result(result)
        times.append((time.perf_counter() - t0) * 1000)

    tr = TimingResult(name=label, times_ms=times)
    del sim
    return tr, t_jit, n_deps


def run_feature_sweep(detector_config, data_path, event_idx,
                      total_pad, response_chunk, bucketed, max_buckets,
                      hits_chunk, max_keys, num_runs, warmup):
    """Sweep feature combinations and print results table."""
    print('\n  Feature Sweep')
    print('  ' + '─' * 65)

    results = {}
    for label, noise, elec, th, dig in FEATURE_MODES:
        tr, t_jit, n_deps = benchmark_config(
            detector_config, data_path, event_idx,
            total_pad, response_chunk, bucketed, max_buckets,
            noise, elec, th, dig, hits_chunk, max_keys,
            num_runs, warmup, label=label)
        results[label] = tr
        print(f'    {label:<25} {tr.mean_ms:>8.1f} ± {tr.std_ms:>5.1f} ms  '
              f'(JIT: {t_jit:.1f}s, {n_deps:,} deps)')

    # Overhead relative to baseline
    baseline = results['Baseline'].mean_ms
    print()
    print(f'  {"Mode":<25} {"Time (ms)":>10} {"Overhead":>10}')
    print('  ' + '─' * 50)
    for label, _, _, _, _ in FEATURE_MODES:
        tr = results[label]
        overhead = tr.mean_ms - baseline
        print(f'  {label:<25} {tr.mean_ms:>10.1f} {overhead:>+10.1f}')

    return results


def run_param_sweep(detector_config, data_path, event_idx,
                    param_name, values,
                    total_pad, response_chunk, bucketed, max_buckets,
                    hits_chunk, max_keys, num_runs, warmup):
    """Sweep a single numeric parameter and print results."""
    print(f'\n  Parameter Sweep: {param_name}')
    print(f'  Values: {values}')
    print('  ' + '─' * 65)

    results = {}
    for val in values:
        kwargs = {
            'total_pad': total_pad,
            'response_chunk': response_chunk,
            'bucketed': bucketed,
            'max_buckets': max_buckets,
            'hits_chunk': hits_chunk,
            'max_keys': max_keys,
        }
        kwargs[param_name] = val

        # Auto-fix: response_chunk must divide total_pad
        tp = kwargs['total_pad']
        rc = kwargs['response_chunk']
        if tp % rc != 0:
            # Find nearest divisor
            for delta in range(rc):
                if tp % (rc - delta) == 0:
                    kwargs['response_chunk'] = rc - delta
                    break

        label = f'{param_name}={val:,}'
        try:
            tr, t_jit, n_deps = benchmark_config(
                detector_config, data_path, event_idx,
                kwargs['total_pad'], kwargs['response_chunk'],
                kwargs['bucketed'], kwargs['max_buckets'],
                False, False, False, False,
                kwargs['hits_chunk'], kwargs['max_keys'],
                num_runs, warmup, label=label)
            results[val] = tr
            print(f'    {label:<30} {tr.mean_ms:>8.1f} ± {tr.std_ms:>5.1f} ms  '
                  f'(JIT: {t_jit:.1f}s)')
        except Exception as e:
            print(f'    {label:<30} FAILED: {e}')

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark JAXTPC simulation timing')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--event', type=int, default=0, help='Event index (default: 0)')
    parser.add_argument('--runs', type=int, default=7, help='Timed runs (default: 7)')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup runs (default: 2)')
    parser.add_argument('--total-pad', type=int, default=200_000)
    parser.add_argument('--response-chunk', type=int, default=50_000)
    parser.add_argument('--bucketed', action='store_true')
    parser.add_argument('--max-buckets', type=int, default=1000)
    parser.add_argument('--hits-chunk', type=int, default=25_000)
    parser.add_argument('--max-keys', type=int, default=4_000_000)
    parser.add_argument('--sweep', nargs='+', default=None,
                        help='Sweep a parameter: --sweep total_pad 100000 200000 500000')
    parser.add_argument('--features-only', action='store_true',
                        help='Only run the feature combination sweep')

    args = parser.parse_args()

    print('=' * 70)
    print(' JAXTPC — Simulation Benchmark')
    print('=' * 70)
    print(f'  Data:     {args.data}')
    print(f'  Config:   {args.config}')
    print(f'  Event:    {args.event}')
    print(f'  Runs:     {args.runs}, Warmup: {args.warmup}')
    print(f'  Device:   {jax.devices()[0]}')

    detector_config = generate_detector(args.config)

    if args.sweep and not args.features_only:
        param_name = args.sweep[0]
        values = [int(v) for v in args.sweep[1:]]
        run_param_sweep(
            detector_config, args.data, args.event,
            param_name, values,
            args.total_pad, args.response_chunk, args.bucketed, args.max_buckets,
            args.hits_chunk, args.max_keys, args.runs, args.warmup)
    else:
        run_feature_sweep(
            detector_config, args.data, args.event,
            args.total_pad, args.response_chunk, args.bucketed, args.max_buckets,
            args.hits_chunk, args.max_keys, args.runs, args.warmup)


if __name__ == '__main__':
    main()
