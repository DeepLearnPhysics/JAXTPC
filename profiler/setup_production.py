"""
One-shot production setup: scan data for total_pad, estimate max_keys,
find optimal chunks, save config.

Steps:
  1. Scan all files → total_pad (no sim)
  2. Estimate max_keys from deposit geometry across all files (no sim)
  3. Find optimal response_chunk and hits_chunk (sim on single event)
  4. Save everything to a production config YAML

Accepts multiple --data files or a directory. Steps 1-2 scan everything;
step 3 uses a single random event for chunk benchmarking.

Usage:
    python3 -m profiler.setup_production --data events.h5 --config config.yaml
    python3 -m profiler.setup_production --data run1.h5 run2.h5 --config config.yaml
    python3 -m profiler.setup_production --data /path/to/h5_dir/ --config config.yaml
"""

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np

from profiler.production_config import save_config


def round_up_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def _resolve_data_paths(data_arg):
    """Resolve a list of paths/directories to HDF5 files."""
    if isinstance(data_arg, str):
        data_arg = [data_arg]
    paths = []
    for p in data_arg:
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, '*.h5'))))
        else:
            paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description='One-shot production config setup')
    parser.add_argument('--data', required=True, nargs='+',
                        help='Input HDF5 file(s) or directory')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('-o', '--output', default=None,
                        help='Output config path (default: config/production_<name>.yaml)')
    parser.add_argument('--events-pad', type=int, default=None,
                        help='Events per file to scan for total_pad/max_keys (default: all)')
    parser.add_argument('--use-p999', action='store_true',
                        help='Use p99.9 deposit count instead of max for total_pad')
    parser.add_argument('--bucketed', action='store_true')
    parser.add_argument('--probe-max-buckets', type=int, default=100_000,
                        help='Large max_buckets for probing active tiles (default: 100k)')
    parser.add_argument('--headroom', type=float, default=1.3,
                        help='Multiply observed max entries by this (default: 1.3)')
    parser.add_argument('--probe-events', type=int, default=5,
                        help='Events to probe for max_buckets (default: 5)')
    parser.add_argument('--n-coarse', type=int, default=3)
    parser.add_argument('--n-fine', type=int, default=10)
    parser.add_argument('--lo', type=int, default=1_000)
    parser.add_argument('--hi', type=int, default=100_000)
    parser.add_argument('--skip-hits', action='store_true',
                        help='Skip hits_chunk optimization')
    parser.add_argument('--group-size', type=int, default=5)
    parser.add_argument('--gap-threshold', type=float, default=5.0)
    parser.add_argument('--tag', default=None,
                        help='Tag for figure filenames (default: config name)')

    args = parser.parse_args()

    h5_files = _resolve_data_paths(args.data)
    if not h5_files:
        print('No HDF5 files found!')
        return

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f'config/production_{base}.yaml'

    tag = args.tag or os.path.splitext(os.path.basename(args.config))[0]

    print('=' * 70)
    print(' JAXTPC — Production Setup')
    print('=' * 70)
    print(f'  Files:   {len(h5_files)}')
    for p in h5_files:
        print(f'    {p}')
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
    total_scanned = 0
    for fpath in h5_files:
        fname = os.path.basename(fpath)
        with h5py.File(fpath, 'r') as f:
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
            total_scanned += n_events
        if len(h5_files) > 1:
            print(f'  {fname}: {n_events} events')

    counts_array = np.array(all_counts)
    max_per_event = counts_array.max(axis=1)
    pcts = np.percentile(max_per_event, [50, 99.9, 100])

    print(f'  Scanned {total_scanned} events across {len(h5_files)} file(s)')
    print(f'  Max-across-volumes: P50={int(pcts[0]):,}, P99.9={int(pcts[1]):,}, Max={int(pcts[2]):,}')

    raw_pad = int(pcts[1]) if args.use_p999 else int(pcts[2])
    total_pad_10k = round_up_to_multiple(raw_pad, 10_000)
    label = 'p99.9' if args.use_p999 else 'max'
    print(f'  Using {label}: {raw_pad:,} → rounded to 10k: {total_pad_10k:,}')

    from profiler.find_optimal_chunks import divisors_in_range
    candidates = divisors_in_range(total_pad_10k, args.lo, args.hi)
    if not candidates:
        total_pad = round_up_to_multiple(total_pad_10k, 50_000)
        print(f'  No divisors in [{args.lo:,}, {args.hi:,}] for {total_pad_10k:,}, '
              f'bumped to {total_pad:,}')
    else:
        total_pad = total_pad_10k

    from profiler.plots import plot_deposit_distribution
    plot_deposit_distribution(counts_array, total_pad, tag=tag)

    # ── Step 2: Estimate max_keys ────────────────────────────────────────

    print('\n' + '─' * 70)
    print(' Step 2: Estimating max_keys from deposit geometry (all files)')
    print('─' * 70)

    from profiler.estimate_max_keys import estimate_max_keys

    max_keys, keys_info = estimate_max_keys(
        h5_files, args.config,
        events_per_file=args.events_pad,
        total_pad=total_pad,
        group_size=args.group_size,
        gap_threshold=args.gap_threshold)

    print(f'  {keys_info["n_events"]} events across {keys_info["n_files"]} file(s)')
    print(f'  Max observed keys:     {keys_info["max_observed_keys"]:,}')
    print(f'  Upper-half max ratio:  {keys_info["upper_max_ratio"]:.3f}')
    print(f'  Extrapolated to {total_pad:,}: {keys_info["extrapolated"]:,}')
    print(f'  Rounded:               {max_keys:,}')

    from profiler.plots import plot_keys_vs_deposits, plot_keys_ratio
    plot_keys_vs_deposits(keys_info['all_deps'], keys_info['all_keys'],
                          total_pad, max_keys, keys_info['upper_max_ratio'], tag=tag)
    plot_keys_ratio(keys_info['all_deps'], keys_info['all_keys'], tag=tag)

    # ── Step 3: Probe max_buckets (if bucketed) ─────────────────────────

    readout_type = detector_config['volumes'][0].get('readout', {}).get('type', 'wire')
    needs_bucketed = args.bucketed or readout_type == 'pixel'
    max_buckets = 1000

    if needs_bucketed:
        print('\n' + '─' * 70)
        print(' Step 3: Probing max_buckets for bucketed accumulation')
        print('─' * 70)

        import gc
        from tools.simulation import DetectorSimulator
        from tools.loader import load_event
        from profiler.timing import sync_result

        jax.clear_caches()
        gc.collect()

        temp_response_chunk = candidates[len(candidates) // 2] if candidates else 50_000

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
        bench_file = h5_files[0]
        n_probe = min(args.probe_events, total_scanned)
        for i in range(n_probe):
            key, subkey = jax.random.split(key)
            deposits = load_event(bench_file, probe_bucket_sim.config, event_idx=i)
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
    from profiler.plots import plot_chunk_timing
    import h5py

    bench_file = h5_files[0]
    with h5py.File(bench_file, 'r') as f:
        n_bench_events = f['pstep/lar_vol'].shape[0]
    bench_event = np.random.RandomState(42).randint(0, n_bench_events)
    print(f'  Benchmark: {os.path.basename(bench_file)} event {bench_event}')

    # Phase 1: response_chunk (track_hits OFF)
    print('\n  Phase 1: response_chunk_size (track_hits OFF)')
    best_response, resp_coarse, _ = auto_search(
        detector_config, bench_file, bench_event, total_pad,
        'response_chunk', args.lo, args.hi,
        include_track_hits=False, fixed_response_chunk=50_000,
        max_keys=max_keys, bucketed=args.bucketed,
        n_coarse=args.n_coarse, n_fine=args.n_fine)

    if not best_response:
        best_response = 50_000
        print(f'  No optimal found, using default: {best_response:,}')
    else:
        print(f'  Best response_chunk: {best_response:,}')

    if resp_coarse:
        vals = sorted(resp_coarse.keys())
        plot_chunk_timing(vals, [(resp_coarse[v], 0) for v in vals],
                          'response_chunk_size', best_response, tag=tag)

    # Re-align total_pad to response_chunk if needed
    if total_pad % best_response != 0:
        total_pad = round_up_to_multiple(total_pad, best_response)
        print(f'  Re-aligned total_pad to {total_pad:,}')

    # Phase 2: hits_chunk (track_hits ON, uses real max_keys)
    hits_divs = divisors_in_range(total_pad, 1000, 25_000)
    best_hits = hits_divs[-1] if hits_divs else best_response
    if not args.skip_hits:
        print('\n  Phase 2: hits_chunk_size (track_hits ON)')
        found, hits_coarse, _ = auto_search(
            detector_config, bench_file, bench_event, total_pad,
            'hits_chunk', args.lo, args.hi,
            include_track_hits=True, fixed_response_chunk=best_response,
            max_keys=max_keys, bucketed=args.bucketed,
            n_coarse=args.n_coarse, n_fine=args.n_fine)
        if found:
            best_hits = found
            print(f'  Best hits_chunk: {best_hits:,}')
        if hits_coarse:
            vals = sorted(hits_coarse.keys())
            plot_chunk_timing(vals, [(hits_coarse[v], 0) for v in vals],
                              'hits_chunk_size', best_hits, tag=tag)

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
    print(f'    python3 production/run_batch.py --data <file.h5> '
          f'--config {args.config} --production-config {args.output}')
    print()


if __name__ == '__main__':
    main()
