"""
Dense chunk-size sweep with error bars.

Single-pass benchmark over divisors of total_pad with many iterations per
point, producing mean ± std timing curves. Useful for visualizing the
chunk-size timing landscape beyond the two-pass best-of-three search in
``find_optimal_chunks``.

Usage:
    python3 -m profiler.sweep_chunks --data file.h5 \\
        --config config/cubic_wireplane_config.yaml \\
        --production-config config/production_cubic_wireplane_doraemon.yaml \\
        --event 102 --tag doraemon --n-iter 12
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

from tools.geometry import generate_detector

from profiler.find_optimal_chunks import divisors_in_range, bench_one
from profiler.production_config import load_config
from profiler.plots import plot_chunk_timing


def sweep(detector_config, data_path, event_idx, total_pad,
          chunk_label, lo, hi, include_track_hits,
          fixed_response_chunk, max_keys, bucketed, n_iter):
    candidates = divisors_in_range(total_pad, lo, hi)
    if not candidates:
        print(f'  No divisors of {total_pad:,} in [{lo:,}, {hi:,}]!', flush=True)
        return {}
    print(f'\n  Candidates ({len(candidates)} in [{lo:,}, {hi:,}]): '
          f'{candidates}', flush=True)
    print(f'  n_iter = {n_iter} per point\n', flush=True)
    results = {}
    for val in candidates:
        if chunk_label == 'response_chunk':
            rc, hc = val, 25_000
        else:
            rc, hc = fixed_response_chunk, val
        mean, std, _ = bench_one(
            detector_config, data_path, event_idx, total_pad,
            rc, hc, include_track_hits, max_keys, bucketed,
            n_iter, label=f'{val:>8,}')
        results[val] = (mean, std)
    return results


def main():
    p = argparse.ArgumentParser(description='Dense chunk-size sweep')
    p.add_argument('--data', required=True, help='HDF5 file')
    p.add_argument('--config', required=True, help='Detector geometry YAML')
    p.add_argument('--production-config', required=True,
                   help='Production YAML providing total_pad and max_keys')
    p.add_argument('--event', type=int, default=0)
    p.add_argument('--n-iter', type=int, default=12,
                   help='Timed iterations per candidate (default: 12)')
    p.add_argument('--lo-response', type=int, default=5_000)
    p.add_argument('--hi-response', type=int, default=200_000)
    p.add_argument('--lo-hits', type=int, default=20_000)
    p.add_argument('--hi-hits', type=int, default=200_000)
    p.add_argument('--bucketed', action='store_true')
    p.add_argument('--max-keys', type=int, default=None,
                   help='Override max_keys from production config (e.g. for '
                        'sensitivity studies)')
    p.add_argument('--tag', default=None,
                   help='Suffix for figure filenames (default: <config>_sweep)')
    p.add_argument('--skip-response', action='store_true')
    p.add_argument('--skip-hits', action='store_true')
    args = p.parse_args()

    detector_config = generate_detector(args.config)
    prod = load_config(args.production_config)
    total_pad = int(prod['total_pad'])
    max_keys = int(args.max_keys) if args.max_keys is not None else int(prod['max_keys'])
    seed_response = int(prod['response_chunk'])
    tag_base = args.tag or os.path.splitext(os.path.basename(args.config))[0]
    tag = f'{tag_base}_sweep'

    print('=' * 70, flush=True)
    print(' JAXTPC — Dense Chunk Sweep', flush=True)
    print('=' * 70, flush=True)
    print(f'  Data:        {args.data}  (event {args.event})', flush=True)
    print(f'  total_pad:   {total_pad:,}', flush=True)
    print(f'  max_keys:    {max_keys:,}', flush=True)
    print(f'  seed response_chunk: {seed_response:,}', flush=True)
    print(f'  n_iter:      {args.n_iter}', flush=True)
    print(f'  Device:      {jax.devices()[0]}', flush=True)

    if not args.skip_response:
        print('\n  -- response_chunk sweep (track_hits OFF) --', flush=True)
        resp = sweep(
            detector_config, args.data, args.event, total_pad,
            'response_chunk', args.lo_response, args.hi_response,
            include_track_hits=False, fixed_response_chunk=seed_response,
            max_keys=max_keys, bucketed=args.bucketed, n_iter=args.n_iter)
        if resp:
            best = min(resp, key=lambda k: resp[k][0])
            m, s = resp[best]
            print(f'\n  Best response_chunk: {best:,}  ({m:.1f} ± {s:.1f} ms)',
                  flush=True)
            vals = sorted(resp.keys())
            plot_chunk_timing(
                vals, [resp[v] for v in vals],
                'response_chunk_size', best, tag=tag)

    if not args.skip_hits:
        print('\n  -- hits_chunk sweep (track_hits ON) --', flush=True)
        hits = sweep(
            detector_config, args.data, args.event, total_pad,
            'hits_chunk', args.lo_hits, args.hi_hits,
            include_track_hits=True, fixed_response_chunk=seed_response,
            max_keys=max_keys, bucketed=args.bucketed, n_iter=args.n_iter)
        if hits:
            best = min(hits, key=lambda k: hits[k][0])
            m, s = hits[best]
            print(f'\n  Best hits_chunk: {best:,}  ({m:.1f} ± {s:.1f} ms)',
                  flush=True)
            vals = sorted(hits.keys())
            plot_chunk_timing(
                vals, [hits[v] for v in vals],
                'hits_chunk_size', best, tag=tag)


if __name__ == '__main__':
    main()
