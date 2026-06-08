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


def select_candidates(total, lo, hi, max_candidates=7):
    """Pick geometrically-spaced divisors of total in [lo, hi]."""
    all_divs = divisors_in_range(total, lo, hi)
    if len(all_divs) <= max_candidates:
        return all_divs
    indices = np.linspace(0, len(all_divs) - 1, max_candidates).astype(int)
    return [all_divs[i] for i in np.unique(indices)]


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
              max_keys, bucketed, n_timed, label='',
              maxg=200_000, box_kw=None, inter_thresh=1.0, deposits=None):
    """Benchmark a single chunk configuration. Returns (mean_ms, std_ms, times).

    Pass maxg + box_kw (e.g. {'box_bpy':8,...} or {'box_bw':12,'box_btw':27})
    to size the track-hits box exactly as production does — the default box
    (6/6/88 or 16/96, maxg 200k) mistimes the hits scatter-add otherwise. Pass
    a preloaded `deposits` to avoid reloading the (slow) event per candidate.
    """
    jax.clear_caches()
    gc.collect()

    track_config = None
    if include_track_hits:
        track_config = create_track_hits_config(
            max_keys=max_keys, hits_chunk_size=hits_chunk,
            inter_thresh=inter_thresh, box_enabled=True, maxg=maxg,
            **(box_kw or {}))

    sim = DetectorSimulator(
        detector_config,
        total_pad=total_pad,
        response_chunk_size=response_chunk,
        include_track_hits=include_track_hits,
        track_config=track_config,
        use_bucketed=bucketed,
    )
    sim.warm_up()

    if deposits is None:
        deposits = load_event(data_path, sim.config, event_idx=event_idx)
    n_deps = sum(int(v.n_actual) for v in deposits.volumes)

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
                n_coarse=3, n_fine=10,
                maxg=200_000, box_kw=None, inter_thresh=1.0, deposits=None):
    """Two-pass search over divisors of total_pad in [lo, hi]."""
    candidates = select_candidates(total_pad, lo, hi)
    if not candidates:
        print(f'  No divisors of {total_pad:,} in [{lo:,}, {hi:,}]!')
        return None, {}, {}

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
            n_coarse, label=f'{val:>8,}',
            maxg=maxg, box_kw=box_kw, inter_thresh=inter_thresh, deposits=deposits)
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
            n_fine, label=f'{val:>8,}',
            maxg=maxg, box_kw=box_kw, inter_thresh=inter_thresh, deposits=deposits)
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

    return best, coarse, fine


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal chunk sizes for JAXTPC simulation')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', default=None,
                        help='Detector geometry YAML (or taken from --prod-config)')
    parser.add_argument('--prod-config', default=None,
                        help='Production config YAML. Reads the REAL total_pad / '
                             'max_keys / maxg / box dims so the hits scatter-add is '
                             'timed at production box size (the default 6/6/88 or '
                             '16/96 box mistimes it), and saves chunks back to it.')
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
    parser.add_argument('--skip-response', action='store_true',
                        help='Skip response_chunk optimization (auto for pixel)')
    parser.add_argument('--save-config', default=None,
                        help='Save results to production config YAML')

    args = parser.parse_args()

    # Resolve capacities/box from the production config when given, so the box
    # is sized exactly as production runs it.
    total_pad, max_keys = args.total_pad, args.max_keys
    maxg, inter_thresh, box_kw, readout = 200_000, 1.0, None, None
    det_path, save_to, prev_rc = args.config, args.save_config, 50_000
    if args.prod_config:
        import yaml
        pc = yaml.safe_load(open(args.prod_config))
        det_path = det_path or pc.get('detector_config')
        total_pad = int(pc.get('total_pad', total_pad))
        max_keys = int(pc.get('max_keys', max_keys))
        maxg = int(pc.get('maxg', maxg))
        inter_thresh = float(pc.get('inter_thresh', 1.0))
        prev_rc = int(pc.get('response_chunk', prev_rc))
        if 'box_bpy' in pc:
            readout = 'pixel'
            box_kw = dict(box_bpy=pc['box_bpy'], box_bpz=pc['box_bpz'], box_bt=pc['box_bt'])
        elif 'box_bw' in pc:
            readout = 'wire'
            box_kw = dict(box_bw=pc['box_bw'], box_btw=pc['box_btw'])
        save_to = save_to or args.prod_config
    if det_path is None:
        parser.error('need --config or --prod-config with a detector_config')

    detector_config = generate_detector(det_path)
    bucketed = args.bucketed
    # Pixel is a single track_hits pass: response_chunk is never used -> skip it.
    skip_response = args.skip_response or (readout == 'pixel')

    print('=' * 70)
    print(' JAXTPC — Find Optimal Chunk Sizes')
    print('=' * 70)
    print(f'  Data:      {args.data}')
    print(f'  Config:    {det_path}  (readout={readout or "?"})')
    print(f'  total_pad: {total_pad:,}  max_keys: {max_keys:,}  maxg: {maxg:,}')
    print(f'  box:       {box_kw or "default"}  inter_thresh: {inter_thresh}')
    print(f'  Range:     [{args.lo:,}, {args.hi:,}]   Device: {jax.devices()[0]}')

    # Load the (slow) event ONCE and reuse it across every candidate.
    from tools.config import create_sim_config
    load_cfg = create_sim_config(detector_config, total_pad=total_pad,
                                 use_bucketed=bucketed)
    deposits = load_event(args.data, load_cfg, event_idx=args.event)
    n_deps = sum(int(v.n_actual) for v in deposits.volumes)
    print(f'  Event {args.event}: {n_deps:,} deposits (loaded once)')

    # Phase 1: response_chunk_size (track_hits OFF)
    best_response, resp_coarse = prev_rc, {}
    if not skip_response:
        print('\n  Phase 1: response_chunk_size (track_hits OFF)')
        best_response, resp_coarse, _ = auto_search(
            detector_config, args.data, args.event, total_pad,
            'response_chunk', args.lo, args.hi,
            include_track_hits=False, fixed_response_chunk=50_000,
            max_keys=max_keys, bucketed=bucketed,
            n_coarse=args.n_coarse, n_fine=args.n_fine,
            maxg=maxg, box_kw=box_kw, inter_thresh=inter_thresh, deposits=deposits)
        if best_response:
            print(f'\n  Best response_chunk_size: {best_response:,}')
    else:
        print(f'\n  Phase 1 skipped (pixel single-pass: response_chunk unused, '
              f'kept at {best_response:,})')

    # Phase 2: hits_chunk_size (track_hits ON)
    best_hits, hits_coarse = None, {}
    if not args.skip_hits and best_response:
        print('\n  Phase 2: hits_chunk_size (track_hits ON)')
        best_hits, hits_coarse, _ = auto_search(
            detector_config, args.data, args.event, total_pad,
            'hits_chunk', args.lo, args.hi,
            include_track_hits=True, fixed_response_chunk=best_response,
            max_keys=max_keys, bucketed=bucketed,
            n_coarse=args.n_coarse, n_fine=args.n_fine,
            maxg=maxg, box_kw=box_kw, inter_thresh=inter_thresh, deposits=deposits)
        if best_hits:
            print(f'\n  Best hits_chunk_size: {best_hits:,}')

    # Summary
    print('\n' + '=' * 70)
    print('  RESULTS')
    print('=' * 70)
    if not skip_response:
        print(f'  --response-chunk {best_response}')
    if best_hits:
        print(f'  --hits-chunk {best_hits}')

    # Figures
    from profiler.plots import plot_chunk_timing
    if resp_coarse:
        vals = sorted(resp_coarse.keys())
        plot_chunk_timing(vals, [(resp_coarse[v], 0) for v in vals],
                          'response_chunk_size', best_response)
    if hits_coarse:
        vals = sorted(hits_coarse.keys())
        plot_chunk_timing(vals, [(hits_coarse[v], 0) for v in vals],
                          'hits_chunk_size', best_hits)

    if save_to:
        from profiler.production_config import update_config
        updates = {}
        if not skip_response and best_response:
            updates['response_chunk'] = best_response
        if best_hits:
            updates['hits_chunk'] = best_hits
        if updates:
            update_config(save_to, updates, detector_config_path=det_path)
            print(f'  Saved to {save_to}')

    print()


if __name__ == '__main__':
    main()
