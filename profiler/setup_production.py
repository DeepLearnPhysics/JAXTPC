"""
One-shot production setup: scan data, benchmark sim, save config.

Steps:
  1. Combined data scan (one pass) → total_pad, maxg, max_keys
  2. (optional) Probe max_buckets for bucketed accumulation
  3. Find optimal response_chunk and hits_chunk (sim benchmark)
  4. Benchmark time(maxg) → fit linear cost model → compute maxg_medium
  5. (optional) Threshold analysis
  6. Save everything to a production config YAML

Accepts multiple --data files or a directory. Step 1 scans everything;
steps 3-4 use a single event for benchmarking.

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


def _pick_bench_event(bench_file):
    """A moderate-density (~p40) event so the small maxg test points fit it
    without clipping (clipping would distort the time(maxg) slope)."""
    import h5py
    with h5py.File(bench_file, 'r') as f:
        lv = f['pstep/lar_vol']
        lens = np.array([lv[i].shape[0] for i in range(lv.shape[0])])
    return int(np.argsort(lens)[int(0.4 * len(lens))])


def benchmark_maxg_medium(detector_config, bench_file, *, total_pad, max_keys,
                          maxg, response_chunk, hits_chunk,
                          inter_thresh=1.0, ng, n_repeats=5,
                          group_size=5, gap_threshold_mm=5.0):
    """Time the sim at several maxg values on one moderate event, then minimize
    the amortized tiered cost over the n_groups CDF:

        amortized(m) = F(m)·time(m) + (1-F(m))·time(maxg_high)

    The time(maxg) slope is the O(maxg × cells) box reduction (event- and
    padding-independent given chunking), so one moderate event suffices.
    Returns (maxg_medium|None, mg_arr, t_arr, time_high).
    """
    import gc
    import time as _time
    from tools.simulation import DetectorSimulator
    from tools.config import create_track_hits_config
    from tools.loader import load_event

    bench_event = _pick_bench_event(bench_file)

    test_pctiles = [50, 75, 90, 99]
    maxg_test_values = {round_up_to_multiple(int(np.percentile(ng, p)), 10_000)
                        for p in test_pctiles}
    maxg_test_values.add(maxg)

    # Keep only maxg values the bench event actually fits (else the box clips).
    _check_tc = create_track_hits_config(
        max_keys=max_keys, hits_chunk_size=hits_chunk, inter_thresh=inter_thresh,
        box_enabled=True, maxg=maxg)
    _check_sim = DetectorSimulator(
        detector_config, track_config=_check_tc, total_pad=total_pad,
        response_chunk_size=response_chunk, include_track_hits=True,
        group_size=group_size, gap_threshold_mm=gap_threshold_mm)
    _check_dep = load_event(bench_file, _check_sim.config, event_idx=bench_event)
    bench_max_gid = max(int(v.group_ids.max()) for v in _check_dep.volumes)
    del _check_sim
    print(f'  Benchmark event {bench_event}: max_group_id = {bench_max_gid:,}')

    maxg_test_values = sorted(m for m in maxg_test_values if m > bench_max_gid)
    if maxg not in maxg_test_values:
        maxg_test_values.append(maxg)
        maxg_test_values.sort()
    print(f'  Testing maxg values: {maxg_test_values}')

    maxg_times = {}
    for mg in maxg_test_values:
        tc = create_track_hits_config(
            max_keys=max_keys, hits_chunk_size=hits_chunk, inter_thresh=inter_thresh,
            box_enabled=True, maxg=mg)
        sim = DetectorSimulator(
            detector_config, track_config=tc, total_pad=total_pad,
            response_chunk_size=response_chunk, include_track_hits=True,
            group_size=group_size, gap_threshold_mm=gap_threshold_mm)
        sim.warm_up()
        bench_dep = load_event(bench_file, sim.config, event_idx=bench_event)
        out = sim.process_event(bench_dep, key=jax.random.PRNGKey(0))
        jax.tree.map(lambda x: x.block_until_ready()
                     if hasattr(x, 'block_until_ready') else None, out[0])
        times = []
        for r in range(n_repeats):
            t0 = _time.time()
            out = sim.process_event(bench_dep, key=jax.random.PRNGKey(r + 100))
            jax.tree.map(lambda x: x.block_until_ready()
                         if hasattr(x, 'block_until_ready') else None, out[0])
            times.append(_time.time() - t0)
        maxg_times[mg] = float(np.mean(times))
        print(f'    maxg={mg:>9,}  mean={maxg_times[mg]:.3f}s')
        del sim, bench_dep
        jax.clear_caches()
        gc.collect()

    print('\n  n_groups CDF:')
    for p in [50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        print(f'    p{p:<6}= {int(np.percentile(ng, p)):>9,}')

    mg_arr = np.array(sorted(maxg_times.keys()), float)
    t_arr = np.array([maxg_times[int(m)] for m in mg_arr], float)
    time_high = maxg_times[maxg]
    maxg_medium = None
    if len(mg_arr) >= 2:
        best_amort = time_high
        for mc in np.arange(mg_arr[0], maxg, 10_000):
            mc = int(mc)
            t_mc = float(np.interp(mc, mg_arr, t_arr))
            frac = float(np.mean(ng < mc))
            amort = frac * t_mc + (1 - frac) * time_high
            if amort < best_amort:
                best_amort = amort
                maxg_medium = mc
        if maxg_medium is not None:
            maxg_medium = round_up_to_multiple(maxg_medium, 10_000)
            frac_m = float(np.mean(ng < maxg_medium))
            t_med = float(np.interp(maxg_medium, mg_arr, t_arr))
            final_amort = frac_m * t_med + (1 - frac_m) * time_high
            print(f'\n  Optimal maxg_medium = {maxg_medium:,}')
            print(f'    {frac_m*100:.1f}% medium ({t_med:.3f}s), '
                  f'{(1-frac_m)*100:.1f}% high ({time_high:.3f}s)')
            print(f'    amortized = {final_amort:.3f}s  '
                  f'(saving {time_high - final_amort:.3f}s/event vs single-tier)')
        else:
            print('\n  No beneficial maxg_medium found (single tier is optimal)')
    else:
        print('\n  Not enough benchmark points for maxg_medium')
    return maxg_medium, mg_arr, t_arr, time_high


def _remake_maxg_medium(args, h5_files):
    """Standalone: re-derive ONLY maxg_medium for an existing production config,
    keeping its chunks / max_keys / maxg / box. Geometry-only n_groups scan +
    benchmark_maxg_medium. Updates the config in place."""
    import yaml as _yaml
    from tools.geometry import generate_detector
    from profiler.find_optimal_maxg import find_optimal_maxg
    from profiler.production_config import update_config

    cfg = _yaml.safe_load(open(args.output))
    det_path = cfg['detector_config']
    detector_config = generate_detector(det_path)

    print('=' * 70)
    print(' Re-derive maxg_medium (standalone)')
    print('=' * 70)
    print(f'  Config:  {args.output}  (maxg={cfg["maxg"]:,})')
    print(f'  n_groups scan: {len(h5_files)} file(s), geometry-only')
    _, info = find_optimal_maxg(h5_files, det_path, group_size=args.group_size,
                                n_workers=args.workers, dim_files=1)
    ng = info['n_groups']

    mm, *_ = benchmark_maxg_medium(
        detector_config, h5_files[0], total_pad=cfg['total_pad'],
        max_keys=cfg['max_keys'], maxg=cfg['maxg'],
        response_chunk=cfg['response_chunk'], hits_chunk=cfg['hits_chunk'],
        inter_thresh=float(cfg.get('inter_thresh', 1.0)), ng=ng,
        group_size=args.group_size, gap_threshold_mm=args.gap_threshold)
    if mm is not None:
        update_config(args.output, {'maxg_medium': mm}, detector_config_path=det_path)
        print(f'\n  Updated maxg_medium = {mm:,} → {args.output}')
    else:
        print('\n  No beneficial split — maxg_medium left unchanged')


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
    parser.add_argument('--headroom', type=float, default=1.3,
                        help='Multiply observed max_keys by this (default: 1.3)')
    parser.add_argument('--cstar', type=float, default=None,
                        help='Threshold multiplier for the charge-aware max_keys '
                             'estimate (c* x inter_thresh). Default by readout: '
                             'pixel 2.5 (-> ~1.0x actual), wire 1.0.')
    parser.add_argument('--divisor', type=float, default=None,
                        help='Per-readout overlap divisor on the max_keys estimate. '
                             'Default by readout: pixel 1.0, wire 3.79 (wire overlap '
                             'is structural; the threshold cannot remove it).')
    parser.add_argument('--n-coarse', type=int, default=3)
    parser.add_argument('--n-fine', type=int, default=10)
    parser.add_argument('--lo', type=int, default=1_000)
    parser.add_argument('--hi', type=int, default=100_000)
    parser.add_argument('--skip-hits', action='store_true',
                        help='Skip hits_chunk optimization')
    parser.add_argument('--group-size', type=int, default=5)
    parser.add_argument('--gap-threshold', type=float, default=5.0)
    parser.add_argument('--maxg-dim-files', type=int, default=20,
                        help='Files to full-group for box dims (MAXG uses fast-P2 '
                             'over all files). Box-dim extents are stable, so a '
                             'subset suffices; avoids the group sort on every file '
                             '(default: 20)')
    parser.add_argument('--tag', default=None,
                        help='Tag for figure filenames (default: config name)')
    parser.add_argument('--skip-chunks', action='store_true',
                        help='Skip chunk optimization, use --default-chunk value')
    parser.add_argument('--default-chunk', type=int, default=5_000,
                        help='Default chunk size when skipping optimization (default: 5000)')
    parser.add_argument('--run-thresholds', action='store_true',
                        help='Run threshold analysis to calibrate corr_threshold and threshold_adc')
    parser.add_argument('--threshold-events', type=int, default=3,
                        help='Events to use for threshold analysis (default: 3)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker processes for Step 1+2 file scanning '
                             '(default: 1 = serial; set ~= CPU count for speed)')
    parser.add_argument('--remake-maxg-medium', action='store_true',
                        help='Standalone: re-derive ONLY maxg_medium for the '
                             'existing -o/--output config (keeps chunks/max_keys/'
                             'maxg/box). Use when the data changed but the geometry '
                             'and capacities did not.')

    args = parser.parse_args()

    h5_files = _resolve_data_paths(args.data)
    if not h5_files:
        print('No HDF5 files found!')
        return

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f'config/production_{base}.yaml'

    # Standalone maxg_medium re-derivation (skips the full pipeline).
    if args.remake_maxg_medium:
        if not os.path.exists(args.output):
            print(f'--remake-maxg-medium needs an existing config at {args.output}')
            return
        _remake_maxg_medium(args, h5_files)
        return

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

    # ── Step 1: Combined scan — total_pad + MAXG + box dims + max_keys ──
    # One pass over all files yields deposit counts (→ total_pad), n_groups
    # distribution (→ maxg), per-group footprint extents (→ box dims), and
    # per-event key estimates (→ max_keys). Previously three separate scans.

    print('\n' + '─' * 70)
    print(' Step 1: Scanning events (total_pad + MAXG + box dims + max_keys)')
    print('─' * 70)

    from tools.geometry import generate_detector
    from tools.config import create_sim_config
    from profiler.find_optimal_maxg import find_optimal_maxg
    from profiler.estimate_max_keys import (build_wire_element_table,
        build_pixel_element_table, build_pixel_value_table,
        build_wire_value_table, build_charge_model)
    import yaml as _yaml

    detector_config = generate_detector(args.config)
    sc = create_sim_config(detector_config, total_pad=2_000_000)

    element_tables = {}
    for v, vol_geom in enumerate(sc.volumes):
        if vol_geom.readout_type == 'pixel':
            element_tables[v] = build_pixel_element_table(sc, vol_geom)
        else:
            element_tables[v] = build_wire_element_table(vol_geom.diffusion)
    for v, tbl in element_tables.items():
        print(f'  Vol {v} ({sc.volumes[v].readout_type}) element table: {tbl.tolist()}')

    # Charge-aware max_keys: per-deposit footprint at threshold c* x inter_thresh,
    # summed, then divided by a per-readout overlap factor. Both are calibrated so
    # the estimate ~= the actual box key count (verify with compare_max_keys):
    #   pixel: c*=2.5, divisor=1   (a higher threshold removes the kernel-tail overlap)
    #   wire : c*=1,   divisor=3.79 (overlap is structural from the 1-D wire
    #                                projection; the threshold plateaus, so use a factor)
    # These defaults are calibrated on the doraemon dataset -- re-check for very
    # different data. Value tables carry the kernel's cell-value distribution; the
    # charge model gives per-deposit intensity (recombination x drift attenuation).
    readout = sc.volumes[0].readout_type
    cstar = args.cstar if args.cstar is not None else (2.5 if readout == 'pixel' else 1.0)
    divisor = (args.divisor if args.divisor is not None
               else (1.0 if readout == 'pixel' else 3.79))
    charge_model = build_charge_model(_yaml.safe_load(open(args.config)))
    value_tables = {}
    for v, vol_geom in enumerate(sc.volumes):
        if vol_geom.readout_type == 'pixel':
            value_tables[v] = build_pixel_value_table(sc, vol_geom)
        else:
            value_tables[v] = build_wire_value_table(vol_geom.diffusion)
    key_thresh = cstar * 1.0  # c* x box inter_thresh (production inter_thresh=1.0)
    print(f'  max_keys: charge-aware ({readout}), c*={cstar}, divisor={divisor} '
          f'(threshold={key_thresh})')

    maxg, maxg_info = find_optimal_maxg(
        h5_files, args.config, group_size=args.group_size,
        events_per_file=args.events_pad, n_workers=args.workers,
        dim_files=args.maxg_dim_files, element_tables=element_tables,
        value_tables=value_tables, charge_model=charge_model,
        key_thresh=key_thresh)

    total_scanned = maxg_info['n_events']
    dep = maxg_info['deposits']
    ng = maxg_info['n_groups']
    keys = maxg_info['keys']

    print(f'  Scanned {total_scanned} events across {maxg_info["n_files"]} file(s) '
          f'in {maxg_info["elapsed_s"]:.0f}s')

    # Deposit distribution → total_pad
    dep_pcts = np.percentile(dep, [50, 99.9, 100])
    print(f'\n  Deposits per volume: P50={int(dep_pcts[0]):,}, '
          f'P99.9={int(dep_pcts[1]):,}, Max={int(dep_pcts[2]):,}')

    raw_pad = int(dep_pcts[1]) if args.use_p999 else int(dep_pcts[2])
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

    # MAXG distribution
    print(f'\n  MAXG (group count, readout-independent):')
    for p in [50, 90, 99, 99.9, 99.95, 100]:
        print(f'    p{p:<6}= {int(np.percentile(ng, p)):>9,}')
    print(f'  Suggested MAXG = {maxg:,}')

    # max_keys from combined scan (apply the per-readout overlap divisor)
    est_max = int(keys.max()) if len(keys) > 0 else 0
    max_observed_keys = int(est_max / divisor)
    max_keys = int(math.ceil(max_observed_keys * args.headroom / 100_000) * 100_000)
    print(f'\n  max_keys: est max = {est_max:,} / divisor {divisor} = '
          f'{max_observed_keys:,}, x {args.headroom} headroom → {max_keys:,}')

    max_buckets = 1000

    # ── Step 2: Find optimal chunks ─────────────────────────────────────

    if args.skip_chunks:
        best_response = args.default_chunk
        best_hits = args.default_chunk
        if total_pad % best_response != 0:
            total_pad = round_up_to_multiple(total_pad, best_response)
        if total_pad % best_hits != 0:
            from profiler.find_optimal_chunks import divisors_in_range as _div
            hits_divs = _div(total_pad, 1000, best_hits)
            best_hits = hits_divs[-1] if hits_divs else best_response
        print(f'\n  Skipping chunk optimization — using defaults:')
        print(f'    response_chunk = {best_response:,}')
        print(f'    hits_chunk     = {best_hits:,}')
    else:

        print('\n' + '─' * 70)
        print(' Step 2: Finding optimal chunk sizes')
        print('─' * 70)
        print(f'  total_pad: {total_pad:,}, max_keys: {max_keys:,}, maxg: {maxg:,}')

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
            n_coarse=args.n_coarse, n_fine=args.n_fine,
            group_size=args.group_size, gap_threshold_mm=args.gap_threshold)

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

        # Phase 2: hits_chunk (track_hits ON, uses maxg_high)
        hits_divs = divisors_in_range(total_pad, 1000, 25_000)
        best_hits = hits_divs[-1] if hits_divs else best_response
        if not args.skip_hits:
            print('\n  Phase 2: hits_chunk_size (track_hits ON)')
            found, hits_coarse, _ = auto_search(
                detector_config, bench_file, bench_event, total_pad,
                'hits_chunk', args.lo, args.hi,
                include_track_hits=True, fixed_response_chunk=best_response,
                max_keys=max_keys, bucketed=args.bucketed,
                n_coarse=args.n_coarse, n_fine=args.n_fine,
                group_size=args.group_size, gap_threshold_mm=args.gap_threshold)
            if found:
                best_hits = found
                print(f'  Best hits_chunk: {best_hits:,}')
            if hits_coarse:
                vals = sorted(hits_coarse.keys())
                plot_chunk_timing(vals, [(hits_coarse[v], 0) for v in vals],
                                  'hits_chunk_size', best_hits, tag=tag)

    # ── Step 3: Benchmark time(maxg) → maxg_medium ───────────────────
    # Sweep several maxg values, measure sim time for each, then combine with
    # the n_groups CDF for the optimal tiered split (see benchmark_maxg_medium).

    print('\n' + '─' * 70)
    print(' Step 3: Benchmarking time(maxg) → maxg_medium')
    print('─' * 70)

    maxg_medium, mg_arr, t_arr, time_high = benchmark_maxg_medium(
        detector_config, h5_files[0], total_pad=total_pad, max_keys=max_keys,
        maxg=maxg, response_chunk=best_response,
        hits_chunk=best_hits, inter_thresh=1.0, ng=ng,
        group_size=args.group_size, gap_threshold_mm=args.gap_threshold)

    # ── Plots for Step 1 + Step 3 ──────────────────────────────────────
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'profiler', 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # 1) n_groups distribution + maxg lines
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ng, bins=80, color='steelblue', alpha=0.8, label='n_groups')
        ax.axvline(maxg, color='red', ls='--', lw=1.5, label=f'maxg={maxg:,}')
        if maxg_medium is not None:
            ax.axvline(maxg_medium, color='orange', ls='--', lw=1.5,
                       label=f'maxg_medium={maxg_medium:,}')
        ax.set_xlabel('n_groups per event-volume')
        ax.set_ylabel('count')
        ax.set_title(f'n_groups distribution ({maxg_info["readout"]}, {len(ng):,} event-volumes)')
        ax.legend()
        fig.tight_layout()
        p1 = os.path.join(fig_dir, f'maxg_distribution_{tag}.png')
        fig.savefig(p1, dpi=120); plt.close(fig)
        print(f'  Saved: {p1}')

        # 2) time(maxg) cost curve
        if len(mg_arr) >= 2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(mg_arr / 1e3, t_arr, 'o-', color='steelblue', markersize=8)
            if maxg_medium is not None:
                t_med_interp = float(np.interp(maxg_medium, mg_arr, t_arr))
                ax.axvline(maxg_medium / 1e3, color='orange', ls='--', lw=1.5,
                           label=f'medium={maxg_medium:,}')
                ax.plot(maxg_medium / 1e3, t_med_interp, 's', color='orange',
                        markersize=10, zorder=5)
            ax.set_xlabel('maxg (thousands)')
            ax.set_ylabel('sim time (s)')
            ax.set_title(f'Sim time vs maxg ({maxg_info["readout"]})')
            ax.legend()
            fig.tight_layout()
            p2 = os.path.join(fig_dir, f'maxg_cost_{tag}.png')
            fig.savefig(p2, dpi=120); plt.close(fig)
            print(f'  Saved: {p2}')

        # 3) amortized cost vs maxg_medium candidate
        if len(mg_arr) >= 2:
            m_sweep = np.arange(mg_arr[0], maxg, 5_000)
            amort_sweep = []
            for mc in m_sweep:
                t_mc = float(np.interp(mc, mg_arr, t_arr))
                frac = float(np.mean(ng < mc))
                amort_sweep.append(frac * t_mc + (1 - frac) * time_high)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(m_sweep / 1e3, amort_sweep, '-', color='steelblue', lw=2)
            ax.axhline(time_high, color='gray', ls=':', lw=1,
                       label=f'single-tier ({time_high:.3f}s)')
            if maxg_medium is not None:
                ax.axvline(maxg_medium / 1e3, color='orange', ls='--', lw=1.5,
                           label=f'optimal={maxg_medium:,}')
            ax.set_xlabel('maxg_medium (thousands)')
            ax.set_ylabel('amortized sim time (s)')
            ax.set_title(f'Tiered routing optimization ({maxg_info["readout"]})')
            ax.legend()
            fig.tight_layout()
            p3 = os.path.join(fig_dir, f'maxg_medium_opt_{tag}.png')
            fig.savefig(p3, dpi=120); plt.close(fig)
            print(f'  Saved: {p3}')

        # 4) deposits distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dep, bins=80, color='steelblue', alpha=0.8)
        ax.axvline(total_pad, color='red', ls='--', lw=1.5,
                   label=f'total_pad={total_pad:,}')
        ax.set_xlabel('deposits per volume')
        ax.set_ylabel('count')
        ax.set_title(f'Deposit distribution ({len(dep):,} event-volumes)')
        ax.legend()
        fig.tight_layout()
        p4 = os.path.join(fig_dir, f'deposit_distribution_{tag}.png')
        fig.savefig(p4, dpi=120); plt.close(fig)
        print(f'  Saved: {p4}')

    except Exception as e:
        print(f'  (plots skipped: {e})')

    # ── Step 4: Threshold analysis (optional) ─────────────────────────

    chosen_corr = 25.0
    chosen_adc = 2.0

    if args.run_thresholds:
        print('\n' + '─' * 70)
        print(' Step 4: Threshold analysis')
        print('─' * 70)

        import gc
        from tools.simulation import DetectorSimulator
        from tools.config import create_track_hits_config
        from tools.loader import load_event
        from profiler.timing import sync_result
        from profiler.threshold_analysis import (
            collect_corr_values, collect_signal_values, auto_thresholds,
            analyze_corr_threshold, analyze_adc_threshold,
            print_corr_results, print_adc_results,
        )
        from profiler.plots import plot_corr_threshold, plot_adc_threshold

        jax.clear_caches()
        gc.collect()

        # Box dims are derived analytically by the simulator from the group
        # definition (group_size/gap_threshold) — same as production — so the
        # thresholds are calibrated on the exact distribution production produces.
        track_config = create_track_hits_config(
            max_keys=max_keys, hits_chunk_size=best_hits,
            inter_thresh=1.0, box_enabled=True, maxg=maxg)
        thresh_sim = DetectorSimulator(
            detector_config,
            total_pad=total_pad,
            response_chunk_size=best_response,
            include_track_hits=True,
            track_config=track_config,
            group_size=args.group_size,
            gap_threshold_mm=args.gap_threshold,
        )
        thresh_sim.warm_up()

        bench_file = h5_files[0]
        key = jax.random.PRNGKey(42)
        n_thresh = min(args.threshold_events, total_scanned)

        # Probe first event for auto thresholds
        key, subkey = jax.random.split(key)
        probe_dep = load_event(bench_file, thresh_sim.config, event_idx=0)
        probe_resp, probe_hits, _ = thresh_sim.process_event(probe_dep, key=subkey)
        sync_result(probe_resp)

        corr_vals = collect_corr_values(probe_hits)
        corr_thresholds = auto_thresholds(corr_vals)
        sig_vals = collect_signal_values(probe_resp, thresh_sim.config)
        adc_thresholds = auto_thresholds(sig_vals)

        print(f'  Auto corr thresholds: {[f"{v:.2f}" for v in corr_thresholds]}')
        print(f'  Auto ADC thresholds:  {[f"{v:.2f}" for v in adc_thresholds]}')
        del probe_resp, probe_hits, probe_dep

        # Accumulate across events
        all_corr = None
        all_adc = None
        key = jax.random.PRNGKey(42)

        for i in range(n_thresh):
            key, subkey = jax.random.split(key)
            deposits = load_event(bench_file, thresh_sim.config, event_idx=i)
            n_deps = sum(v.n_actual for v in deposits.volumes)
            print(f'  Event {i}: {n_deps:,} deposits')

            response_signals, track_hits_raw, _ = thresh_sim.process_event(
                deposits, key=subkey)
            sync_result(response_signals)

            corr_results = analyze_corr_threshold(
                track_hits_raw, thresh_sim.config, corr_thresholds)
            if all_corr is None:
                all_corr = [{k: 0.0 for k in r} for r in corr_results]
                for j, r in enumerate(corr_results):
                    all_corr[j]['threshold'] = r['threshold']
            for j, r in enumerate(corr_results):
                all_corr[j]['total_charge'] += r['total_charge']
                all_corr[j]['kept_charge'] += r['kept_charge']
                all_corr[j]['total_entries'] += r['total_entries']
                all_corr[j]['kept_entries'] += r['kept_entries']

            adc_results = analyze_adc_threshold(
                response_signals, thresh_sim.config, adc_thresholds)
            if all_adc is None:
                all_adc = [{k: 0.0 for k in r} for r in adc_results]
                for j, r in enumerate(adc_results):
                    all_adc[j]['threshold'] = r['threshold']
            for j, r in enumerate(adc_results):
                all_adc[j]['total_signal'] += r['total_signal']
                all_adc[j]['kept_signal'] += r['kept_signal']
                all_adc[j]['total_bins'] += r['total_bins']
                all_adc[j]['kept_bins'] += r['kept_bins']

        del thresh_sim

        # Recompute fractions and print
        for r in all_corr:
            tc = r['total_charge']
            r['charge_lost_frac'] = (tc - r['kept_charge']) / tc if tc > 0 else 0
            te = r['total_entries']
            r['entries_dropped_frac'] = 1.0 - (r['kept_entries'] / te) if te > 0 else 0
        for r in all_adc:
            ts = r['total_signal']
            r['signal_lost_frac'] = (ts - r['kept_signal']) / ts if ts > 0 else 0
            tb = r['total_bins']
            r['bins_dropped_frac'] = 1.0 - (r['kept_bins'] / tb) if tb > 0 else 0

        print(f'\n  Correspondence Threshold')
        print_corr_results(all_corr)
        print(f'\n  Signal Threshold')
        print_adc_results(all_adc)

        # Pick thresholds: highest that keeps >= 99% charge/signal
        for r in reversed(all_corr):
            if r['charge_lost_frac'] <= 0.01:
                chosen_corr = r['threshold']
                break
        for r in reversed(all_adc):
            if r['signal_lost_frac'] <= 0.01:
                chosen_adc = r['threshold']
                break

        print(f'\n  Chosen corr_threshold: {chosen_corr:.3g} (keeps >= 99% charge)')
        print(f'  Chosen threshold_adc:  {chosen_adc:.3g} (keeps >= 99% signal)')

        # Plots
        corr_kept = [1.0 - r['charge_lost_frac'] for r in all_corr]
        adc_kept = [1.0 - r['signal_lost_frac'] for r in all_adc]
        plot_corr_threshold([r['threshold'] for r in all_corr], corr_kept, tag=tag)
        plot_adc_threshold([r['threshold'] for r in all_adc], adc_kept, tag=tag)

    # ── Save ────────────────────────────────────────────────────────────

    config_values = {
        'total_pad': total_pad,
        'response_chunk': best_response,
        'hits_chunk': best_hits,
        'max_keys': max_keys,
        'maxg': maxg,
        'inter_thresh': 1.0,
        'threshold_adc': chosen_adc,
        'corr_threshold': chosen_corr,
        'max_buckets': max_buckets,
    }
    if maxg_medium is not None:
        config_values['maxg_medium'] = maxg_medium
    # Box (group-as-bucket) per-group dims are NOT stored: the simulator derives
    # them analytically from the group definition + geometry at construction
    # (tools.track_hits.compute_box_dims), including for the timing sims above.

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
