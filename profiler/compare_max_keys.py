"""
Compare the geometry-based max_keys ESTIMATE against the ACTUAL runtime
track-hits count, on the same raw edepsim events that production runs on.

Two modes:

  (default) sequential — walk events [event, event+events) of one file:
      ACTUAL   = final_count per (vol, plane) from the real simulator
      ESTIMATE = estimate_keys_for_event() per (vol, plane) from geometry

  --rank-validate — scan a whole dataset (parallel) to find the TOP-K events
      by estimated keys, then run the real simulator only on those tail events
      and compare actual final_count vs the estimate. This validates whether
      the estimate's heavy tail is real (so max_keys must cover it) or an
      overcount (so a smaller value is safe).
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import jax

from tools.geometry import generate_detector
from tools.config import create_sim_config, create_track_hits_config
from tools.simulation import DetectorSimulator
from tools.loader import load_event

from profiler.estimate_max_keys import (
    build_wire_element_table, build_pixel_element_table,
    estimate_keys_for_event,
)
from profiler.find_optimal_pad import (
    get_volume_ranges, count_deposits_per_volume,
)


def _build_element_tables(sim_config, num_s=32):
    tables = {}
    for v, vg in enumerate(sim_config.volumes):
        if vg.readout_type == 'pixel':
            tables[v] = build_pixel_element_table(sim_config, vg, num_s=num_s)
        else:
            tables[v] = build_wire_element_table(vg.diffusion, num_s=num_s)
    return tables


def _resolve_files(data_arg):
    paths = []
    for p in ([data_arg] if isinstance(data_arg, str) else data_arg):
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, '*.h5'))))
        else:
            paths.append(p)
    return paths


# ── ranking (parallel, estimate-only) ──────────────────────────────────────
_rank_state = {}


def _init_rank_worker(config_path, element_tables, group_size, gap, rank_by):
    dc = generate_detector(config_path)
    _rank_state['rank_by'] = rank_by
    if rank_by == 'estimate':
        _rank_state['sim_config'] = create_sim_config(dc)
        _rank_state['element_tables'] = element_tables
        _rank_state['group_size'] = group_size
        _rank_state['gap'] = gap
    else:  # deposits (segment count)
        _rank_state['volume_ranges'] = get_volume_ranges(dc)


def _rank_file(args):
    """Return list of (metric, fpath, idx) for one file.

    metric = max-over-volumes value (deposits or estimated keys), the
    quantity that drives the worst per-(vol,plane) max_keys.
    """
    fpath, max_events = args
    rank_by = _rank_state['rank_by']
    recs = []
    with h5py.File(fpath, 'r') as f:
        ds = f['pstep/lar_vol']
        n = ds.shape[0] if max_events is None else min(ds.shape[0], max_events)
        if rank_by == 'estimate':
            sc = _rank_state['sim_config']
            et = _rank_state['element_tables']
            gs = _rank_state['group_size']
            gap = _rank_state['gap']
            for i in range(n):
                keys, _ = estimate_keys_for_event(
                    ds[i], sc, et, group_size=gs, gap_threshold_mm=gap)
                recs.append((max(keys.values()) if keys else 0, fpath, i))
        else:
            vr = _rank_state['volume_ranges']
            for i in range(n):
                row = ds[i]
                pos = np.column_stack([row['x'].astype(np.float32),
                                       row['y'].astype(np.float32),
                                       row['z'].astype(np.float32)])
                recs.append((max(count_deposits_per_volume(pos, vr)), fpath, i))
    return recs


def rank_events(h5_files, config_path, element_tables, group_size, gap,
                n_workers, events_per_file, top_k, rank_by):
    args_list = [(fp, events_per_file) for fp in h5_files]
    all_recs = []
    if n_workers <= 1 or len(h5_files) <= 1:
        _init_rank_worker(config_path, element_tables, group_size, gap, rank_by)
        for done, a in enumerate(args_list, 1):
            all_recs.extend(_rank_file(a))
            if done % 25 == 0 or done == len(h5_files):
                print(f'  [rank {done}/{len(h5_files)} files]', flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=_init_rank_worker,
                initargs=(config_path, element_tables, group_size, gap,
                          rank_by)) as ex:
            futs = [ex.submit(_rank_file, a) for a in args_list]
            for done, fut in enumerate(as_completed(futs), 1):
                all_recs.extend(fut.result())
                if done % 25 == 0 or done == len(h5_files):
                    print(f'  [rank {done}/{len(h5_files)} files]', flush=True)
    all_recs.sort(key=lambda r: r[0], reverse=True)
    return all_recs[:top_k]


def _build_sim(detector_config, args):
    track_config = create_track_hits_config(
        max_keys=args.probe_max_keys, hits_chunk_size=args.hits_chunk,
        inter_thresh=args.inter_thresh,
        box_enabled=args.box, maxg=args.maxg,
        box_bpy=args.box_bpy, box_bpz=args.box_bpz, box_bt=args.box_bt,
        box_bw=args.box_bw, box_btw=args.box_btw)
    sim = DetectorSimulator(
        detector_config,
        total_pad=args.total_pad,
        response_chunk_size=args.response_chunk,
        use_bucketed=args.bucketed,
        max_active_buckets=args.probe_max_buckets,
        include_track_hits=True,
        track_config=track_config)
    sim.warm_up()
    return sim


def _actual_for_event(sim, fpath, idx, key):
    deposits = load_event(fpath, sim.config, event_idx=idx)
    n_deps = sum(int(v.n_actual) for v in deposits.volumes)
    _, track_hits_raw, _ = sim.process_event(deposits, key=key)
    actual = {k: int(raw[4]) for k, raw in track_hits_raw.items()
              if isinstance(k, tuple)}
    return n_deps, actual


def run_rank_validate(args):
    import yaml as _yaml
    from profiler.estimate_max_keys import (build_pixel_value_table,
        build_wire_value_table, build_charge_model)
    detector_config = generate_detector(args.config)
    sim_config_est = create_sim_config(detector_config)
    element_tables = _build_element_tables(sim_config_est)
    charge_model = build_charge_model(_yaml.safe_load(open(args.config)))
    value_tables = {v: (build_pixel_value_table(sim_config_est, vg)
                        if vg.readout_type == 'pixel'
                        else build_wire_value_table(vg.diffusion))
                    for v, vg in enumerate(sim_config_est.volumes)}
    h5_files = _resolve_files(args.data)

    if args.random:
        top = random_events(h5_files, args.random, args.seed)
        print(f'  Random sample: {len(top)} events across {len(h5_files)} '
              f'files (seed={args.seed})', flush=True)
    else:
        metric = 'segments(max-vol deposits)' if args.rank_by == 'deposits' \
            else 'estimate'
        print(f'  Ranking {len(h5_files)} files (top-{args.top_k} by {metric}, '
              f'{args.rank_workers} workers, '
              f'{"all" if args.rank_events_per_file is None else args.rank_events_per_file}'
              f' events/file)...', flush=True)
        top = rank_events(h5_files, args.config, element_tables,
                          args.group_size, args.gap_threshold,
                          args.rank_workers, args.rank_events_per_file,
                          args.top_k, args.rank_by)
        print(f'  Top metric range: {top[0][0]:,} (max) .. {top[-1][0]:,} '
              f'(#{len(top)})', flush=True)

    print(f'\n  Device: {jax.devices()[0]}\n  Building actual simulator '
          f'(total_pad={args.total_pad:,}, probe max_keys={args.probe_max_keys:,})...',
          flush=True)
    sim = _build_sim(detector_config, args)
    key = jax.random.PRNGKey(42)

    print(f'\n  Validating top-{args.top_k} events with ACTUAL simulator:',
          flush=True)
    max_actual = 0
    max_actual_loc = None
    rows = []
    n_overflow = 0
    for rank, (m, fp, idx) in enumerate(top, 1):
        key, sub = jax.random.split(key)
        deposits = load_event(fp, sim.config, event_idx=idx)
        n_deps = sum(int(v.n_actual) for v in deposits.volumes)
        try:
            _, thr, _ = sim.process_event(deposits, key=sub)
            actual = {k: int(raw[4]) for k, raw in thr.items()
                      if isinstance(k, tuple)}
            act_max = max(actual.values()) if actual else 0
            over = ''
        except RuntimeError as e:
            # sim raises on track_hits overflow; record as >= probe, continue
            act_max = args.probe_max_keys
            over = f' *** OVERFLOW >= probe ({str(e)[-40:].strip()}) ***'
            n_overflow += 1
        with h5py.File(fp, 'r') as _f:
            pstep = _f['pstep/lar_vol'][idx]
        est_keys, _ = estimate_keys_for_event(
            pstep, sim_config_est, element_tables,
            group_size=args.group_size, gap_threshold_mm=args.gap_threshold)
        est_max = max(est_keys.values()) if est_keys else 0
        estc_keys, _ = estimate_keys_for_event(
            pstep, sim_config_est, element_tables,
            group_size=args.group_size, gap_threshold_mm=args.gap_threshold,
            value_tables=value_tables, charge_model=charge_model,
            inter_thresh=args.inter_thresh)
        estc_max = max(estc_keys.values()) if estc_keys else 0
        if act_max > max_actual:
            max_actual = act_max
            max_actual_loc = f'{os.path.basename(fp)} evt {idx}'
        rows.append((m, est_max, act_max, n_deps, estc_max))
        print(f'    [{rank:>3}] {os.path.basename(fp)} evt {idx}: '
              f'deps={n_deps:>9,}  PERDEP={est_max:>9,}  CHG={estc_max:>9,}  '
              f'ACTUAL={act_max:>9,}{over}', flush=True)

    import math
    arr = np.array(rows, float)  # m, perdep, act, deps, chg
    rec = int(math.ceil(int(max_actual * args.headroom) / 100_000) * 100_000)
    ratio = arr[:, 1] / np.maximum(arr[:, 2], 1)   # perdep/act
    ratioc = arr[:, 4] / np.maximum(arr[:, 2], 1)  # charge-aware/act
    print('\n' + '=' * 64)
    print(f'  events run:            {len(rows)}')
    if n_overflow:
        print(f'  OVERFLOWED probe:      {n_overflow} event(s) >= '
              f'{args.probe_max_keys:,} -- max is a LOWER BOUND; raise probe')
    print(f'  Max ACTUAL keys:       {max_actual:,}  ({max_actual_loc})')
    print(f'  Max PERDEP / CHG est:  {int(arr[:,1].max()):,} / {int(arr[:,4].max()):,}')
    print(f'  PERDEP/ACTUAL ratio:   median={np.median(ratio):.3f}  '
          f'min={ratio.min():.3f}  max={ratio.max():.3f}')
    print(f'  CHG/ACTUAL ratio:      median={np.median(ratioc):.3f}  '
          f'min={ratioc.min():.3f}  max={ratioc.max():.3f}')
    print(f'  Recommended max_keys (actual x {args.headroom}): {rec:,}')
    print('=' * 64, flush=True)


def run_sequential(args):
    detector_config = generate_detector(args.config)
    sim_config_est = create_sim_config(detector_config)
    element_tables = _build_element_tables(sim_config_est)
    for v, t in element_tables.items():
        print(f'  Vol {v} element_table: {t.tolist()}', flush=True)

    sim = _build_sim(detector_config, args)
    cfg = sim.config
    key = jax.random.PRNGKey(42)

    rows = []
    with h5py.File(args.data, 'r') as h:
        ds = h['pstep/lar_vol']
        for i in range(args.events):
            evt = args.event + i
            key, sub = jax.random.split(key)
            n_deps, actual = _actual_for_event(sim, args.data, evt, sub)
            est_keys, _ = estimate_keys_for_event(
                ds[evt], sim_config_est, element_tables,
                group_size=args.group_size, gap_threshold_mm=args.gap_threshold)
            a = max(actual.values()) if actual else 0
            e = max(est_keys.values()) if est_keys else 0
            rows.append((a, e))
            print(f'  evt {evt}: deps={n_deps:>9,}  ACTUAL_max={a:>9,}  '
                  f'EST_max={e:>9,}  EST/ACT={e/max(a,1):.2f}', flush=True)

    act = np.array([r[0] for r in rows], float)
    est = np.array([r[1] for r in rows], float)
    print('\n  EST/ACTUAL ratio: median='
          f'{np.median(est/np.maximum(act,1)):.3f}  '
          f'max={(est/np.maximum(act,1)).max():.3f}')
    print(f'  ACTUAL max: {int(act.max()):,}  ESTIMATE max: {int(est.max()):,}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, nargs='+',
                    help='File (sequential mode) or file(s)/dir (rank-validate)')
    ap.add_argument('--config', required=True)
    ap.add_argument('--event', type=int, default=0)
    ap.add_argument('--events', type=int, default=12)
    ap.add_argument('--total-pad', type=int, default=1_330_000)
    ap.add_argument('--response-chunk', type=int, default=33_250)
    ap.add_argument('--hits-chunk', type=int, default=133_000)
    ap.add_argument('--probe-max-keys', type=int, default=10_000_000)
    ap.add_argument('--bucketed', action='store_true',
                    help='Bucketed response accumulation (required for pixel)')
    ap.add_argument('--probe-max-buckets', type=int, default=200_000,
                    help='Large max_buckets when --bucketed (default: 200k)')
    ap.add_argument('--inter-thresh', type=float, default=1.0)
    # box-mode (production path) — match the production config dims
    ap.add_argument('--no-box', dest='box', action='store_false', default=True,
                    help='Use the legacy merge path instead of box (box is default)')
    ap.add_argument('--maxg', type=int, default=110_000)
    ap.add_argument('--box-bpy', type=int, default=8)
    ap.add_argument('--box-bpz', type=int, default=8)
    ap.add_argument('--box-bt', type=int, default=83)
    ap.add_argument('--box-bw', type=int, default=12)
    ap.add_argument('--box-btw', type=int, default=27)
    ap.add_argument('--group-size', type=int, default=5)
    ap.add_argument('--gap-threshold', type=float, default=5.0)
    ap.add_argument('--headroom', type=float, default=1.5)
    # rank-validate mode
    ap.add_argument('--rank-validate', action='store_true',
                    help='Rank dataset, validate top-K with the actual sim')
    ap.add_argument('--rank-by', choices=['deposits', 'estimate'],
                    default='deposits',
                    help='Rank metric: deposits=segment count (cheap), '
                         'estimate=geometry keys (default: deposits)')
    ap.add_argument('--top-k', type=int, default=500)
    ap.add_argument('--random', type=int, default=None,
                    help='Validate N random events instead of top-K')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rank-workers', type=int, default=16)
    ap.add_argument('--rank-events-per-file', type=int, default=None,
                    help='Cap events/file during ranking (default: all)')
    args = ap.parse_args()

    if args.rank_validate:
        run_rank_validate(args)
    else:
        args.data = args.data[0]
        run_sequential(args)


if __name__ == '__main__':
    main()
