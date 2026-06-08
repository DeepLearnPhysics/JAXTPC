"""
Find optimal MAXG (group-bucket capacity) AND the readout-correct box dims for
the group-as-bucket track-hits path, WITHOUT the simulation.

Two parameters, two axes (orthogonal):
  MAXG       = max group count per (event, volume).  Readout-INDEPENDENT
               (groups come from deposits/tracks).  Set at p99.95; the sim
               raises maxg_overflow for rarer events -> log + reprocess.
  box dims   = per-group local footprint + kernel.  Readout-SPECIFIC:
               pixel -> (BPY, BPZ, BT) from (py, pz, tick) extents + 3x3x69 kernel
               wire  -> (BW, BT)       from (wire, tick) extents + (2Kw+1)x(2Kt+1)

Cost note (why modes matter): the input is uncompressed vlen-of-compound HDF5
(~8.5 GB/file, mean ~665k steps/event), and a vlen dataset has no field-selective
read — every scan reads the full 64-byte record per step. The scan is therefore
disk-I/O bound; the per-event group sort roughly *doubles* per-worker time.

Modes:
  --fast            P2 (n_actual/group_size + n_tracks), no group sort, no box
                    dims. ~1.8x faster; MAXG is ~4% conservative (safe, absorbed
                    by rounding). Use when you only need MAXG.
  --dim-files N     Hybrid (recommended for full datasets): P2 MAXG over ALL
                    files + a full group sort over the first N files for box
                    dims. Box-dim extents are a stable, tight local property, so
                    a subset suffices; a margin guards against unscanned outliers.
  (default)         Full group sort over every file: exact MAXG + global-max box
                    dims. Most accurate, slowest.

Usage:
    python3 -m profiler.find_optimal_maxg --data run_*/ --config config/cubic_pixel_config.yaml \
        --workers 28 --dim-files 20
"""
import argparse, glob, math, os, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
from tools.geometry import generate_detector
from tools.config import create_sim_config
from tools.loader import compute_group_ids
from profiler.estimate_max_keys import estimate_keys_for_event

# Events per vlen read. Batching amortizes h5py per-element overhead (~1.3x);
# memory stays bounded since we drop each event after reducing it.
_READ_BATCH = 20


def _resolve(data):
    paths = []
    for p in ([data] if isinstance(data, str) else data):
        paths.extend(sorted(glob.glob(os.path.join(p, '*.h5'))) if os.path.isdir(p) else [p])
    return paths


_state = {}


def _init(config_path, group_size, gap, events_per_file, exact_counts,
          element_tables=None, value_tables=None, charge_model=None,
          key_thresh=1.0):
    sc = create_sim_config(generate_detector(config_path), total_pad=2_000_000)
    _state.update(sc=sc, gs=group_size, gap=gap, epf=events_per_file,
                  exact=exact_counts, readout=sc.volumes[0].readout_type,
                  etables=element_tables, vtables=value_tables,
                  cmodel=charge_model, kthresh=key_thresh)


def _grp_extents(coord, starts):
    """max-min+1 per group for a coordinate (sorted by group)."""
    return (np.maximum.reduceat(coord, starts) - np.minimum.reduceat(coord, starts) + 1)


def _scan(arg):
    """Return (fpath, nev, ng_list, ext_list, dep_list, keys_list).

    ng_list: one n_groups per (event, volume) — P2 estimate unless exact_counts.
    ext_list: [wire/py ext, pz ext, tick ext] per (event, volume), only if do_dims.
    dep_list: deposit count per (event, volume) — always returned.
    keys_list: per-event max keys (max across volumes/planes) — only if etables provided.
    """
    fpath, do_dims = arg
    sc = _state['sc']; gs = _state['gs']; gap = _state['gap']
    epf = _state['epf']; exact = _state['exact']; readout = _state['readout']
    etables = _state.get('etables')
    vtables = _state.get('vtables'); cmodel = _state.get('cmodel')
    do_keys = etables is not None or vtables is not None
    need_groups = exact or do_dims
    ng_list, ext_list, dep_list, keys_list = [], [], [], []
    with h5py.File(fpath, 'r') as f:
        ds = f['pstep/lar_vol']
        nev = ds.shape[0] if epf is None else min(epf, ds.shape[0])
        for b0 in range(0, nev, _READ_BATCH):
            for row in ds[b0:min(b0 + _READ_BATCH, nev)]:   # batched vlen read
                pc = np.column_stack([row['x'], row['y'], row['z']]).astype(np.float32) / 10.0
                tid = row['track_id'].astype(np.int32); de = row['de'].astype(np.float32)
                t0 = (row['t'].astype(np.float32) / 1000.0) if 't' in row.dtype.names \
                    else np.zeros(len(de), np.float32)
                for vi, vg in enumerate(sc.volumes):
                    (x0, x1), (y0, y1), (z0, z1) = vg.ranges_cm
                    m = ((pc[:, 0] >= x0) & (pc[:, 0] < x1) & (pc[:, 1] >= y0) &
                         (pc[:, 1] < y1) & (pc[:, 2] >= z0) & (pc[:, 2] < z1) & (de > 0))
                    nact = int(m.sum())
                    dep_list.append(nact)
                    if nact == 0:
                        ng_list.append(0)
                        continue
                    ntr = int(np.unique(tid[m]).size)
                    g = ng = None
                    if need_groups:
                        g, _, ng = compute_group_ids(
                            np.column_stack([row['x'], row['y'], row['z']]).astype(np.float32)[m],
                            tid[m], np.ones(nact, bool), group_size=gs, gap_threshold_mm=gap)
                    ng_list.append(ng if (exact or need_groups) else (nact / gs + ntr))

                    if not do_dims:
                        continue
                    g = g.astype(np.int64)
                    drift = np.abs(pc[m][:, 0] - vg.x_anode_cm)
                    vel = max(vg.diffusion.velocity_cm_us, 1e-9)
                    tick = np.floor(((drift / vel)
                                     + t0[m] + sc.pre_window_us) / sc.time_step_us).astype(np.int64)
                    order = np.argsort(g, kind='stable')
                    gs_, tk_ = g[order], tick[order]
                    _, starts = np.unique(gs_, return_index=True)
                    te = _grp_extents(tk_, starts).max()
                    if readout == 'pixel':
                        oy, oz = vg.pixel_origins_cm
                        py = np.floor((pc[m][:, 1] - oy) / vg.pixel_pitch_cm).astype(np.int64)[order]
                        pz = np.floor((pc[m][:, 2] - oz) / vg.pixel_pitch_cm).astype(np.int64)[order]
                        ext_list.append([_grp_extents(py, starts).max(),
                                         _grp_extents(pz, starts).max(), te])
                    else:
                        vyz = pc[m][:, 1:3] - np.array(vg.yz_center_cm, np.float32)
                        ew = 0
                        for p in range(vg.n_planes):
                            rp = (vyz[:, 0] * np.sin(vg.angles_rad[p])
                                  + vyz[:, 1] * np.cos(vg.angles_rad[p]))
                            widx = (np.round(rp / vg.wire_spacings_cm[p]).astype(np.int64)
                                    + vg.index_offsets[p])[order]
                            ew = max(ew, int(_grp_extents(widx, starts).max()))
                        ext_list.append([ew, 0, te])

                # max_keys estimation: call the existing function per event
                if do_keys:
                    event_keys, _ = estimate_keys_for_event(
                        row, sc, etables, group_size=gs, gap_threshold_mm=gap,
                        value_tables=vtables, charge_model=cmodel,
                        inter_thresh=_state.get('kthresh', 1.0))
                    event_max = max(event_keys.values()) if event_keys else 0
                    keys_list.append(event_max)

    return fpath, nev, ng_list, ext_list, dep_list, keys_list


def find_optimal_maxg(data, config_path, group_size=5, gap=5.0, events_per_file=None,
                      n_workers=1, round_to=10_000, pctile=99.95, fast=False,
                      dim_files=None, dim_margin=2, element_tables=None,
                      value_tables=None, charge_model=None, key_thresh=1.0):
    """Estimate MAXG + box dims + deposits + (optionally) max_keys in one pass.

    value_tables + charge_model -> charge-aware max_keys estimate (per-deposit
    footprint at threshold key_thresh = c* x box_inter_thresh; c*~2.5 -> ~1.0x
    the actual box key count). Falls back to element_tables (geometry only) when
    value_tables is None.

    fast        -> P2 MAXG only, no box dims.
    dim_files=N -> hybrid: P2 MAXG over all files, box dims from first N files.
    else        -> full group sort over all files (exact MAXG + global-max dims).
    dim_margin  -> cells added to each footprint extent before adding the kernel
                   (guards subset-sampled dims against unscanned outliers).
    element_tables -> {vol_idx: (num_s,) int32 array}. When provided, each
                      event-volume also gets a max_keys estimate (returned in
                      info['keys']). Avoids a second scan pass.
    """
    files = _resolve(data)
    sc = create_sim_config(generate_detector(config_path), total_pad=2_000_000)
    readout = sc.volumes[0].readout_type
    if readout == 'pixel':
        from tools.kernels import load_pixel_response_kernel
        pkp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'pixel_response.npz')
        pk = load_pixel_response_kernel(pkp, pixel_pitch_cm=sc.volumes[0].pixel_pitch_cm,
                                        time_spacing=sc.time_step_us,
                                        max_sigma_trans_unitless=1.0, max_sigma_long_unitless=1.0)
        ker = (pk.kernel_py, pk.kernel_pz, pk.kernel_time)
    else:
        d = sc.volumes[0].diffusion
        ker = (2 * d.K_wire + 1, 0, 2 * d.K_time + 1)

    # per-file: should this file contribute box-dim extents?
    if fast:
        exact_counts = False; dim_set = set()
    elif dim_files is not None:
        exact_counts = False; dim_set = set(files[:max(0, dim_files)])
    else:
        exact_counts = True; dim_set = set(files)
    tasks = [(fp, fp in dim_set) for fp in files]

    ng_all, ext_all, dep_all, keys_all, total_ev, t0 = [], [], [], [], 0, time.time()
    if n_workers <= 1 or len(files) <= 1:
        _init(config_path, group_size, gap, events_per_file, exact_counts,
              element_tables=element_tables, value_tables=value_tables,
              charge_model=charge_model, key_thresh=key_thresh)
        for i, task in enumerate(tasks, 1):
            _, nev, ng, ext, dep, keys = _scan(task)
            ng_all.extend(ng); ext_all.extend(ext); dep_all.extend(dep)
            keys_all.extend(keys); total_ev += nev
            if len(files) > 1:
                print(f'  [{i}/{len(files)}] ({time.time()-t0:.0f}s)', flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init,
                                 initargs=(config_path, group_size, gap, events_per_file,
                                           exact_counts, element_tables, value_tables,
                                           charge_model, key_thresh)) as ex:
            futs = [ex.submit(_scan, t) for t in tasks]
            for done, fut in enumerate(as_completed(futs), 1):
                _, nev, ng, ext, dep, keys = fut.result()
                ng_all.extend(ng); ext_all.extend(ext); dep_all.extend(dep)
                keys_all.extend(keys); total_ev += nev
                if done % 20 == 0 or done == len(files):
                    print(f'  [{done}/{len(files)} files] ({time.time()-t0:.0f}s)', flush=True)

    ng = np.array(ng_all, float)
    maxg = int(math.ceil(np.percentile(ng, pctile) / round_to) * round_to)
    dep = np.array(dep_all, float)
    keys = np.array(keys_all, float) if keys_all else np.empty(0)
    info = {'readout': readout, 'kernel': ker, 'n_events': total_ev, 'n_files': len(files),
            'elapsed_s': time.time() - t0, 'n_groups': ng, 'deposits': dep, 'keys': keys,
            'extents': np.array(ext_all, float) if ext_all else np.empty((0, 3)),
            'p99.95': float(np.percentile(ng, 99.95)), 'max': float(ng.max()),
            'max_ng': float(ng.max()), 'suggested_maxg': maxg, 'fast': fast,
            'dim_files': dim_files, 'dim_margin': dim_margin, 'n_dim_files': len(dim_set)}
    if ext_all:
        ext = np.array(ext_all, float)
        e1m = int(ext[:, 0].max()) + dim_margin
        e2m = int(ext[:, 1].max()) + dim_margin
        etm = int(ext[:, 2].max()) + dim_margin
        if readout == 'pixel':
            info['box_dims'] = {'BPY': e1m + ker[0], 'BPZ': e2m + ker[1], 'BT': etm + ker[2]}
            info['extent_max'] = {'py': e1m - dim_margin, 'pz': e2m - dim_margin, 'tick': etm - dim_margin}
        else:
            info['box_dims'] = {'BW': e1m + ker[0], 'BT': etm + ker[2]}
            info['extent_max'] = {'wire': e1m - dim_margin, 'tick': etm - dim_margin}
    return maxg, info


def main():
    ap = argparse.ArgumentParser(description='Estimate MAXG + box dims for the box track-hits path')
    ap.add_argument('--data', required=True, nargs='+')
    ap.add_argument('--config', required=True)
    ap.add_argument('--group-size', type=int, default=5)
    ap.add_argument('--gap-threshold', type=float, default=5.0)
    ap.add_argument('--events', type=int, default=None)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--pctile', type=float, default=99.95)
    ap.add_argument('--round-to', type=int, default=10_000)
    ap.add_argument('--fast', action='store_true', help='P2 MAXG only (no grouping, no box dims)')
    ap.add_argument('--dim-files', type=int, default=None,
                    help='Hybrid: P2 MAXG over all files + box dims from the first N files')
    ap.add_argument('--dim-margin', type=int, default=2,
                    help='Cells added to each footprint extent before the kernel (default: 2)')
    ap.add_argument('--save', default=None)
    ap.add_argument('--tag', default='box')
    args = ap.parse_args()

    files = _resolve(args.data)
    mode = ('FAST (P2, MAXG only)' if args.fast else
            f'hybrid (P2 MAXG all files, dims from {args.dim_files} files)'
            if args.dim_files is not None else 'full (group sort, MAXG+dims)')
    print('=' * 66)
    print(' JAXTPC — Find Optimal MAXG + box dims (group-as-bucket path)')
    print('=' * 66)
    print(f'  Files: {len(files)}  workers: {args.workers}  mode: {mode}')

    maxg, info = find_optimal_maxg(args.data, args.config, group_size=args.group_size,
                                   gap=args.gap_threshold, events_per_file=args.events,
                                   n_workers=args.workers, round_to=args.round_to,
                                   pctile=args.pctile, fast=args.fast,
                                   dim_files=args.dim_files, dim_margin=args.dim_margin)
    ng = info['n_groups']
    print(f'\n  readout: {info["readout"]}   kernel window: {info["kernel"]}')
    print(f'  scanned {info["n_events"]:,} events ({len(ng):,} event-volumes) in '
          f'{info["elapsed_s"]:.0f}s ({info["n_events"]/max(info["elapsed_s"],1e-9):.0f} ev/s)')
    print(f'\n  MAXG (group count, readout-independent):')
    for p in [50, 90, 99, 99.9, 99.95, 100]:
        print(f'    p{p:<6}= {int(np.percentile(ng, p)):>9,}')
    print(f'  Suggested MAXG (p{args.pctile} -> {args.round_to:,}) = {maxg:,}  '
          f'(overflow {100*np.mean(ng>maxg):.3f}% -> reprocess)')
    if info.get('box_dims'):
        print(f'\n  box dims (readout-specific, footprint + margin {info["dim_margin"]} + kernel,'
              f' from {info["n_dim_files"]} files):')
        print(f'    extent max: {info["extent_max"]}')
        print(f'    -> {info["box_dims"]}')

    save = args.save or f'experiments/merge/maxg_dist_{args.tag}.npz'
    os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
    np.savez(save, n_groups=ng, extents=info['extents'], suggested_maxg=maxg,
             box_dims=str(info.get('box_dims', {})), readout=info['readout'],
             kernel=info['kernel'])
    print(f'\n  Saved -> {save}')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ng, bins=80, color='steelblue'); ax.axvline(maxg, color='r', ls='--', label=f'MAXG={maxg:,}')
        ax.set_xlabel('n_groups per event-volume'); ax.set_ylabel('count'); ax.legend()
        ax.set_title(f'MAXG distribution ({info["readout"]})')
        png = save.replace('.npz', '.png'); fig.tight_layout(); fig.savefig(png, dpi=110)
        print(f'  Saved -> {png}')
    except Exception as e:
        print(f'  (plot skipped: {e})')


if __name__ == '__main__':
    main()
