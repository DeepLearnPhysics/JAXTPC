"""
Batch simulation: run events and save to structured HDF5 files.

Produces three file types per batch:
    {dataset}_resp_{NNNN}.h5  — sparse thresholded wire signals
    {dataset}_seg_{NNNN}.h5   — 3D truth deposits (per-volume)
    {dataset}_corr_{NNNN}.h5  — 3D→2D correspondence + track labels

See README.md for pipeline details, output schema, and threading architecture.

Usage (from project root):
    python3 production/run_batch.py
    python3 production/run_batch.py --data mpvmpr_20.h5 --dataset mpvmpr --threshold-adc 5.0
    python3 production/run_batch.py --events 100 --events-per-file 50
    python3 production/run_batch.py --no-track-hits
"""

import argparse
import os
import sys
import time
import gc
import threading
import queue
from functools import partial

# Add project root to path so tools/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tools.simulation import DetectorSimulator
from tools.config import create_track_hits_config
from tools.geometry import generate_detector
from tools.loader import ParticleStepExtractor, build_deposit_data

from production.save import (
    write_config_resp, write_config_seg, write_config_corr,
    save_event_resp, save_event_seg, save_event_corr,
    encode_correspondence_csr, _plane_label,
)

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# EVENT LOADING
# =============================================================================

def load_deposit(extractor, event_idx, sim_config,
                 group_size=5, gap_threshold_mm=5.0):
    """Load one event from an open extractor, build DepositData.

    Uses the extractor directly (file stays open across events) then
    passes raw arrays to build_deposit_data for volume splitting,
    grouping, and padding.

    Returns DepositData (multi-volume, padded, grouped).
    """
    step_data = extractor.extract_step_arrays(event_idx)
    positions_mm = np.asarray(
        step_data.get('position', np.empty((0, 3))), dtype=np.float32)
    n = positions_mm.shape[0]

    # GEANT4 stores time in nanoseconds; convert to microseconds
    t_ns = np.asarray(step_data.get('t', np.zeros((n,))), dtype=np.float32)
    t0_us = t_ns / 1000.0

    return build_deposit_data(
        positions_mm,
        np.asarray(step_data.get('de', np.zeros((n,))), dtype=np.float32),
        np.asarray(step_data.get('dx', np.zeros((n,))), dtype=np.float32),
        sim_config,
        theta=np.asarray(step_data.get('theta', np.zeros((n,))), dtype=np.float32),
        phi=np.asarray(step_data.get('phi', np.zeros((n,))), dtype=np.float32),
        track_ids=np.asarray(step_data.get('track_id', np.ones((n,))), dtype=np.int32),
        t0_us=t0_us,
        group_size=group_size,
        gap_threshold_mm=gap_threshold_mm,
    )


def get_num_events(data_path):
    with h5py.File(data_path, 'r') as f:
        return f['pstep/lar_vol'].shape[0]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Batch TPC simulation (v2)')
    parser.add_argument('--data', default='mpvmpr_20.h5', help='Input HDF5 file')
    parser.add_argument('--config', default='config/cubic_wireplane_config.yaml')
    parser.add_argument('--dataset', default='sim', help='Dataset name for output files')
    parser.add_argument('--outdir', default='.', help='Output directory')
    parser.add_argument('--events', type=int, default=None, help='Number of events (default: all)')
    parser.add_argument('--events-per-file', type=int, default=1000,
                        help='Events per output file (default: 1000)')
    parser.add_argument('--threshold-adc', type=float, default=2.0,
                        help='Threshold in ADC for sparse signal output (default: 2.0)')
    # Physics toggles (default: noise OFF, electronics OFF, digitization ON)
    parser.add_argument('--noise', action='store_true', help='Enable intrinsic noise')
    parser.add_argument('--electronics', action='store_true', help='Enable electronics response')
    parser.add_argument('--no-digitize', action='store_true', help='Disable ADC digitization')
    parser.add_argument('--no-track-hits', action='store_true', help='Disable track correspondence')
    parser.add_argument('--sce', default=None, help='Path to SCE HDF5 map for E-field distortions')
    # Grouping
    parser.add_argument('--group-size', type=int, default=5)
    parser.add_argument('--gap-threshold', type=float, default=5.0,
                        help='Gap threshold in mm for group splitting')
    parser.add_argument('--corr-threshold', type=float, default=25.0,
                        help='Charge threshold in electrons for correspondence entries (default: 25)')
    parser.add_argument('--total-pad', type=int, default=500_000)
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of save worker threads (0=serial, default: 2)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    include_noise = args.noise
    include_electronics = args.electronics
    include_digitize = not args.no_digitize
    include_track_hits = not args.no_track_hits
    include_sce = args.sce is not None
    events_per_file = args.events_per_file
    threshold_adc = args.threshold_adc
    dataset_name = args.dataset

    total_events = get_num_events(args.data)
    num_events = min(args.events, total_events) if args.events else total_events
    num_files = (num_events + events_per_file - 1) // events_per_file

    # Output directories
    resp_dir = os.path.join(args.outdir, 'resp')
    seg_dir = os.path.join(args.outdir, 'seg')
    corr_dir = os.path.join(args.outdir, 'corr') if include_track_hits else None
    for d in [resp_dir, seg_dir]:
        os.makedirs(d, exist_ok=True)
    if corr_dir:
        os.makedirs(corr_dir, exist_ok=True)

    print('=' * 60)
    print(' JAXTPC Batch Simulation v2')
    print('=' * 60)
    print(f'  Data:          {args.data} ({num_events}/{total_events} events)')
    print(f'  Dataset:       {dataset_name}')
    print(f'  Events/file:   {events_per_file}')
    print(f'  Num files:     {num_files}')
    print(f'  Threshold:     {threshold_adc} ADC')
    print(f'  Noise:         {"ON" if include_noise else "OFF"}')
    print(f'  Electronics:   {"ON" if include_electronics else "OFF"}')
    print(f'  Digitization:  {"ON" if include_digitize else "OFF"}')
    print(f'  SCE:           {args.sce if include_sce else "OFF"}')
    print(f'  Track hits:    {"ON" if include_track_hits else "OFF"}')
    print(f'  Group size:    {args.group_size}')
    print(f'  Total pad:     {args.total_pad:,}')
    print(f'  Workers:       {args.workers} {"(serial)" if args.workers == 0 else "(threaded)"}')
    print(f'  Device:        {jax.devices()[0]}')
    print(f'  Output:        {args.outdir}/{{resp,seg,corr}}/')
    print()

    # ---- Create simulator ----
    detector_config = generate_detector(args.config)
    track_config = create_track_hits_config() if include_track_hits else None

    t_create = time.time()
    simulator = DetectorSimulator(
        detector_config,
        track_config=track_config,
        total_pad=args.total_pad,
        include_noise=include_noise,
        include_electronics=include_electronics,
        include_track_hits=include_track_hits,
        include_digitize=include_digitize,
        include_electric_dist=include_sce,
        electric_dist_path=args.sce,
    )
    t_create = time.time() - t_create

    cfg = simulator.config
    params = simulator.default_sim_params
    dig_config = getattr(simulator, 'digitization_config', None)

    t_warmup = time.time()
    simulator.warm_up()
    t_warmup = time.time() - t_warmup

    print(f'\n  Simulator creation: {t_create:.1f}s')
    print(f'  JIT warmup:        {t_warmup:.1f}s')

    # ---- Real-data warmup ----
    print("  Real-data warmup...", end='', flush=True)
    t0 = time.time()
    warmup_dep = load_deposit(
        ParticleStepExtractor(args.data), 0, cfg,
        args.group_size, args.gap_threshold)
    warmup_r, _, warmup_dep = simulator.process_event(warmup_dep, key=jax.random.PRNGKey(0))
    for a in warmup_r.values():
        jax.block_until_ready(a)
    del warmup_r, warmup_dep
    gc.collect()
    print(f" {time.time() - t0:.1f}s\n")

    # ---- Save helpers ----
    key = jax.random.PRNGKey(args.seed)
    total_start = time.time()

    num_workers = args.workers
    file_lock = threading.Lock()

    def save_one_event(f_resp, f_seg, f_corr, item):
        """Save a single event (CSR encode + HDF5 write). Thread-safe."""
        (event_key, response_np, track_hits_raw, deposits, source_idx) = item

        # CSR encoding (numpy, GIL-free — runs in parallel across workers)
        corr_data = None
        if include_track_hits and f_corr is not None:
            corr_data = {}
            for plane_key, raw in track_hits_raw.items():
                if not isinstance(plane_key, tuple):
                    continue
                pk, gid, ch, count, _ = raw
                corr_data[plane_key] = encode_correspondence_csr(
                    pk, gid, ch, count, cfg.num_time_steps,
                    threshold=args.corr_threshold)

        # HDF5 write (serialized through file lock)
        with file_lock:
            save_event_resp(f_resp, event_key, response_np, threshold_adc,
                            source_idx, deposits,
                            digitized=include_digitize)
            save_event_seg(f_seg, event_key, deposits, source_idx)
            if corr_data is not None and f_corr is not None:
                _write_corr_event(f_corr, event_key, corr_data,
                                  deposits, source_idx)

    def _write_corr_event(f, event_key, corr_data, deposits, source_idx):
        """Write pre-encoded correspondence to HDF5."""
        evt = f.create_group(event_key)
        evt.attrs['source_event_idx'] = source_idx
        evt.attrs['n_volumes'] = len(deposits.volumes)
        evt.attrs['threshold'] = args.corr_threshold

        for v in range(len(deposits.volumes)):
            vol_grp = evt.create_group(f'volume_{v}')
            g2t = deposits.group_to_track[v]
            if g2t is not None:
                vol_grp.create_dataset('group_to_track',
                                       data=g2t, compression='gzip')
            for (vi, pi), csr in corr_data.items():
                if vi != v:
                    continue
                g = vol_grp.create_group(_plane_label(pi))
                for k, arr in csr.items():
                    g.create_dataset(k, data=arr, compression='gzip')
                g.attrs['n_groups_plane'] = len(csr['group_ids'])
                g.attrs['n_entries'] = len(csr['delta_wires'])

    def save_worker(f_resp, f_seg, f_corr, save_queue):
        """Worker thread: pull items from queue, encode + save."""
        while True:
            item = save_queue.get()
            if item is None:
                break
            save_one_event(f_resp, f_seg, f_corr, item)
            save_queue.task_done()

    # ---- Process events ----
    with ParticleStepExtractor(args.data) as extractor:
        for file_idx in range(num_files):
            event_start = file_idx * events_per_file
            event_end = min(event_start + events_per_file, num_events)
            n_in_file = event_end - event_start

            resp_path = os.path.join(resp_dir,
                f'{dataset_name}_resp_{file_idx:04d}.h5')
            seg_path = os.path.join(seg_dir,
                f'{dataset_name}_seg_{file_idx:04d}.h5')
            corr_path = os.path.join(corr_dir,
                f'{dataset_name}_corr_{file_idx:04d}.h5') if corr_dir else None

            print(f'File {file_idx:04d}: events {event_start}–{event_end-1} '
                  f'({n_in_file} events)')

            with h5py.File(resp_path, 'w') as f_resp, \
                 h5py.File(seg_path, 'w') as f_seg:

                f_corr_ctx = h5py.File(corr_path, 'w') if corr_path else None
                try:
                    write_config_resp(
                        f_resp, cfg, params, simulator.recomb_model,
                        dataset_name, file_idx, args.data,
                        n_in_file, event_start, threshold_adc,
                        digitization_config=dig_config)
                    write_config_seg(
                        f_seg, cfg, dataset_name, file_idx, args.data,
                        n_in_file, event_start,
                        args.group_size, args.gap_threshold)
                    if f_corr_ctx:
                        write_config_corr(
                            f_corr_ctx, cfg, dataset_name, file_idx, args.data,
                            n_in_file, event_start,
                            args.group_size, args.gap_threshold)

                    # Start workers (if threaded)
                    save_queue = None
                    workers = []
                    if num_workers > 0:
                        save_queue = queue.Queue(maxsize=num_workers + 2)
                        for w in range(num_workers):
                            t = threading.Thread(
                                target=save_worker,
                                args=(f_resp, f_seg, f_corr_ctx, save_queue))
                            t.daemon = True
                            t.start()
                            workers.append(t)

                    for idx in range(event_start, event_end):
                        key, subkey = jax.random.split(key)
                        local_idx = idx - event_start
                        event_key = f'event_{local_idx:03d}'

                        # Load + build DepositData (volume split, group, pad)
                        t_load = time.time()
                        deposits = load_deposit(
                            extractor, idx, cfg,
                            args.group_size, args.gap_threshold)
                        t_load = time.time() - t_load
                        n_deposits = sum(v.n_actual for v in deposits.volumes)

                        # Simulate
                        t_sim = time.time()
                        response_signals, track_hits, deposits = \
                            simulator.process_event(deposits, key=subkey)
                        for arr in response_signals.values():
                            jax.block_until_ready(arr)
                        t_sim = time.time() - t_sim

                        # GPU → CPU transfer for signals (large arrays)
                        # track_hits and qs_per_volume convert lazily in save fns
                        response_np = {k: np.asarray(v)
                                       for k, v in response_signals.items()}

                        item = (event_key, response_np, track_hits, deposits, idx)

                        # Save (serial or queued)
                        t_save = time.time()
                        if num_workers > 0:
                            save_queue.put(item)
                        else:
                            save_one_event(f_resp, f_seg, f_corr_ctx, item)
                        t_save = time.time() - t_save

                        t_total = t_load + t_sim + t_save
                        print(f'  [{local_idx+1:3d}/{n_in_file}] event {idx:6d}  '
                              f'{n_deposits:6,} deps  '
                              f'load={t_load:.2f}s  sim={t_sim:.2f}s  '
                              f'save={t_save:.2f}s  total={t_total:.1f}s')

                        del response_signals
                        gc.collect()

                    # Wait for workers to finish
                    if num_workers > 0:
                        for _ in range(num_workers):
                            save_queue.put(None)
                        for t in workers:
                            t.join()

                finally:
                    if f_corr_ctx:
                        f_corr_ctx.close()

            # Print file sizes
            resp_mb = os.path.getsize(resp_path) / (1024 * 1024)
            seg_mb = os.path.getsize(seg_path) / (1024 * 1024)
            print(f'  → resp: {resp_mb:.1f} MB, seg: {seg_mb:.1f} MB', end='')
            if corr_path and os.path.exists(corr_path):
                corr_mb = os.path.getsize(corr_path) / (1024 * 1024)
                print(f', corr: {corr_mb:.1f} MB')
            else:
                print()
            print()

    total_elapsed = time.time() - total_start
    print(f'{"=" * 60}')
    print(f'  Done. {num_events} events in {total_elapsed:.1f}s')
    print(f'  Average: {total_elapsed/num_events:.2f}s/event')
    print(f'  Files:   {num_files} × 3 in {args.outdir}/{{resp,seg,corr}}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
