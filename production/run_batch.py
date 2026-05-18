"""
Batch simulation: run events and save to structured HDF5 files.

Produces three file types per batch (canonical names; see
docs/DATASET_DESIGN.md in particle-imaging-models):
    {dataset}_sensor_{NNNN}.h5  — sparse thresholded raw readout
    {dataset}_edep_{NNNN}.h5    — 3D truth energy deposits (per-volume)
    {dataset}_hits_{NNNN}.h5    — per-particle charge attribution at sensor elements

See README.md for pipeline details, output schema, and threading architecture.

Usage (from project root):
    python3 production/run_batch.py
    python3 production/run_batch.py --data mpvmpr_20.h5 --dataset mpvmpr --threshold-adc 5.0
    python3 production/run_batch.py --events 100 --events-per-file 50
    python3 production/run_batch.py --no-track-hits
"""

import argparse
import os
import re
import subprocess
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
from tools.loader import ParticleStepExtractor, build_deposit_data, compute_interaction_ids

from production.save import (
    write_config_sensor, write_config_edep, write_config_hits,
    save_event_sensor, save_event_edep, save_event_hits,
    encode_correspondence_csr, encode_correspondence_csr_pixel,
)

sys.stdout.reconfigure(line_buffering=True)


def _get_git_info():
    """Get repo URL, commit hash, and dirty status from git."""
    def _run(cmd):
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    repo = _run(['git', 'remote', 'get-url', 'origin'])
    commit = _run(['git', 'rev-parse', 'HEAD'])
    dirty_output = _run(['git', 'status', '--porcelain'])
    dirty = dirty_output is not None and len(dirty_output) > 0
    return repo, commit, dirty


def _parse_edepsim_path(data_path):
    """Try to extract identifiers from an edep-sim path convention.

    Matches: .../{production_version}/run_{run_id}/edepsim_{file_idx}.h5
    Returns (production_version, run_id, file_idx) or (None, None, None).
    """
    m = re.search(r'/([^/]+)/run_(\d+)/edepsim_(\d+)\.h5$', data_path)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


def _read_edepsim_event_table(data_path):
    """Read run_id/event_id from edep-sim event/geant4 if present."""
    with h5py.File(data_path, 'r') as f:
        if 'event/geant4' not in f:
            return None
        return f['event/geant4'][:]


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
    pdata = getattr(extractor, '_last_particle_data', None) or {}
    interaction_ids = compute_interaction_ids(
        extractor.file, event_idx,
        root_track_ids=step_data.get('root_track_id'),
        particle_track_ids=pdata.get('track_id'),
        particle_parent_ids=pdata.get('parent_track_id'))
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
        interaction_ids=interaction_ids,
        root_track_ids=np.asarray(step_data.get('root_track_id', np.zeros((n,))), dtype=np.int32),
        pdg=np.asarray(step_data.get('pdg', np.zeros((n,))), dtype=np.int32),
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
    parser.add_argument('--intrinsic', action='store_true', help='Enable intrinsic (electronics) noise')
    parser.add_argument('--coherent', action='store_true', help='Enable coherent noise (per-group)')
    parser.add_argument('--electronics', action='store_true', help='Enable electronics response')
    parser.add_argument('--no-digitize', action='store_true', help='Disable ADC digitization')
    parser.add_argument('--no-track-hits', action='store_true', help='Disable track correspondence')
    parser.add_argument('--max-keys', type=int, default=4_000_000,
                        help='Max unique hits for track labeling (default: 4M)')
    parser.add_argument('--hits-chunk', type=int, default=25_000,
                        help='Deposits per track-hits fori_loop chunk (must divide total-pad)')
    parser.add_argument('--inter-thresh', type=float, default=1.0,
                        help='Track hits intermediate pruning threshold (default: 1.0)')
    parser.add_argument('--sce', default=None, help='Path to SCE HDF5 map for E-field distortions')
    # Grouping
    parser.add_argument('--group-size', type=int, default=5)
    parser.add_argument('--gap-threshold', type=float, default=5.0,
                        help='Gap threshold in mm for group splitting')
    parser.add_argument('--hits-threshold', type=float, default=1.0,
                        help='Charge threshold for hits (per-particle) '
                             'entries (default: 1.0)')
    parser.add_argument('--total-pad', type=int, default=500_000)
    parser.add_argument('--response-chunk', type=int, default=50_000,
                        help='Deposits per fori_loop batch (must divide total-pad)')
    parser.add_argument('--bucketed', action='store_true', help='Use bucketed accumulation')
    parser.add_argument('--max-buckets', type=int, default=1000,
                        help='Max active buckets per plane (bucketed mode)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of save worker threads (0=serial, default: 2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--production-config', default=None,
                        help='Load optimized params from profiler config YAML')
    # Event identification
    parser.add_argument('--production-version', default=None,
                        help='Production version string (e.g. test_00_00_01). '
                             'Auto-parsed from --data path if not set.')
    parser.add_argument('--run-id', type=int, default=None,
                        help='Edep-sim run ID (e.g. 26628546). '
                             'Auto-parsed from --data path if not set.')
    parser.add_argument('--event-id-offset', type=int, default=None,
                        help='Offset added to source_event_idx to compute '
                             'globally unique event_id within a run '
                             '(e.g. edepsim_file_idx * events_per_file). '
                             'Auto-computed from filename if not set.')
    parser.add_argument('--edepsim-ids', action='store_true',
                        help='Read run_id and event_id from the input file\'s '
                             'event/geant4 dataset instead of CLI/path. '
                             'Use when edep-sim files carry meaningful IDs.')
    args = parser.parse_args()

    # Apply production config (overrides defaults, CLI args re-override)
    if args.production_config:
        from profiler.production_config import load_config, apply_to_args
        prod_cfg = load_config(args.production_config)
        apply_to_args(args, prod_cfg)
        print(f'  Loaded production config: {args.production_config}')

    include_intrinsic_noise = args.intrinsic
    include_coherent_noise = args.coherent
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

    # Detect readout type early for pixel-specific defaults
    detector_config = generate_detector(args.config)
    readout_type = detector_config['volumes'][0].get('readout', {}).get('type', 'wire')

    # Pixel readout: track hits are mandatory (single response pass)
    if readout_type == 'pixel' and not include_track_hits:
        print('  NOTE: pixel readout requires track hits — enabling')
        include_track_hits = True

    # Output directories
    sensor_dir = os.path.join(args.outdir, 'sensor')
    edep_dir = os.path.join(args.outdir, 'edep')
    hits_dir = os.path.join(args.outdir, 'hits') if include_track_hits else None
    for d in [sensor_dir, edep_dir]:
        os.makedirs(d, exist_ok=True)
    if hits_dir:
        os.makedirs(hits_dir, exist_ok=True)

    print('=' * 60)
    print(' JAXTPC Batch Simulation v2')
    print('=' * 60)
    print(f'  Data:          {args.data} ({num_events}/{total_events} events)')
    print(f'  Dataset:       {dataset_name}')
    print(f'  Events/file:   {events_per_file}')
    print(f'  Num files:     {num_files}')
    print(f'  Threshold:     {threshold_adc} ADC')
    print(f'  Intrinsic:     {"ON" if include_intrinsic_noise else "OFF"}')
    print(f'  Coherent:      {"ON" if include_coherent_noise else "OFF"}')
    print(f'  Electronics:   {"ON" if include_electronics else "OFF"}')
    print(f'  Digitization:  {"ON" if include_digitize else "OFF"}')
    print(f'  SCE:           {args.sce if include_sce else "OFF"}')
    print(f'  Track hits:    {"ON" if include_track_hits else "OFF"}')
    print(f'  Group size:    {args.group_size}')
    print(f'  Total pad:     {args.total_pad:,}')
    if readout_type == 'wire':
        print(f'  Bucketed:      {"ON (max_buckets=" + str(args.max_buckets) + ")" if args.bucketed else "OFF"}')
    print(f'  Workers:       {args.workers} {"(serial)" if args.workers == 0 else "(threaded)"}')
    print(f'  Device:        {jax.devices()[0]}')
    print(f'  Output:        {args.outdir}/{{sensor,edep,hits}}/')
    print(f'  Readout:       {readout_type}')
    print()

    # Pixel defaults: smaller hits_chunk for optimal merge performance
    hits_chunk = args.hits_chunk
    if readout_type == 'pixel' and args.hits_chunk == 25_000:
        hits_chunk = 5_000

    track_config = create_track_hits_config(
        max_keys=args.max_keys, hits_chunk_size=hits_chunk,
        inter_thresh=args.inter_thresh,
    ) if include_track_hits else None

    t_create = time.time()
    simulator = DetectorSimulator(
        detector_config,
        track_config=track_config,
        total_pad=args.total_pad,
        response_chunk_size=args.response_chunk,
        use_bucketed=args.bucketed if readout_type == 'wire' else False,
        max_active_buckets=args.max_buckets,
        include_intrinsic_noise=include_intrinsic_noise,
        include_coherent_noise=include_coherent_noise,
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
        if isinstance(a, dict):
            for arr in a.values():
                jax.block_until_ready(arr)
        elif isinstance(a, tuple):
            jax.block_until_ready(a[0])
        else:
            jax.block_until_ready(a)
    del warmup_r, warmup_dep
    gc.collect()
    print(f" {time.time() - t0:.1f}s\n")

    # ---- Event identification ----
    batch_timestamp = int(time.time())
    batch_ts_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(batch_timestamp))

    production_version = args.production_version
    run_id = args.run_id
    event_id_offset = args.event_id_offset
    edepsim_event_table = None

    if args.edepsim_ids:
        edepsim_event_table = _read_edepsim_event_table(args.data)
        if edepsim_event_table is not None:
            file_run_id = int(edepsim_event_table[0]['run_id'])
            if run_id is None and file_run_id != 0:
                run_id = file_run_id
            print(f'  Edepsim IDs:   ON (file run_id={file_run_id}, '
                  f'event_id range {int(edepsim_event_table["event_id"].min())}'
                  f'-{int(edepsim_event_table["event_id"].max())})')
        else:
            print('  WARNING: --edepsim-ids set but event/geant4 not found; '
                  'falling back to offset mode')

    # Auto-parse from path if not explicitly set
    if production_version is None or run_id is None or event_id_offset is None:
        path_version, path_run_id, path_file_idx = _parse_edepsim_path(
            os.path.abspath(args.data))
        if path_version is not None:
            if production_version is None:
                production_version = path_version
            if run_id is None:
                run_id = path_run_id
            if event_id_offset is None and not args.edepsim_ids:
                event_id_offset = path_file_idx * total_events
                print(f'  Auto offset:   {event_id_offset} '
                      f'(file {path_file_idx} x {total_events} events)')

    if event_id_offset is None:
        event_id_offset = 0

    print(f'  Batch time:    {batch_timestamp} ({batch_ts_str} UTC)')
    if production_version is not None:
        print(f'  Prod version:  {production_version}')
    if run_id is not None:
        print(f'  Run ID:        {run_id}')
    print(f'  Event offset:  {event_id_offset}')

    # ---- Git provenance ----
    git_repo, git_commit, git_dirty = _get_git_info()
    git_info = {}
    if git_repo:
        git_info['git_repo'] = git_repo
    if git_commit:
        git_info['git_commit'] = git_commit
    git_info['git_dirty'] = git_dirty
    print(f'  Git commit:    {git_commit[:12] if git_commit else "unknown"}'
          f'{"  (dirty)" if git_dirty else ""}')

    # ---- Provenance dict (passed to all write_config_* calls) ----
    provenance = {
        'production_version': production_version,
        'run_id': run_id,
        'batch_timestamp': batch_timestamp,
        'git_info': git_info,
    }

    # ---- Save helpers ----
    key = jax.random.PRNGKey(args.seed)
    total_start = time.time()

    num_workers = args.workers
    file_lock = threading.Lock()

    def save_one_event(f_sensor, f_edep, f_hits, item):
        """Save a single event (CSR encode + HDF5 write). Thread-safe."""
        (event_key, response_np, track_hits_raw, deposits,
         source_idx, ev_id) = item

        # CSR encoding (numpy, GIL-free — runs in parallel across workers)
        hits_data = None
        if include_track_hits and f_hits is not None:
            hits_data = {}
            for plane_key, raw in track_hits_raw.items():
                if not isinstance(plane_key, tuple):
                    continue
                vol_idx, plane_idx = plane_key
                sk, tk, gid, ch, count, _ = raw
                if cfg.volumes[vol_idx].readout_type == 'pixel':
                    num_pz = cfg.volumes[vol_idx].pixel_shape[1]
                    hits_data[plane_key] = encode_correspondence_csr_pixel(
                        sk, tk, gid, ch, count, num_pz,
                        threshold=args.hits_threshold)
                else:
                    pk = sk * cfg.num_time_steps + tk
                    hits_data[plane_key] = encode_correspondence_csr(
                        pk, gid, ch, count, cfg.num_time_steps,
                        threshold=args.hits_threshold)

        # HDF5 write (serialized through file lock)
        with file_lock:
            save_event_sensor(f_sensor, event_key, response_np, threshold_adc,
                            source_idx, deposits, cfg=cfg,
                            digitized=include_digitize, event_id=ev_id)
            save_event_edep(f_edep, event_key, deposits, source_idx, cfg=cfg,
                            event_id=ev_id)
            if hits_data is not None and f_hits is not None:
                save_event_hits(f_hits, event_key, hits_data, deposits,
                                source_idx,
                                hits_threshold=args.hits_threshold, cfg=cfg,
                                event_id=ev_id)

    def save_worker(f_sensor, f_edep, f_hits, save_queue):
        """Worker thread: pull items from queue, encode + save."""
        while True:
            item = save_queue.get()
            if item is None:
                break
            save_one_event(f_sensor, f_edep, f_hits, item)
            save_queue.task_done()

    # ---- Process events ----
    with ParticleStepExtractor(args.data) as extractor:
        for file_idx in range(num_files):
            event_start = file_idx * events_per_file
            event_end = min(event_start + events_per_file, num_events)
            n_in_file = event_end - event_start

            sensor_path = os.path.join(sensor_dir,
                f'{dataset_name}_sensor_{file_idx:04d}.h5')
            edep_path = os.path.join(edep_dir,
                f'{dataset_name}_edep_{file_idx:04d}.h5')
            hits_path = os.path.join(hits_dir,
                f'{dataset_name}_hits_{file_idx:04d}.h5') if hits_dir else None

            print(f'File {file_idx:04d}: events {event_start}–{event_end-1} '
                  f'({n_in_file} events)')

            with h5py.File(sensor_path, 'w') as f_sensor, \
                 h5py.File(edep_path, 'w') as f_edep:

                f_hits_ctx = h5py.File(hits_path, 'w') if hits_path else None
                try:
                    write_config_sensor(
                        f_sensor, cfg, params, simulator.recomb_model,
                        dataset_name, file_idx, args.data,
                        n_in_file, event_start, threshold_adc,
                        digitization_config=dig_config,
                        provenance=provenance)
                    write_config_edep(
                        f_edep, cfg, dataset_name, file_idx, args.data,
                        n_in_file, event_start,
                        args.group_size, args.gap_threshold,
                        provenance=provenance)
                    if f_hits_ctx:
                        write_config_hits(
                            f_hits_ctx, cfg, dataset_name, file_idx, args.data,
                            n_in_file, event_start,
                            args.group_size, args.gap_threshold,
                            provenance=provenance)

                    # Start workers (if threaded)
                    save_queue = None
                    workers = []
                    if num_workers > 0:
                        save_queue = queue.Queue(maxsize=num_workers + 2)
                        for w in range(num_workers):
                            t = threading.Thread(
                                target=save_worker,
                                args=(f_sensor, f_edep, f_hits_ctx, save_queue))
                            t.daemon = True
                            t.start()
                            workers.append(t)

                    for idx in range(event_start, event_end):
                        key, subkey = jax.random.split(key)
                        local_idx = idx - event_start
                        event_key = f'event_{local_idx:03d}'

                        # Compute event_id
                        if edepsim_event_table is not None:
                            event_id = int(edepsim_event_table[idx]['event_id'])
                        else:
                            event_id = event_id_offset + idx

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
                            if isinstance(arr, dict):
                                for a in arr.values():
                                    jax.block_until_ready(a)
                            elif isinstance(arr, tuple):
                                jax.block_until_ready(arr[0])
                            else:
                                jax.block_until_ready(arr)
                        t_sim = time.time() - t_sim

                        # Convert all formats → sparse before saving
                        from tools.output import to_sparse
                        response_signals = to_sparse(
                            response_signals, cfg, threshold_adc=threshold_adc)

                        # GPU → CPU transfer for signals
                        response_np = {}
                        for k, v in response_signals.items():
                            if isinstance(v, dict):
                                response_np[k] = {fk: np.asarray(fv) for fk, fv in v.items()}
                            else:
                                response_np[k] = np.asarray(v)

                        item = (event_key, response_np, track_hits, deposits,
                                idx, event_id)

                        # Save (serial or queued)
                        t_save = time.time()
                        if num_workers > 0:
                            save_queue.put(item)
                        else:
                            save_one_event(f_sensor, f_edep, f_hits_ctx, item)
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
                    if f_hits_ctx:
                        f_hits_ctx.close()

            # Print file sizes
            sensor_mb = os.path.getsize(sensor_path) / (1024 * 1024)
            edep_mb = os.path.getsize(edep_path) / (1024 * 1024)
            print(f'  → sensor: {sensor_mb:.1f} MB, edep: {edep_mb:.1f} MB', end='')
            if hits_path and os.path.exists(hits_path):
                hits_mb = os.path.getsize(hits_path) / (1024 * 1024)
                print(f', hits: {hits_mb:.1f} MB')
            else:
                print()
            print()

    total_elapsed = time.time() - total_start
    print(f'{"=" * 60}')
    print(f'  Done. {num_events} events in {total_elapsed:.1f}s')
    print(f'  Average: {total_elapsed/num_events:.2f}s/event')
    print(f'  Files:   {num_files} × 3 in {args.outdir}/{{sensor,edep,hits}}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
