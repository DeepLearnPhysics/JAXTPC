"""
Batch simulation: run events and save to structured HDF5 files.

Produces three file types per batch (canonical names; see
docs/DATASET_DESIGN.md in particle-imaging-models):
    {dataset}_sensor_{NNNN}.h5  — sparse thresholded raw readout
    {dataset}_step_{NNNN}.h5    — 3D truth energy deposits (per-volume)
    {dataset}_hits_{NNNN}.h5    — per-particle charge attribution at sensor elements

See README.md for pipeline details, output schema, and threading architecture.

Usage (from project root):
    python3 production/run_batch.py
    python3 production/run_batch.py --data mpvmpr_20.h5 --dataset mpvmpr --threshold-adc 5.0
    python3 production/run_batch.py --events 100 --events-per-file 50
    python3 production/run_batch.py --no-track-hits
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
import gc
import threading
import queue
import traceback
from collections import defaultdict
from functools import partial

# Per-event save-phase profiling (off by default). Enable with
# JAXTPC_PROFILE_SAVE=1 to print encode / lockwait / write-sensor / write-step /
# write-hits times from every save worker, plus queue depth from the main loop.
_PROFILE_SAVE = os.environ.get('JAXTPC_PROFILE_SAVE') == '1'

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
    write_config_sensor, write_config_step, write_config_hits,
    save_event_sensor, save_event_step, save_event_hits,
    encode_correspondence_csr, encode_correspondence_csr_pixel,
    set_codec,
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
# MULTI-FILE / SHARD / OVERFLOW-LOG HELPERS
# =============================================================================

def _resolve_source_files(paths_arg):
    """Expand a list of paths (files, dirs, globs) into a sorted list of .h5 files."""
    if isinstance(paths_arg, str):
        paths_arg = [paths_arg]
    out = []
    for p in paths_arg:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, '*.h5'))))
        elif any(c in p for c in '*?['):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    # de-dup while preserving order
    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _apply_subset(files, shard_id, num_shards, file_range):
    """Filter file list by --shard-id/--num-shards (round-robin) then --file-range."""
    if num_shards > 1:
        files = [f for i, f in enumerate(files) if i % num_shards == shard_id]
    if file_range:
        # Python slice: "start:stop" or "start:stop:step"
        parts = file_range.split(':')
        sl = slice(*(int(p) if p else None for p in parts))
        files = files[sl]
    return files


def _init_overflow_log(path):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'source_path', 'src_file_idx', 'event_idx',
                'event_id', 'n_deposits', 'error_type', 'error_message',
            ])


def _log_overflow(path, source_path, src_file_idx, event_idx, event_id,
                  n_deposits, error_type, message):
    """Append one row to the overflow CSV."""
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([
            int(time.time()), source_path,
            src_file_idx if src_file_idx is not None else -1,
            event_idx, event_id, n_deposits,
            error_type, message[:500].replace('\n', ' '),
        ])


def _classify_runtime_error(message):
    m = message.lower()
    if 'deposits >' in m or 'total_pad' in m:
        return 'total_pad_overflow'
    if 'maxg overflow' in m or 'maxg=' in m:
        return 'maxg_overflow'
    if 'track_hits overflow' in m or 'max_keys' in m:
        return 'max_keys_overflow'
    if 'bucket overflow' in m or 'max_buckets' in m or 'max_active_buckets' in m:
        return 'max_buckets_overflow'
    return 'runtime_error'


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Batch TPC simulation (v2)')
    parser.add_argument('--data', nargs='+', default=['mpvmpr_20.h5'],
                        help='Input HDF5 file(s), directory(ies), or glob(s). '
                             'Multiple sources processed sequentially in one process.')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='This shard index, 0-based (default: 0). Combined with '
                             '--num-shards selects every Nth file round-robin.')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Total number of shards to split files across (default: 1).')
    parser.add_argument('--file-range', default=None,
                        help='Python slice "start:stop[:step]" into the resolved file list, '
                             'applied AFTER --shard-id/--num-shards selection.')
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
                        help='Track hits in-JIT pruning threshold. Units depend on '
                             'readout: ENC for wire (kernel is dimensionless pre-electronics), '
                             'ADC for pixel (kernel bakes in chip gain). Default: 1.0.')
    # Box (group-as-bucket) track-hits path — production default. --no-box uses
    # the sort-merge fallback. maxg comes from profiler.setup_production; the
    # per-group box dims are derived analytically by the simulator from the
    # group definition (group_size/gap_threshold) + geometry — no longer set here.
    parser.add_argument('--no-box', action='store_true',
                        help='Use the sort-merge track-hits fallback instead of the box path')
    parser.add_argument('--maxg', type=int, default=200_000,
                        help='Box path: max group-bucket capacity per event-volume')
    parser.add_argument('--maxg-medium', type=int, default=None,
                        help='Tiered mode: medium-tier maxg (e.g. 180000). When set, '
                             'builds two simulators and routes per-event on n_groups. '
                             'Events with max(n_groups) < this use the faster medium '
                             'sim; the rest use --maxg.')
    parser.add_argument('--distortion', default=None,
                        help='Path to a drift-field distortion file (trained SIREN/distortion field)')
    # Grouping
    parser.add_argument('--group-size', type=int, default=5)
    parser.add_argument('--gap-threshold', type=float, default=5.0,
                        help='Gap threshold in mm for group splitting')
    parser.add_argument('--hits-threshold', type=float, default=1.0,
                        help='Charge threshold for per-particle hits at CSR encode '
                             '(YAML key: corr_threshold). Units: ENC for wire (signed, '
                             'chs > t), ADC for pixel (|chs| > t, bipolar). Readout-aware '
                             'default: 1.0 ADC (pixel), 25.0 ENC (wire).')
    parser.add_argument('--total-pad', type=int, default=500_000)
    parser.add_argument('--response-chunk', type=int, default=50_000,
                        help='Deposits per fori_loop batch (must divide total-pad)')
    parser.add_argument('--bucketed', action='store_true', help='Use bucketed accumulation')
    parser.add_argument('--max-buckets', type=int, default=1000,
                        help='Max active buckets per plane (bucketed mode)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of save worker threads (0=serial, default: 2)')
    parser.add_argument('--per-worker-files', action='store_true',
                        help='Each save worker writes its OWN output files '
                             '(suffixed _wNN) so HDF5 writes parallelize instead '
                             'of serializing on one file lock. Needed to make the '
                             'pipeline sim-bound on one GPU with many CPU workers. '
                             'Output becomes N files per source (read as a set).')
    parser.add_argument('--read-workers', type=int, default=1,
                        help='Parallel reader/prefetch threads, each with its own '
                             'file handle, splitting events round-robin. >1 keeps '
                             'load (HDF5 read + build_deposit_data grouping) off the '
                             'critical path so the loop stays sim-bound on dense data.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Resume mode: skip any output file already marked '
                             'done (a per-file marker under {outdir}/.done/, '
                             'written only AFTER its sensor/step/hits are fully '
                             'closed). Safe to re-run any range/subset — finished '
                             'files are not recomputed and nothing is overwritten. '
                             'A file that crashed mid-write has no marker and is '
                             'redone.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--codec', default='blosc-zstd',
                        help='Output compression codec: gzip, gzip-1, lzf, '
                             'lz4, zstd, blosc-lz4, blosc-zstd (default). '
                             'blosc-zstd is smaller than gzip AND ~2.3x faster '
                             'to read; blosc-lz4 is ~4x faster (+19%% disk). '
                             'Non-gzip needs hdf5plugin (read + write).')
    parser.add_argument('--production-config', default=None,
                        help='Load optimized params from profiler config YAML')
    # Event identification
    parser.add_argument('--production-version', default=None,
                        help='Production version string (e.g. test_00_00_01). '
                             'Auto-parsed from --data path if not set.')
    parser.add_argument('--run-id', type=int, default=None,
                        help='Step-sim run ID (e.g. 26628546). '
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

    # Production config fills any field NOT explicitly given on the CLI. Precedence:
    # explicit CLI args > production config > argparse defaults. `explicit` is the
    # set of arg dests the user actually typed (matched against sys.argv).
    explicit = {a.dest for a in parser._actions
                if any(opt in sys.argv[1:] for opt in a.option_strings)}
    prod_cfg = {}
    if args.production_config:
        from profiler.production_config import load_config, apply_to_args
        prod_cfg = load_config(args.production_config)
        apply_to_args(args, prod_cfg, explicit)
        print(f'  Loaded production config: {args.production_config}')

    # Output compression codec (default blosc-zstd; gzip fallback in save.py
    # if hdf5plugin is unavailable).
    set_codec(args.codec)

    include_intrinsic_noise = args.intrinsic
    include_coherent_noise = args.coherent
    include_electronics = args.electronics
    include_digitize = not args.no_digitize
    include_track_hits = not args.no_track_hits
    include_distortion = args.distortion is not None
    events_per_file = args.events_per_file
    threshold_adc = args.threshold_adc
    dataset_name = args.dataset

    # ---- Resolve source files and apply shard/range subset ----
    source_files_all = _resolve_source_files(args.data)
    if not source_files_all:
        print(f'No source HDF5 files found from: {args.data}')
        return
    source_files = _apply_subset(
        source_files_all, args.shard_id, args.num_shards, args.file_range)
    if not source_files:
        print(f'No files selected for shard {args.shard_id}/{args.num_shards} '
              f'(range={args.file_range}). Total available: {len(source_files_all)}.')
        return

    # Detect readout type early for pixel-specific defaults
    detector_config = generate_detector(args.config)
    readout_type = detector_config['volumes'][0].get('readout', {}).get('type', 'wire')

    # Pixel readout: track hits are mandatory (single response pass)
    if readout_type == 'pixel' and not include_track_hits:
        print('  NOTE: pixel readout requires track hits — enabling')
        include_track_hits = True

    # Top-level output dirs (per-source subfolders by run_id added inside loop)
    base_sensor_dir = os.path.join(args.outdir, 'sensor')
    base_step_dir = os.path.join(args.outdir, 'step')
    base_hits_dir = os.path.join(args.outdir, 'hits') if include_track_hits else None
    logs_dir = os.path.join(args.outdir, 'logs')
    # Per-file done markers (resume): written only after an output file's HDF5
    # is fully closed, so they are a crash-safe "done" signal (output files
    # themselves exist empty from the moment processing starts).
    done_dir = os.path.join(args.outdir, '.done')
    for d in [base_sensor_dir, base_step_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)
    if base_hits_dir:
        os.makedirs(base_hits_dir, exist_ok=True)
    if args.skip_existing:
        os.makedirs(done_dir, exist_ok=True)

    # Overflow log + per-shard summary path
    overflow_csv = os.path.join(
        logs_dir, f'overflow_events_shard{args.shard_id:03d}.csv')
    _init_overflow_log(overflow_csv)
    summary_path = os.path.join(
        logs_dir, f'summary_shard{args.shard_id:03d}.txt')

    print('=' * 60)
    print(' JAXTPC Batch Simulation v2')
    print('=' * 60)
    print(f'  Sources:       {len(source_files)} / {len(source_files_all)} '
          f'(shard {args.shard_id}/{args.num_shards}'
          f'{", range " + args.file_range if args.file_range else ""})')
    print(f'  Dataset:       {dataset_name}')
    print(f'  Events/file:   {events_per_file}')
    print(f'  Threshold:     {threshold_adc} ADC')
    print(f'  Intrinsic:     {"ON" if include_intrinsic_noise else "OFF"}')
    print(f'  Coherent:      {"ON" if include_coherent_noise else "OFF"}')
    print(f'  Electronics:   {"ON" if include_electronics else "OFF"}')
    print(f'  Digitization:  {"ON" if include_digitize else "OFF"}')
    print(f'  Distortion:    {args.distortion if include_distortion else "OFF"}')
    print(f'  Track hits:    {"ON" if include_track_hits else "OFF"}')
    print(f'  Group size:    {args.group_size}')
    print(f'  Total pad:     {args.total_pad:,}')
    if readout_type == 'wire':
        print(f'  Bucketed:      {"ON (max_buckets=" + str(args.max_buckets) + ")" if args.bucketed else "OFF"}')
    print(f'  Workers:       {args.workers} {"(serial)" if args.workers == 0 else "(threaded)"}')
    print(f'  Device:        {jax.devices()[0]}')
    print(f'  Output base:   {args.outdir}/')
    print(f'  Overflow log:  {overflow_csv}')
    print(f'  Readout:       {readout_type}')
    print()

    # Pixel default: smaller hits_chunk for optimal merge performance. Apply only
    # when neither the CLI nor the production config set hits_chunk (so an explicit
    # value — even one that equals the argparse default — is honored).
    hits_chunk = args.hits_chunk
    if (readout_type == 'pixel' and 'hits_chunk' not in explicit
            and 'hits_chunk' not in prod_cfg):
        hits_chunk = 5_000

    # corr_threshold (hits charge cut) is readout-specific, not one global
    # default: pixel is ADC (the 1.0 default is its ~1-ADC floor), wire is ENC
    # (~25 e- floor). Apply the wire floor only when neither the CLI
    # (--hits-threshold) nor the config (corr_threshold) set it.
    if (readout_type == 'wire' and 'hits_threshold' not in explicit
            and 'corr_threshold' not in prod_cfg):
        args.hits_threshold = 25.0
        print('  NOTE: wire readout — corr_threshold default → 25.0 ENC')

    track_config = create_track_hits_config(
        max_keys=args.max_keys, hits_chunk_size=hits_chunk,
        inter_thresh=args.inter_thresh,
        box_enabled=not args.no_box, maxg=args.maxg,
    ) if include_track_hits else None

    use_tiered = (args.maxg_medium is not None
                  and include_track_hits and not args.no_box)

    if include_track_hits:
        if args.no_box:
            print('  Track hits:    sort-merge (fallback)')
        elif use_tiered:
            print(f'  Track hits:    box TIERED (medium={args.maxg_medium:,}, '
                  f'high={args.maxg:,}; dims derived analytically by sim)')
        else:
            print(f'  Track hits:    box (maxg={args.maxg:,}; '
                  f'dims derived analytically by sim)')

    sim_medium = None
    sim_high = None

    def _build_sim(tc):
        return DetectorSimulator(
            detector_config,
            track_config=tc,
            total_pad=args.total_pad,
            response_chunk_size=args.response_chunk,
            use_bucketed=args.bucketed if readout_type == 'wire' else False,
            max_active_buckets=args.max_buckets,
            include_intrinsic_noise=include_intrinsic_noise,
            include_coherent_noise=include_coherent_noise,
            include_electronics=include_electronics,
            include_track_hits=include_track_hits,
            include_digitize=include_digitize,
            distortion=args.distortion,
            group_size=args.group_size,
            gap_threshold_mm=args.gap_threshold,
        )

    t_create = time.time()
    if use_tiered:
        track_config_medium = create_track_hits_config(
            max_keys=args.max_keys, hits_chunk_size=hits_chunk,
            inter_thresh=args.inter_thresh,
            box_enabled=True, maxg=args.maxg_medium,
        )
        sim_medium = _build_sim(track_config_medium)
        sim_high = _build_sim(track_config)
        simulator = sim_medium
    else:
        simulator = _build_sim(track_config)
    t_create = time.time() - t_create

    cfg = simulator.config
    params = simulator.default_sim_params
    dig_config = getattr(simulator, 'digitization_config', None)

    t_warmup = time.time()
    simulator.warm_up()
    if use_tiered:
        sim_high.warm_up()
    t_warmup = time.time() - t_warmup

    print(f'\n  Simulator creation: {t_create:.1f}s')
    print(f'  JIT warmup:        {t_warmup:.1f}s'
          f'{" (2 tiers)" if use_tiered else ""}')

    # ---- Real-data warmup (uses first source file) ----
    print("  Real-data warmup...", end='', flush=True)
    t0 = time.time()
    try:
        warmup_dep = load_deposit(
            ParticleStepExtractor(source_files[0]), 0, cfg,
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
    except RuntimeError as e:
        print(f'\n  Warmup event hit a limit: {str(e)[:140]}')
        print('  Continuing — affected events will be logged and skipped.')
    gc.collect()
    print(f" {time.time() - t0:.1f}s\n")

    # ---- Batch timestamp + git provenance (shared across all sources) ----
    batch_timestamp = int(time.time())
    batch_ts_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(batch_timestamp))

    git_repo, git_commit, git_dirty = _get_git_info()
    git_info = {}
    if git_repo:
        git_info['git_repo'] = git_repo
    if git_commit:
        git_info['git_commit'] = git_commit
    git_info['git_dirty'] = git_dirty

    print(f'  Batch time:    {batch_timestamp} ({batch_ts_str} UTC)')
    print(f'  Git commit:    {git_commit[:12] if git_commit else "unknown"}'
          f'{"  (dirty)" if git_dirty else ""}')

    # ---- Save helpers ----
    key = jax.random.PRNGKey(args.seed)
    total_start = time.time()

    num_workers = args.workers
    # Per-file locks (previously: single file_lock for all three). Splitting
    # lets one worker write sensor while another writes step/hits — necessary
    # because sen+step+hits write under-lock totals ~2.7s/event and main
    # produces at ~3.5s/event, so a single lock was the throughput ceiling.
    # Per-file locks raise the ceiling from 1/(sen+step+hits) to 1/max(...).
    sen_lock = threading.Lock()
    step_lock = threading.Lock()
    hits_lock = threading.Lock()

    def save_one_event(f_sensor, f_step, f_hits, item, locks):
        """Save a single event (CSR encode + HDF5 write). Thread-safe.

        `locks` is a (sensor, step, hits) lock trio. Shared-file mode passes the
        same three locks to every worker (writes serialize per file); per-worker
        mode passes each worker its own (uncontended) trio so writes parallelize.
        """
        # to_sparse runs here (off the main thread), pulling the dense device
        # result to host and thresholding it.
        (event_key, response_payload, track_hits_raw, deposits,
         source_idx, ev_id) = item
        if _PROFILE_SAVE:
            _tid = threading.get_ident() & 0xFFFF
            _t_enter = time.time()
        if _PROFILE_SAVE: _t0 = time.time()
        from tools.output import to_sparse
        _sparse = to_sparse(response_payload, cfg, threshold_adc=threshold_adc)
        response_np = {}
        for _k, _v in _sparse.items():
            if isinstance(_v, dict):
                response_np[_k] = {fk: np.asarray(fv) for fk, fv in _v.items()}
            else:
                response_np[_k] = np.asarray(_v)
        if _PROFILE_SAVE: _t_post = time.time() - _t0

        # CSR encoding (numpy, GIL-free — runs in parallel across workers)
        if _PROFILE_SAVE: _t0 = time.time()
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
        if _PROFILE_SAVE: _t_enc = time.time() - _t0

        # HDF5 writes — each file under its own lock so different files can
        # overlap across workers. No deadlock risk: every worker takes the
        # locks in the same order (sen → step → hits) and holds at most one
        # at a time.
        if _PROFILE_SAVE: _t0 = time.time()
        with locks[0]:
            if _PROFILE_SAVE:
                _lw_sen = time.time() - _t0; _t0 = time.time()
            save_event_sensor(f_sensor, event_key, response_np, threshold_adc,
                            source_idx, deposits, cfg=cfg,
                            digitized=include_digitize, event_id=ev_id)
            if _PROFILE_SAVE: _t_sen = time.time() - _t0

        if _PROFILE_SAVE: _t0 = time.time()
        with locks[1]:
            if _PROFILE_SAVE:
                _lw_step = time.time() - _t0; _t0 = time.time()
            save_event_step(f_step, event_key, deposits, source_idx, cfg=cfg,
                            event_id=ev_id)
            if _PROFILE_SAVE: _t_step = time.time() - _t0

        if hits_data is not None and f_hits is not None:
            if _PROFILE_SAVE: _t0 = time.time()
            with locks[2]:
                if _PROFILE_SAVE:
                    _lw_hits = time.time() - _t0; _t0 = time.time()
                save_event_hits(f_hits, event_key, hits_data, deposits,
                                source_idx,
                                hits_threshold=args.hits_threshold, cfg=cfg,
                                event_id=ev_id)
                if _PROFILE_SAVE: _t_hits = time.time() - _t0
        elif _PROFILE_SAVE:
            _lw_hits = 0.0; _t_hits = 0.0

        if _PROFILE_SAVE:
            _t_total = time.time() - _t_enter
            _lw_total = _lw_sen + _lw_step + _lw_hits
            print(f'    [SAVE W{_tid:04x}] {event_key}  post={_t_post:.2f}s  '
                  f'enc={_t_enc:.2f}s  '
                  f'lockwait={_lw_total:.2f}s (sen={_lw_sen:.2f} '
                  f'step={_lw_step:.2f} hits={_lw_hits:.2f})  '
                  f'sen={_t_sen:.2f}s step={_t_step:.2f}s hits={_t_hits:.2f}s  '
                  f'total={_t_total:.2f}s', flush=True)

    def save_worker(f_sensor, f_step, f_hits, save_queue, locks):
        """Worker thread: pull items from queue, encode + save."""
        while True:
            item = save_queue.get()
            if item is None:
                break
            save_one_event(f_sensor, f_step, f_hits, item, locks)
            save_queue.task_done()

    # ---- Process events across all source files ----
    total_processed = 0
    total_skipped = 0
    files_skipped_done = 0
    tier_medium_count = 0
    tier_high_count = 0
    by_error = defaultdict(int)

    for src_pos, source_path in enumerate(source_files):
        print('#' * 60)
        print(f' Source {src_pos + 1}/{len(source_files)}: {source_path}')
        print('#' * 60)

        # Parse run_id / file_idx / production_version from path
        prod_version, run_id, src_file_idx = _parse_edepsim_path(
            os.path.abspath(source_path))
        if args.production_version is not None:
            prod_version = args.production_version
        if args.run_id is not None:
            run_id = args.run_id

        # Number of events in this source
        try:
            total_events = get_num_events(source_path)
        except Exception as e:
            print(f'  ERROR opening {source_path}: {e}')
            _log_overflow(overflow_csv, source_path, src_file_idx, -1, -1, -1,
                          'file_open_error', str(e))
            continue
        num_events = min(args.events, total_events) if args.events else total_events
        num_out_files = (num_events + events_per_file - 1) // events_per_file

        # Per-source event_id_offset / edepsim table
        edepsim_event_table = None
        event_id_offset = args.event_id_offset
        if args.edepsim_ids:
            edepsim_event_table = _read_edepsim_event_table(source_path)
            if edepsim_event_table is not None:
                file_run_id = int(edepsim_event_table[0]['run_id'])
                if run_id is None and file_run_id != 0:
                    run_id = file_run_id
        if event_id_offset is None:
            if src_file_idx is not None and not args.edepsim_ids:
                event_id_offset = src_file_idx * total_events
            else:
                event_id_offset = 0

        # Per-run output subdirs
        run_subdir = (f'run_{run_id:010d}' if run_id is not None
                      else 'run_unknown')
        sensor_dir = os.path.join(base_sensor_dir, run_subdir)
        step_dir = os.path.join(base_step_dir, run_subdir)
        hits_dir = os.path.join(base_hits_dir, run_subdir) if base_hits_dir else None
        os.makedirs(sensor_dir, exist_ok=True)
        os.makedirs(step_dir, exist_ok=True)
        if hits_dir:
            os.makedirs(hits_dir, exist_ok=True)
        done_run_dir = os.path.join(done_dir, run_subdir)
        if args.skip_existing:
            os.makedirs(done_run_dir, exist_ok=True)

        provenance = {
            'production_version': prod_version,
            'run_id': run_id,
            'batch_timestamp': batch_timestamp,
            'git_info': git_info,
        }

        print(f'  Run ID:        {run_id}')
        print(f'  Events:        {num_events}/{total_events} '
              f'(offset={event_id_offset})')
        print(f'  Output dirs:   .../{{sensor,step,hits}}/{run_subdir}/')

        # Loop output files within this source (usually 1, depends on events_per_file)
        try:
            with ParticleStepExtractor(source_path) as extractor:
                for out_file_idx in range(num_out_files):
                    event_start = out_file_idx * events_per_file
                    event_end = min(event_start + events_per_file, num_events)
                    n_in_file = event_end - event_start

                    # Output naming: src_file_idx primary; sub-index if split
                    if num_out_files == 1 and src_file_idx is not None:
                        suffix = f'{src_file_idx:04d}'
                    elif src_file_idx is not None:
                        suffix = f'{src_file_idx:04d}_{out_file_idx:02d}'
                    else:
                        suffix = f'{src_pos:04d}_{out_file_idx:02d}'

                    sensor_path = os.path.join(
                        sensor_dir, f'{dataset_name}_sensor_{suffix}.h5')
                    step_path = os.path.join(
                        step_dir, f'{dataset_name}_step_{suffix}.h5')
                    hits_path = (os.path.join(
                        hits_dir, f'{dataset_name}_hits_{suffix}.h5')
                        if hits_dir else None)
                    done_marker = os.path.join(
                        done_run_dir, f'{dataset_name}_{suffix}.done')

                    # Resume: skip output files already finished in a prior run.
                    if args.skip_existing and os.path.exists(done_marker):
                        print(f'  Output {suffix}: SKIP (done)')
                        files_skipped_done += 1
                        continue

                    print(f'  Output {suffix}: events {event_start}–{event_end-1} '
                          f'({n_in_file} events)')

                    cfg_file_idx = (src_file_idx if src_file_idx is not None
                                    else src_pos)

                    with h5py.File(sensor_path, 'w') as f_sensor, \
                         h5py.File(step_path, 'w') as f_step:

                        f_hits_ctx = h5py.File(hits_path, 'w') if hits_path else None
                        # --per-worker-files: each save worker gets its OWN output
                        # file set (suffixed _wNN) so HDF5 writes run on separate
                        # cores instead of serializing on one file's lock. The win
                        # case is one GPU + many CPU workers (the GPU is the ceiling;
                        # the save just has to keep up). Worker 0 keeps the canonical
                        # name; workers 1..N-1 get extra files (read as a set).
                        per_worker = args.per_worker_files and num_workers > 1
                        extra_sets = []
                        try:
                            def _write_headers(fs, fe, fh):
                                write_config_sensor(
                                    fs, cfg, params, simulator.recomb_model,
                                    dataset_name, cfg_file_idx, source_path,
                                    n_in_file, event_start, threshold_adc,
                                    digitization_config=dig_config,
                                    provenance=provenance)
                                write_config_step(
                                    fe, cfg, dataset_name, cfg_file_idx,
                                    source_path, n_in_file, event_start,
                                    args.group_size, args.gap_threshold,
                                    provenance=provenance)
                                if fh:
                                    write_config_hits(
                                        fh, cfg, dataset_name, cfg_file_idx,
                                        source_path, n_in_file, event_start,
                                        args.group_size, args.gap_threshold,
                                        provenance=provenance)

                            _write_headers(f_sensor, f_step, f_hits_ctx)
                            wsets = [(f_sensor, f_step, f_hits_ctx)]
                            if per_worker:
                                for w in range(1, num_workers):
                                    fs = h5py.File(sensor_path[:-3] + f'_w{w:02d}.h5', 'w')
                                    fe = h5py.File(step_path[:-3] + f'_w{w:02d}.h5', 'w')
                                    fh = (h5py.File(hits_path[:-3] + f'_w{w:02d}.h5', 'w')
                                          if hits_path else None)
                                    _write_headers(fs, fe, fh)
                                    extra_sets.append((fs, fe, fh))
                                    wsets.append((fs, fe, fh))

                            save_queue = None
                            workers = []
                            if num_workers > 0:
                                save_queue = queue.Queue(maxsize=num_workers + 2)
                                for w in range(num_workers):
                                    fs, fe, fh = wsets[w] if per_worker else wsets[0]
                                    wl = ((threading.Lock(), threading.Lock(),
                                           threading.Lock()) if per_worker
                                          else (sen_lock, step_lock, hits_lock))
                                    t = threading.Thread(
                                        target=save_worker,
                                        args=(fs, fe, fh, save_queue, wl))
                                    t.daemon = True
                                    t.start()
                                    workers.append(t)

                            # ── reader prefetch (parallel) ──────────────────
                            # N reader threads, each with its OWN extractor (h5py
                            # handle), split events round-robin. The heavy part
                            # (build_deposit_data grouping) is numpy and releases
                            # the GIL, so readers parallelize — keeps the main
                            # loop from going load-bound on dense events.
                            n_read = max(1, args.read_workers)
                            read_q = queue.Queue(maxsize=n_read + 2)

                            def _read_loop(ex, rank):
                                for r_idx in range(event_start + rank, event_end, n_read):
                                    try:
                                        t0 = time.time()
                                        d = load_deposit(
                                            ex, r_idx, cfg,
                                            args.group_size, args.gap_threshold)
                                        read_q.put((r_idx, d, time.time() - t0))
                                    except Exception as exc:
                                        read_q.put((r_idx, exc, 0.0))

                            def _reader(rank):
                                if rank == 0:
                                    _read_loop(extractor, rank)
                                else:
                                    with ParticleStepExtractor(source_path) as ex:
                                        _read_loop(ex, rank)
                                read_q.put(None)

                            reader_threads = []
                            for _rk in range(n_read):
                                rt = threading.Thread(target=_reader, args=(_rk,),
                                                       daemon=True)
                                rt.start()
                                reader_threads.append(rt)

                            n_readers_done = 0
                            while True:
                                t_wait = time.time()
                                item = read_q.get()
                                t_wait = time.time() - t_wait
                                if item is None:
                                    n_readers_done += 1
                                    if n_readers_done >= n_read:
                                        break
                                    continue
                                idx, deposits_or_exc, t_load_work = item
                                key, subkey = jax.random.split(key)
                                local_idx = idx - event_start
                                event_key = f'event_{local_idx:03d}'

                                if edepsim_event_table is not None:
                                    event_id = int(edepsim_event_table[idx]['event_id'])
                                else:
                                    event_id = event_id_offset + idx

                                n_deposits = -1
                                try:
                                    if isinstance(deposits_or_exc, Exception):
                                        raise deposits_or_exc
                                    deposits = deposits_or_exc
                                    n_deposits = sum(v.n_actual for v in deposits.volumes)

                                    if use_tiered:
                                        max_ng = max(len(g2t) for g2t in deposits.group_to_track)
                                        active_sim = sim_high if max_ng > args.maxg_medium else sim_medium
                                        tier_label = 'H' if max_ng > args.maxg_medium else 'M'
                                    else:
                                        active_sim = simulator
                                        tier_label = ''

                                    t_sim = time.time()
                                    response_signals, track_hits, deposits = \
                                        active_sim.process_event(deposits, key=subkey)
                                    # block here = per-event sync; the next sim
                                    # then waits behind the workers' transfers.
                                    for arr in response_signals.values():
                                        if isinstance(arr, dict):
                                            for a in arr.values():
                                                jax.block_until_ready(a)
                                        elif isinstance(arr, tuple):
                                            jax.block_until_ready(arr[0])
                                        else:
                                            jax.block_until_ready(arr)
                                    t_sim = time.time() - t_sim

                                    # Hand off the raw device result; the workers do
                                    # to_sparse (the dense host pull) off the main thread.
                                    item = (event_key, response_signals, track_hits,
                                            deposits, idx, event_id)

                                    t_save = time.time()
                                    if num_workers > 0:
                                        _qd_before = save_queue.qsize() if _PROFILE_SAVE else 0
                                        save_queue.put(item)
                                    else:
                                        _qd_before = 0
                                        save_one_event(
                                            f_sensor, f_step, f_hits_ctx, item,
                                            (sen_lock, step_lock, hits_lock))
                                    t_save = time.time() - t_save

                                    # Wall on main = wait(get) + sim + save(put). Reader's
                                    # load_work runs in parallel; wait≈0 ⇒ prefetch hides it.
                                    t_total = t_wait + t_sim + t_save
                                    _qd_str = (f'  qd_before_put={_qd_before}/{save_queue.maxsize}'
                                               if _PROFILE_SAVE and num_workers > 0 else '')
                                    _tier_str = f'  [{tier_label}]' if tier_label else ''
                                    print(f'    [{local_idx+1:3d}/{n_in_file}] '
                                          f'event {idx:6d}  {n_deposits:6,} deps  '
                                          f'wait={t_wait:.2f}s (load_work={t_load_work:.2f}s)  '
                                          f'sim={t_sim:.2f}s  '
                                          f'save={t_save:.2f}s  '
                                          f'total={t_total:.1f}s{_tier_str}{_qd_str}')
                                    total_processed += 1
                                    if tier_label == 'M':
                                        tier_medium_count += 1
                                    elif tier_label == 'H':
                                        tier_high_count += 1

                                    del response_signals

                                except RuntimeError as e:
                                    err_type = _classify_runtime_error(str(e))
                                    by_error[err_type] += 1
                                    total_skipped += 1
                                    _log_overflow(
                                        overflow_csv, source_path, src_file_idx,
                                        idx, event_id, n_deposits, err_type, str(e))
                                    print(f'    [SKIP {idx:6d}] {err_type}: '
                                          f'{str(e)[:140]}')

                                # Per-event gc.collect() starves the GPU (full heap
                                # scan, growing over the run = the slowdown). Amortize
                                # it: collect every 50 events instead of every event.
                                if (local_idx + 1) % 50 == 0:
                                    gc.collect()

                            for rt in reader_threads:
                                rt.join()

                            if num_workers > 0:
                                for _ in range(num_workers):
                                    save_queue.put(None)
                                for t in workers:
                                    t.join()

                        finally:
                            for fs, fe, fh in extra_sets:
                                fs.close()
                                fe.close()
                                if fh:
                                    fh.close()
                            if f_hits_ctx:
                                f_hits_ctx.close()

                    # File sizes (sum across per-worker shards if any)
                    try:
                        def _tot_mb(p):
                            files = [p] + (glob.glob(p[:-3] + '_w*.h5')
                                           if args.per_worker_files else [])
                            return sum(os.path.getsize(f) for f in files
                                       if os.path.exists(f)) / (1024 * 1024)
                        sensor_mb = _tot_mb(sensor_path)
                        step_mb = _tot_mb(step_path)
                        msg = f'  → sensor: {sensor_mb:.1f} MB, step: {step_mb:.1f} MB'
                        if hits_path and os.path.exists(hits_path):
                            hits_mb = _tot_mb(hits_path)
                            msg += f', hits: {hits_mb:.1f} MB'
                        print(msg)
                    except OSError:
                        pass

                    # Output fully written + closed -> drop the resume marker.
                    if args.skip_existing:
                        with open(done_marker, 'w'):
                            pass
                    print()
        except Exception as e:
            print(f'  ERROR processing {source_path}: {e}')
            traceback.print_exc()
            _log_overflow(overflow_csv, source_path, src_file_idx,
                          -1, -1, -1, 'source_loop_error', str(e))
            continue

    total_elapsed = time.time() - total_start

    # ---- Final summary ----
    print('=' * 60)
    print(f'  Shard {args.shard_id}/{args.num_shards} done in {total_elapsed:.1f}s')
    print(f'  Sources:          {len(source_files)}')
    if args.skip_existing:
        print(f'  Files skipped (done): {files_skipped_done}')
    print(f'  Events processed: {total_processed}')
    print(f'  Events skipped:   {total_skipped}')
    if use_tiered:
        print(f'  Tier medium:      {tier_medium_count}')
        print(f'  Tier high:        {tier_high_count}')
    for err_type, n in sorted(by_error.items()):
        print(f'    {err_type}: {n}')
    if total_processed > 0:
        print(f'  Average:          {total_elapsed/total_processed:.2f}s/event')
    print('=' * 60)

    with open(summary_path, 'w') as f:
        f.write(f'shard_id={args.shard_id}\n')
        f.write(f'num_shards={args.num_shards}\n')
        f.write(f'file_range={args.file_range or ""}\n')
        f.write(f'sources_total={len(source_files_all)}\n')
        f.write(f'sources_processed={len(source_files)}\n')
        f.write(f'files_skipped_done={files_skipped_done}\n')
        f.write(f'events_processed={total_processed}\n')
        f.write(f'events_skipped={total_skipped}\n')
        for err_type, n in sorted(by_error.items()):
            f.write(f'skipped_{err_type}={n}\n')
        f.write(f'elapsed_sec={total_elapsed:.1f}\n')
        f.write(f'overflow_log={overflow_csv}\n')
        f.write(f'batch_timestamp={batch_timestamp}\n')
    print(f'  Summary: {summary_path}')


if __name__ == '__main__':
    main()
