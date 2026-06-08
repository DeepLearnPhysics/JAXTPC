"""
Single-config GPU fit + timing probe for ONE DetectorSimulator configuration.

Runs in its own process (the sweep driver launches one per cell) because JAX has
no peak-memory reset and a device OOM can leave the process unusable. A fresh
process per point gives a clean GPU and a faithful high-water mark.

Measures, for one benchmark event:
  - peak device memory  (XLA_PYTHON_CLIENT_PREALLOCATE=false -> BFC grows on
    demand, so 'completes' == 'fits on this GPU' and peak_bytes_in_use is the
    real high-water mark instead of the 75% preallocation grab)
  - per-event timing     (mean +/- std over N process_event iterations)
  - status               (ok | oom | overflow | disk_full | error)

Emits exactly one machine-readable line to stdout:
    RESULT:{...json...}
so the parent can parse it even amid other simulator chatter.

Usage:
    python3 -m profiler.bench_fit \\
        --config config/cubic_pixel_config.yaml \\
        --data /path/edepsim_000000.h5 --event 3 \\
        --total-pad 1330000 --response-chunk 66500 --hits-chunk 66500 \\
        --max-keys 14000000 --maxg 240000 \\
        --box-bpy 9 --box-bpz 8 --box-bt 83 --n-iter 5
"""
import os
# Must be set before importing jax. preallocate=false => faithful OOM + real peak.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep XLA's PTX/compile temp OFF node-local /lscratch (it filled and crashed a
# prior production run) by pointing TMPDIR at a repo-local dir, unless already set.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if 'TMPDIR' not in os.environ:
    _tmp = os.path.join(_REPO, 'experiments', '_xla_tmp')
    os.makedirs(_tmp, exist_ok=True)
    os.environ['TMPDIR'] = _tmp

import jax
import numpy as np

from tools.geometry import generate_detector
from tools.config import create_track_hits_config
from tools.simulation import DetectorSimulator
from tools.loader import load_event


def _mem():
    """(peak_mb, in_use_mb, limit_mb) for device 0."""
    s = jax.devices()[0].memory_stats() or {}
    return (s.get('peak_bytes_in_use', 0) / 1e6,
            s.get('bytes_in_use', 0) / 1e6,
            s.get('bytes_limit', 0) / 1e6)


def _sync(result):
    """block_until_ready over the response_signals (mirrors run_batch)."""
    for a in result[0].values():
        if isinstance(a, dict):
            for x in a.values():
                jax.block_until_ready(x)
        elif isinstance(a, tuple):
            jax.block_until_ready(a[0])
        else:
            jax.block_until_ready(a)


def _classify(exc):
    """Map an exception to a status string."""
    msg = str(exc)
    low = msg.lower()
    if 'no space left' in low:
        return 'disk_full'
    if 'resource_exhausted' in low or 'out of memory' in low:
        # disk_full already handled above; remaining RESOURCE_EXHAUSTED is GPU OOM
        return 'oom'
    if isinstance(exc, RuntimeError) and (
            'overflow' in low or 'max_keys' in low or 'maxg' in low
            or 'total_pad' in low or 'deposits >' in low):
        return 'overflow'
    return 'error'


def main():
    p = argparse.ArgumentParser(description='Single-config GPU fit + timing probe')
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--event', type=int, default=0)
    p.add_argument('--total-pad', type=int, required=True)
    p.add_argument('--response-chunk', type=int, required=True)
    p.add_argument('--hits-chunk', type=int, required=True)
    p.add_argument('--max-keys', type=int, required=True)
    p.add_argument('--maxg', type=int, default=240000)
    p.add_argument('--box-bpy', type=int, default=9)
    p.add_argument('--box-bpz', type=int, default=8)
    p.add_argument('--box-bt', type=int, default=83)
    p.add_argument('--box-bw', type=int, default=12)
    p.add_argument('--box-btw', type=int, default=27)
    p.add_argument('--inter-thresh', type=float, default=1.0)
    p.add_argument('--no-box', action='store_true')
    p.add_argument('--bucketed', action='store_true')
    p.add_argument('--max-buckets', type=int, default=1000)
    p.add_argument('--no-track-hits', action='store_true')
    p.add_argument('--n-iter', type=int, default=5)
    p.add_argument('--tag', default='')
    args = p.parse_args()

    rec = {
        'tag': args.tag, 'config': args.config, 'event': args.event,
        'total_pad': args.total_pad, 'response_chunk': args.response_chunk,
        'hits_chunk': args.hits_chunk, 'max_keys': args.max_keys,
        'maxg': args.maxg, 'box_bpy': args.box_bpy, 'box_bpz': args.box_bpz,
        'box_bt': args.box_bt, 'box_bw': args.box_bw, 'box_btw': args.box_btw,
        'bucketed': args.bucketed, 'no_box': args.no_box, 'n_iter': args.n_iter,
        'status': 'error', 'error': None,
    }

    def emit():
        print('RESULT:' + json.dumps(rec), flush=True)

    try:
        detector_config = generate_detector(args.config)
        readout = detector_config['volumes'][0].get('readout', {}).get('type', 'wire')
        rec['readout'] = readout

        include_track_hits = (readout == 'pixel') or (not args.no_track_hits)
        track_config = None
        if include_track_hits:
            track_config = create_track_hits_config(
                max_keys=args.max_keys, hits_chunk_size=args.hits_chunk,
                inter_thresh=args.inter_thresh,
                box_enabled=not args.no_box, maxg=args.maxg,
                box_bpy=args.box_bpy, box_bpz=args.box_bpz, box_bt=args.box_bt,
                box_bw=args.box_bw, box_btw=args.box_btw)
        use_bucketed = args.bucketed if readout == 'wire' else False

        t0 = time.perf_counter()
        sim = DetectorSimulator(
            detector_config, track_config=track_config,
            total_pad=args.total_pad, response_chunk_size=args.response_chunk,
            use_bucketed=use_bucketed, max_active_buckets=args.max_buckets,
            include_track_hits=include_track_hits,
            include_digitize=True)            # production default (run_batch)
        sim.warm_up()
        rec['compile_s'] = round(time.perf_counter() - t0, 1)
        pk, inuse, lim = _mem()
        rec['mem_after_warmup_mb'] = round(pk, 1)
        rec['mem_limit_mb'] = round(lim, 1)

        deposits = load_event(args.data, sim.config, event_idx=args.event)
        rec['n_deps'] = int(sum(int(v.n_actual) for v in deposits.volumes))

        key = jax.random.PRNGKey(42)
        _sync(sim.process_event(deposits, key=key))   # real-data warmup

        times = []
        for _ in range(args.n_iter):
            t = time.perf_counter()
            _sync(sim.process_event(deposits, key=key))
            times.append((time.perf_counter() - t) * 1000)
        rec['mean_ms'] = round(float(np.mean(times)), 1)
        rec['std_ms'] = round(float(np.std(times)), 1)
        pk, inuse, lim = _mem()
        rec['peak_mem_mb'] = round(pk, 1)
        rec['status'] = 'ok'

    except BaseException as e:                  # incl. XlaRuntimeError (OOM)
        rec['status'] = _classify(e)
        rec['error'] = (type(e).__name__ + ': ' + str(e))[:300]
        try:
            pk, _, lim = _mem()
            rec['peak_mem_mb'] = round(pk, 1)
            rec.setdefault('mem_limit_mb', round(lim, 1))
        except Exception:
            pass

    emit()


if __name__ == '__main__':
    main()
