"""Golden-output regression harness for the production sim path.

Runs DetectorSimulator.process_event deterministically (no random noise, fixed
key) and reduces the output to a flat dict of numpy arrays so two runs can be
compared exactly.

Modes
-----
  capture   run one (config, dims) and save the canonical output to an .npz golden
  compare   re-run and assert it matches a saved golden (allclose for floats)
  equiv     run the SAME event twice with two box-dim sets and assert identical
            output  (the core invariant: box dims only need to be >= the per-group
            footprint, so profiled vs analytic dims must give bit-identical hits)

The third mode needs no stored file and validates the central refactor claim on
the *current* code before anything changes.

Usage
-----
  python3 -m tests.golden.golden equiv  --config config/cubic_wireplane_config.yaml \
      --data <edepsim.h5> --event 22 \
      --total-pad 450000 --response-chunk 28125 --hits-chunk 28125 \
      --max-keys 3100000 --maxg 110000 \
      --dims-a 12 27 --dims-b 12 38            # wire: BW BTW
  python3 -m tests.golden.golden equiv --config config/cubic_pixel_config.yaml ... \
      --dims-a 8 8 83 --dims-b 8 8 94          # pixel: BPY BPZ BT

  python3 -m tests.golden.golden capture --out tests/golden/data/wire.npz ... --dims 12 27
  python3 -m tests.golden.golden compare --golden tests/golden/data/wire.npz ... --dims 12 27
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Deterministic reductions/scatter so goldens are bit-reproducible across runs
# (without this, GPU segment-sum/scatter order drifts at the ~1 ADC / 1e-3 ENC
# level and the comparison sees spurious mismatches).
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import jax

from tools.geometry import generate_detector
from tools.config import create_track_hits_config
from tools.simulation import DetectorSimulator
from tools.loader import load_event


def _flatten(obj, prefix, out):
    """Recursively reduce a nested dict/tuple/array structure to {path: np.ndarray}."""
    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=repr):
            _flatten(obj[k], f'{prefix}/{k!r}', out)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            _flatten(v, f'{prefix}[{i}]', out)
    elif hasattr(obj, 'shape') or np.isscalar(obj):
        out[prefix] = np.asarray(obj)
    # else: skip non-array leaves (e.g. None)


def _build_sim(args, dims):
    detector_config = generate_detector(args.config)
    readout = detector_config['volumes'][0].get('readout', {}).get('type', 'wire')
    # dims=None -> let the simulator compute analytic box dims (production path)
    if dims is None:
        box_kw = {}
    elif readout == 'pixel':
        box_kw = {'box_bpy': dims[0], 'box_bpz': dims[1], 'box_bt': dims[2]}
    else:
        box_kw = {'box_bw': dims[0], 'box_btw': dims[1]}
    tc = create_track_hits_config(
        max_keys=args.max_keys, hits_chunk_size=args.hits_chunk,
        inter_thresh=args.inter_thresh, box_enabled=True, maxg=args.maxg, **box_kw)
    sim = DetectorSimulator(
        detector_config, track_config=tc, total_pad=args.total_pad,
        response_chunk_size=args.response_chunk, include_track_hits=True,
        include_digitize=True)
    sim.warm_up()
    return sim, readout


def _canon(args, dims):
    """Run process_event deterministically and return the flattened output."""
    sim, readout = _build_sim(args, dims)
    deposits = load_event(args.data, sim.config, event_idx=args.event)
    result = sim.process_event(deposits, key=jax.random.PRNGKey(42))
    flat = {}
    _flatten(result[0], 'sensor', flat)        # response_signals
    _flatten(result[1], 'hits', flat)          # track_hits_raw
    flat = {k: jax.block_until_ready(v) for k, v in flat.items()}
    return {k: np.asarray(v) for k, v in flat.items()}, readout


def _diff(a, b, rtol=0, atol=0):
    """Return list of (key, reason) mismatches between two flat dicts."""
    bad = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        if k not in a:
            bad.append((k, 'missing in A')); continue
        if k not in b:
            bad.append((k, 'missing in B')); continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            bad.append((k, f'shape {x.shape} != {y.shape}')); continue
        if np.issubdtype(x.dtype, np.floating):
            if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                d = np.nanmax(np.abs(x - y))
                bad.append((k, f'float max|Δ|={d:.6g}'))
        else:
            if not np.array_equal(x, y):
                bad.append((k, f'int mismatch ({int((x != y).sum())} cells)'))
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['capture', 'compare', 'equiv'])
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--event', type=int, default=0)
    p.add_argument('--total-pad', type=int, required=True)
    p.add_argument('--response-chunk', type=int, required=True)
    p.add_argument('--hits-chunk', type=int, required=True)
    p.add_argument('--max-keys', type=int, required=True)
    p.add_argument('--maxg', type=int, required=True)
    p.add_argument('--inter-thresh', type=float, default=1.0)
    p.add_argument('--dims', type=int, nargs='+', help='box dims (wire: BW BTW; pixel: BPY BPZ BT)')
    p.add_argument('--dims-a', type=int, nargs='+', help='equiv mode: first dim set')
    p.add_argument('--dims-b', type=int, nargs='+', help='equiv mode: second dim set')
    p.add_argument('--out', help='capture: output golden .npz')
    p.add_argument('--golden', help='compare: golden .npz to check against')
    # float tolerance for compare (0 = exact)
    p.add_argument('--rtol', type=float, default=0.0)
    p.add_argument('--atol', type=float, default=0.0)
    args = p.parse_args()

    if args.mode == 'equiv':
        a, ro = _canon(args, args.dims_a)
        b, _ = _canon(args, args.dims_b)
        bad = _diff(a, b)
        print(f'[equiv {ro}] dims {args.dims_a} vs {args.dims_b}: '
              f'{len(a)} arrays compared')
        if bad:
            print('MISMATCH:')
            for k, why in bad[:30]:
                print(f'  {k}: {why}')
            sys.exit(1)
        print('IDENTICAL — box-dim invariant holds.')

    elif args.mode == 'capture':
        a, ro = _canon(args, args.dims)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, **a)
        print(f'[capture {ro}] {len(a)} arrays -> {args.out}')

    elif args.mode == 'compare':
        g = dict(np.load(args.golden, allow_pickle=False))
        a, ro = _canon(args, args.dims)
        bad = _diff(g, a, rtol=args.rtol, atol=args.atol)
        print(f'[compare {ro}] vs {args.golden}: {len(a)} arrays')
        if bad:
            print('REGRESSION:')
            for k, why in bad[:30]:
                print(f'  {k}: {why}')
            sys.exit(1)
        print('MATCH — no regression.')


if __name__ == '__main__':
    main()
