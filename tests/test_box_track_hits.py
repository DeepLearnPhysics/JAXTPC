"""Tests for the group-as-bucket ("box") track-hits path.

Box is the production default (run_batch); merge is the dormant fallback. These
tests guard that the box path:
  1. activates via TrackHitsConfig.box_enabled (config, not env),
  2. produces output structurally identical and numerically equivalent to the
     sort-merge fallback (so downstream consumers can't tell which ran), and
  3. raises maxg_overflow (caught/logged/reprocessed by run_batch) when a
     group_id would exceed the box capacity instead of silently clipping.

CPU-only, synthetic deposits, but needs the wire response kernels.
"""
import numpy as np
import jax
import pytest

from tools.config import create_sim_config, create_track_hits_config
from tools.loader import build_deposit_data
from tools.simulation import DetectorSimulator

pytestmark = [pytest.mark.slow, pytest.mark.requires_kernels]

TOTAL_PAD = 2000
RESP_CHUNK = 1000
HITS_CHUNK = 1000


@pytest.fixture(scope="module")
def wire_config():
    return {
        'volumes': [{
            'id': 0,
            'geometry': {'ranges': [[-20.0, 0.0], [-20.0, 20.0], [-20.0, 20.0]],
                         'drift_direction': -1},
            'planes': [
                {'plane_id': 0, 'type': 'first_induction', 'angle': 60.0,
                 'wire_spacing': 0.3, 'distance_from_anode': 0.6, 'bias_voltage': -200.0},
                {'plane_id': 1, 'type': 'second_induction', 'angle': -60.0,
                 'wire_spacing': 0.3, 'distance_from_anode': 0.3, 'bias_voltage': -200.0},
                {'plane_id': 2, 'type': 'collection', 'angle': 0.0,
                 'wire_spacing': 0.3, 'distance_from_anode': 0.0, 'bias_voltage': 500.0},
            ]}],
        'readout': {'sampling_rate': 2.0, 'electrons_per_adc': 182},
        'simulation': {
            'drift': {'velocity': 1.6, 'longitudinal_diffusion': 6.2,
                      'transverse_diffusion': 16.3, 'electron_lifetime': 10.0},
            'charge_recombination': {'model': 'modified_box',
                                     'recomb_parameters': {'alpha': 0.93, 'beta': 0.212}}},
        'medium': {'type': 'liquid_argon',
                   'properties': {'density': 1.396, 'ionization_energy': 23.6,
                                  'excitation_ratio': 0.21},
                   'temperature': 87.0, 'pressure': 1.0},
        'electric_field': {'field_strength': 500.0},
    }


def _deposits(wire_config):
    """Several synthetic tracks inside volume 0 -> a few dozen groups."""
    rng = np.random.RandomState(7)
    segs = []
    for t in range(8):
        p0 = rng.uniform([-18, -18, -18], [-2, 18, 18])
        d = rng.uniform(-1, 1, 3); d /= np.linalg.norm(d)
        npts = rng.randint(20, 40)
        s = np.linspace(0, rng.uniform(4, 10), npts)[:, None]
        pts = p0[None, :] + s * d[None, :]
        pts[:, 0] = np.clip(pts[:, 0], -19.9, -0.1)
        segs.append((pts.astype(np.float32), np.full(npts, t, np.int32)))
    pos = (np.concatenate([p for p, _ in segs]) * 10.0).astype(np.float32)
    tid = np.concatenate([i for _, i in segs])
    n = len(tid)
    de = rng.uniform(0.5, 3.0, n).astype(np.float32)
    dx = rng.uniform(0.05, 0.4, n).astype(np.float32)
    sc = create_sim_config(wire_config, total_pad=TOTAL_PAD, include_track_hits=False)
    dep = build_deposit_data(pos, de, dx, sc, track_ids=tid,
                             group_size=5, gap_threshold_mm=5.0)
    n_groups = max(int(v.group_ids.max()) for v in dep.volumes) + 1
    return dep, n_groups


def _run(wire_config, box_enabled, maxg=2000):
    tc = create_track_hits_config(max_keys=200_000, hits_chunk_size=HITS_CHUNK,
                                  inter_thresh=1.0, box_enabled=box_enabled,
                                  maxg=maxg, box_bw=16, box_btw=96)
    sim = DetectorSimulator(wire_config, track_config=tc, total_pad=TOTAL_PAD,
                            response_chunk_size=RESP_CHUNK, include_track_hits=True)
    dep, n_groups = _deposits(wire_config)
    _, track_hits, _ = sim.process_event(dep, key=jax.random.PRNGKey(0))
    return track_hits, n_groups


def test_box_matches_merge(wire_config):
    """Box (via config) is structurally identical and numerically ~= merge."""
    th_merge, _ = _run(wire_config, box_enabled=False)
    th_box, _ = _run(wire_config, box_enabled=True)

    keys = sorted(k for k in th_merge if isinstance(k, tuple))
    assert keys, "no per-(vol,plane) track-hits produced"
    assert keys == sorted(k for k in th_box if isinstance(k, tuple))

    for k in keys:
        m, b = th_merge[k], th_box[k]
        # identical tuple shape + dtypes -> downstream cannot tell which ran
        assert len(m) == len(b) == 6
        for i in range(len(m)):
            assert np.asarray(m[i]).dtype == np.asarray(b[i]).dtype
        cm, cb = int(m[4]), int(b[4])
        assert cm > 0
        # counts within a handful of keys (threshold-boundary / pruning drift)
        assert abs(cm - cb) <= max(5, int(0.001 * cm))
        # total charge agrees to <0.1%
        chm = float(np.asarray(m[3])[:cm].astype(np.float64).sum())
        chb = float(np.asarray(b[3])[:cb].astype(np.float64).sum())
        assert abs(chm - chb) <= 1e-3 * max(abs(chm), 1.0)


def test_maxg_overflow_raises(wire_config):
    """maxg below the group count raises (not silent clip); message classifiable."""
    with pytest.raises(RuntimeError, match="maxg overflow"):
        _run(wire_config, box_enabled=True, maxg=5)
