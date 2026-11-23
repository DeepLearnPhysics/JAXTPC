"""Format 3: step file reading, group refinement, hits encoding, presupera."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from production import stepfile
from production.load import load_hits3, load_step3
from production.save import encode_hits, save_event_hits3, save_event_presupera

RANGES = (((-20., 0.), (-20., 20.), (-20., 20.)), ((0., 20.), (-20., 20.), (-20., 20.)))


def write_step(path, pos, track, runs):
    k = np.floor(pos / 0.01).astype(np.int64)
    with h5py.File(path, 'w') as f:
        f.create_group('schema').attrs.update(dict(role='step', group_size=5, group_gap_mm=3.0))
        g = f.create_group('entry_000000')
        d = g.create_dataset('positions', data=np.diff(k, axis=0, prepend=np.zeros((1, 3), int)).astype('i4'))
        d.attrs.update(dict(scale_factor=0.01, cell_offset=0.5))
        for c, t in (('t', 'f4'), ('de', 'f4'), ('dx', 'f2'), ('theta', 'f2'), ('phi', 'f2'), ('p', 'f4')):
            g[c] = np.full(len(pos), 0.25, t)
        g['track_id'] = np.asarray(track, 'i4')
        g['group_run_lengths'] = np.asarray(runs, 'u2')


def test_step_roundtrip_and_groups(tmp_path):
    # one track: 3 steps at x < 0, 1 at x > 0, back to x < 0 (one group 1 of 5), then a second track
    x = np.array([-0.9, -0.6, -0.3, 0.2, -0.1, 5.0, 5.3])
    pos = np.stack([x, x * 0 + 1., x * 0 + 2.], 1)
    p = str(tmp_path / 's.h5')
    write_step(p, pos, [0] * 5 + [3] * 2, [5, 2])
    assert stepfile.is_step_file(p)
    s = load_step3(p, 0, RANGES)
    assert np.abs(s['positions_mm'] - pos).max() <= 0.005 + 1e-9
    assert s['group1'].tolist() == [1] * 5 + [2] * 2
    assert s['volume'].tolist() == [0, 0, 0, 1, 0, 1, 1]
    # the pieces before and after the excursion are different groups
    assert s['group2'].tolist() == [1, 1, 1, 1, 2, 2, 2]
    _, maps = stepfile.refine_groups(s['group1'], s['volume'], 2)
    assert [m.tolist() for m in maps] == [[1, 1], [1, 2]]


@pytest.mark.parametrize('side', [-1, 1])
def test_boundary_side_is_kept(tmp_path, side):
    x = side * np.array([1e-7, 1e-4, 0.004999, 0.005, 0.00999, 0.01])
    if side > 0:
        x = np.r_[0.0, x]
    p = str(tmp_path / 's.h5')
    write_step(p, np.stack([x, x * 0, x * 0], 1), [0] * len(x), [len(x)])
    assert (load_step3(p, 0, RANGES)['volume'] == (0 if side < 0 else 1)).all()


def deposits(n_groups):
    vol = SimpleNamespace(n_actual=4, group_ids=np.array([1, 1, 2, 3]))
    return SimpleNamespace(volumes=(vol,), group_to_track=(np.r_[0, np.arange(10, 10 + n_groups)],))


@pytest.mark.parametrize('readout', ['wire', 'pixel'])
def test_hits_roundtrip(tmp_path, readout):
    rng = np.random.default_rng(0)
    n, T = 4000, 10.0
    names = ('wire', 'time') if readout == 'wire' else ('py', 'pz', 'time')
    gid = rng.integers(1, 40, n)
    co = {k: (gid * 20 + rng.integers(-200, 200, n)).astype(np.int32) for k in names}   # offsets > int8
    q = rng.normal(0, 60, n).astype(np.float32)
    enc = encode_hits(gid, co, q, T)
    assert enc['offset_time'].dtype == np.int16
    cfg = SimpleNamespace(plane_names=((('U',) if readout == 'wire' else ('Pixel',)),),
                          volumes=(SimpleNamespace(readout_type=readout),))
    p = str(tmp_path / 'h.h5')
    with h5py.File(p, 'w') as f:
        save_event_hits3(f, 'entry_000000', {(0, 0): enc}, deposits(3), 0, T, cfg, event_id=7,
                         attrs={'job_id': 0, 'job_event': 7})
    got = load_hits3(p, 'entry_000000')[(0, cfg.plane_names[0][0])]
    keep = np.abs(q) > T
    o = np.argsort(gid[keep], kind='stable')
    assert np.array_equal(got['group'], gid[keep][o])
    for k in names:
        assert np.array_equal(got[k], co[k][keep][o])
    assert np.abs(got['charge'] - q[keep][o]).max() <= T / 32 + 1e-4


def test_hits_empty_and_threshold():
    e = np.zeros(0, np.int32)
    enc = encode_hits(e, {'wire': e, 'time': e}, e.astype(np.float32), 10.0)
    assert all(len(v) == 0 for v in enc.values())
    with pytest.raises(ValueError):
        encode_hits(e, {'wire': e, 'time': e}, e.astype(np.float32), 0.0)


def test_presupera(tmp_path):
    p = str(tmp_path / 'p.h5')
    with h5py.File(p, 'w') as f:
        save_event_presupera(f, 'entry_000000', deposits(3), [np.array([1, 1, 2], 'i4')], 0, None)
        save_event_presupera(f, 'entry_000001', deposits(3), None, 0, None)
        g = f['entry_000000/volume_0']
        assert g['group_to_track'][()].tolist() == [10, 11, 12]
        assert g['group2_to_group1'][()].tolist() == [1, 1, 2] and 'deposit_to_group' not in g
        assert f['entry_000001/volume_0/deposit_to_group'][()].tolist() == [1, 1, 2, 3]
