"""
edep-sim step files (format 3) as simulation input.

A step file has one group per event (``entry_NNNNNN``) with flat arrays, rows
track by track: positions, t, de, dx, theta, phi, p, track_id and
``group_run_lengths`` (group 1: runs of consecutive rows, 1-based ids over the
whole event).  Positions are cells of a fixed grid: ``(cumsum(k) + cell_offset) *
scale_factor``; every reader must use these decoded values, also to assign steps
to volumes.

The simulation refines group 1 into group 2: a new group at every change of
group 1 or volume in row order, numbered 1, 2, ... in each volume.  hits use group
2; presupera stores group2_to_group1, so step -> group 2 can be rebuilt.
"""

import h5py
import numpy as np

from production.entries import entry_key

FORMAT_VERSION = 3


def is_step_file(path):
    with h5py.File(path, 'r') as f:
        return 'schema' in f and f['schema'].attrs.get('role') in ('step', b'step')


def decode_cells(ds):
    k = np.cumsum(ds[()].astype(np.int64), axis=0)
    return (k + float(ds.attrs['cell_offset'])) * float(ds.attrs['scale_factor'])


def read_step(f, index):
    """Columns of one event; positions in mm (float64), group1 per row."""
    g = f[entry_key(index)]
    out = {k: g[k][()] for k in ('t', 'de', 'dx', 'theta', 'phi', 'p', 'track_id')}
    out['positions_mm'] = decode_cells(g['positions'])
    runs = g['group_run_lengths'][()].astype(np.int64)
    out['group1'] = np.repeat(np.arange(1, len(runs) + 1, dtype=np.int32), runs)
    return out


def volume_of(positions_mm, ranges_cm):
    """Volume index per row (-1 outside), as build_deposit_data assigns it."""
    pos_cm = np.asarray(positions_mm, dtype=np.float32) / 10.0
    out = np.full(len(pos_cm), -1, dtype=np.int32)
    for v, r in enumerate(ranges_cm):
        m = np.ones(len(pos_cm), dtype=bool)
        for ax in range(3):
            m &= (pos_cm[:, ax] >= r[ax][0]) & (pos_cm[:, ax] < r[ax][1])
        out[m] = v
    return out


def refine_groups(group1, volume, n_volumes):
    """Group 2 per row (0 outside all volumes) and, per volume, group2 -> group1
    (row g-1 is group g)."""
    n = len(group1)
    start = np.ones(n, dtype=bool)
    start[1:] = (group1[1:] != group1[:-1]) | (volume[1:] != volume[:-1])
    group2 = np.zeros(n, dtype=np.int32)
    maps = []
    for v in range(n_volumes):
        m = volume == v
        group2[m] = np.cumsum(start[m])
        maps.append(group1[m & start].astype(np.int32))
    return group2, maps


class StepFileExtractor:
    """Open step file; the counterpart of ParticleStepExtractor."""

    def __init__(self, path):
        self.file = h5py.File(path, 'r')

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def load(self, index, sim_config, regroup=None):
        """DepositData of one event and the per-volume group2 -> group1 maps.

        regroup: (group_size, gap_threshold_mm) to define groups per volume here
        instead of refining the step file's; the maps are then None.
        """
        from tools.loader import build_deposit_data
        s = read_step(self.file, index)
        pos = s['positions_mm'].astype(np.float32)
        kw = dict(theta=s['theta'], phi=s['phi'], track_ids=s['track_id'],
                  t0_us=s['t'] / np.float32(1000.0))
        maps = None
        if regroup is None:
            ranges = [v.ranges_cm for v in sim_config.volumes]
            vol = volume_of(pos, ranges)
            group2, maps = refine_groups(s['group1'], vol, len(ranges))
            kw['group_ids'] = group2
        else:
            kw.update(group_size=regroup[0], gap_threshold_mm=regroup[1])
        return build_deposit_data(pos, s['de'], s['dx'], sim_config, **kw), maps
