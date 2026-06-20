"""
Tests for DepositData construction, volume splitting, grouping, and padding.

Replaces: test_layer5_split_pad.py
"""

import numpy as np
import jax.numpy as jnp
import pytest

from tools.geometry import generate_detector
from tools.config import create_sim_config, DepositData, VolumeDeposits
from tools.loader import build_deposit_data


CONFIG_PATH = 'config/cubic_wireplane_config.yaml'


@pytest.fixture(scope="module")
def full_config():
    detector = generate_detector(CONFIG_PATH)
    return create_sim_config(detector, include_track_hits=True)


@pytest.fixture(scope="module")
def raw_data():
    """Synthetic edepsim-style step data (self-contained, no external HDF5).

    Each track is a contiguous run of deposits spanning x = -180 -> +180 cm, so
    every track crosses the x=0 volume boundary: this populates BOTH volumes and
    keeps per-track deposits contiguous (needed for group-id computation). Mirrors
    what load_particle_step_data('out.h5') used to provide.
    """
    rng = np.random.RandomState(1234)
    n_tracks = 8
    step_cm = 0.4
    P, DE, DX, TH, PH, TID = [], [], [], [], [], []
    for t in range(n_tracks):
        x0, x1 = -180.0, 180.0
        start = np.array([x0, rng.uniform(-150, 150), rng.uniform(-150, 150)])
        end = np.array([x1, rng.uniform(-150, 150), rng.uniform(-150, 150)])
        seg = end - start
        length = np.linalg.norm(seg)
        d = seg / length
        n = int(length / step_cm)
        s = np.arange(n) * step_cm
        pts = start[None, :] + s[:, None] * d[None, :]
        dedx = rng.uniform(1.8, 2.6)
        P.append(pts)
        DE.append(np.full(n, dedx * step_cm, np.float32))
        DX.append(np.full(n, step_cm, np.float32))
        TH.append(np.full(n, np.arccos(np.clip(d[0], -1, 1)), np.float32))
        PH.append(np.full(n, np.arctan2(d[2], d[1]), np.float32))
        TID.append(np.full(n, t, np.int32))
    return {
        'positions_mm': (np.concatenate(P) * 10.0).astype(np.float32),
        'de': np.concatenate(DE),
        'dx': np.concatenate(DX),
        'theta': np.concatenate(TH),
        'phi': np.concatenate(PH),
        'track_ids': np.concatenate(TID),
    }


class TestBuildFromFile:
    """Build DepositData from HDF5 loaded data."""

    def test_structure(self, raw_data, full_config):
        deposits = build_deposit_data(
            raw_data['positions_mm'], raw_data['de'], raw_data['dx'], full_config,
            theta=raw_data['theta'], phi=raw_data['phi'],
            track_ids=raw_data['track_ids'])
        assert isinstance(deposits, DepositData)
        assert isinstance(deposits.volumes[0], VolumeDeposits)
        assert len(deposits.volumes) == 2

    def test_no_deposits_lost(self, raw_data, full_config):
        deposits = build_deposit_data(
            raw_data['positions_mm'], raw_data['de'], raw_data['dx'], full_config,
            track_ids=raw_data['track_ids'])
        n_total = sum(v.n_actual for v in deposits.volumes)
        n_input = len(raw_data['de'])
        assert n_total == n_input

    def test_padded_shapes(self, raw_data, full_config):
        deposits = build_deposit_data(
            raw_data['positions_mm'], raw_data['de'], raw_data['dx'], full_config)
        for v in deposits.volumes:
            assert v.de.shape == (full_config.total_pad,)
            assert v.positions_mm.shape == (full_config.total_pad, 3)

    def test_padding_is_zero(self, raw_data, full_config):
        deposits = build_deposit_data(
            raw_data['positions_mm'], raw_data['de'], raw_data['dx'], full_config)
        for v in deposits.volumes:
            n = v.n_actual
            assert float(jnp.sum(v.de[n:])) == 0.0

    def test_x_ranges(self, raw_data, full_config):
        """Stored positions are in the volume-LOCAL frame: x_local in
        [0, drift_length] (x_local = drift_dir * (x_anode - x_global) >= 0)."""
        deposits = build_deposit_data(
            raw_data['positions_mm'], raw_data['de'], raw_data['dx'], full_config)
        for vi, vol in enumerate(deposits.volumes):
            n = vol.n_actual
            if n > 0:
                x_local_cm = np.asarray(vol.positions_mm[:n, 0]) / 10.0
                x_min, x_max = full_config.volumes[vi].ranges_cm[0]
                drift_length_cm = abs(x_max - x_min)
                assert np.all(x_local_cm >= -1e-3), \
                    f"vol {vi}: negative local x ({x_local_cm.min():.3f} cm)"
                assert np.all(x_local_cm <= drift_length_cm + 1e-3), \
                    f"vol {vi}: local x {x_local_cm.max():.3f} cm exceeds drift length {drift_length_cm} cm"


class TestGroupIds:
    """Per-volume group ID computation."""

    def test_groups_start_from_one(self, raw_data, full_config):
        deposits = build_deposit_data(
            raw_data['positions_mm'][:1000], raw_data['de'][:1000],
            raw_data['dx'][:1000], full_config,
            track_ids=raw_data['track_ids'][:1000])
        for v in deposits.volumes:
            n = v.n_actual
            if n > 0:
                gids = np.asarray(v.group_ids[:n])
                assert gids.min() >= 1

    def test_padding_has_group_zero(self, raw_data, full_config):
        deposits = build_deposit_data(
            raw_data['positions_mm'][:500], raw_data['de'][:500],
            raw_data['dx'][:500], full_config,
            track_ids=raw_data['track_ids'][:500])
        for v in deposits.volumes:
            n = v.n_actual
            assert np.all(np.asarray(v.group_ids[n:]) == 0)

    def test_group_to_track_valid(self, raw_data, full_config):
        deposits = build_deposit_data(
            raw_data['positions_mm'][:1000], raw_data['de'][:1000],
            raw_data['dx'][:1000], full_config,
            track_ids=raw_data['track_ids'][:1000])
        for vi in range(len(deposits.volumes)):
            g2t = deposits.group_to_track[vi]
            assert g2t is not None and len(g2t) > 0

    def test_precomputed_group_ids(self, full_config):
        N = 100
        pos = np.zeros((N, 3), dtype=np.float32)
        pos[:, 0] = -500  # all in volume 0
        gids = np.arange(N, dtype=np.int32) // 10 + 1
        deposits = build_deposit_data(pos, np.ones(N, dtype=np.float32),
                                       np.ones(N, dtype=np.float32) * 0.5,
                                       full_config, group_ids=gids)
        assert np.all(np.asarray(deposits.volumes[0].group_ids[:N]) == gids)

    def test_custom_group_size(self, raw_data, full_config):
        deps_g3 = build_deposit_data(
            raw_data['positions_mm'][:200], raw_data['de'][:200],
            raw_data['dx'][:200], full_config,
            track_ids=raw_data['track_ids'][:200], group_size=3)
        deps_g10 = build_deposit_data(
            raw_data['positions_mm'][:200], raw_data['de'][:200],
            raw_data['dx'][:200], full_config,
            track_ids=raw_data['track_ids'][:200], group_size=10)
        # Smaller group_size → more groups (use whichever volume has deposits)
        for vi in range(2):
            if deps_g3.volumes[vi].n_actual > 10:
                n3 = int(np.asarray(deps_g3.volumes[vi].group_ids[:deps_g3.volumes[vi].n_actual]).max())
                n10 = int(np.asarray(deps_g10.volumes[vi].group_ids[:deps_g10.volumes[vi].n_actual]).max())
                assert n3 > n10
                break


class TestPixelOriginLocalFrame:
    """Pixel grid origin must be in the volume-LOCAL frame.

    Deposits are transformed to local/centered coords (y_local = y_global -
    y_center) before digitization, so pixel_origins_cm must be local too
    (= range_min - center = -half_extent). Otherwise an off-center pixel volume
    digitizes against the wrong frame and silently mis-attributes / drops charge.
    Regression test for the global-origin-vs-local-positions bug.
    """

    def _pixel_cfg(self, yz_range):
        cfg = generate_detector('config/cubic_pixel_config.yaml')
        cfg['volumes'] = [cfg['volumes'][0]]
        cfg['volumes'][0]['geometry']['ranges'] = [
            [-216.0, 0.0], list(yz_range), list(yz_range)]
        return create_sim_config(cfg, total_pad=100, include_track_hits=False)

    def test_origin_is_local_half_extent(self):
        for yz in [(-216.0, 216.0), (0.0, 432.0), (1000.0, 1432.0)]:
            sc = self._pixel_cfg(yz)
            half = (yz[1] - yz[0]) / 2.0
            oy, oz = sc.volumes[0].pixel_origins_cm
            assert oy == pytest.approx(-half) and oz == pytest.approx(-half), \
                f"range {yz}: origin ({oy}, {oz}) != local -half_extent {-half}"

    def test_center_deposit_maps_to_grid_center(self):
        """A deposit at the volume's physical center (local origin) maps to the
        middle of the pixel grid, regardless of where the volume sits globally."""
        from tools.wires import digitize_pixel_positions
        for yz in [(-216.0, 216.0), (0.0, 432.0), (1000.0, 1432.0)]:
            sc = self._pixel_cfg(yz)
            vol = sc.volumes[0]
            # deposit at volume center -> local (0, 0)
            py, pz, _, _ = digitize_pixel_positions(
                jnp.array([[0.0, 0.0]]), vol.pixel_pitch_cm,
                jnp.array(vol.pixel_origins_cm))
            mid = vol.pixel_shape[0] // 2
            assert abs(int(py[0]) - mid) <= 1 and abs(int(pz[0]) - mid) <= 1, \
                f"range {yz}: center deposit -> pixel ({int(py[0])},{int(pz[0])}), expected ~{mid}"


class TestEdgeCases:
    """Edge cases for build_deposit_data."""

    def test_all_one_volume(self, full_config):
        N = 50
        pos = np.ones((N, 3), dtype=np.float32) * 500  # x=50cm → volume 1
        deposits = build_deposit_data(pos, np.ones(N, dtype=np.float32),
                                       np.ones(N, dtype=np.float32) * 0.5,
                                       full_config)
        assert deposits.volumes[0].n_actual == 0
        assert deposits.volumes[1].n_actual == N

    def test_outside_all_volumes(self, full_config):
        N = 10
        pos = np.ones((N, 3), dtype=np.float32) * 5000  # x=500cm, outside
        deposits = build_deposit_data(pos, np.ones(N, dtype=np.float32),
                                       np.ones(N, dtype=np.float32) * 0.5,
                                       full_config)
        assert sum(v.n_actual for v in deposits.volumes) == 0

    def test_zero_deposits(self, full_config):
        pos = np.zeros((0, 3), dtype=np.float32)
        deposits = build_deposit_data(pos, np.zeros(0, dtype=np.float32),
                                       np.zeros(0, dtype=np.float32),
                                       full_config)
        assert sum(v.n_actual for v in deposits.volumes) == 0
        assert deposits.volumes[0].de.shape == (full_config.total_pad,)

    def test_single_deposit(self, full_config):
        pos = np.array([[-500, 0, 0]], dtype=np.float32)
        deposits = build_deposit_data(pos, np.array([2.0], dtype=np.float32),
                                       np.array([0.5], dtype=np.float32),
                                       full_config)
        assert deposits.volumes[0].n_actual == 1
        assert deposits.volumes[1].n_actual == 0

    def test_original_indices(self, full_config):
        pos = np.array([[-100, 0, 0], [100, 0, 0], [-200, 0, 0], [200, 0, 0],
                         [5000, 0, 0]], dtype=np.float32)
        deposits = build_deposit_data(pos, np.ones(5, dtype=np.float32),
                                       np.ones(5, dtype=np.float32) * 0.5,
                                       full_config)
        assert list(deposits.original_indices[0]) == [0, 2]
        assert list(deposits.original_indices[1]) == [1, 3]

    def test_scalar_dx(self, full_config):
        pos = np.ones((50, 3), dtype=np.float32) * -500
        deposits = build_deposit_data(pos, np.ones(50, dtype=np.float32),
                                       0.5, full_config)
        assert deposits.volumes[0].dx.shape == (full_config.total_pad,)

    def test_boundary_deposit_no_duplication(self, full_config):
        """Deposits at exact volume boundary should appear in exactly one volume."""
        # Volume 0: x ∈ [-216, 0) cm, Volume 1: x ∈ [0, 216) cm
        # x=0 boundary: should go to volume 1 (>= x_min of vol 1)
        x_values_mm = np.array([0.0, -0.01, 0.01], dtype=np.float32)
        pos = np.zeros((3, 3), dtype=np.float32)
        pos[:, 0] = x_values_mm
        deposits = build_deposit_data(pos, np.ones(3, dtype=np.float32),
                                       np.ones(3, dtype=np.float32) * 0.5,
                                       full_config)
        total = sum(v.n_actual for v in deposits.volumes)
        assert total == 3, f"Expected 3 deposits, got {total} (duplication or loss)"

    def test_padding_sentinel_values(self, full_config):
        """Padding region should have de=0, dx=1mm, track_ids=-1."""
        pos = np.array([[-500, 0, 0]], dtype=np.float32)
        deposits = build_deposit_data(pos, np.array([2.0], dtype=np.float32),
                                       np.array([0.5], dtype=np.float32),
                                       full_config, track_ids=np.array([7], dtype=np.int32))
        v = deposits.volumes[0]
        n = v.n_actual
        assert n == 1
        pad_de = np.asarray(v.de[n:])
        pad_dx = np.asarray(v.dx[n:])
        pad_tids = np.asarray(v.track_ids[n:])
        assert np.all(pad_de == 0.0)
        assert np.all(pad_dx == 1.0)
        assert np.all(pad_tids == -1)

    def test_track_crossing_boundary(self, full_config):
        pos = np.array([[-100, 0, 0], [-50, 0, 0], [50, 0, 0], [100, 0, 0]],
                        dtype=np.float32)
        tids = np.array([1, 1, 1, 1], dtype=np.int32)
        deposits = build_deposit_data(pos, np.ones(4, dtype=np.float32),
                                       np.ones(4, dtype=np.float32) * 0.5,
                                       full_config, track_ids=tids)
        # Same track in both volumes → separate groups per volume
        g2t_0 = deposits.group_to_track[0]
        g2t_1 = deposits.group_to_track[1]
        gids_0 = np.asarray(deposits.volumes[0].group_ids[:deposits.volumes[0].n_actual])
        gids_1 = np.asarray(deposits.volumes[1].group_ids[:deposits.volumes[1].n_actual])
        assert g2t_0[gids_0[0]] == 1
        assert g2t_1[gids_1[0]] == 1
