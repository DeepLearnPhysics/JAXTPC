"""
Tests for production save/load roundtrip.

Files under test: production/save.py, production/load.py
"""

import os
import tempfile
import numpy as np
import jax.numpy as jnp
import pytest

from production.save import (
    save_event_sensor, write_config_sensor, encode_correspondence_csr,
    save_event_edep, write_config_edep,
)
from production.load import (
    load_event_sensor, load_event_edep, build_viz_config,
)


@pytest.fixture
def mock_sim_config():
    """Minimal SimConfig-like object for save functions."""
    from collections import namedtuple
    VG = namedtuple('VG', ['num_wires', 'n_planes', 'wire_lengths_m', 'ranges_cm',
                           'readout_type'])
    Cfg = namedtuple('Cfg', ['volumes', 'n_volumes', 'num_time_steps', 'time_step_us',
                              'pre_window_us', 'post_window_us',
                              'electrons_per_adc', 'include_intrinsic_noise',
                              'include_coherent_noise',
                              'include_electronics', 'include_digitize',
                              'plane_names'])
    Params = namedtuple('Params', ['velocity_cm_us', 'lifetime_us'])

    vg0 = VG(num_wires=(50, 50, 40), n_planes=3, wire_lengths_m=(np.ones(50),) * 3,
             ranges_cm=((-20, 0), (-20, 20), (-20, 20)), readout_type='wire')
    vg1 = VG(num_wires=(50, 50, 40), n_planes=3, wire_lengths_m=(np.ones(50),) * 3,
             ranges_cm=((0, 20), (-20, 20), (-20, 20)), readout_type='wire')
    return (
        Cfg(
            volumes=(vg0, vg1),
            n_volumes=2,
            num_time_steps=100,
            time_step_us=0.5,
            pre_window_us=0.0,
            post_window_us=0.0,
            electrons_per_adc=182.0,
            include_intrinsic_noise=False,
            include_coherent_noise=False,
            include_electronics=False,
            include_digitize=False,
            plane_names=(('U', 'V', 'Y'), ('U', 'V', 'Y')),
        ),
        Params(velocity_cm_us=0.16, lifetime_us=10000.0),
    )


def _mock_deposits(n_per_vol=50):
    """Create a minimal mock DepositData for save tests."""
    from collections import namedtuple
    VD = namedtuple('VD', ['n_actual'])
    DD = namedtuple('DD', ['volumes', 'group_to_track', 'original_indices'])
    return DD(
        volumes=(VD(n_actual=n_per_vol), VD(n_actual=n_per_vol)),
        group_to_track=(None, None),
        original_indices=(None, None),
    )


def _mock_edep_deposits(n_per_vol=20):
    """Create mock DepositData with full physics fields for edep save tests."""
    from collections import namedtuple
    rng = np.random.RandomState(42)
    VD = namedtuple('VD', ['n_actual', 'positions_mm', 'de', 'dx',
                           'theta', 'phi', 't0_us', 'charge', 'photons'])
    DD = namedtuple('DD', ['volumes', 'group_to_track', 'original_indices'])
    vols = []
    for _ in range(2):
        n = n_per_vol
        vols.append(VD(
            n_actual=n,
            positions_mm=rng.uniform(-500, 500, (n, 3)).astype(np.float32),
            de=rng.uniform(0.5, 5.0, n).astype(np.float32),
            dx=rng.uniform(0.01, 0.5, n).astype(np.float32),
            theta=rng.uniform(0, np.pi, n).astype(np.float32),
            phi=rng.uniform(-np.pi, np.pi, n).astype(np.float32),
            t0_us=rng.uniform(0, 10.0, n).astype(np.float32),
            charge=rng.uniform(100, 10000, n).astype(np.float32),
            photons=rng.uniform(100, 10000, n).astype(np.float32),
        ))
    return DD(volumes=tuple(vols), group_to_track=(None, None),
              original_indices=(None, None))


class TestRespRoundtrip:
    """Test response save → load roundtrip."""

    def test_basic_roundtrip(self, mock_sim_config):
        """Save and load response, verify values match."""
        cfg, params = mock_sim_config
        import h5py

        # Create test signals with known values
        signals = {}
        for s in range(2):
            for p in range(3):
                nw = cfg.volumes[s].num_wires[p]
                arr = np.zeros((nw, cfg.num_time_steps), dtype=np.float32)
                arr[10, 20] = 42.0
                arr[15, 30] = -7.5
                arr[0, 0] = 100.0
                signals[(s, p)] = arr

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                write_config_sensor(f, cfg, params, 'emb', 'test', 0,
                                  'test.h5', 1, 0, 1.0)
                save_event_sensor(f, 'event_000', signals, 1.0, 0, _mock_deposits())

            dense, attrs, _ = load_event_sensor(tmp_path, 0)

            for (s, p), orig in signals.items():
                loaded = dense[(s, p)]
                # Only values above threshold should survive
                mask = np.abs(orig) >= 1.0
                np.testing.assert_array_almost_equal(
                    loaded[mask], orig[mask], decimal=4,
                    err_msg=f"Plane ({s},{p}) values don't match")
                # Below-threshold values should be zero
                assert np.all(loaded[~mask] == 0)
        finally:
            os.unlink(tmp_path)

    def test_delta_encoding_correctness(self, mock_sim_config):
        """Verify delta encoding handles non-sequential wire/time indices."""
        cfg, params = mock_sim_config
        import h5py

        arr = np.zeros((50, 100), dtype=np.float32)
        # Scattered pixels — tests that delta encoding handles gaps
        arr[5, 90] = 10.0
        arr[45, 10] = 20.0
        arr[25, 50] = 30.0
        signals = {(0, 0): arr}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                write_config_sensor(f, cfg, params, 'emb', 'test', 0,
                                  'test.h5', 1, 0, 1.0)
                save_event_sensor(f, 'event_000', signals, 1.0, 0, _mock_deposits())

            dense, _, _ = load_event_sensor(tmp_path, 0)

            np.testing.assert_allclose(dense[(0, 0)][5, 90], 10.0)
            np.testing.assert_allclose(dense[(0, 0)][45, 10], 20.0)
            np.testing.assert_allclose(dense[(0, 0)][25, 50], 30.0)
        finally:
            os.unlink(tmp_path)


class TestVizConfig:
    """Test build_viz_config from metadata."""

    def test_viz_config_fields(self, mock_sim_config):
        """build_viz_config should produce usable config for visualization."""
        cfg, params = mock_sim_config
        import h5py

        signals = {(0, 0): np.zeros((50, 100), dtype=np.float32)}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                write_config_sensor(f, cfg, params, 'emb', 'test', 0,
                                  'test.h5', 1, 0, 1.0)
                save_event_sensor(f, 'event_000', signals, 1.0, 0, _mock_deposits())

            viz_cfg = build_viz_config(tmp_path)

            assert viz_cfg.num_time_steps == 100
            assert viz_cfg.time_step_us == 0.5
            assert viz_cfg.electrons_per_adc == 182.0
            assert viz_cfg.volumes[0].num_wires[0] == 50
            assert viz_cfg.volumes[1].num_wires[2] == 40
        finally:
            os.unlink(tmp_path)


class TestCSREncoding:
    """Test CSR correspondence encoding."""

    def test_basic_encoding(self):
        """Encode simple correspondence and verify structure."""
        pk = np.array([100, 100, 200, 200, 300], dtype=np.int32)
        gid = np.array([0, 1, 0, 1, 2], dtype=np.int32)
        ch = np.array([10.0, 5.0, 8.0, 3.0, 15.0], dtype=np.float32)

        csr = encode_correspondence_csr(pk, gid, ch, 5, num_time_steps=50)

        assert 'group_ids' in csr
        assert 'group_sizes' in csr
        assert 'center_wires' in csr
        assert 'peak_charges' in csr
        assert 'charges_u16' in csr

        # 3 unique groups
        assert len(csr['group_ids']) == 3
        # Total entries = 5
        assert len(csr['delta_wires']) == 5

    def test_peak_charge_is_max(self):
        """Peak charge for each group should be the maximum."""
        pk = np.array([100, 101, 102], dtype=np.int32)
        gid = np.array([0, 0, 0], dtype=np.int32)
        ch = np.array([5.0, 20.0, 10.0], dtype=np.float32)

        csr = encode_correspondence_csr(pk, gid, ch, 3, num_time_steps=50)

        assert float(csr['peak_charges'][0]) == 20.0

    def test_threshold_prunes(self):
        """Entries below threshold should be removed."""
        pk = np.array([100, 200, 300], dtype=np.int32)
        gid = np.array([0, 0, 0], dtype=np.int32)
        ch = np.array([50.0, 1.0, 30.0], dtype=np.float32)

        csr = encode_correspondence_csr(pk, gid, ch, 3, num_time_steps=50,
                                         threshold=10.0)

        assert len(csr['delta_wires']) == 2  # 1.0 pruned

    def test_uint16_roundtrip(self):
        """charges_u16 / 65535 * peak should recover original charges."""
        pk = np.array([100, 200, 300], dtype=np.int32)
        gid = np.array([0, 0, 0], dtype=np.int32)
        ch = np.array([100.0, 50.0, 25.0], dtype=np.float32)

        csr = encode_correspondence_csr(pk, gid, ch, 3, num_time_steps=50)

        peak = csr['peak_charges'][0]
        recovered = peak * csr['charges_u16'].astype(np.float32) / 65535.0

        np.testing.assert_allclose(recovered, ch, rtol=1e-3)

    def test_csr_encode_decode_roundtrip_via_hdf5(self):
        """Full CSR encode → HDF5 write → HDF5 read → decode roundtrip."""
        import h5py
        from production.load import _decode_plane_hits

        num_time_steps = 50
        # Two groups: group 0 has 3 entries, group 1 has 2 entries
        wires = np.array([10, 11, 12, 20, 21], dtype=np.int32)
        times = np.array([5, 6, 7, 30, 31], dtype=np.int32)
        pk = wires * num_time_steps + times
        gid = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        ch = np.array([100.0, 50.0, 25.0, 80.0, 40.0], dtype=np.float32)

        csr = encode_correspondence_csr(pk, gid, ch, 5, num_time_steps)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                g = f.create_group('plane_test')
                for key, val in csr.items():
                    g.create_dataset(key, data=val)

            with h5py.File(tmp_path, 'r') as f:
                dec_pk, dec_gid, dec_ch, dec_n = _decode_plane_hits(
                    f['plane_test'], num_time_steps)

            assert dec_n == 5
            # Recovered pk should match original (wire * nts + time encoding)
            dec_wires = dec_pk // num_time_steps
            dec_times = dec_pk % num_time_steps
            # Sort both by pk for comparison (decode may reorder within group)
            orig_order = np.argsort(pk)
            dec_order = np.argsort(dec_pk)
            np.testing.assert_array_equal(dec_pk[dec_order], pk[orig_order])
            np.testing.assert_array_equal(dec_gid[dec_order], gid[orig_order])
            np.testing.assert_allclose(dec_ch[dec_order], ch[orig_order], rtol=1e-3)
        finally:
            os.unlink(tmp_path)


class TestEdepRoundtrip:
    """Test edep save → load roundtrip."""

    def test_position_roundtrip_tolerance(self, mock_sim_config):
        """Saved positions should be recoverable within voxelization tolerance."""
        cfg, _ = mock_sim_config
        import h5py

        deposits = _mock_edep_deposits(n_per_vol=30)
        pos_step_mm = 0.3

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                write_config_edep(f, cfg, 'test', 0, 'test.h5', 1, 0, 5, 5.0)
                save_event_edep(f, 'event_000', deposits, 0, pos_step_mm=pos_step_mm)

            volumes = load_event_edep(tmp_path, 0)

            for vi in range(2):
                orig = deposits.volumes[vi].positions_mm[:deposits.volumes[vi].n_actual]
                loaded = volumes[vi]['positions_mm']
                assert loaded.shape == orig.shape
                np.testing.assert_allclose(
                    loaded, orig, atol=pos_step_mm / 2 + 0.01,
                    err_msg=f"Volume {vi} positions exceed voxelization tolerance")
        finally:
            os.unlink(tmp_path)

    def test_physics_fields_roundtrip(self, mock_sim_config):
        """de, dx, charge, photons should survive float16/float32 roundtrip."""
        cfg, _ = mock_sim_config
        import h5py

        deposits = _mock_edep_deposits(n_per_vol=15)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                write_config_edep(f, cfg, 'test', 0, 'test.h5', 1, 0, 5, 5.0)
                save_event_edep(f, 'event_000', deposits, 0)

            volumes = load_event_edep(tmp_path, 0)

            for vi in range(2):
                vol = volumes[vi]
                orig = deposits.volumes[vi]
                n = orig.n_actual
                # charge/photons are float32 → exact
                np.testing.assert_allclose(vol['charge'], orig.charge[:n], rtol=1e-6)
                np.testing.assert_allclose(vol['photons'], orig.photons[:n], rtol=1e-6)
                # de/dx/theta/phi/t0 are float16 → wider tolerance
                np.testing.assert_allclose(vol['de'], orig.de[:n], rtol=5e-3, atol=1e-3)
        finally:
            os.unlink(tmp_path)

    def test_empty_volume(self, mock_sim_config):
        """A volume with n_actual=0 should save and load without error."""
        cfg, _ = mock_sim_config
        import h5py
        from collections import namedtuple

        VD = namedtuple('VD', ['n_actual', 'positions_mm', 'de', 'dx',
                               'theta', 'phi', 't0_us', 'charge', 'photons'])
        DD = namedtuple('DD', ['volumes', 'group_to_track', 'original_indices'])

        empty = VD(n_actual=0, positions_mm=np.zeros((0, 3), dtype=np.float32),
                   de=np.zeros(0, np.float32), dx=np.zeros(0, np.float32),
                   theta=np.zeros(0, np.float32), phi=np.zeros(0, np.float32),
                   t0_us=np.zeros(0, np.float32), charge=np.zeros(0, np.float32),
                   photons=np.zeros(0, np.float32))
        deposits = DD(volumes=(empty, empty), group_to_track=(None, None),
                      original_indices=(None, None))

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                write_config_edep(f, cfg, 'test', 0, 'test.h5', 1, 0, 5, 5.0)
                save_event_edep(f, 'event_000', deposits, 0)

            volumes = load_event_edep(tmp_path, 0)

            for vi in range(2):
                assert volumes[vi]['n_actual'] == 0
        finally:
            os.unlink(tmp_path)

    def test_multi_event_isolation(self, mock_sim_config):
        """Multiple events in one file should load independently."""
        cfg, _ = mock_sim_config
        import h5py

        dep0 = _mock_edep_deposits(n_per_vol=10)
        dep1 = _mock_edep_deposits(n_per_vol=15)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, 'w') as f:
                write_config_edep(f, cfg, 'test', 0, 'test.h5', 2, 0, 5, 5.0)
                save_event_edep(f, 'event_000', dep0, 0)
                save_event_edep(f, 'event_001', dep1, 1)

            vol0 = load_event_edep(tmp_path, 0)
            vol1 = load_event_edep(tmp_path, 1)

            assert vol0[0]['n_actual'] == 10
            assert vol1[0]['n_actual'] == 15
            # Verify they're actually different
            assert not np.array_equal(vol0[0]['de'], vol1[0]['de'][:10])
        finally:
            os.unlink(tmp_path)
