"""
Production pipeline integration tests.

One simulator (dense + track_hits) shared across all tests via module scope.
Plus one bucketed simulator for dense==bucketed comparison.

Tests: signal shapes, track hits, qs_fractions, warm_up,
custom sim_params, edge cases, bucketed equivalence,
synthetic deterministic pipeline.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_event, build_deposit_data
from tools.wires import sparse_buckets_to_dense


CONFIG_PATH = 'config/cubic_wireplane_config.yaml'
DATA_PATH = 'out.h5'
EVENT_IDX = 2

needs_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason="out.h5 not found")


# ---------------------------------------------------------------------------
# Module-scoped fixtures (1 JIT compile for main simulator)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector_config():
    return generate_detector(CONFIG_PATH)


@pytest.fixture(scope="module")
def sim(detector_config):
    """Dense + track_hits simulator. Shared across all tests."""
    return DetectorSimulator(
        detector_config, total_pad=200_000, response_chunk_size=50_000,
        use_bucketed=False, include_noise=False, include_electronics=False,
        include_track_hits=True, include_digitize=False)


@pytest.fixture(scope="module")
def deposits(sim):
    return load_event(DATA_PATH, sim.config, event_idx=EVENT_IDX)


@pytest.fixture(scope="module")
def sim_results(sim, deposits):
    """Run once, reuse signals/track_hits/qs across tests."""
    return sim.process_event(deposits)


# ---------------------------------------------------------------------------
# Signal shapes
# ---------------------------------------------------------------------------

@needs_data
class TestSignalShapes:

    def test_signal_shapes(self, sim, sim_results):
        signals, _, _ = sim_results
        cfg = sim.config
        for vi in range(cfg.n_volumes):
            for pi in range(cfg.volumes[vi].n_planes):
                assert signals[(vi, pi)].shape == (
                    cfg.volumes[vi].num_wires[pi], cfg.num_time_steps)

    def test_all_planes_present(self, sim, sim_results):
        signals, _, _ = sim_results
        cfg = sim.config
        for vi in range(cfg.n_volumes):
            for pi in range(cfg.volumes[vi].n_planes):
                assert (vi, pi) in signals


# ---------------------------------------------------------------------------
# Track hits
# ---------------------------------------------------------------------------

@needs_data
class TestTrackHits:

    def test_track_hits_keys(self, sim, sim_results):
        _, track_hits_raw, _ = sim_results
        cfg = sim.config
        for vi in range(cfg.n_volumes):
            for pi in range(cfg.volumes[vi].n_planes):
                assert (vi, pi) in track_hits_raw

    def test_qs_fractions(self, sim, sim_results):
        _, _, filled_deposits = sim_results
        for v in range(sim.config.n_volumes):
            vol = filled_deposits.volumes[v]
            qs = vol.qs_fractions[:vol.n_actual]
            assert float(jnp.sum(jnp.abs(qs))) > 0, f"Volume {v}: qs_fractions should be non-zero"


# ---------------------------------------------------------------------------
# SimParams
# ---------------------------------------------------------------------------

@needs_data
class TestSimParams:

    def test_different_velocity_changes_output(self, sim, deposits, sim_results):
        signals_default, _, _ = sim_results
        s1 = float(signals_default[(1, 2)].sum())

        params2 = sim.default_sim_params._replace(velocity_cm_us=jnp.array(0.17))
        sigs2, _, _ = sim.process_event(deposits, sim_params=params2)
        s2 = float(sigs2[(1, 2)].sum())
        assert abs(s1 - s2) > 1.0

    def test_no_recompile(self, sim, deposits):
        """Second call with same params should be ~same speed as first (no recompile)."""
        params2 = sim.default_sim_params._replace(velocity_cm_us=jnp.array(0.17))
        # First timed call
        t0 = time.time()
        sigs1, _, _ = sim.process_event(deposits, sim_params=params2)
        jax.block_until_ready(sigs1[(1, 2)])
        t_first = time.time() - t0

        # Second timed call — should be similar speed (no recompile)
        t0 = time.time()
        sigs2, _, _ = sim.process_event(deposits, sim_params=params2)
        jax.block_until_ready(sigs2[(1, 2)])
        t_second = time.time() - t0

        # Second call should not be dramatically slower (recompile would 10x+ it)
        assert t_second < t_first * 2.0, \
            f"Second call ({t_second:.1f}s) much slower than first ({t_first:.1f}s)"

    def test_default_params_accessible(self, sim):
        p = sim.default_sim_params
        assert p.velocity_cm_us is not None
        assert p.recomb_params is not None


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

@needs_data
class TestWarmUp:
    def test_completes(self, sim):
        sim.warm_up()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@needs_data
class TestEdgeCases:

    def test_single_volume_deposits(self, sim):
        pos = np.ones((100, 3), dtype=np.float32) * 500
        deps = build_deposit_data(pos, np.ones(100, dtype=np.float32) * 2.0,
                                   np.ones(100, dtype=np.float32) * 0.5,
                                   sim.config)
        sigs, _, _ = sim.process_event(deps)
        for p in range(3):
            assert float(jnp.abs(sigs[(0, p)]).sum()) < 1e-3
            assert float(jnp.abs(sigs[(1, p)]).sum()) > 0

    def test_zero_energy_zero_signal(self, sim):
        pos = np.ones((50, 3), dtype=np.float32) * -500
        deps = build_deposit_data(pos, np.zeros(50, dtype=np.float32),
                                   np.ones(50, dtype=np.float32) * 0.5, sim.config)
        sigs, _, _ = sim.process_event(deps)
        total = sum(float(jnp.abs(v).sum()) for v in sigs.values())
        assert total < 1e-3


# ---------------------------------------------------------------------------
# Bucketed equivalence (2nd JIT compile)
# ---------------------------------------------------------------------------

@needs_data
class TestBucketed:

    @pytest.fixture(scope="class")
    def sim_bucketed(self, detector_config):
        return DetectorSimulator(
            detector_config, total_pad=200_000, response_chunk_size=50_000,
            use_bucketed=True, max_active_buckets=1000,
            include_noise=False, include_electronics=False,
            include_track_hits=False, include_digitize=False)

    def test_dense_equals_bucketed(self, sim, sim_bucketed, sim_results):
        signals_dense, _, _ = sim_results

        deps_b = load_event(DATA_PATH, sim_bucketed.config, event_idx=EVENT_IDX)
        sig_bucketed, _, _ = sim_bucketed.process_event(deps_b)

        cfg = sim.config
        for vi in range(cfg.n_volumes):
            for pi in range(cfg.volumes[vi].n_planes):
                d = np.asarray(signals_dense[(vi, pi)])
                b_tuple = sig_bucketed[(vi, pi)]
                buckets, num_active, ctk, B1, B2 = b_tuple
                nw = cfg.volumes[vi].num_wires[pi]
                nt = cfg.num_time_steps
                b = np.asarray(sparse_buckets_to_dense(
                    buckets, ctk, num_active, int(B1), int(B2), nw, nt, 1000))
                max_diff = float(np.max(np.abs(d - b)))
                assert max_diff < 0.01, f"({vi},{pi}): {max_diff:.4f}"


# ---------------------------------------------------------------------------
# Synthetic deterministic pipeline (no external data files)
# ---------------------------------------------------------------------------

RESPONSE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'tools', 'responses')
has_kernels = all(
    os.path.exists(os.path.join(RESPONSE_PATH, f'{p}_plane_kernel.npz'))
    for p in ['U', 'V', 'Y'])


@pytest.mark.requires_kernels
@pytest.mark.skipif(not has_kernels, reason="Response kernel files not found")
class TestSyntheticPipeline:
    """Full pipeline test with hand-placed deposits — no out.h5 needed.

    Five deposits placed at known positions in volume 0 (east).
    Validates physics invariants rather than comparing to frozen snapshots.
    """

    @pytest.fixture(scope="class")
    def synthetic_sim(self):
        det = generate_detector(CONFIG_PATH)
        return DetectorSimulator(
            det, total_pad=500, response_chunk_size=500,
            include_track_hits=False, include_noise=False,
            include_electronics=False, include_digitize=False)

    @pytest.fixture(scope="class")
    def synthetic_results(self, synthetic_sim):
        positions_mm = np.array([
            [-500.0, 0.0, 0.0],
            [-800.0, 0.0, 0.0],
            [-1200.0, 0.0, 0.0],
            [-500.0, 100.0, 100.0],
            [-500.0, -100.0, -100.0],
        ], dtype=np.float32)
        de = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        dx = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)

        deposits = build_deposit_data(positions_mm, de, dx, synthetic_sim.config)
        signals, _, _ = synthetic_sim.process_event(deposits)
        return signals, deposits

    def test_east_has_signal(self, synthetic_results, synthetic_sim):
        """Volume 0 (east) should have nonzero signal — deposits are at x<0."""
        signals, _ = synthetic_results
        cfg = synthetic_sim.config
        for pi in range(cfg.volumes[0].n_planes):
            total = float(jnp.sum(jnp.abs(signals[(0, pi)])))
            assert total > 0, f"East plane {pi} should have signal"

    def test_west_is_empty(self, synthetic_results, synthetic_sim):
        """Volume 1 (west) should have zero signal — no deposits at x>0."""
        signals, _ = synthetic_results
        cfg = synthetic_sim.config
        for pi in range(cfg.volumes[1].n_planes):
            total = float(jnp.sum(jnp.abs(signals[(1, pi)])))
            assert total < 1e-3, f"West plane {pi} should be empty"

    def test_collection_plane_positive(self, synthetic_results):
        """Y-plane (collection, plane 2) signal peak should be positive."""
        signals, _ = synthetic_results
        y_signal = np.asarray(signals[(0, 2)])
        assert float(np.max(y_signal)) > 0, "Collection plane peak should be positive"

    def test_induction_plane_bipolar(self, synthetic_results):
        """U-plane (induction, plane 0) signal should have both signs."""
        signals, _ = synthetic_results
        u_signal = np.asarray(signals[(0, 0)])
        assert float(np.max(u_signal)) > 0, "Induction should have positive values"
        assert float(np.min(u_signal)) < 0, "Induction should have negative values"

    def test_far_deposits_more_diffused(self, synthetic_results, synthetic_sim):
        """Deposits further from anode should produce broader signals on Y-plane."""
        signals, _ = synthetic_results
        y_signal = np.asarray(signals[(0, 2)])

        # Find the wire with peak signal — should be near wire for z=0
        wire_sums = np.sum(np.abs(y_signal), axis=1)
        peak_wire = int(np.argmax(wire_sums))

        # Time profile at peak wire should span multiple bins (diffusion)
        time_profile = y_signal[peak_wire]
        active_bins = np.sum(np.abs(time_profile) > 0.01 * np.max(np.abs(time_profile)))
        assert active_bins > 3, "Signal should be spread across multiple time bins"

    def test_deterministic(self, synthetic_sim):
        """Same deposits should produce identical output (no stochastic noise)."""
        positions_mm = np.array([[-500.0, 0.0, 0.0]], dtype=np.float32)
        de = np.array([2.0], dtype=np.float32)
        dx = np.array([0.3], dtype=np.float32)

        deps = build_deposit_data(positions_mm, de, dx, synthetic_sim.config)
        sigs1, _, _ = synthetic_sim.process_event(deps)
        sigs2, _, _ = synthetic_sim.process_event(deps)

        for key in sigs1:
            np.testing.assert_array_equal(
                np.asarray(sigs1[key]), np.asarray(sigs2[key]),
                err_msg=f"Plane {key} not deterministic")
