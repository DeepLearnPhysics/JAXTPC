"""
Production pipeline integration tests.

One simulator (dense + track_hits) shared across all tests via module scope.
Plus one bucketed simulator for dense==bucketed comparison.

Tests: baseline regression, track hits, qs_fractions, warm_up,
custom sim_params, edge cases, bucketed equivalence.
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
# Baseline regression
# ---------------------------------------------------------------------------

@needs_data
class TestBaselineRegression:

    def test_signals_match_baselines(self, sim, sim_results):
        signals, _, _ = sim_results
        baseline = np.load('tests/baselines/baseline_signals_minimal.npz')
        for vi in range(2):
            for pi in range(3):
                ref = baseline[f'signal_{vi}_{pi}']
                new = np.asarray(signals[(vi, pi)])
                max_diff = float(np.max(np.abs(new - ref)))
                assert max_diff < 0.5, f"({vi},{pi}): max_diff={max_diff:.4f}"

    def test_signal_shapes(self, sim, sim_results):
        signals, _, _ = sim_results
        cfg = sim.config
        for vi in range(cfg.n_volumes):
            for pi in range(cfg.volumes[vi].n_planes):
                assert signals[(vi, pi)].shape == (
                    cfg.volumes[vi].num_wires[pi], cfg.num_time_steps)


# ---------------------------------------------------------------------------
# Track hits
# ---------------------------------------------------------------------------

@needs_data
class TestTrackHits:

    def test_track_hits_match_baselines(self, sim, deposits):
        # Run fresh — finalize_track_hits mutates the dict (pops group_to_track)
        _, track_hits_raw, _ = sim.process_event(deposits)
        track_hits = sim.finalize_track_hits(track_hits_raw)
        baseline = np.load('tests/baselines/baseline_track_hits.npz')
        for vi in range(2):
            for pi in range(3):
                ref = int(baseline[f'num_hits_{vi}_{pi}'])
                new = int(track_hits[(vi, pi)]['num_hits'])
                # Allow small tolerance — t0_us windowing may shift edge deposits
                assert abs(ref - new) <= 2, f"({vi},{pi}): ref={ref}, new={new}, diff={abs(ref-new)}"

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
