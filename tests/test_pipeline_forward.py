"""
Differentiable forward path integration tests.

One diff simulator shared across all tests via module scope.

Tests: signals, baseline match, gradients, no-recompile,
edge cases, forward vs production consistency.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data
from tools.config import pad_deposit_data


CONFIG_PATH = 'config/cubic_wireplane_config.yaml'
DATA_PATH = 'out.h5'
N = 1000

needs_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason="out.h5 not found")


# ---------------------------------------------------------------------------
# Module-scoped fixtures (1 JIT compile)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector_config():
    return generate_detector(CONFIG_PATH)


@pytest.fixture(scope="module")
def sim(detector_config):
    return DetectorSimulator(detector_config, differentiable=True, n_segments=N)


@pytest.fixture(scope="module")
def raw():
    return load_particle_step_data(DATA_PATH, event_idx=2)


@pytest.fixture(scope="module")
def deposits(raw, sim):
    return build_deposit_data(
        raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim.config,
        theta=raw['theta'][:N], phi=raw['phi'][:N])


@pytest.fixture(scope="module")
def signals(sim, deposits):
    """Run forward once, reuse across tests."""
    return sim.forward(sim.default_sim_params, deposits)


@pytest.fixture(scope="module")
def grads(sim, deposits):
    """Compute gradients once, reuse across tests."""
    def loss(params):
        sigs = sim.forward(params, deposits)
        return sum(jnp.sum(s ** 2) for s in sigs)
    return jax.grad(loss)(sim.default_sim_params)


# ---------------------------------------------------------------------------
# Basic forward
# ---------------------------------------------------------------------------

@needs_data
class TestForwardBasic:

    def test_six_signals(self, signals):
        assert len(signals) == 6

    def test_shapes(self, sim, signals):
        cfg = sim.config
        expected = [(cfg.volumes[v].num_wires[p], cfg.num_time_steps)
                    for v in range(cfg.n_volumes)
                    for p in range(cfg.volumes[v].n_planes)]
        for i, sig in enumerate(signals):
            assert sig.shape == expected[i]

    def test_nonzero_signal(self, signals):
        total = sum(float(jnp.sum(jnp.abs(s))) for s in signals)
        assert total > 0

    def test_baseline_match(self, signals):
        baseline = np.load('tests/baselines/baseline_diff_signals.npz')
        for i, sig in enumerate(signals):
            ref = baseline[f'signal_{i}']
            ref_sum = float(np.sum(np.abs(ref)))
            if ref_sum < 1e-6:
                continue
            max_diff = float(np.max(np.abs(np.asarray(sig) - ref)))
            assert max_diff / ref_sum < 0.02, f"Signal {i}: rel_diff={max_diff/ref_sum:.4f}"


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

@needs_data
class TestGradients:

    def test_velocity_nonzero(self, grads):
        assert float(grads.velocity_cm_us) != 0.0

    def test_lifetime_nonzero(self, grads):
        assert float(grads.lifetime_us) != 0.0

    def test_recomb_alpha_nonzero(self, grads):
        assert float(grads.recomb_params.alpha) != 0.0

    def test_gradients_finite(self, grads):
        """All gradient fields should be finite (no NaN/Inf)."""
        assert np.isfinite(float(grads.velocity_cm_us))
        assert np.isfinite(float(grads.lifetime_us))
        assert np.isfinite(float(grads.diffusion_trans_cm2_us))
        assert np.isfinite(float(grads.diffusion_long_cm2_us))


# ---------------------------------------------------------------------------
# No recompile
# ---------------------------------------------------------------------------

@needs_data
class TestNoRecompile:

    def test_second_call_fast(self, sim, deposits):
        sim.forward(sim.default_sim_params, deposits)  # warm
        t0 = time.time()
        sim.forward(sim.default_sim_params, deposits)
        assert time.time() - t0 < 2.0

    def test_different_velocity_different_output(self, sim, deposits):
        params2 = sim.default_sim_params._replace(velocity_cm_us=jnp.array(0.17))
        sigs1 = sim.forward(sim.default_sim_params, deposits)
        sigs2 = sim.forward(params2, deposits)
        diff = sum(float(jnp.sum(jnp.abs(s1 - s2)))
                   for s1, s2 in zip(sigs1, sigs2))
        assert diff > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@needs_data
class TestEdgeCases:

    def test_single_volume_zeros_other(self, sim):
        pos = np.ones((N, 3), dtype=np.float32) * 500
        deps = build_deposit_data(pos, np.ones(N, dtype=np.float32) * 2.0,
                                   np.ones(N, dtype=np.float32) * 0.5, sim.config)
        sigs = sim.forward(sim.default_sim_params, deps)
        vol0 = sum(float(jnp.abs(sigs[i]).sum()) for i in range(3))
        vol1 = sum(float(jnp.abs(sigs[i]).sum()) for i in range(3, 6))
        assert vol0 < 1e-6
        assert vol1 > 0

    def test_zero_energy_zero_signal(self, sim):
        pos = np.ones((N, 3), dtype=np.float32) * -500
        deps = build_deposit_data(pos, np.zeros(N, dtype=np.float32),
                                   np.ones(N, dtype=np.float32) * 0.5, sim.config)
        sigs = sim.forward(sim.default_sim_params, deps)
        total = sum(float(jnp.abs(s).sum()) for s in sigs)
        assert total < 1e-6


# ---------------------------------------------------------------------------
# Forward vs production consistency
# ---------------------------------------------------------------------------

@needs_data
class TestForwardVsProduction:

    def test_signals_match(self, detector_config, raw):
        sim_prod = DetectorSimulator(
            detector_config, total_pad=N, response_chunk_size=N,
            include_track_hits=False, include_noise=False,
            include_electronics=False, include_digitize=False)

        deps_prod = build_deposit_data(
            raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim_prod.config,
            theta=raw['theta'][:N], phi=raw['phi'][:N])
        sigs_prod, _, _ = sim_prod.process_event(deps_prod)

        sim_diff = DetectorSimulator(
            detector_config, differentiable=True, n_segments=N)
        deps_diff = build_deposit_data(
            raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim_diff.config,
            theta=raw['theta'][:N], phi=raw['phi'][:N])
        deps_diff = pad_deposit_data(deps_diff, sim_diff.config.total_pad)
        sigs_diff = sim_diff.forward(sim_diff.default_sim_params, deps_diff)

        cfg = sim_prod.config
        plane_order = [(v, p)
            for v in range(cfg.n_volumes)
            for p in range(cfg.volumes[v].n_planes)]
        for i, (vol, plane) in enumerate(plane_order):
            prod = np.asarray(sigs_prod[(vol, plane)])
            diff = np.asarray(sigs_diff[i])
            prod_sum = float(np.sum(np.abs(prod)))
            if prod_sum < 1e-6:
                continue
            rel = float(np.max(np.abs(prod - diff))) / prod_sum
            assert rel < 0.01, f"({vol},{plane}): rel={rel:.4f}"
