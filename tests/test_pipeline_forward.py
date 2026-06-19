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


# ---------------------------------------------------------------------------
# Finite-difference gradient verification (no external data)
# ---------------------------------------------------------------------------

RESPONSE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'tools', 'responses')
has_kernels = all(
    os.path.exists(os.path.join(RESPONSE_PATH, f'{p}_plane_kernel.npz'))
    for p in ['U', 'V', 'Y'])


@pytest.mark.requires_kernels
@pytest.mark.skipif(not has_kernels, reason="Response kernel files not found")
class TestFiniteDifferenceGradients:
    """Verify AD gradients match finite-difference approximation.

    Uses forward_segments with a tiny synthetic track — no out.h5 needed.
    This catches gradient correctness, not just "gradients are nonzero."
    """

    @pytest.fixture(scope="class")
    def _x64(self):
        """Run these gradient checks in float64.

        Comparing an AD gradient to a CENTRAL finite difference of a float32
        loss suffers catastrophic cancellation: e.g. lifetime ~10000 us with a
        small eps makes loss(+eps) and loss(-eps) agree to within float32
        rounding, so the FD value is platform/JAX-version-dependent noise (it
        passed on jax 0.5.3 / py3.10 but produced a garbage FD=1000.0 on the
        CI py3.11 runner). In float64 the FD converges to the true gradient
        (rel_err ~ 0) regardless of platform, so the comparison is sound.
        """
        was_enabled = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", was_enabled)

    @pytest.fixture(scope="class")
    def fd_sim(self, _x64):
        det = generate_detector(CONFIG_PATH)
        return DetectorSimulator(
            det, differentiable=True, n_segments=50)

    @pytest.fixture(scope="class")
    def fd_track(self, _x64):
        """Straight track at x=-500mm, spanning y/z = [-50, 50]mm."""
        N = 50
        t = jnp.linspace(0, 1, N)
        positions = jnp.stack([
            jnp.full(N, -500.0),
            t * 100.0 - 50.0,
            t * 100.0 - 50.0,
        ], axis=-1)
        de = jnp.ones(N) * 2.0
        return positions, de

    def _loss(self, sim, params, positions, de):
        sigs = sim.forward_segments(params, positions, de, 5.0)
        return sum(float(jnp.sum(s ** 2)) for s in sigs)

    def _loss_jax(self, sim, params, positions, de):
        sigs = sim.forward_segments(params, positions, de, 5.0)
        return sum(jnp.sum(s ** 2) for s in sigs)

    def test_velocity_gradient_sign_and_order(self, fd_sim, fd_track):
        """AD and FD velocity gradients should agree in sign and order of magnitude.

        Velocity shifts signals across discrete time bins, so the AD gradient
        through kernel interpolation has inherent discretization error (~2x).
        We check sign agreement and same order of magnitude, not tight match.
        """
        positions, de = fd_track
        params = fd_sim.default_sim_params
        v0 = float(params.velocity_cm_us)
        eps = v0 * 1e-5

        ad_grad = jax.grad(
            lambda p: self._loss_jax(fd_sim, p, positions, de)
        )(params)
        ad_val = float(ad_grad.velocity_cm_us)

        params_plus = params._replace(
            velocity_cm_us=params.velocity_cm_us + eps)
        params_minus = params._replace(
            velocity_cm_us=params.velocity_cm_us - eps)
        fd_val = (self._loss(fd_sim, params_plus, positions, de)
                  - self._loss(fd_sim, params_minus, positions, de)) / (2 * eps)

        assert abs(ad_val) > 0, "AD gradient should be nonzero"
        assert abs(fd_val) > 0, "FD gradient should be nonzero"
        assert np.sign(ad_val) == np.sign(fd_val), \
            f"Sign mismatch: AD={ad_val:.2e}, FD={fd_val:.2e}"
        ratio = abs(ad_val / fd_val)
        assert 0.1 < ratio < 10, \
            f"Order of magnitude mismatch: AD={ad_val:.2e}, FD={fd_val:.2e}, ratio={ratio:.2f}"

    def test_lifetime_gradient(self, fd_sim, fd_track):
        """AD gradient of lifetime should match finite-difference."""
        positions, de = fd_track
        params = fd_sim.default_sim_params
        eps = 1e-3

        ad_grad = jax.grad(
            lambda p: self._loss_jax(fd_sim, p, positions, de)
        )(params)
        ad_val = float(ad_grad.lifetime_us)

        params_plus = params._replace(
            lifetime_us=params.lifetime_us + eps)
        params_minus = params._replace(
            lifetime_us=params.lifetime_us - eps)
        fd_val = (self._loss(fd_sim, params_plus, positions, de)
                  - self._loss(fd_sim, params_minus, positions, de)) / (2 * eps)

        assert abs(ad_val) > 0, "AD gradient should be nonzero"
        rel_err = abs(ad_val - fd_val) / max(abs(ad_val), abs(fd_val), 1e-10)
        assert rel_err < 0.05, \
            f"lifetime grad: AD={ad_val:.6f}, FD={fd_val:.6f}, rel_err={rel_err:.4f}"

    def test_recomb_alpha_gradient(self, fd_sim, fd_track):
        """AD gradient of recombination alpha should match finite-difference."""
        positions, de = fd_track
        params = fd_sim.default_sim_params
        eps = 1e-4

        ad_grad = jax.grad(
            lambda p: self._loss_jax(fd_sim, p, positions, de)
        )(params)
        ad_val = float(ad_grad.recomb_params.alpha)

        rp = params.recomb_params
        params_plus = params._replace(
            recomb_params=rp._replace(alpha=rp.alpha + eps))
        params_minus = params._replace(
            recomb_params=rp._replace(alpha=rp.alpha - eps))
        fd_val = (self._loss(fd_sim, params_plus, positions, de)
                  - self._loss(fd_sim, params_minus, positions, de)) / (2 * eps)

        assert abs(ad_val) > 0, "AD gradient should be nonzero"
        rel_err = abs(ad_val - fd_val) / max(abs(ad_val), abs(fd_val))
        assert rel_err < 0.05, \
            f"alpha grad: AD={ad_val:.4f}, FD={fd_val:.4f}, rel_err={rel_err:.4f}"
