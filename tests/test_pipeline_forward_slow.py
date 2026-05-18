"""
Slow differentiable tests: multi-model gradients, all-field SimParams,
muon optimization through position generation.

Skip with: pytest -m "not slow"

2 diff simulators (modified_box + emb) shared via module scope.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data


CONFIG_PATH = 'config/cubic_wireplane_config.yaml'
DATA_PATH = 'out.h5'
N = 500

needs_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason="out.h5 not found")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def detector_config():
    return generate_detector(CONFIG_PATH)


@pytest.fixture(scope="module")
def raw():
    return load_particle_step_data(DATA_PATH, event_idx=2)


@pytest.fixture(scope="module")
def sim_box(detector_config):
    return DetectorSimulator(
        detector_config, differentiable=True, n_segments=N,
        recombination_model='modified_box')


@pytest.fixture(scope="module")
def sim_emb(detector_config):
    return DetectorSimulator(
        detector_config, differentiable=True, n_segments=N,
        recombination_model='emb')


# ---------------------------------------------------------------------------
# Multi-model recombination
# ---------------------------------------------------------------------------

@needs_data
class TestRecombModels:

    def test_different_models_different_output(self, sim_box, sim_emb, raw):
        deps_box = build_deposit_data(
            raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim_box.config,
            theta=raw['theta'][:N], phi=raw['phi'][:N])
        deps_emb = build_deposit_data(
            raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim_emb.config,
            theta=raw['theta'][:N], phi=raw['phi'][:N])

        sigs_box = sim_box.forward(sim_box.default_sim_params, deps_box)
        sigs_emb = sim_emb.forward(sim_emb.default_sim_params, deps_emb)

        box_sum = sum(float(jnp.sum(jnp.abs(s))) for s in sigs_box)
        emb_sum = sum(float(jnp.sum(jnp.abs(s))) for s in sigs_emb)
        assert abs(box_sum - emb_sum) > 1.0

    def test_box_alpha_beta_gradients(self, sim_box, raw):
        deps = build_deposit_data(
            raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim_box.config,
            theta=raw['theta'][:N], phi=raw['phi'][:N])

        def loss(params):
            return sum(jnp.sum(s ** 2) for s in sim_box.forward(params, deps))

        grads = jax.grad(loss)(sim_box.default_sim_params)
        assert float(grads.recomb_params.alpha) != 0.0
        assert float(grads.recomb_params.beta) != 0.0

    def test_emb_alpha_beta90_R_gradients(self, sim_emb, raw):
        deps = build_deposit_data(
            raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim_emb.config,
            theta=raw['theta'][:N], phi=raw['phi'][:N])

        def loss(params):
            return sum(jnp.sum(s ** 2) for s in sim_emb.forward(params, deps))

        grads = jax.grad(loss)(sim_emb.default_sim_params)
        assert float(grads.recomb_params.alpha) != 0.0
        assert float(grads.recomb_params.beta_90) != 0.0
        assert float(grads.recomb_params.R) != 0.0


# ---------------------------------------------------------------------------
# All SimParams fields (uses EMB — has more recomb params)
# ---------------------------------------------------------------------------

@needs_data
class TestAllSimParamsGradients:

    @pytest.fixture(scope="class")
    def grads(self, sim_emb, raw):
        deps = build_deposit_data(
            raw['positions_mm'][:N], raw['de'][:N], raw['dx'][:N], sim_emb.config,
            theta=raw['theta'][:N], phi=raw['phi'][:N])

        def loss(params):
            return sum(jnp.sum(s ** 2) for s in sim_emb.forward(params, deps))

        return jax.grad(loss)(sim_emb.default_sim_params)

    def test_velocity(self, grads):
        assert float(grads.velocity_cm_us) != 0.0

    def test_lifetime(self, grads):
        assert float(grads.lifetime_us) != 0.0

    def test_diffusion_trans(self, grads):
        assert float(grads.diffusion_trans_cm2_us) != 0.0

    def test_diffusion_long(self, grads):
        assert float(grads.diffusion_long_cm2_us) != 0.0

    def test_recomb_alpha(self, grads):
        assert float(grads.recomb_params.alpha) != 0.0

    def test_recomb_beta90(self, grads):
        assert float(grads.recomb_params.beta_90) != 0.0

    def test_recomb_R(self, grads):
        assert float(grads.recomb_params.R) != 0.0


# ---------------------------------------------------------------------------
# Muon optimization (gradient through position generation)
# ---------------------------------------------------------------------------

@needs_data
class TestMuonOptimization:

    @pytest.fixture(scope="class")
    def sim_muon(self, detector_config):
        return DetectorSimulator(
            detector_config, differentiable=True, n_segments=200)

    def test_gradient_flows_through_positions(self, sim_muon):
        def generate(start_x, angle, n):
            t = jnp.linspace(0, 100, n)
            x = start_x + t * jnp.cos(angle)
            y = t * jnp.sin(angle) * 0.5
            z = t * jnp.sin(angle) * 0.866
            return jnp.stack([x, y, z], axis=-1), jnp.ones(n) * 0.5

        params = sim_muon.default_sim_params
        pp = jnp.array([500.0, 0.3])

        def fwd(pp):
            pos, de = generate(pp[0], pp[1], 200)
            return sum(jnp.sum(s ** 2) for s in
                       sim_muon.forward_segments(params, pos, de, 5.0))

        grads = jax.grad(fwd)(pp)
        assert not jnp.any(jnp.isnan(grads))
        assert jnp.any(grads != 0)

    def test_optimization_step_reduces_loss(self, sim_muon):
        def generate(start_x, angle, n):
            t = jnp.linspace(0, 100, n)
            x = start_x + t * jnp.cos(angle)
            y = t * jnp.sin(angle) * 0.5
            z = t * jnp.sin(angle) * 0.866
            return jnp.stack([x, y, z], axis=-1), jnp.ones(n) * 0.5

        params = sim_muon.default_sim_params
        pp = jnp.array([500.0, 0.3])

        def fwd(pp):
            pos, de = generate(pp[0], pp[1], 200)
            return sum(jnp.sum(s ** 2) for s in
                       sim_muon.forward_segments(params, pos, de, 5.0))

        loss0 = float(fwd(pp))
        grads = jax.grad(fwd)(pp)
        pp_new = pp - 1e-6 * grads
        loss1 = float(fwd(pp_new))
        assert loss1 <= loss0 + 1e-6
