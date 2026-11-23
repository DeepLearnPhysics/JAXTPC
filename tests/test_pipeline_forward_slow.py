"""
Slow differentiable tests: Modified Box gradients, recombination model selection,
gradients with respect to deposit positions. Synthetic deposits only.

Skip with: pytest -m "not slow"
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from gradient_checks import check_param_gradient, param_gradients
from tools.loader import build_deposit_data
from tools.simulation import DetectorSimulator

pytestmark = pytest.mark.slow

N = 64


@pytest.fixture(scope="module")
def x64():
    was_enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was_enabled)


@pytest.fixture(scope="module")
def sims(x64, compact_detector):
    return {m: DetectorSimulator(compact_detector(), differentiable=True,
                                 n_segments=N, recombination_model=m)
            for m in ('modified_box', 'emb')}


# ---------------------------------------------------------------------------
# Recombination model selection and Modified Box gradients
# ---------------------------------------------------------------------------

def _survival(model, dedx, cos_drift):
    """Survival fraction written out independently of tools/recombination.py."""
    field_density = 1.396 * 0.5            # g/cm3 * kV/cm
    if model == 'modified_box':
        alpha, beta = 0.93, 0.212
    else:
        alpha, r = 0.904, 1.25
        beta = 0.204 / np.sqrt(1 - cos_drift ** 2 + cos_drift ** 2 / r ** 2)
    xi = beta * dedx / field_density
    return np.log(alpha + xi) / xi


class TestRecombModels:

    def test_signal_ratio_is_survival_ratio(self, sims):
        """emb / box signals differ by exactly the ratio of survival fractions."""
        dedx, cos_drift = 8.0, np.cos(0.4)
        sigs = {}
        for model, sim in sims.items():
            pos = np.stack([np.full(N, -700.0), np.linspace(-150, 150, N),
                            np.linspace(-120, 200, N)], axis=1).astype(np.float32)
            # track along x rotated by 0.4 rad about z
            deps = build_deposit_data(
                pos, np.full(N, dedx * 0.3, np.float32), np.full(N, 3.0, np.float32),
                sim.config, theta=np.full(N, np.pi / 2, np.float32),
                phi=np.full(N, 0.4, np.float32))
            sigs[model] = sim.forward(sim.default_sim_params, deps)
        expected = _survival('emb', dedx, cos_drift) / _survival('modified_box', dedx, cos_drift)
        assert abs(expected - 1) > 0.05        # the models must be distinguishable here
        for p in range(3):                     # volume 0 holds all the charge
            box, emb = (np.asarray(sigs[m][p]) for m in ('modified_box', 'emb'))
            scale = np.vdot(box, emb) / np.vdot(box, box)
            assert scale == pytest.approx(expected, rel=1e-4)
            assert np.linalg.norm(emb - scale * box) < 1e-5 * np.linalg.norm(emb)


class TestBoxGradients:

    @pytest.fixture(scope="class")
    def box_gradients(self, sims, synthetic_tracks):
        return param_gradients(sims['modified_box'], synthetic_tracks(sims['modified_box'].config))

    @pytest.mark.parametrize("field", ['recomb_params.alpha', 'recomb_params.beta'])
    def test_matches_finite_difference(self, box_gradients, field):
        check_param_gradient(box_gradients, field)


# ---------------------------------------------------------------------------
# Gradients through deposit positions
# ---------------------------------------------------------------------------

class TestPositionGradients:

    def test_track_parameter_gradients(self, sims):
        """d loss / d (x0, y0, z0, angle) through forward_segments == central FD."""
        sim = sims['emb']
        params = sim.default_sim_params
        t = jnp.linspace(0.0, 100.0, N)

        @jax.jit
        def loss(q):
            x = q[0] + t * jnp.cos(q[3])
            y = q[1] + t * jnp.sin(q[3]) * 0.5
            z = q[2] + t * jnp.sin(q[3]) * 0.866
            pos = jnp.stack([x, y, z], axis=-1)
            sigs = sim.forward_segments(params, pos, jnp.full(N, 0.5), 5.0)
            return sum(jnp.sum(s ** 2) for s in sigs)

        q0 = jnp.array([503.17, 3.1, -7.3, 0.3])
        ad = np.asarray(jax.grad(loss)(q0))
        fd = np.empty(4)
        for i, eps in enumerate([1e-5, 1e-5, 1e-5, 1e-7]):
            d = jnp.zeros(4).at[i].set(eps)
            fd[i] = float(loss(q0 + d) - loss(q0 - d)) / (2 * eps)
        assert np.all(np.abs(fd) > 0), fd
        np.testing.assert_allclose(ad, fd, rtol=1e-3)
