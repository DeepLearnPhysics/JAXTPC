"""
Differentiable forward path integration tests (synthetic deposits).

- forward() reproduces the production response on every plane, at uniform and at
  mixed wire pitch
- every SimParams gradient matches a finite difference
- forward() does not recompile when SimParams change
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from gradient_checks import check_param_gradient, param_gradients
from tools.loader import build_deposit_data
from tools.simulation import DetectorSimulator


N = 64
PLANES = [(v, p) for v in range(2) for p in range(3)]


# ---------------------------------------------------------------------------
# Forward vs production consistency
# ---------------------------------------------------------------------------

def _run_both(detector, synthetic_tracks):
    prod = DetectorSimulator(
        detector, total_pad=N, response_chunk_size=N // 2,
        include_track_hits=False)
    diff = DetectorSimulator(detector, differentiable=True, n_segments=N)
    deposits = synthetic_tracks(prod.config)
    sig_prod, _, _ = prod.process_event(deposits)
    sig_diff = diff.forward(diff.default_sim_params, deposits)
    return diff, deposits, sig_prod, sig_diff


def _rel_l2(prod, diff):
    """||prod - diff|| / ||prod||: independent of the number of deposits or ticks."""
    return float(np.linalg.norm(prod - diff) / np.linalg.norm(prod))


@pytest.fixture(scope="module")
def uniform(compact_detector, synthetic_tracks):
    return _run_both(compact_detector(), synthetic_tracks)


@pytest.fixture(scope="module")
def mixed(compact_detector, synthetic_tracks):
    """ICARUS-like pitches: plane 0 at half the pitch of the other two."""
    return _run_both(compact_detector((0.15, 0.3, 0.3)), synthetic_tracks)


class TestForwardVsProduction:

    @pytest.mark.parametrize("vol,plane", PLANES)
    def test_uniform_pitch(self, uniform, vol, plane):
        _, _, sig_prod, sig_diff = uniform
        prod = np.asarray(sig_prod[(vol, plane)])
        diff = np.asarray(sig_diff[PLANES.index((vol, plane))])
        assert len(sig_diff) == len(PLANES)
        assert prod.shape == diff.shape
        assert np.abs(prod).max() > 1.0   # electrons; not a comparison of zeros
        assert _rel_l2(prod, diff) < 1e-5

    @pytest.mark.parametrize("vol,plane", [
        pytest.param(v, p, marks=[] if p == 0 else pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason="production scales every plane's transverse diffusion by "
                   "wire_spacings_cm[0] (tools/simulation.py): sigma_T on Y is "
                   "0.363 cm, analytic 0.180 cm. Relative L2 vs forward() is "
                   + {1: "0.28-0.31 on V", 2: "0.39-0.42 on Y"}[p]))
        for v, p in PLANES])
    def test_mixed_pitch(self, mixed, vol, plane):
        _, _, sig_prod, sig_diff = mixed
        prod = np.asarray(sig_prod[(vol, plane)])
        diff = np.asarray(sig_diff[PLANES.index((vol, plane))])
        if prod.shape != diff.shape or np.abs(prod).max() <= 1.0:
            pytest.fail("unexpected shape or empty plane")   # not the known bug
        assert _rel_l2(prod, diff) < 1e-5


class TestForwardSegments:

    def test_matches_forward(self, uniform):
        """Global positions through forward_segments == the same deposits via forward.

        forward_segments fixes theta = phi = 0 (track along z), so forward gets the
        same angles.
        """
        sim = uniform[0]
        rng = np.random.RandomState(11)
        n = 40
        pos = rng.uniform([-2000, -250, -250], [2000, 250, 250], (n, 3)).astype(np.float32)
        de = np.linspace(0.5, 1.5, n, dtype=np.float32)
        deposits = build_deposit_data(
            pos, de, np.full(n, 3.0, np.float32), sim.config)
        params = sim.default_sim_params
        reference = sim.forward(params, deposits)
        segments = sim.forward_segments(params, jnp.asarray(pos), jnp.asarray(de), 3.0)
        for ref, seg in zip(reference, segments, strict=True):
            assert np.abs(np.asarray(ref)).max() > 1.0
            assert _rel_l2(np.asarray(ref), np.asarray(seg)) < 1e-5


class TestNoRecompile:

    def test_params_change_without_recompile(self, uniform, count_compiles):
        sim, deposits, _, reference = uniform
        fwd = jax.jit(sim.forward)
        # Positive control: the first call must be seen to compile
        assert count_compiles(
            lambda: jax.block_until_ready(fwd(sim.default_sim_params, deposits))) > 0
        params = sim.default_sim_params
        params2 = params._replace(velocity_cm_us=params.velocity_cm_us * 1.06,
                                  lifetime_us=params.lifetime_us * 0.5)
        out = {}
        assert count_compiles(
            lambda: out.update(s=jax.block_until_ready(fwd(params2, deposits)))) == 0
        # ... and the new values were used, not baked in
        for new, old in zip(out['s'], reference):
            assert _rel_l2(np.asarray(old), np.asarray(new)) > 0.1

    def test_eager_forward_does_not_recompile(self, uniform, count_compiles):
        sim, deposits, _, _ = uniform
        p2 = sim.default_sim_params._replace(lifetime_us=jnp.array(5000.0))
        jax.block_until_ready(sim.forward(p2, deposits))
        p3 = sim.default_sim_params._replace(lifetime_us=jnp.array(7000.0))
        assert count_compiles(
            lambda: jax.block_until_ready(sim.forward(p3, deposits))) == 0


# ---------------------------------------------------------------------------
# Every SimParams gradient vs finite differences
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def emb_gradients(compact_detector, synthetic_tracks):
    """Gradients in float64; 
    In float32 the difference of two nearly equal losses is rounding noise."""
    was_enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        sim = DetectorSimulator(compact_detector(), differentiable=True, n_segments=N,
                                recombination_model='emb')
        yield param_gradients(sim, synthetic_tracks(sim.config))
    finally:
        jax.config.update("jax_enable_x64", was_enabled)


class TestParamGradients:

    @pytest.mark.parametrize("field", [
        'velocity_cm_us', 'lifetime_us',
        'diffusion_trans_cm2_us', 'diffusion_long_cm2_us',
        'recomb_params.alpha', 'recomb_params.beta_90', 'recomb_params.R'])
    def test_matches_finite_difference(self, emb_gradients, field):
        check_param_gradient(emb_gradients, field)
