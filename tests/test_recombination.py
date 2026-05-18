"""
Tests for charge and light recombination.

File under test: tools/recombination.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from tools.recombination import compute_quanta, XI_FN
from tools.config import ModifiedBoxParams, EMBParams


class TestComputeQuantaModifiedBox:
    """Tests for compute_quanta with Modified Box xi function."""

    @pytest.fixture
    def xi_fn(self):
        return XI_FN['modified_box']

    def test_mip_survival_fraction(self, recomb_params, xi_fn):
        """MIP at dE/dx=2.1 MeV/cm should give R ~ 0.705 analytically."""
        field_kVcm = float(recomb_params.field_strength_Vcm) / 1000.0
        dE_dx = 2.1
        xi = (float(recomb_params.beta) / float(recomb_params.density)) * dE_dx / field_kVcm
        expected_R = np.log(float(recomb_params.alpha) + xi) / xi

        de = jnp.array([dE_dx * 1.0])
        dx = jnp.array([1.0])
        phi = jnp.zeros(1)
        e_field = recomb_params.field_strength_Vcm

        Q, L = compute_quanta(de, dx, phi, e_field, recomb_params, xi_fn)
        jax.block_until_ready(Q)

        w_value_mev = float(recomb_params.w_value) * 1e-6
        initial_charge = float(de[0]) / w_value_mev
        actual_R = float(Q[0]) / initial_charge

        np.testing.assert_allclose(actual_R, expected_R, rtol=1e-4)

    def test_higher_dedx_lower_survival(self, recomb_params, xi_fn):
        """Higher dE/dx should give lower survival fraction."""
        de_low = jnp.array([2.1])
        de_high = jnp.array([10.0])
        dx = jnp.array([1.0])
        phi = jnp.zeros(1)
        e_field = recomb_params.field_strength_Vcm

        Q_low, _ = compute_quanta(de_low, dx, phi, e_field, recomb_params, xi_fn)
        Q_high, _ = compute_quanta(de_high, dx, phi, e_field, recomb_params, xi_fn)
        jax.block_until_ready(Q_low)
        jax.block_until_ready(Q_high)

        w_value_mev = float(recomb_params.w_value) * 1e-6
        R_low = float(Q_low[0]) / (float(de_low[0]) / w_value_mev)
        R_high = float(Q_high[0]) / (float(de_high[0]) / w_value_mev)

        assert R_high < R_low

    def test_dx_zero_returns_zero(self, recomb_params, xi_fn):
        """dx=0 should return zero charge and zero light."""
        de = jnp.array([5.0])
        dx = jnp.array([0.0])
        phi = jnp.zeros(1)

        Q, L = compute_quanta(de, dx, phi, recomb_params.field_strength_Vcm,
                              recomb_params, xi_fn)
        jax.block_until_ready(Q)

        assert float(Q[0]) == 0.0
        assert float(L[0]) == 0.0

    def test_negative_de_returns_zero(self, recomb_params, xi_fn):
        """Negative energy deposit should return zero charge and light."""
        de = jnp.array([-5.0])
        dx = jnp.array([1.0])
        phi = jnp.zeros(1)

        Q, L = compute_quanta(de, dx, phi, recomb_params.field_strength_Vcm,
                              recomb_params, xi_fn)
        jax.block_until_ready(Q)

        assert float(Q[0]) == 0.0
        assert float(L[0]) == 0.0

    def test_survival_fraction_in_0_1(self, recomb_params, xi_fn):
        """Survival fraction should always be in [0, 1] for physical inputs."""
        rng = np.random.RandomState(99)
        n = 500
        de = jnp.array(rng.uniform(0.01, 50.0, size=n), dtype=jnp.float32)
        dx = jnp.array(rng.uniform(0.01, 5.0, size=n), dtype=jnp.float32)
        phi = jnp.zeros(n)

        Q, L = compute_quanta(de, dx, phi, recomb_params.field_strength_Vcm,
                              recomb_params, xi_fn)
        jax.block_until_ready(Q)

        w_value_mev = float(recomb_params.w_value) * 1e-6
        initial_charge = de / w_value_mev
        R = Q / initial_charge

        assert jnp.all(R >= 0.0)
        assert jnp.all(R <= 1.0)

    def test_vectorized_correctness(self, recomb_params, xi_fn):
        """Vectorized calculation should match element-wise loop."""
        rng = np.random.RandomState(42)
        n = 100
        de = jnp.array(rng.uniform(0.5, 10.0, size=n), dtype=jnp.float32)
        dx = jnp.array(rng.uniform(0.01, 1.0, size=n), dtype=jnp.float32)
        phi = jnp.zeros(n)

        Q_vec, L_vec = compute_quanta(de, dx, phi, recomb_params.field_strength_Vcm,
                                      recomb_params, xi_fn)
        jax.block_until_ready(Q_vec)

        Q_singles = []
        L_singles = []
        for i in range(n):
            q, l = compute_quanta(de[i:i+1], dx[i:i+1], phi[i:i+1],
                                  recomb_params.field_strength_Vcm, recomb_params, xi_fn)
            jax.block_until_ready(q)
            Q_singles.append(float(q[0]))
            L_singles.append(float(l[0]))

        np.testing.assert_allclose(np.array(Q_vec), np.array(Q_singles), rtol=1e-4)
        np.testing.assert_allclose(np.array(L_vec), np.array(L_singles), rtol=1e-4)


class TestChargeLightAnticorrelation:
    """Tests for the Q + L = ΔE/W_ph conservation law."""

    @pytest.fixture
    def xi_fn_box(self):
        return XI_FN['modified_box']

    @pytest.fixture
    def emb_params(self):
        return EMBParams(
            density=jnp.array(1.396),
            w_value=jnp.array(23.6),
            excitation_ratio=jnp.array(0.21),
            field_strength_Vcm=jnp.array(500.0),
            alpha=jnp.array(0.904),
            beta_90=jnp.array(0.204),
            R=jnp.array(1.25),
        )

    @pytest.fixture
    def xi_fn_emb(self):
        return XI_FN['emb']

    def test_q_plus_l_equals_total_quanta_box(self, recomb_params, xi_fn_box):
        """Q + L should equal ΔE / W_ph for Modified Box."""
        rng = np.random.RandomState(7)
        n = 200
        de = jnp.array(rng.uniform(0.5, 20.0, size=n), dtype=jnp.float32)
        dx = jnp.array(rng.uniform(0.01, 2.0, size=n), dtype=jnp.float32)
        phi = jnp.zeros(n)

        Q, L = compute_quanta(de, dx, phi, recomb_params.field_strength_Vcm,
                              recomb_params, xi_fn_box)
        jax.block_until_ready(Q)

        W_ph_mev = float(recomb_params.w_value) * 1e-6 / (1.0 + float(recomb_params.excitation_ratio))
        N_total = de / W_ph_mev

        np.testing.assert_allclose(np.array(Q + L), np.array(N_total), rtol=1e-5)

    def test_q_plus_l_equals_total_quanta_emb(self, emb_params, xi_fn_emb):
        """Q + L should equal ΔE / W_ph for EMB (with angular variation)."""
        rng = np.random.RandomState(8)
        n = 200
        de = jnp.array(rng.uniform(0.5, 20.0, size=n), dtype=jnp.float32)
        dx = jnp.array(rng.uniform(0.01, 2.0, size=n), dtype=jnp.float32)
        phi = jnp.array(rng.uniform(0, np.pi/2, size=n), dtype=jnp.float32)

        Q, L = compute_quanta(de, dx, phi, emb_params.field_strength_Vcm,
                              emb_params, xi_fn_emb)
        jax.block_until_ready(Q)

        W_ph_mev = float(emb_params.w_value) * 1e-6 / (1.0 + float(emb_params.excitation_ratio))
        N_total = de / W_ph_mev

        np.testing.assert_allclose(np.array(Q + L), np.array(N_total), rtol=1e-5)

    def test_more_charge_means_less_light(self, recomb_params, xi_fn_box):
        """Higher E-field → more charge escapes → fewer photons."""
        de = jnp.array([5.0])
        dx = jnp.array([1.0])
        phi = jnp.zeros(1)

        Q_low_E, L_low_E = compute_quanta(de, dx, phi, jnp.array(250.0),
                                           recomb_params, xi_fn_box)
        Q_high_E, L_high_E = compute_quanta(de, dx, phi, jnp.array(750.0),
                                             recomb_params, xi_fn_box)
        jax.block_until_ready(Q_low_E)
        jax.block_until_ready(Q_high_E)

        # Higher field → more charge
        assert float(Q_high_E[0]) > float(Q_low_E[0])
        # Higher field → less light (anti-correlation)
        assert float(L_high_E[0]) < float(L_low_E[0])

    def test_emb_angle_affects_partition(self, emb_params, xi_fn_emb):
        """EMB: parallel tracks (φ→0) should have more recombination → more light."""
        de = jnp.array([5.0, 5.0])
        dx = jnp.array([1.0, 1.0])
        phi_parallel = jnp.array([0.01])  # near-parallel to field
        phi_perp = jnp.array([np.pi / 2])  # perpendicular to field

        Q_par, L_par = compute_quanta(de[:1], dx[:1], phi_parallel,
                                       emb_params.field_strength_Vcm, emb_params, xi_fn_emb)
        Q_perp, L_perp = compute_quanta(de[:1], dx[:1], phi_perp,
                                         emb_params.field_strength_Vcm, emb_params, xi_fn_emb)
        jax.block_until_ready(Q_par)
        jax.block_until_ready(Q_perp)

        # Parallel → more recombination → less charge, more light
        assert float(Q_par[0]) < float(Q_perp[0])
        assert float(L_par[0]) > float(L_perp[0])

    def test_light_always_non_negative(self, recomb_params, xi_fn_box):
        """Photon count should never be negative for physical inputs."""
        rng = np.random.RandomState(42)
        n = 1000
        de = jnp.array(rng.uniform(0.01, 50.0, size=n), dtype=jnp.float32)
        dx = jnp.array(rng.uniform(0.01, 5.0, size=n), dtype=jnp.float32)
        phi = jnp.zeros(n)

        Q, L = compute_quanta(de, dx, phi, recomb_params.field_strength_Vcm,
                              recomb_params, xi_fn_box)
        jax.block_until_ready(L)

        assert jnp.all(L >= 0.0), f"Negative photons found: min={float(jnp.min(L))}"
