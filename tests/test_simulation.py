"""
Tests for simulation module - integration tests and physics validation.

File under test: tools/simulation.py
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from tools.simulation import DetectorSimulator
from tools.config import DepositData
from tools.loader import build_deposit_data, load_event


RESPONSE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools', 'responses')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'cubic_wireplane_config.yaml')

has_kernels = all(
    os.path.exists(os.path.join(RESPONSE_PATH, f'{p}_plane_kernel.npz'))
    for p in ['U', 'V', 'Y']
)
has_config = os.path.exists(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Integration tests requiring response kernels
# ---------------------------------------------------------------------------

def _make_deposit(n, sim_config, x_range=(-100, 100), de_val=2.0, dx_val=0.3,
                  theta_val=np.pi/4, phi_val=0.0, track_id=0):
    """Create a DepositData with uniform parameters using build_deposit_data."""
    rng = np.random.RandomState(42)
    x_mm = rng.uniform(x_range[0], x_range[1], size=n).astype(np.float32)
    y_mm = rng.uniform(-50, 50, size=n).astype(np.float32)
    z_mm = rng.uniform(-50, 50, size=n).astype(np.float32)
    positions_mm = np.stack([x_mm, y_mm, z_mm], axis=1)

    return build_deposit_data(
        positions_mm,
        np.full(n, de_val, dtype=np.float32),
        np.full(n, dx_val, dtype=np.float32),
        sim_config,
        theta=np.full(n, theta_val, dtype=np.float32),
        phi=np.full(n, phi_val, dtype=np.float32),
        track_ids=np.full(n, track_id, dtype=np.int32),
    )


def _make_single_deposit(x_mm, y_mm, z_mm, sim_config, de=2.0, dx=0.3,
                          theta=np.pi/4, phi=0.0, track_id=0):
    """Create a DepositData with a single particle."""
    return build_deposit_data(
        np.array([[x_mm, y_mm, z_mm]], dtype=np.float32),
        np.array([de], dtype=np.float32),
        np.array([dx], dtype=np.float32),
        sim_config,
        theta=np.array([theta], dtype=np.float32),
        phi=np.array([phi], dtype=np.float32),
        track_ids=np.array([track_id], dtype=np.int32),
    )


@pytest.fixture(scope="module")
def simulator():
    """Create a DetectorSimulator (expensive, shared across module)."""
    if not has_config or not has_kernels:
        pytest.skip("Config or kernel files not found")

    from tools.geometry import generate_detector
    detector = generate_detector(CONFIG_PATH)
    if detector is None:
        pytest.skip("Failed to load detector config")

    sim = DetectorSimulator(
        detector,
        response_path=RESPONSE_PATH,
        total_pad=25_000,
        response_chunk_size=25_000,
    )
    return sim


@pytest.mark.requires_kernels
@pytest.mark.slow
class TestBuildDepositData:
    """Tests for build_deposit_data volume splitting."""

    def test_volume_splitting(self, simulator):
        """Deposits spanning both volumes → verify per-volume counts."""
        deposit = _make_deposit(100, simulator.config, x_range=(-200, 200))
        n0 = deposit.volumes[0].n_actual
        n1 = deposit.volumes[1].n_actual
        assert n0 + n1 == 100
        assert n0 > 0 and n1 > 0

    def test_padded_data_preserves_entries(self, simulator):
        """Padded data should preserve original de values."""
        deposit = _make_deposit(50, simulator.config, x_range=(-200, -1))
        n = deposit.volumes[0].n_actual
        assert n == 50
        np.testing.assert_allclose(
            np.array(deposit.volumes[0].de[:n]), np.full(n, 2.0), rtol=1e-4)


@pytest.mark.requires_kernels
@pytest.mark.slow
class TestIntegration:
    """Integration tests running full simulation pipeline."""

    def test_zero_charge_zero_output(self, simulator):
        """de=0 everywhere should produce all-zero response signals."""
        deposit = _make_deposit(10, simulator.config, de_val=0.0)

        response_signals, track_hits, _ = simulator.process_event(deposit)

        for key, signals in response_signals.items():
            total = float(jnp.sum(jnp.abs(signals)))
            assert total == 0.0, f"Expected zero signal for plane {key}, got total={total}"

    def test_single_deposit_localization(self, simulator):
        """Single deposit should produce signal peak near expected wire/time."""
        # Place deposit at known location on east side
        deposit = _make_single_deposit(-50.0, 0.0, 0.0, simulator.config)

        response_signals, _, _ = simulator.process_event(deposit)

        # Y-plane (plane_idx=2) on east side (side_idx=0)
        y_signal = response_signals.get((0, 2))
        if y_signal is not None:
            # Signal should be non-zero
            total = float(jnp.sum(jnp.abs(y_signal)))
            assert total > 0, "Y-plane should have non-zero signal"

            # Peak should be localized (not spread uniformly)
            peak_val = float(jnp.max(jnp.abs(y_signal)))
            assert peak_val > total / y_signal.size * 10, \
                "Signal should be localized, not uniform"

    def test_recombination_applied(self, simulator):
        """dE/dx=2 vs dE/dx=10: signal ratio should reflect recombination."""
        # Low dE/dx
        deposit_low = _make_deposit(20, simulator.config, x_range=(-100, -1), de_val=2.0, dx_val=1.0)
        # High dE/dx
        deposit_high = _make_deposit(20, simulator.config, x_range=(-100, -1), de_val=10.0, dx_val=1.0)

        response_low, _, _ = simulator.process_event(deposit_low)
        response_high, _, _ = simulator.process_event(deposit_high)

        # Compare total signal on Y-plane east
        total_low = float(jnp.sum(jnp.abs(response_low.get((0, 2), jnp.zeros(1)))))
        total_high = float(jnp.sum(jnp.abs(response_high.get((0, 2), jnp.zeros(1)))))

        if total_low > 0 and total_high > 0:
            # Higher dE/dx deposits more energy but has lower survival fraction
            # So ratio of signals should differ from ratio of de
            ratio_de = 10.0 / 2.0  # 5x energy
            ratio_signal = total_high / total_low
            # Signal ratio should be less than de ratio (recombination eats more at high dE/dx)
            assert ratio_signal < ratio_de, \
                f"Recombination not applied: signal ratio {ratio_signal:.2f} >= de ratio {ratio_de}"


@pytest.mark.requires_kernels
@pytest.mark.slow
class TestPhysicsValidation:
    """Physics validation tests (cross-cutting)."""

    def test_more_drift_more_attenuation(self, simulator):
        """Charge farther from anode should have weaker signal."""
        # Close to east anode (x ≈ -200, short drift)
        deposit_near = _make_single_deposit(-1900.0, 0.0, 0.0, simulator.config)
        # Far from east anode (x ≈ 0, long drift)
        deposit_far = _make_single_deposit(-50.0, 0.0, 0.0, simulator.config)

        response_near, _, _ = simulator.process_event(deposit_near)
        response_far, _, _ = simulator.process_event(deposit_far)

        total_near = float(jnp.sum(jnp.abs(response_near.get((0, 2), jnp.zeros(1)))))
        total_far = float(jnp.sum(jnp.abs(response_far.get((0, 2), jnp.zeros(1)))))

        if total_near > 0 and total_far > 0:
            assert total_near > total_far, \
                f"Near-anode signal ({total_near:.2f}) should be stronger than far ({total_far:.2f})"

    def test_linear_charge_scaling(self, simulator):
        """de=1 vs de=2 → output should scale by recombined charge ratio."""
        deposit_1 = _make_deposit(20, simulator.config, x_range=(-100, -1), de_val=1.0, dx_val=1.0)
        deposit_2 = _make_deposit(20, simulator.config, x_range=(-100, -1), de_val=2.0, dx_val=1.0)

        response_1, _, _ = simulator.process_event(deposit_1)
        response_2, _, _ = simulator.process_event(deposit_2)

        total_1 = float(jnp.sum(jnp.abs(response_1.get((0, 2), jnp.zeros(1)))))
        total_2 = float(jnp.sum(jnp.abs(response_2.get((0, 2), jnp.zeros(1)))))

        if total_1 > 0 and total_2 > 0:
            # Compute expected ratio using recombination directly
            from tools.recombination import compute_quanta, XI_FN
            params = simulator.default_sim_params
            xi_fn = XI_FN[simulator.recomb_model]
            E_field = params.recomb_params.field_strength_Vcm
            phi_drift = jnp.array([np.pi / 4])
            Q1, _ = compute_quanta(
                jnp.array([1.0]), jnp.array([0.1]), phi_drift, E_field,
                params.recomb_params, xi_fn)
            Q2, _ = compute_quanta(
                jnp.array([2.0]), jnp.array([0.1]), phi_drift, E_field,
                params.recomb_params, xi_fn)
            expected_ratio = float(Q2[0]) / float(Q1[0])

            actual_ratio = total_2 / total_1
            np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.15,
                                       err_msg=f"Signal ratio {actual_ratio:.3f} differs from charge ratio {expected_ratio:.3f}")

    def test_y_plane_signal_polarity(self, simulator):
        """Y-plane (collection) signals should be predominantly positive (unipolar)."""
        deposit = _make_deposit(30, simulator.config, x_range=(-150, -10))

        response_signals, _, _ = simulator.process_event(deposit)
        y_signal = response_signals.get((0, 2))

        if y_signal is not None:
            total_positive = float(jnp.sum(jnp.maximum(y_signal, 0.0)))
            total_negative = float(jnp.sum(jnp.minimum(y_signal, 0.0)))

            if total_positive > 0:
                # Collection plane should have net positive signal
                assert total_positive > abs(total_negative), \
                    f"Y-plane should be net positive: pos={total_positive:.2f}, neg={total_negative:.2f}"

    def test_diffusion_broadens_with_distance(self, simulator):
        """Signal from farther charge should be broader (larger RMS spread)."""
        # Short drift (~20cm from east anode at x=-216)
        deposit_near = _make_single_deposit(-1960.0, 0.0, 0.0, simulator.config)
        # Long drift (~180cm)
        deposit_far = _make_single_deposit(-400.0, 0.0, 0.0, simulator.config)

        response_near, _, _ = simulator.process_event(deposit_near)
        response_far, _, _ = simulator.process_event(deposit_far)

        y_near = response_near.get((0, 2))
        y_far = response_far.get((0, 2))

        if y_near is not None and y_far is not None:
            # Measure temporal width using RMS of signal distribution
            total_near = float(jnp.sum(jnp.abs(y_near)))
            total_far = float(jnp.sum(jnp.abs(y_far)))

            if total_near > 0 and total_far > 0:
                # Compute RMS width in time direction
                time_axis = jnp.arange(y_near.shape[1], dtype=jnp.float32)
                profile_near = jnp.sum(jnp.abs(y_near), axis=0)
                profile_far = jnp.sum(jnp.abs(y_far), axis=0)

                mean_near = jnp.sum(profile_near * time_axis) / jnp.sum(profile_near)
                mean_far = jnp.sum(profile_far * time_axis) / jnp.sum(profile_far)

                rms_near = jnp.sqrt(jnp.sum(profile_near * (time_axis - mean_near)**2) / jnp.sum(profile_near))
                rms_far = jnp.sqrt(jnp.sum(profile_far * (time_axis - mean_far)**2) / jnp.sum(profile_far))

                assert float(rms_far) > float(rms_near), \
                    f"Far deposit should have broader signal: rms_far={float(rms_far):.2f} <= rms_near={float(rms_near):.2f}"


@pytest.mark.requires_kernels
@pytest.mark.slow
class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_determinism(self, simulator):
        """Same input twice should produce bit-identical output."""
        deposit = _make_deposit(30, simulator.config, x_range=(-100, -1))
        resp1, _, _ = simulator.process_event(deposit)
        resp2, _, _ = simulator.process_event(deposit)

        for key in resp1:
            np.testing.assert_array_equal(
                np.array(resp1[key]), np.array(resp2[key]),
                err_msg=f"Plane {key}: non-deterministic output")

    def test_all_east_no_west_signal(self, simulator):
        """All-east deposits should produce zero signal on west planes."""
        deposit = _make_deposit(30, simulator.config, x_range=(-200, -1))
        resp, _, _ = simulator.process_event(deposit)

        for (side_idx, plane_idx), sig in resp.items():
            if side_idx == 1:  # West
                total = float(jnp.sum(jnp.abs(sig)))
                assert total == 0.0, \
                    f"West plane ({side_idx},{plane_idx}) should be zero, got {total}"

    def test_cathode_deposit(self, simulator):
        """Deposit at x=0 should be assigned to west side and produce signal."""
        deposit = _make_single_deposit(0.0, 0.0, 0.0, simulator.config)
        resp, _, _ = simulator.process_event(deposit)

        # x>=0 goes west (side 1); east should be empty
        for (side_idx, plane_idx), sig in resp.items():
            if side_idx == 0:
                assert float(jnp.sum(jnp.abs(sig))) == 0.0, \
                    "East side should be empty for x=0 deposit"

        west_y = resp.get((1, 2))
        if west_y is not None:
            assert float(jnp.sum(jnp.abs(west_y))) > 0, \
                "West Y-plane should have signal for x=0 deposit"

    def test_uv_bipolarity(self, simulator):
        """U/V (induction) plane integrals should be much smaller than Y (collection)."""
        deposit = _make_deposit(30, simulator.config, x_range=(-150, -10))
        resp, _, _ = simulator.process_event(deposit)

        y_integral = abs(float(jnp.sum(resp.get((0, 2), jnp.zeros(1)))))

        if y_integral > 0:
            for plane_idx in [0, 1]:  # U, V
                sig = resp.get((0, plane_idx))
                if sig is not None:
                    uv_integral = abs(float(jnp.sum(sig)))
                    assert uv_integral < 0.5 * y_integral, \
                        f"Plane {plane_idx} |integral|={uv_integral:.2f} should be << Y |integral|={y_integral:.2f}"
