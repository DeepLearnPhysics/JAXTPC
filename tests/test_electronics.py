"""
Tests for electronics response module.

File under test: tools/electronics.py
Cross-references: tools/wires.py (sparse_buckets_to_dense)
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.signal import fftconvolve

from tools.electronics import (
    create_rcrc_response,
    compute_fft_size,
    electronics_response_core,
    electronics_convolve_active,
    buckets_to_active_wires,
)
from tools.wires import sparse_buckets_to_dense
from tools.config import DepositData
from tools.simulation import DetectorSimulator


RESPONSE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools', 'responses')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'cubic_wireplane_config.yaml')

has_kernels = all(
    os.path.exists(os.path.join(RESPONSE_PATH, f'{p}_plane_kernel.npz'))
    for p in ['U', 'V', 'Y']
)
has_config = os.path.exists(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deposit(n, sim_config, x_range=(-100, 100), de_val=2.0, dx_val=0.3,
                  theta_val=np.pi / 4, phi_val=0.0, track_id=0):
    """Create a DepositData with uniform parameters using build_deposit_data."""
    from tools.loader import build_deposit_data
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


# =========================================================================
# Class 1: TestRCRCKernel
# =========================================================================

class TestRCRCKernel:
    """Validates create_rcrc_response kernel physics."""

    def test_kernel_length(self):
        """R = n_tau * tau / dt = 6000 for defaults."""
        kernel = create_rcrc_response(tau_us=1000.0, time_step_us=0.5, n_tau=3.0)
        assert len(kernel) == 6000

    def test_kernel_first_element(self):
        """delta(0) + continuous at t=0 ~ 1 - 2*dt/tau."""
        tau = 1000.0
        dt = 0.5
        kernel = create_rcrc_response(tau_us=tau, time_step_us=dt, n_tau=3.0)
        expected_first = 1.0 - 2.0 * dt / tau
        np.testing.assert_allclose(kernel[0], expected_first, rtol=1e-5)

    def test_kernel_decays(self):
        """Tail near zero; second element negative (shaping)."""
        kernel = create_rcrc_response(tau_us=1000.0, time_step_us=0.5, n_tau=3.0)
        assert abs(kernel[-1]) < 1e-3, f"Tail should be near zero, got {kernel[-1]}"
        assert kernel[1] < 0, f"Second element should be negative (shaping), got {kernel[1]}"


# =========================================================================
# Class 2: TestFFTConvolution
# =========================================================================

class TestFFTConvolution:
    """Validates electronics_response_core and electronics_convolve_active against scipy."""

    @pytest.fixture
    def fft_test_data(self, jax_key):
        """Synthetic signals: 50 wires, 251 time, 5 active wires with random data."""
        num_wires = 50
        num_time = 251
        active_wires = [3, 10, 22, 35, 48]

        signals = np.zeros((num_wires, num_time), dtype=np.float32)
        rng = np.random.RandomState(99)
        for w in active_wires:
            signals[w] = rng.randn(num_time).astype(np.float32)

        kernel = create_rcrc_response(tau_us=100.0, time_step_us=0.5, n_tau=3.0)
        fft_size = compute_fft_size(num_time, len(kernel))

        return {
            'signals': jnp.array(signals),
            'signals_np': signals,
            'kernel': jnp.array(kernel),
            'kernel_np': kernel,
            'num_wires': num_wires,
            'num_time': num_time,
            'active_wires': active_wires,
            'fft_size': fft_size,
            'chunk_size': num_wires,
        }

    def test_dense_matches_scipy(self, fft_test_data):
        """Per-wire FFT conv matches scipy.signal.fftconvolve."""
        d = fft_test_data
        output = electronics_response_core(
            d['signals'], d['kernel'], 0.0,
            d['chunk_size'], d['fft_size'], d['num_time']
        )
        output_np = np.array(output)

        for w in d['active_wires']:
            scipy_ref = fftconvolve(d['signals_np'][w], d['kernel_np'], mode='full')[:d['num_time']]
            np.testing.assert_allclose(
                output_np[w], scipy_ref, rtol=1e-4, atol=1e-5,
                err_msg=f"Dense mismatch on wire {w}"
            )

    def test_inactive_wires_stay_zero(self, fft_test_data):
        """Wires with no signal remain zero after convolution."""
        d = fft_test_data
        output = electronics_response_core(
            d['signals'], d['kernel'], 0.0,
            d['chunk_size'], d['fft_size'], d['num_time']
        )

        all_wires = set(range(d['num_wires']))
        inactive_wires = all_wires - set(d['active_wires'])

        for w in inactive_wires:
            assert jnp.all(output[w] == 0), f"Wire {w} should be zero"

    def test_active_matches_scipy(self, fft_test_data):
        """electronics_convolve_active matches scipy per-row."""
        d = fft_test_data
        active_signals = d['signals'][jnp.array(d['active_wires'])]
        chunk = len(d['active_wires'])

        # Pad to chunk_size
        padded = jnp.zeros((chunk, d['num_time']), dtype=jnp.float32)
        padded = padded.at[:chunk].set(active_signals)

        fft_size = compute_fft_size(d['num_time'], len(d['kernel_np']))

        output = electronics_convolve_active(
            padded, d['kernel'], jnp.array(chunk),
            chunk, fft_size, d['num_time']
        )
        output_np = np.array(output)

        for i, w in enumerate(d['active_wires']):
            scipy_ref = fftconvolve(d['signals_np'][w], d['kernel_np'], mode='full')[:d['num_time']]
            np.testing.assert_allclose(
                output_np[i], scipy_ref, rtol=1e-4, atol=1e-5,
                err_msg=f"Active mismatch on row {i} (wire {w})"
            )

    def test_dense_and_active_consistency(self, fft_test_data):
        """Both functions produce same result for same wire data."""
        d = fft_test_data

        # Dense path
        dense_output = electronics_response_core(
            d['signals'], d['kernel'], 0.0,
            d['chunk_size'], d['fft_size'], d['num_time']
        )

        # Active path: gather active wires
        active_idx = jnp.array(d['active_wires'])
        active_signals = d['signals'][active_idx]
        chunk = len(d['active_wires'])
        fft_size = compute_fft_size(d['num_time'], len(d['kernel_np']))

        active_output = electronics_convolve_active(
            active_signals, d['kernel'], jnp.array(chunk),
            chunk, fft_size, d['num_time']
        )

        for i, w in enumerate(d['active_wires']):
            np.testing.assert_allclose(
                np.array(active_output[i]),
                np.array(dense_output[w]),
                rtol=1e-5, atol=1e-6,
                err_msg=f"Dense vs active mismatch on wire {w}"
            )


# =========================================================================
# Class 3: TestBucketsToActiveWires
# =========================================================================

class TestBucketsToActiveWires:
    """Validates buckets_to_active_wires with hand-crafted bucket data."""

    @pytest.fixture
    def bucket_test_data(self):
        """Small bucket data: 20 wires, 100 time, B1=4, B2=10, 3 active buckets."""
        num_wires = 20
        num_time = 100
        B1 = 4
        B2 = 10
        max_buckets = 10
        chunk_size = num_wires  # conservative

        NUM_BUCKETS_T = (num_time + B2 - 1) // B2  # = 10

        # Bucket 0: wires 0-3, time 0-9  (key = bw=0, bt=0 => key=0*10+0=0)
        # Bucket 1: wires 8-11, time 20-29 (key = bw=2, bt=2 => key=2*10+2=22)
        # Bucket 2: wires 0-3, time 50-59  (key = bw=0, bt=5 => key=0*10+5=5)
        compact_to_key = np.zeros(max_buckets, dtype=np.int32)
        compact_to_key[0] = 0
        compact_to_key[1] = 22
        compact_to_key[2] = 5
        num_active = 3

        buckets = np.zeros((max_buckets, B1, B2), dtype=np.float32)
        # Bucket 0: known values
        buckets[0, 0, 0] = 1.0   # wire 0, time 0
        buckets[0, 1, 5] = 2.5   # wire 1, time 5
        buckets[0, 3, 9] = 3.0   # wire 3, time 9
        # Bucket 1: known values
        buckets[1, 0, 0] = 4.0   # wire 8, time 20
        buckets[1, 2, 7] = 1.5   # wire 10, time 27
        # Bucket 2: known values
        buckets[2, 0, 0] = 7.0   # wire 0, time 50
        buckets[2, 1, 2] = 0.5   # wire 1, time 52

        return {
            'buckets': jnp.array(buckets),
            'num_active': jnp.array(num_active),
            'compact_to_key': jnp.array(compact_to_key),
            'B1': B1,
            'B2': B2,
            'num_wires': num_wires,
            'num_time': num_time,
            'chunk_size': chunk_size,
            'max_buckets': max_buckets,
        }

    def test_active_wire_count(self, bucket_test_data):
        """Occupancy detection counts unique wires correctly."""
        d = bucket_test_data
        _, _, n_active_wires = buckets_to_active_wires(
            d['buckets'], d['num_active'], d['compact_to_key'],
            d['B1'], d['B2'], d['num_wires'], d['num_time'],
            d['chunk_size'], d['max_buckets']
        )
        # Bucket 0 covers wires 0-3, bucket 1 covers wires 8-11, bucket 2 covers wires 0-3
        # Unique wires: {0,1,2,3, 8,9,10,11} = 8
        assert int(n_active_wires) == 8

    def test_data_correctly_scattered(self, bucket_test_data):
        """Signal values land at correct (wire, time) positions."""
        d = bucket_test_data
        active_signals, wire_indices, n_active_wires = buckets_to_active_wires(
            d['buckets'], d['num_active'], d['compact_to_key'],
            d['B1'], d['B2'], d['num_wires'], d['num_time'],
            d['chunk_size'], d['max_buckets']
        )

        # Build a lookup: global_wire -> row in active_signals
        wire_indices_np = np.array(wire_indices)
        n = int(n_active_wires)
        wire_to_row = {int(wire_indices_np[i]): i for i in range(n)}

        active_np = np.array(active_signals)

        # Check specific values from bucket 0: wire 0 time 0 = 1.0
        assert 0 in wire_to_row
        row0 = wire_to_row[0]
        np.testing.assert_allclose(active_np[row0, 0], 1.0, atol=1e-6)

        # wire 1 time 5 = 2.5
        assert 1 in wire_to_row
        row1 = wire_to_row[1]
        np.testing.assert_allclose(active_np[row1, 5], 2.5, atol=1e-6)

        # wire 3 time 9 = 3.0
        assert 3 in wire_to_row
        row3 = wire_to_row[3]
        np.testing.assert_allclose(active_np[row3, 9], 3.0, atol=1e-6)

        # Bucket 1: wire 8 time 20 = 4.0
        assert 8 in wire_to_row
        row8 = wire_to_row[8]
        np.testing.assert_allclose(active_np[row8, 20], 4.0, atol=1e-6)

        # wire 10 time 27 = 1.5
        assert 10 in wire_to_row
        row10 = wire_to_row[10]
        np.testing.assert_allclose(active_np[row10, 27], 1.5, atol=1e-6)

        # Bucket 2: wire 0 time 50 = 7.0
        np.testing.assert_allclose(active_np[row0, 50], 7.0, atol=1e-6)

        # wire 1 time 52 = 0.5
        np.testing.assert_allclose(active_np[row1, 52], 0.5, atol=1e-6)

    def test_overlapping_buckets_accumulate(self, bucket_test_data):
        """Two buckets writing same (wire, time) -> values summed."""
        d = bucket_test_data
        num_wires = d['num_wires']
        num_time = d['num_time']
        B1, B2 = d['B1'], d['B2']
        max_buckets = d['max_buckets']

        NUM_BUCKETS_T = (num_time + B2 - 1) // B2

        # Create two buckets that overlap on wire 2, time 3
        # Bucket A: wires 0-3, time 0-9 (key=0)
        # Bucket B: wires 0-3, time 0-9 (key=0) -- same bucket key!
        # Actually, same key means they'd be the same bucket.
        # Instead: use two different buckets that map to different time ranges
        # but we want overlap. Let's use bucket positions that share a wire.
        #
        # Better: Bucket A covers wires 0-3, time 0-9 (key 0)
        #         Bucket B covers wires 0-3, time 0-9 (key 0) -- same key, different slot
        # Actually in sparse bucketing, same key just gets accumulated. Let's test
        # with two different bucket slots both mapping to the same key.
        compact_to_key = np.zeros(max_buckets, dtype=np.int32)
        compact_to_key[0] = 0  # wires 0-3, time 0-9
        compact_to_key[1] = 0  # same bucket key -> overlapping
        num_active = 2

        buckets = np.zeros((max_buckets, B1, B2), dtype=np.float32)
        buckets[0, 0, 0] = 5.0  # wire 0, time 0
        buckets[1, 0, 0] = 3.0  # wire 0, time 0 (overlapping)

        active_signals, wire_indices, n_active_wires = buckets_to_active_wires(
            jnp.array(buckets), jnp.array(num_active), jnp.array(compact_to_key),
            B1, B2, num_wires, num_time, num_wires, max_buckets
        )

        wire_indices_np = np.array(wire_indices)
        n = int(n_active_wires)
        wire_to_row = {int(wire_indices_np[i]): i for i in range(n)}

        active_np = np.array(active_signals)
        row0 = wire_to_row[0]
        np.testing.assert_allclose(active_np[row0, 0], 8.0, atol=1e-6,
                                   err_msg="Overlapping bucket values should accumulate (5+3=8)")

    def test_matches_sparse_buckets_to_dense(self, bucket_test_data):
        """Scatter wire-sparse back to dense matches sparse_buckets_to_dense from tools/wires.py."""
        d = bucket_test_data

        # Get wire-sparse output
        active_signals, wire_indices, n_active_wires = buckets_to_active_wires(
            d['buckets'], d['num_active'], d['compact_to_key'],
            d['B1'], d['B2'], d['num_wires'], d['num_time'],
            d['chunk_size'], d['max_buckets']
        )

        # Reconstruct dense from wire-sparse output
        n = int(n_active_wires)
        reconstructed = np.zeros((d['num_wires'], d['num_time']), dtype=np.float32)
        wire_indices_np = np.array(wire_indices)
        active_np = np.array(active_signals)
        for i in range(n):
            w = int(wire_indices_np[i])
            reconstructed[w] = active_np[i]

        # Get reference dense from sparse_buckets_to_dense
        reference_dense = sparse_buckets_to_dense(
            d['buckets'], d['compact_to_key'], d['num_active'],
            d['B1'], d['B2'], d['num_wires'], d['num_time'], d['max_buckets']
        )

        np.testing.assert_allclose(
            reconstructed, np.array(reference_dense), atol=1e-6,
            err_msg="Wire-sparse->dense should match sparse_buckets_to_dense"
        )


# =========================================================================
# Module-scoped fixtures for simulator tests (expensive init, shared)
# =========================================================================

def _skip_if_missing():
    if not has_config or not has_kernels:
        pytest.skip("Config or kernel files not found")


@pytest.fixture(scope="module")
def _electronics_detector():
    """Detector config shared by all electronics simulator tests."""
    _skip_if_missing()
    from tools.geometry import generate_detector
    det = generate_detector(CONFIG_PATH)
    if det is None:
        pytest.skip("Failed to load detector config")
    return det


@pytest.fixture(scope="module")
def sim_dense_electronics(_electronics_detector):
    """Dense + electronics simulator (created once per module)."""
    return DetectorSimulator(
        _electronics_detector,
        response_path=RESPONSE_PATH,
        include_electronics=True,
        total_pad=25_000,
        response_chunk_size=25_000,
    )


@pytest.fixture(scope="module")
def sim_bucketed_electronics(_electronics_detector):
    """Bucketed + electronics simulator (created once per module)."""
    return DetectorSimulator(
        _electronics_detector,
        response_path=RESPONSE_PATH,
        use_bucketed=True,
        include_electronics=True,
        total_pad=25_000,
        response_chunk_size=25_000,
    )


@pytest.fixture(scope="module")
def sim_bucketed(_electronics_detector):
    """Bucketed, no electronics simulator (created once per module)."""
    return DetectorSimulator(
        _electronics_detector,
        response_path=RESPONSE_PATH,
        use_bucketed=True,
        total_pad=25_000,
        response_chunk_size=25_000,
    )


# =========================================================================
# Class 4+5: TestSimulatorElectronics (init + integration, shared fixtures)
# =========================================================================

@pytest.mark.requires_kernels
@pytest.mark.slow
class TestSimulatorElectronics:
    """Init checks and end-to-end integration for all electronics mode combos.

    Simulators are module-scoped: each config is built once (~18 s for
    load_response_kernels) and reused for both attribute checks and
    process_event integration tests.  Total: 3 inits instead of 7.
    """

    def test_dense_electronics(self, sim_dense_electronics, _electronics_detector):
        """Dense + electronics: 2D output with RC-RC high-pass shaping."""
        deposit = _make_deposit(20, sim_dense_electronics.config, x_range=(-100, -1))
        resp, _, _ = sim_dense_electronics.process_event(deposit, key=jax.random.PRNGKey(0))

        for plane_key, sig in resp.items():
            assert isinstance(sig, jax.Array), \
                f"Plane {plane_key}: expected jax.Array, got {type(sig)}"
            assert sig.ndim == 2, \
                f"Plane {plane_key}: expected 2D, got {sig.ndim}D"

        y_signal = resp.get((0, 2))
        assert y_signal is not None, "Expected Y-plane east signal"
        peak = float(jnp.max(jnp.abs(y_signal)))
        assert peak > 0, "Signal should be non-zero"
        total = float(jnp.sum(y_signal))
        assert abs(total) < peak * y_signal.shape[0], \
            f"DC suppression: |integral|={abs(total):.2f} vs peak*W={peak * y_signal.shape[0]:.2f}"

    def test_bucketed_electronics(self, sim_bucketed_electronics, _electronics_detector):
        """Bucketed + electronics: 3-tuple output with valid wire indices."""
        deposit = _make_deposit(20, sim_bucketed_electronics.config, x_range=(-100, -1))
        resp, _, _ = sim_bucketed_electronics.process_event(deposit, key=jax.random.PRNGKey(0))

        for plane_key, sig in resp.items():
            assert isinstance(sig, tuple) and len(sig) == 3, \
                f"Plane {plane_key}: expected 3-tuple, got {type(sig)}"

            active_signals, wire_indices, n_active = sig
            n = int(n_active)
            if n > 0:
                num_wires = sim_bucketed_electronics.config.volumes[plane_key[0]].num_wires[plane_key[1]]
                wire_idx_np = np.array(wire_indices[:n])
                assert np.all(wire_idx_np >= 0) and np.all(wire_idx_np < num_wires), \
                    f"Plane {plane_key}: wire indices out of bounds"

        for plane_idx in range(3):
            sig = resp.get((0, plane_idx))
            if sig is not None:
                active_signals, _, n_active = sig
                n = int(n_active)
                assert n > 0 and float(jnp.sum(jnp.abs(active_signals[:n]))) > 0, \
                    f"East plane {plane_idx}: expected non-zero active signals"

    def test_bucketed_no_electronics(self, sim_bucketed):
        """Bucketed without electronics: 5-tuple output."""
        deposit = _make_deposit(20, sim_bucketed.config, x_range=(-100, -1))
        resp, _, _ = sim_bucketed.process_event(deposit, key=jax.random.PRNGKey(0))

        for plane_key, sig in resp.items():
            assert isinstance(sig, tuple) and len(sig) == 5, \
                f"Plane {plane_key}: expected 5-tuple, got {type(sig)}"


class TestElectronicsIntegral:
    """RC-RC shaping filter properties."""

    def test_kernel_dc_component_small(self):
        """RC-RC kernel sum should be small relative to peak (strong DC suppression)."""
        kernel = create_rcrc_response(tau_us=1000.0, time_step_us=0.5, n_tau=3.0)
        dc = abs(float(np.sum(kernel)))
        peak = float(np.max(np.abs(kernel)))
        assert dc / peak < 0.15, \
            f"Kernel DC/peak ratio = {dc/peak:.4f}, expected < 0.15"

    def test_output_has_bipolar_shape(self):
        """RC-RC of a positive pulse should produce both positive and negative values."""
        kernel = create_rcrc_response(tau_us=1000.0, time_step_us=0.5, n_tau=3.0)
        num_time = 2000
        fft_size = compute_fft_size(num_time, len(kernel))

        signals = np.zeros((1, num_time), dtype=np.float32)
        signals[0, 100:200] = 1.0  # positive box pulse

        output = electronics_response_core(
            jnp.array(signals), jnp.array(kernel), 0.0,
            1, fft_size, num_time)

        assert float(jnp.max(output)) > 0, "Should have positive values"
        assert float(jnp.min(output)) < 0, "Should have negative values (bipolar shaping)"

    def test_inactive_wires_stay_zero(self):
        """Wires with no input signal should remain zero after electronics."""
        kernel = create_rcrc_response(tau_us=100.0, time_step_us=0.5, n_tau=3.0)
        num_time = 500
        fft_size = compute_fft_size(num_time, len(kernel))

        signals = np.zeros((10, num_time), dtype=np.float32)
        signals[3, 100] = 50.0  # only wire 3 active

        output = electronics_response_core(
            jnp.array(signals), jnp.array(kernel), 0.0,
            10, fft_size, num_time)

        for w in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
            assert float(jnp.sum(jnp.abs(output[w]))) == 0.0, \
                f"Wire {w} should be zero"


class TestDigitization:
    """Tests for ADC digitization boundary values."""

    def test_output_clamped_to_adc_range(self):
        """Digitized output (unsigned) should be clamped to [0, 2^n_bits - 1]."""
        from tools.electronics import _digitize_signal

        pedestal = 900.0
        gain = 1.0
        n_bits = 12
        adc_max = float((1 << n_bits) - 1)

        extreme_signals = jnp.array([-10000.0, 0.0, 10000.0, 5000.0], dtype=jnp.float32)
        result = _digitize_signal(extreme_signals, gain, pedestal, adc_max)

        # Output is unsigned - pedestal, so range is [-pedestal, adc_max - pedestal]
        assert jnp.all(result >= -pedestal), "Digitized values should be >= -pedestal"
        assert jnp.all(result <= adc_max - pedestal), \
            f"Digitized values should be <= {adc_max - pedestal}"

    def test_zero_input_gives_zero(self):
        """Zero input should digitize to zero (pedestal cancels)."""
        from tools.electronics import _digitize_signal

        result = _digitize_signal(
            jnp.array([0.0], dtype=jnp.float32), 1.0, 900.0, 4095.0)

        assert float(result[0]) == 0.0

    def test_negative_overflow_clamps(self):
        """Large negative values should clamp to -pedestal."""
        from tools.electronics import _digitize_signal

        result = _digitize_signal(
            jnp.array([-10000.0], dtype=jnp.float32), 1.0, 900.0, 4095.0)

        assert float(result[0]) == -900.0

    def test_positive_overflow_clamps(self):
        """Large positive values should clamp to adc_max - pedestal."""
        from tools.electronics import _digitize_signal

        result = _digitize_signal(
            jnp.array([10000.0], dtype=jnp.float32), 1.0, 900.0, 4095.0)

        assert float(result[0]) == 4095.0 - 900.0
