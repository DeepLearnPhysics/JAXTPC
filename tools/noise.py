"""
Noise generation for JAXTPC detector.

Adds realistic intrinsic noise to detector response signals based on
the MicroBooNE noise model (arXiv:1705.07341).

Noise model (Equation 3.6):
    ENC ~ RMS_ADC = sqrt(x^2 + (y + z*L)^2)
    where:
        x = white/parallel noise (ADC)
        y = series noise (ADC)
        z = wire capacitance contribution (ADC/m)
        L = wire length (m)
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
from pathlib import Path

from tools.wires import sparse_buckets_to_dense

# Load noise parameters and empirical spectrum from config
_NOISE_CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'noise_spectrum.npz'
_noise_cfg = np.load(_NOISE_CONFIG_PATH, allow_pickle=True)

_NOISE_X = float(_noise_cfg['noise_param_x'])  # ADC - white/parallel noise
_NOISE_Y = float(_noise_cfg['noise_param_y'])  # ADC - series noise
_NOISE_Z = float(_noise_cfg['noise_param_z'])  # ADC/m - wire capacitance
_EMPIRICAL_FREQS_HZ = _noise_cfg['spectrum_freqs_hz']
_EMPIRICAL_SHAPE = _noise_cfg['spectrum_shape']


def _get_noise_spectrum_shape(num_time_ticks):
    """
    Interpolate empirical MicroBooNE noise spectrum to FFT resolution.

    Parameters
    ----------
    num_time_ticks : int
        Number of time samples.

    Returns
    -------
    spectrum_shape : np.ndarray
        Normalized series noise amplitude spectrum (unit energy).
    """
    num_freq_bins = num_time_ticks // 2 + 1
    sampling_rate = 2e6  # 2 MHz sampling (0.5 us per tick)
    freqs = np.fft.rfftfreq(num_time_ticks, d=1 / sampling_rate)

    spectrum = np.interp(freqs, _EMPIRICAL_FREQS_HZ, _EMPIRICAL_SHAPE)

    energy = np.sum(spectrum**2)
    if energy > 0:
        spectrum = spectrum / np.sqrt(energy) * np.sqrt(num_freq_bins)

    return spectrum.astype(np.float32)


def _noise_core(key, num_wires, num_time_ticks, spectrum_shape, series_rms, white_rms):
    """
    Core noise generation logic. Not JIT-compiled directly — called
    inside jit or vmap externally.

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    num_wires : int
        Number of wires.
    num_time_ticks : int
        Number of time samples.
    spectrum_shape : jax.Array
        Normalized series noise spectrum shape.
    series_rms : jax.Array
        Series noise RMS in ADC for each wire, shape (num_wires,).
    white_rms : float
        White/parallel noise RMS in ADC.

    Returns
    -------
    noise : jax.Array
        Noise array of shape (num_wires, num_time_ticks) in ADC.
    """
    num_freq_bins = num_time_ticks // 2 + 1

    key_real, key_imag, key_white = jax.random.split(key, 3)
    real_parts = jax.random.normal(key_real, (num_wires, num_freq_bins))
    imag_parts = jax.random.normal(key_imag, (num_wires, num_freq_bins))

    real_parts = real_parts * spectrum_shape[None, :]
    imag_parts = imag_parts * spectrum_shape[None, :]

    complex_noise = real_parts + 1j * imag_parts
    complex_noise = complex_noise.at[:, 0].set(complex_noise[:, 0].real)

    nyquist_idx = num_freq_bins - 1
    if num_time_ticks % 2 == 0:
        complex_noise = complex_noise.at[:, nyquist_idx].set(
            complex_noise[:, nyquist_idx].real
        )

    shaped_noise = jnp.fft.irfft(complex_noise, n=num_time_ticks, axis=1)

    current_rms = jnp.std(shaped_noise, axis=1, keepdims=True)
    current_rms = jnp.maximum(current_rms, 1e-10)
    shaped_noise = shaped_noise / current_rms * series_rms[:, None]

    white_noise = jax.random.normal(key_white, (num_wires, num_time_ticks)) * white_rms

    return shaped_noise + white_noise


@partial(jit, static_argnums=(1, 2))
def _generate_noise_for_plane(key, num_wires, num_time_ticks, spectrum_shape,
                              series_rms, white_rms):
    """
    Generate intrinsic noise for a wire plane using JAX.

    Generates two physically distinct noise components:
    - Series noise (shaped by electronics): scales with wire length
    - White/parallel noise (flat spectrum): constant for all wires

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    num_wires : int
        Number of wires (static).
    num_time_ticks : int
        Number of time samples (static).
    spectrum_shape : jax.Array
        Normalized series noise spectrum shape.
    series_rms : jax.Array
        Series noise RMS in ADC for each wire, shape (num_wires,).
    white_rms : float
        White/parallel noise RMS in ADC.

    Returns
    -------
    noise : jax.Array
        Noise array of shape (num_wires, num_time_ticks) in ADC.
    """
    return _noise_core(key, num_wires, num_time_ticks, spectrum_shape,
                       series_rms, white_rms)


@partial(jit, static_argnums=(1, 2, 3))
def _generate_noise_for_buckets(key, max_buckets, B1, B2, spectrum_shape,
                                bucket_series_rms, white_rms):
    """
    Generate noise for all buckets in a plane using vmap.

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    max_buckets : int
        Maximum number of buckets (static).
    B1 : int
        Bucket size in wire direction (static).
    B2 : int
        Bucket size in time direction (static).
    spectrum_shape : jax.Array
        Normalized series noise spectrum shape for B2 ticks.
    bucket_series_rms : jax.Array
        Series noise RMS per wire per bucket, shape (max_buckets, B1).
    white_rms : float
        White/parallel noise RMS in ADC.

    Returns
    -------
    noise : jax.Array
        Noise array of shape (max_buckets, B1, B2) in ADC.
    """
    keys = jax.random.split(key, max_buckets)
    return jax.vmap(
        lambda k, s: _noise_core(k, B1, B2, spectrum_shape, s, white_rms)
    )(keys, bucket_series_rms)


def add_noise(response_signals, detector_config, threshold_enc=0, key=None):
    """
    Add realistic intrinsic noise to detector response signals.

    Parameters
    ----------
    response_signals : dict
        Dictionary mapping (side_idx, plane_idx) -> jnp.ndarray in ADC.
        Each array has shape (num_wires, num_time_steps).
    detector_config : dict
        Detector configuration from generate_detector().
        Must contain 'wire_lengths_m', 'num_time_steps', 'num_wires_actual',
        and 'electrons_per_adc'.
    threshold_enc : float, optional
        Electron threshold for deadband zeroing. Values with |x| < threshold
        (converted to ADC) are set to zero. Default 0 (no zeroing).
    key : jax.Array, optional
        JAX PRNGKey. Default is PRNGKey(0).

    Returns
    -------
    noisy_signals : dict
        Dictionary mapping (side_idx, plane_idx) -> jnp.ndarray with noise added.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    electrons_per_adc = float(detector_config['electrons_per_adc'])
    deadband_adc = threshold_enc / electrons_per_adc
    num_time_ticks = int(detector_config['num_time_steps'])
    wire_lengths_m = detector_config['wire_lengths_m']

    spectrum_shape = _get_noise_spectrum_shape(num_time_ticks)
    spectrum_jax = jnp.array(spectrum_shape)

    noisy_signals = {}

    for (side_idx, plane_idx), signal in response_signals.items():
        num_wires = int(detector_config['num_wires_actual'][side_idx, plane_idx])
        lengths = wire_lengths_m[(side_idx, plane_idx)]

        series_rms = jnp.array(_NOISE_Y + _NOISE_Z * lengths, dtype=jnp.float32)

        key, subkey = jax.random.split(key)
        noise = _generate_noise_for_plane(
            subkey, num_wires, num_time_ticks,
            spectrum_jax, series_rms, _NOISE_X
        )

        noisy = jnp.asarray(signal) + noise

        if deadband_adc > 0:
            noisy = jnp.where(jnp.abs(noisy) < deadband_adc, 0.0, noisy)

        noisy_signals[(side_idx, plane_idx)] = noisy

    return noisy_signals


def generate_noise(detector_config, key=None):
    """
    Generate noise arrays for all wire planes (dense format).

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().
    key : jax.Array, optional
        JAX PRNGKey. Default is PRNGKey(0).

    Returns
    -------
    noise_dict : dict
        Dictionary mapping (side_idx, plane_idx) -> jnp.array (num_wires, num_time_steps).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    num_time_ticks = int(detector_config['num_time_steps'])
    wire_lengths_m = detector_config['wire_lengths_m']

    spectrum_shape = _get_noise_spectrum_shape(num_time_ticks)
    spectrum_jax = jnp.array(spectrum_shape)

    noise_dict = {}

    for side_idx in range(2):
        for plane_idx in range(3):
            num_wires = int(detector_config['num_wires_actual'][side_idx, plane_idx])
            if num_wires == 0:
                continue

            lengths = wire_lengths_m[(side_idx, plane_idx)]
            series_rms = jnp.array(_NOISE_Y + _NOISE_Z * lengths, dtype=jnp.float32)

            key, subkey = jax.random.split(key)
            noise = _generate_noise_for_plane(
                subkey, num_wires, num_time_ticks,
                spectrum_jax, series_rms, _NOISE_X
            )
            noise_dict[(side_idx, plane_idx)] = noise

    return noise_dict


def generate_noise_bucketed(detector_config, bucketed_signals, key=None):
    """
    Generate noise arrays matching bucketed signal format.

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().
    bucketed_signals : dict
        Dictionary mapping (side_idx, plane_idx) -> (buckets, num_active,
        compact_to_key, B1, B2) tuples from bucketed simulation.
    key : jax.Array, optional
        JAX PRNGKey. Default is PRNGKey(0).

    Returns
    -------
    noise_dict : dict
        Dictionary mapping (side_idx, plane_idx) -> jnp.array (max_buckets, B1, B2).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    num_time_steps = int(detector_config['num_time_steps'])
    wire_lengths_m = detector_config['wire_lengths_m']

    noise_dict = {}

    for (si, pi), signal_tuple in bucketed_signals.items():
        buckets, num_active, compact_to_key, B1, B2 = signal_tuple
        B1_int = int(B1)
        B2_int = int(B2)
        max_buckets = buckets.shape[0]
        num_wires = int(detector_config['num_wires_actual'][si, pi])

        all_lengths = wire_lengths_m[(si, pi)]

        # Map each bucket to its wire range
        NUM_BUCKETS_T = (num_time_steps + B2_int - 1) // B2_int
        wire_starts = (compact_to_key // NUM_BUCKETS_T) * B1_int
        wire_indices = wire_starts[:, None] + jnp.arange(B1_int)
        wire_indices = jnp.clip(wire_indices, 0, num_wires - 1)

        bucket_series_rms = _NOISE_Y + _NOISE_Z * all_lengths[wire_indices]
        bucket_series_rms = bucket_series_rms.astype(jnp.float32)

        spectrum_shape = _get_noise_spectrum_shape(B2_int)
        spectrum_jax = jnp.array(spectrum_shape)

        key, subkey = jax.random.split(key)
        noise = _generate_noise_for_buckets(
            subkey, max_buckets, B1_int, B2_int,
            spectrum_jax, bucket_series_rms, _NOISE_X
        )
        noise_dict[(si, pi)] = noise

    return noise_dict


def process_response(response_signals, detector_config, threshold_enc,
                     include_noise=True, key=None):
    """
    Process response signals: optionally add noise, threshold, and decompose
    into sparse output with signal-first partition.

    Parameters
    ----------
    response_signals : dict
        Dictionary mapping (side_idx, plane_idx) -> signal data.
        Dense: jnp.ndarray (num_wires, num_time_steps).
        Bucketed: tuple (buckets, num_active, compact_to_key, B1, B2).
    detector_config : dict
        Detector configuration from generate_detector().
    threshold_enc : float
        Threshold in electrons for deadband zeroing.
    include_noise : bool, optional
        Whether to add intrinsic noise. Default True.
    key : jax.Array, optional
        JAX PRNGKey for noise generation. Default is PRNGKey(0).

    Returns
    -------
    result : dict
        Dictionary mapping (side_idx, plane_idx) -> dict with:
            'indices': (N_c, 2) int32 — surviving pixel locations
            'values': (N_c,) float32 — combined signal+noise values
            'signal': (N_s,) float32 — signal values at first n_signal entries
            'n_signal': int — boundary index ([:n_signal] = signal, [n_signal:] = noise-only)
            'threshold_adc': float — threshold used in ADC units
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    electrons_per_adc = float(detector_config['electrons_per_adc'])
    threshold_adc = threshold_enc / electrons_per_adc
    num_time_steps = int(detector_config['num_time_steps'])

    # Separate bucketed and dense planes
    bucketed_planes = {}
    dense_planes = {}
    for (si, pi), signal_raw in response_signals.items():
        if isinstance(signal_raw, tuple) and len(signal_raw) == 5:
            bucketed_planes[(si, pi)] = signal_raw
        else:
            dense_planes[(si, pi)] = signal_raw

    # Generate noise if needed
    noise_dense = {}
    noise_bucketed = {}
    if include_noise:
        if dense_planes:
            key, subkey = jax.random.split(key)
            noise_dense = generate_noise(detector_config, key=subkey)
        if bucketed_planes:
            key, subkey = jax.random.split(key)
            noise_bucketed = generate_noise_bucketed(
                detector_config, bucketed_planes, key=subkey
            )

    result = {}

    for (si, pi), signal_raw in response_signals.items():
        num_wires = int(detector_config['num_wires_actual'][si, pi])
        is_bucketed = (si, pi) in bucketed_planes

        if is_bucketed:
            buckets, num_active, compact_to_key, B1, B2 = signal_raw
            max_buckets = buckets.shape[0]

            dense_signal = sparse_buckets_to_dense(
                buckets, compact_to_key, num_active,
                int(B1), int(B2), num_wires, num_time_steps, max_buckets
            )

            if include_noise:
                noise_b = noise_bucketed[(si, pi)]
                combined_buckets = buckets + noise_b
                dense_combined = sparse_buckets_to_dense(
                    combined_buckets, compact_to_key, num_active,
                    int(B1), int(B2), num_wires, num_time_steps, max_buckets
                )
            else:
                dense_combined = dense_signal
        else:
            dense_signal = jnp.asarray(signal_raw)

            if include_noise and (si, pi) in noise_dense:
                dense_combined = dense_signal + noise_dense[(si, pi)]
            else:
                dense_combined = dense_signal

        # Threshold + partition using numpy (process_response is host-side)
        combined_np = np.asarray(dense_combined)
        signal_np = np.asarray(dense_signal)

        mask = np.abs(combined_np) >= threshold_adc
        surv_idx = np.argwhere(mask)
        surv_val = combined_np[mask]
        surv_sig = signal_np[mask]

        has_signal = surv_sig != 0.0
        n_signal = int(has_signal.sum())

        # Partition: signal entries first, noise-only entries after
        sig_idx = surv_idx[has_signal]
        noise_idx = surv_idx[~has_signal]
        indices = np.concatenate([sig_idx, noise_idx]) if len(noise_idx) > 0 else sig_idx
        values = np.concatenate([surv_val[has_signal], surv_val[~has_signal]]) if len(noise_idx) > 0 else surv_val[has_signal]
        signal = surv_sig[has_signal]

        result[(si, pi)] = {
            'indices': jnp.array(indices, dtype=jnp.int32),
            'values': jnp.array(values, dtype=jnp.float32),
            'signal': jnp.array(signal, dtype=jnp.float32),
            'n_signal': n_signal,
            'threshold_adc': float(threshold_adc),
        }

    return result


def extract_signal(sparse_output):
    """
    Extract signal-only sparse data from process_response output.

    Returns dict compatible with visualize_wire_signals(sparse_data=True).

    Parameters
    ----------
    sparse_output : dict
        Output from process_response.

    Returns
    -------
    dict
        Dictionary mapping (side, plane) -> (indices, values) tuples
        containing only the signal portion of the data.
    """
    return {
        k: (v['indices'][:v['n_signal']], v['signal'])
        for k, v in sparse_output.items()
    }
