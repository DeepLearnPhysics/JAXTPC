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

Standalone functions (generate_noise, add_noise, etc.) accept SimConfig.
Factory function (create_noise_fn) builds closures for use inside JIT.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np


def load_noise_params(noise_spectrum_path):
    """
    Load noise model parameters from NPZ file.

    Parameters
    ----------
    noise_spectrum_path : str
        Path to noise_spectrum.npz.

    Returns
    -------
    noise_x : float
        White/parallel noise RMS in ADC.
    noise_y : float
        Series noise baseline in ADC.
    noise_z : float
        Wire capacitance coupling in ADC/m.
    empirical_freqs_hz : np.ndarray
        Empirical frequency bins.
    empirical_shape : np.ndarray
        Empirical noise spectral shape.
    """
    cfg = np.load(noise_spectrum_path, allow_pickle=True)
    return (
        float(cfg['noise_param_x']),
        float(cfg['noise_param_y']),
        float(cfg['noise_param_z']),
        cfg['spectrum_freqs_hz'],
        cfg['spectrum_shape'],
    )


def _get_noise_spectrum_shape(num_time_ticks, empirical_freqs_hz, empirical_shape):
    """
    Interpolate empirical noise spectrum to FFT resolution.

    Parameters
    ----------
    num_time_ticks : int
        Number of time samples.
    empirical_freqs_hz : np.ndarray
        Empirical frequency bins from noise spectrum file.
    empirical_shape : np.ndarray
        Empirical noise spectral shape from noise spectrum file.

    Returns
    -------
    spectrum_shape : np.ndarray
        Normalized series noise amplitude spectrum (unit energy).
    """
    num_freq_bins = num_time_ticks // 2 + 1
    sampling_rate = 2e6  # 2 MHz sampling (0.5 us per tick)
    freqs = np.fft.rfftfreq(num_time_ticks, d=1 / sampling_rate)

    spectrum = np.interp(freqs, empirical_freqs_hz, empirical_shape)

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


def add_noise(response_signals, config, threshold_enc=0, key=None):
    """
    Add realistic intrinsic noise to dense detector response signals.

    Parameters
    ----------
    response_signals : dict
        Dictionary mapping (side_idx, plane_idx) -> jnp.ndarray in ADC.
        Each array has shape (num_wires, num_time_steps).
    config : SimConfig
        Simulation configuration (from sim.config).
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

    noise_x, noise_y, noise_z, emp_freqs, emp_shape = load_noise_params(
        config.noise_spectrum_path)

    deadband_adc = threshold_enc / config.electrons_per_adc
    num_time_ticks = config.num_time_steps

    spectrum_shape = _get_noise_spectrum_shape(num_time_ticks, emp_freqs, emp_shape)
    spectrum_jax = jnp.array(spectrum_shape)

    noisy_signals = {}

    for (side_idx, plane_idx), signal in response_signals.items():
        num_wires = config.side_geom[side_idx].num_wires[plane_idx]
        lengths = config.side_geom[side_idx].wire_lengths_m[plane_idx]

        series_rms = jnp.array(noise_y + noise_z * lengths, dtype=jnp.float32)

        key, subkey = jax.random.split(key)
        noise = _generate_noise_for_plane(
            subkey, num_wires, num_time_ticks,
            spectrum_jax, series_rms, noise_x
        )

        noisy = jnp.asarray(signal) + noise

        if deadband_adc > 0:
            noisy = jnp.where(jnp.abs(noisy) < deadband_adc, 0.0, noisy)

        noisy_signals[(side_idx, plane_idx)] = noisy

    return noisy_signals


def generate_noise(config, key=None):
    """
    Generate noise arrays for all wire planes (dense format).

    Parameters
    ----------
    config : SimConfig
        Simulation configuration (from sim.config).
    key : jax.Array, optional
        JAX PRNGKey. Default is PRNGKey(0).

    Returns
    -------
    noise_dict : dict
        Dictionary mapping (side_idx, plane_idx) -> jnp.array (num_wires, num_time_steps).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    noise_x, noise_y, noise_z, emp_freqs, emp_shape = load_noise_params(
        config.noise_spectrum_path)

    num_time_ticks = config.num_time_steps

    spectrum_shape = _get_noise_spectrum_shape(num_time_ticks, emp_freqs, emp_shape)
    spectrum_jax = jnp.array(spectrum_shape)

    noise_dict = {}

    for side_idx in range(2):
        for plane_idx in range(3):
            num_wires = config.side_geom[side_idx].num_wires[plane_idx]
            if num_wires == 0:
                continue

            lengths = config.side_geom[side_idx].wire_lengths_m[plane_idx]
            series_rms = jnp.array(noise_y + noise_z * lengths, dtype=jnp.float32)

            key, subkey = jax.random.split(key)
            noise = _generate_noise_for_plane(
                subkey, num_wires, num_time_ticks,
                spectrum_jax, series_rms, noise_x
            )
            noise_dict[(side_idx, plane_idx)] = noise

    return noise_dict


def generate_noise_bucketed(config, bucketed_signals, key=None):
    """
    Generate noise arrays matching bucketed signal format.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration (from sim.config).
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

    noise_x, noise_y, noise_z, emp_freqs, emp_shape = load_noise_params(
        config.noise_spectrum_path)

    num_time_steps = config.num_time_steps

    noise_dict = {}

    for (side_idx, plane_idx), signal_tuple in bucketed_signals.items():
        buckets, num_active, compact_to_key, B1, B2 = signal_tuple
        B1_int = int(B1)
        B2_int = int(B2)
        max_buckets = buckets.shape[0]
        num_wires = config.side_geom[side_idx].num_wires[plane_idx]

        all_lengths = jnp.array(config.side_geom[side_idx].wire_lengths_m[plane_idx])

        NUM_BUCKETS_T = (num_time_steps + B2_int - 1) // B2_int
        wire_starts = (compact_to_key // NUM_BUCKETS_T) * B1_int
        wire_indices = wire_starts[:, None] + jnp.arange(B1_int)
        wire_indices = jnp.clip(wire_indices, 0, num_wires - 1)

        bucket_series_rms = noise_y + noise_z * all_lengths[wire_indices]
        bucket_series_rms = bucket_series_rms.astype(jnp.float32)

        spectrum_shape = _get_noise_spectrum_shape(B2_int, emp_freqs, emp_shape)
        spectrum_jax = jnp.array(spectrum_shape)

        key, subkey = jax.random.split(key)
        noise = _generate_noise_for_buckets(
            subkey, max_buckets, B1_int, B2_int,
            spectrum_jax, bucket_series_rms, noise_x
        )
        noise_dict[(side_idx, plane_idx)] = noise

    return noise_dict


# =============================================================================
# FACTORY FUNCTION (create closure for use inside JIT)
# =============================================================================

def _noop_noise(key, sig, side_idx, plane_idx, n_wires, n_time):
    """Identity — no noise."""
    return sig


def _build_dense_noise(noise_series_rms, noise_white_rms, spectrum_dense):
    """Build dense-mode noise closure."""
    def noise_fn(key, signal, side_idx, plane_idx, n_wires, n_time):
        series_rms = noise_series_rms[(side_idx, plane_idx)]
        noise = _noise_core(key, n_wires, n_time, spectrum_dense, series_rms, noise_white_rms)
        return signal + noise
    return noise_fn


def _build_bucketed_noise(cfg, spectrum_bucketed, bucket_dims,
                          wire_lengths_jax, noise_x, noise_y, noise_z):
    """Build bucketed-mode noise closure (noise per bucket)."""
    def make_fn(plane_type, side_idx, plane_idx):
        B1, B2 = bucket_dims[plane_type]
        spectrum = spectrum_bucketed[plane_type]
        lengths = wire_lengths_jax[(side_idx, plane_idx)]

        def fn(key, signal_tuple, side_idx, plane_idx, n_wires, n_time):
            buckets, num_active, compact_to_key, _, _ = signal_tuple
            NUM_BUCKETS_T = (n_time + B2 - 1) // B2
            wire_starts = (compact_to_key // NUM_BUCKETS_T) * B1
            wire_indices = wire_starts[:, None] + jnp.arange(B1)
            wire_indices = jnp.clip(wire_indices, 0, n_wires - 1)
            bucket_series_rms = noise_y + noise_z * lengths[wire_indices]
            noise = _generate_noise_for_buckets(
                key, cfg.max_active_buckets, B1, B2,
                spectrum, bucket_series_rms, noise_x
            )
            active_mask = jnp.arange(cfg.max_active_buckets) < num_active
            noise = noise * active_mask[:, None, None]
            return (buckets + noise, num_active, compact_to_key, B1, B2)
        return fn

    plane_fns = {
        (s, p): make_fn(cfg.plane_names[s][p], s, p)
        for s in range(2) for p in range(3)
    }

    def noise_fn(key, sig, side_idx, plane_idx, n_wires, n_time):
        return plane_fns[(side_idx, plane_idx)](key, sig, side_idx, plane_idx, n_wires, n_time)
    return noise_fn


def _build_wire_sparse_noise(wire_lengths_jax, noise_x, noise_y, noise_z,
                             spectrum_dense, e_chunk):
    """Build wire-sparse-mode noise closure (noise on active wire rows)."""
    def make_fn(side_idx, plane_idx):
        lengths = wire_lengths_jax[(side_idx, plane_idx)]

        def fn(key, signal_tuple, side_idx, plane_idx, n_wires, n_time):
            active_signals, wire_indices, n_active = signal_tuple
            active_series_rms = noise_y + noise_z * lengths[wire_indices]
            noise = _noise_core(
                key, e_chunk, n_time, spectrum_dense,
                active_series_rms, noise_x
            )
            valid = jnp.arange(e_chunk) < n_active
            return (active_signals + noise * valid[:, None],
                    wire_indices, n_active)
        return fn

    plane_fns = {
        (s, p): make_fn(s, p)
        for s in range(2) for p in range(3)
    }

    def noise_fn(key, sig, side_idx, plane_idx, n_wires, n_time):
        return plane_fns[(side_idx, plane_idx)](key, sig, side_idx, plane_idx, n_wires, n_time)
    return noise_fn


def create_noise_fn(cfg, response_kernels, e_chunk=None):
    """Create noise generation closure for use inside JIT.

    Pre-computes spectra, wire length arrays, and series RMS from
    cfg.side_geom[s].wire_lengths_m[p].

    Parameters
    ----------
    cfg : SimConfig
        Static simulation config (includes wire lengths via SideGeometry).
    response_kernels : dict
        Loaded response kernels (for bucket dimensions).
    e_chunk : int, optional
        Electronics chunk size (for wire-sparse mode).

    Returns
    -------
    noise_fn : callable
        Signature: (key, sig, side_idx, plane_idx, n_wires, n_time) -> noisy signal.
    """
    if not cfg.include_noise:
        return _noop_noise

    # Load noise parameters from config path
    noise_x, noise_y, noise_z, emp_freqs, emp_shape = load_noise_params(
        cfg.noise_spectrum_path)

    spectrum_dense = jnp.array(
        _get_noise_spectrum_shape(cfg.num_time_steps, emp_freqs, emp_shape))

    wire_lengths_jax = {
        (s, p): jnp.array(cfg.side_geom[s].wire_lengths_m[p], dtype=jnp.float32)
        for s in range(2) for p in range(3)
    }

    if cfg.use_bucketed and cfg.include_electronics:
        return _build_wire_sparse_noise(
            wire_lengths_jax, noise_x, noise_y, noise_z,
            spectrum_dense, e_chunk)

    elif cfg.use_bucketed:
        bucket_dims = {}
        spectrum_bucketed = {}
        for plane_type in ['U', 'V', 'Y']:
            B1 = 2 * response_kernels[plane_type].num_wires
            B2 = 2 * response_kernels[plane_type].kernel_height
            bucket_dims[plane_type] = (B1, B2)
            spectrum_bucketed[plane_type] = jnp.array(
                _get_noise_spectrum_shape(B2, emp_freqs, emp_shape))
        return _build_bucketed_noise(
            cfg, spectrum_bucketed, bucket_dims,
            wire_lengths_jax, noise_x, noise_y, noise_z)

    else:
        noise_series_rms = {
            (s, p): jnp.array(
                noise_y + noise_z * cfg.side_geom[s].wire_lengths_m[p],
                dtype=jnp.float32)
            for s in range(2) for p in range(3)
        }
        return _build_dense_noise(
            noise_series_rms, noise_x, spectrum_dense)
