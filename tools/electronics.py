"""
Electronics response convolution for JAXTPC detector.

Applies RC⊗RC electronics shaping to wire signals using sparse FFT,
processing only active (non-zero) wires for efficiency.

The RC⊗RC impulse response is:
    h(t) = δ(t) + (t/τ - 2) · (1/τ) · e^(-t/τ) · dt

where τ is the RC time constant (typically 1000 μs).
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def compute_fft_size(num_time, kernel_length):
    """
    Compute next power-of-2 FFT size for linear convolution.

    Parameters
    ----------
    num_time : int
        Number of time samples in the signal.
    kernel_length : int
        Length of the convolution kernel.

    Returns
    -------
    int
        FFT size (power of 2).
    """
    return int(2 ** np.ceil(np.log2(num_time + kernel_length - 1)))


def create_rcrc_response(tau_us, time_step_us, n_tau=3.0):
    """
    Create RC⊗RC electronics impulse response kernel.

    Parameters
    ----------
    tau_us : float
        RC time constant in microseconds.
    time_step_us : float
        Time step size in microseconds.
    n_tau : float
        Number of time constants to include in kernel. Default 3.0.

    Returns
    -------
    kernel : np.ndarray
        Impulse response kernel, shape (R,).
    """
    R = int(n_tau * tau_us / time_step_us)
    t = np.arange(R) * time_step_us  # time in μs
    dt = time_step_us

    # RC⊗RC: h(t) = δ(t) + (t/τ - 2) · (1/τ) · e^(-t/τ) · dt
    kernel = np.zeros(R, dtype=np.float32)
    kernel[0] = 1.0  # δ(t) term
    kernel += (t / tau_us - 2.0) * (1.0 / tau_us) * np.exp(-t / tau_us) * dt

    return kernel


def load_electronics_response(time_step_us, tau_us=1000.0, n_tau=3.0):
    """
    Load electronics response kernels for all wire planes.

    Currently uses the same RC⊗RC kernel for all planes.

    Parameters
    ----------
    time_step_us : float
        Time step size in microseconds.
    tau_us : float
        RC time constant in microseconds. Default 1000.0.
    n_tau : float
        Number of time constants in kernel. Default 3.0.

    Returns
    -------
    kernels : dict
        Dictionary with keys 'U', 'V', 'Y', each mapping to the kernel array.
    """
    kernel = create_rcrc_response(tau_us, time_step_us, n_tau)
    return {'U': kernel, 'V': kernel, 'Y': kernel}


@partial(jax.jit, static_argnames=('chunk_size', 'fft_size', 'num_time'))
def electronics_response_core(signals, response, threshold, chunk_size, fft_size, num_time):
    """
    Apply electronics response convolution to dense wire signals.

    Finds active wires (above threshold), gathers them, applies FFT
    convolution, and scatters results back to dense output.

    Parameters
    ----------
    signals : jax.Array
        Dense wire signals, shape (num_wires, num_time).
    response : jax.Array
        Electronics impulse response kernel, shape (R,).
    threshold : float
        Threshold for detecting active wires (max abs value).
    chunk_size : int
        Maximum number of active wires to process (static).
    fft_size : int
        FFT size for convolution (static).
    num_time : int
        Number of time samples in output (static).

    Returns
    -------
    output : jax.Array
        Convolved signals, shape (num_wires, num_time).
    """
    num_wires = signals.shape[0]

    # Find active wires
    max_abs = jnp.max(jnp.abs(signals), axis=1)
    active_indices = jnp.where(max_abs > threshold, size=chunk_size, fill_value=0)[0]
    n_active = jnp.sum(max_abs > threshold)

    # Gather active wires
    active_signals = signals[active_indices]  # (chunk_size, num_time)

    # FFT convolution
    S = jnp.fft.rfft(active_signals, n=fft_size, axis=1)
    R = jnp.fft.rfft(response, n=fft_size)
    convolved = jnp.fft.irfft(S * R[None, :], n=fft_size, axis=1)

    # Trim to original time length
    convolved = convolved[:, :num_time]

    # Mask padding rows (beyond n_active)
    valid_mask = jnp.arange(chunk_size) < n_active
    convolved = convolved * valid_mask[:, None]

    # Scatter back to dense output
    output = jnp.zeros_like(signals)
    output = output.at[active_indices].set(convolved)

    return output


@partial(jax.jit, static_argnames=('chunk_size', 'fft_size', 'num_time'))
def electronics_convolve_active(active_signals, response, n_active, chunk_size, fft_size, num_time):
    """
    Apply electronics response convolution to pre-gathered active wires.

    Used in bucketed mode where active wires have already been extracted
    by buckets_to_active_wires.

    Parameters
    ----------
    active_signals : jax.Array
        Active wire signals, shape (chunk_size, num_time).
    response : jax.Array
        Electronics impulse response kernel, shape (R,).
    n_active : int or jax.Array
        Number of actually active wires (rest are padding).
    chunk_size : int
        Size of active_signals first dimension (static).
    fft_size : int
        FFT size for convolution (static).
    num_time : int
        Number of time samples in output (static).

    Returns
    -------
    convolved : jax.Array
        Convolved active signals, shape (chunk_size, num_time).
    """
    # FFT convolution
    S = jnp.fft.rfft(active_signals, n=fft_size, axis=1)
    R = jnp.fft.rfft(response, n=fft_size)
    convolved = jnp.fft.irfft(S * R[None, :], n=fft_size, axis=1)

    # Trim to original time length
    convolved = convolved[:, :num_time]

    # Mask padding rows
    valid_mask = jnp.arange(chunk_size) < n_active
    convolved = convolved * valid_mask[:, None]

    return convolved


@partial(jax.jit, static_argnames=('B1', 'B2', 'num_wires', 'num_time', 'chunk_size', 'max_buckets'))
def buckets_to_active_wires(buckets, num_active, compact_to_key,
                            B1, B2, num_wires, num_time, chunk_size, max_buckets):
    """
    Convert sparse buckets to individual active wire signals.

    Takes bucketed sparse representation and produces wire-sparse,
    time-dense output: one row per active wire.

    Parameters
    ----------
    buckets : jax.Array
        Sparse bucket contributions, shape (max_buckets, B1, B2).
    num_active : int or jax.Array
        Number of active buckets.
    compact_to_key : jax.Array
        Mapping from compact index to bucket key, shape (max_buckets,).
    B1 : int
        Bucket size in wire direction (static).
    B2 : int
        Bucket size in time direction (static).
    num_wires : int
        Total number of wires (static).
    num_time : int
        Total number of time steps (static).
    chunk_size : int
        Maximum number of active wires in output (static).
    max_buckets : int
        Maximum number of buckets (static).

    Returns
    -------
    active_signals : jax.Array
        Wire signals, shape (chunk_size, num_time).
    wire_indices : jax.Array
        Global wire index for each row, shape (chunk_size,).
    n_active_wires : jax.Array
        Number of actually active wires (scalar).
    """
    NUM_BUCKETS_T = (num_time + B2 - 1) // B2

    # Decode bucket keys to wire starts
    bw = compact_to_key // NUM_BUCKETS_T
    wire_starts = bw * B1  # (max_buckets,)

    # Build boolean occupancy vector to find unique active wires
    # Each bucket covers B1 wires starting at wire_starts[b]
    occupancy = jnp.zeros(num_wires + 1, dtype=jnp.bool_)

    def mark_bucket(i, occ):
        is_active = i < num_active
        ws = wire_starts[i]
        wire_ids = ws + jnp.arange(B1)
        # Clip to valid range; use num_wires as "trash" bin
        wire_ids = jnp.where(
            is_active & (wire_ids < num_wires),
            wire_ids,
            num_wires
        )
        occ = occ.at[wire_ids].set(True)
        return occ

    occupancy = jax.lax.fori_loop(0, max_buckets, mark_bucket, occupancy)
    # Clear trash bin
    occupancy = occupancy.at[num_wires].set(False)

    # Assign ranks via cumsum
    ranks = jnp.cumsum(occupancy[:num_wires]) - 1  # (num_wires,)
    # rank is -1 for inactive wires, 0..n_active_wires-1 for active
    n_active_wires = jnp.sum(occupancy[:num_wires])

    # Build rank_lookup: maps global wire -> row in output
    # Inactive wires map to chunk_size-1 (trash row, will be zeroed)
    rank_lookup = jnp.where(occupancy[:num_wires], ranks, chunk_size - 1)

    # Build wire_indices: for each output row, what global wire is it?
    # Use occupancy to extract wire indices
    wire_positions = jnp.where(occupancy[:num_wires], jnp.arange(num_wires), num_wires)
    # Sort to get active wires first (num_wires values go to end)
    sorted_wires = jnp.sort(wire_positions)
    # Pad to chunk_size if num_wires < chunk_size (fill with num_wires sentinel)
    if num_wires < chunk_size:
        sorted_wires = jnp.concatenate([
            sorted_wires,
            jnp.full(chunk_size - num_wires, num_wires, dtype=sorted_wires.dtype)
        ])
    wire_indices = sorted_wires[:chunk_size]
    # Replace overflow with 0 (will be masked anyway)
    wire_indices = jnp.where(jnp.arange(chunk_size) < n_active_wires, wire_indices, 0)

    # Scatter bucket data to per-wire rows
    active_signals = jnp.zeros((chunk_size, num_time), dtype=jnp.float32)

    def scatter_bucket(i, output):
        is_active = i < num_active
        key = compact_to_key[i]
        bw_i = key // NUM_BUCKETS_T
        bt_i = key % NUM_BUCKETS_T

        w_start = bw_i * B1
        t_start = bt_i * B2

        bucket_data = jnp.where(is_active, buckets[i], 0.0)  # (B1, B2)

        # For each wire in this bucket, scatter to the correct output row
        w_indices = w_start + jnp.arange(B1)  # (B1,)
        # Clip wire indices to valid range
        w_indices_safe = jnp.clip(w_indices, 0, num_wires - 1)
        row_indices = rank_lookup[w_indices_safe]  # (B1,)

        # Build 2D index arrays for scatter
        # row_indices: (B1,) -> (B1, B2) via broadcast
        # time indices: (B2,) -> (B1, B2) via broadcast
        row_2d = jnp.broadcast_to(row_indices[:, None], (B1, B2))
        t_indices = t_start + jnp.arange(B2)
        time_2d = jnp.broadcast_to(t_indices[None, :], (B1, B2))

        # Only scatter if wire and time are in bounds
        valid_w = (w_indices[:, None] < num_wires) & (w_indices[:, None] >= 0)
        valid_t = (t_indices[None, :] < num_time) & (t_indices[None, :] >= 0)
        valid = valid_w & valid_t & is_active

        data = jnp.where(valid, bucket_data, 0.0)
        output = output.at[row_2d, time_2d].add(data, mode='drop')
        return output

    active_signals = jax.lax.fori_loop(0, max_buckets, scatter_bucket, active_signals)

    # Zero out the trash row and any row beyond n_active_wires
    valid_rows = jnp.arange(chunk_size) < n_active_wires
    active_signals = active_signals * valid_rows[:, None]

    return active_signals, wire_indices, n_active_wires


# =============================================================================
# FACTORY FUNCTIONS (create closures for use inside JIT)
# =============================================================================

def _noop_electronics(sig, side_idx, plane_idx, n_wires, n_time):
    """Identity — no electronics processing."""
    return sig


def _build_dense(cfg, e_kernels, e_chunk, e_fft, e_threshold):
    """Build per-plane dense electronics closures."""
    def make_fn(plane_type):
        kernel = e_kernels[plane_type]
        def fn(signal, n_wires, n_time):
            return electronics_response_core(
                signal, kernel, e_threshold, e_chunk, e_fft, n_time)
        return fn

    plane_fns = {
        (s, p): make_fn(cfg.plane_names[s][p])
        for s in range(2) for p in range(3)
    }

    def electronics_fn(sig, side_idx, plane_idx, n_wires, n_time):
        return plane_fns[(side_idx, plane_idx)](sig, n_wires, n_time)
    return electronics_fn


def _build_bucketed(cfg, e_kernels, e_chunk, e_fft, bucket_dims):
    """Build per-plane bucketed electronics closures (bucket → wire-sparse)."""
    max_buckets_e = cfg.max_active_buckets

    def make_fn(plane_type):
        kernel = e_kernels[plane_type]
        B1, B2 = bucket_dims[plane_type]

        def fn(signal_tuple, num_wires_plane, num_time_steps_plane):
            buckets, num_active, compact_to_key, _, _ = signal_tuple
            active_signals, wire_indices, n_active_w = buckets_to_active_wires(
                buckets, num_active, compact_to_key,
                B1, B2, num_wires_plane, num_time_steps_plane,
                e_chunk, max_buckets_e
            )
            active_signals = electronics_convolve_active(
                active_signals, kernel, n_active_w,
                e_chunk, e_fft, num_time_steps_plane
            )
            return (active_signals, wire_indices, n_active_w)
        return fn

    plane_fns = {
        (s, p): make_fn(cfg.plane_names[s][p])
        for s in range(2) for p in range(3)
    }

    def electronics_fn(sig, side_idx, plane_idx, n_wires, n_time):
        return plane_fns[(side_idx, plane_idx)](sig, n_wires, n_time)
    return electronics_fn


def create_electronics_fn(cfg, response_kernels,
                          electronics_chunk_size=None, electronics_threshold=0.0):
    """Create electronics response closure for use inside JIT.

    Pre-computes kernels, FFT size, and bucket dims internally from
    SimConfig and response kernels.

    Parameters
    ----------
    cfg : SimConfig
        Static simulation config.
    response_kernels : dict
        Loaded response kernels (for bucket dimensions).
    electronics_chunk_size : int, optional
        Max active wires to process. Defaults to max num_wires across all planes.
    electronics_threshold : float
        Threshold for active wire detection. Default 0.0.

    Returns
    -------
    electronics_fn : callable
        Signature: (sig, side_idx, plane_idx, n_wires, n_time) -> processed signal.
    metadata : dict
        Pre-computed values for printing/validation:
        {'e_chunk', 'e_fft', 'e_kernels'}. Empty if electronics disabled.
    """
    if not cfg.include_electronics:
        return _noop_electronics, {}

    # Pre-compute from raw inputs
    _TAU_US = 1000.0
    _N_TAU = 3.0
    raw_kernels = load_electronics_response(
        time_step_us=cfg.time_step_us, tau_us=_TAU_US, n_tau=_N_TAU)
    e_kernels = {t: jnp.array(raw_kernels[t]) for t in ['U', 'V', 'Y']}

    R = len(raw_kernels['U'])
    e_fft = compute_fft_size(cfg.num_time_steps, R)

    if electronics_chunk_size is None:
        e_chunk = max(cfg.side_geom[s].num_wires[p]
                      for s in range(2) for p in range(3))
    else:
        e_chunk = electronics_chunk_size

    metadata = {'e_chunk': e_chunk, 'e_fft': e_fft}

    if cfg.use_bucketed:
        bucket_dims = {}
        for plane_type in ['U', 'V', 'Y']:
            B1 = 2 * response_kernels[plane_type].num_wires
            B2 = 2 * response_kernels[plane_type].kernel_height
            bucket_dims[plane_type] = (B1, B2)
        return _build_bucketed(cfg, e_kernels, e_chunk, e_fft, bucket_dims), metadata

    return _build_dense(cfg, e_kernels, e_chunk, e_fft, electronics_threshold), metadata


# =============================================================================
# DIGITIZATION FACTORY
# =============================================================================

def _digitize_signal(signal, gain_scale, pedestal, adc_max):
    """Core digitization: scale, add pedestal, round, clip, subtract pedestal."""
    scaled = signal * gain_scale
    unsigned = scaled + pedestal
    unsigned = jnp.round(unsigned)
    unsigned = jnp.clip(unsigned, 0.0, adc_max)
    return unsigned - pedestal


def _noop_digitize(sig, side_idx, plane_idx):
    """Identity — no digitization."""
    return sig


def _build_digitize_dense(cfg, gain, adc_max, ped_collection, ped_induction):
    """Build per-plane dense digitization closures."""
    def make_fn(plane_type):
        ped = ped_collection if plane_type == 'Y' else ped_induction
        def fn(signal):
            return _digitize_signal(signal, gain, ped, adc_max)
        return fn

    plane_fns = {
        (s, p): make_fn(cfg.plane_names[s][p])
        for s in range(2) for p in range(3)
    }
    def digitize_fn(sig, side_idx, plane_idx):
        return plane_fns[(side_idx, plane_idx)](sig)
    return digitize_fn


def _build_digitize_bucketed(cfg, gain, adc_max, ped_collection, ped_induction):
    """Build per-plane bucketed digitization closures."""
    def make_fn(plane_type):
        ped = ped_collection if plane_type == 'Y' else ped_induction
        def fn(signal_tuple):
            buckets, num_active, compact_to_key, b1, b2 = signal_tuple
            return (_digitize_signal(buckets, gain, ped, adc_max),
                    num_active, compact_to_key, b1, b2)
        return fn

    plane_fns = {
        (s, p): make_fn(cfg.plane_names[s][p])
        for s in range(2) for p in range(3)
    }
    def digitize_fn(sig, side_idx, plane_idx):
        return plane_fns[(side_idx, plane_idx)](sig)
    return digitize_fn


def _build_digitize_wire_sparse(cfg, gain, adc_max, ped_collection, ped_induction):
    """Build per-plane wire-sparse digitization closures."""
    def make_fn(plane_type):
        ped = ped_collection if plane_type == 'Y' else ped_induction
        def fn(signal_tuple):
            active_signals, wire_indices, n_active = signal_tuple
            return (_digitize_signal(active_signals, gain, ped, adc_max),
                    wire_indices, n_active)
        return fn

    plane_fns = {
        (s, p): make_fn(cfg.plane_names[s][p])
        for s in range(2) for p in range(3)
    }
    def digitize_fn(sig, side_idx, plane_idx):
        return plane_fns[(side_idx, plane_idx)](sig)
    return digitize_fn


def create_digitize_fn(cfg, digitization_config=None):
    """Create digitization closure for use inside JIT.

    If digitization_config is None and cfg.include_digitize is True,
    builds a default DigitizationConfig.

    Parameters
    ----------
    cfg : SimConfig
        Static simulation config.
    digitization_config : DigitizationConfig, optional
        Explicit digitization parameters. If None, uses defaults.

    Returns
    -------
    digitize_fn : callable
        Signature: (sig, side_idx, plane_idx) -> digitized signal.
    dig_config : DigitizationConfig or None
        The resolved config (for printing/inspection).
    """
    if not cfg.include_digitize:
        return _noop_digitize, None

    if digitization_config is None:
        from tools.config import create_digitization_config
        digitization_config = create_digitization_config()

    dig_cfg = digitization_config
    gain = float(dig_cfg.gain_scale)
    adc_max = float((1 << dig_cfg.n_bits) - 1)
    ped_collection = float(dig_cfg.pedestal_collection)
    ped_induction = float(dig_cfg.pedestal_induction)

    if cfg.use_bucketed and cfg.include_electronics:
        return _build_digitize_wire_sparse(
            cfg, gain, adc_max, ped_collection, ped_induction), digitization_config
    elif cfg.use_bucketed:
        return _build_digitize_bucketed(
            cfg, gain, adc_max, ped_collection, ped_induction), digitization_config
    return _build_digitize_dense(
        cfg, gain, adc_max, ped_collection, ped_induction), digitization_config
