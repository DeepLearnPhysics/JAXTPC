"""
Shared physics pipeline functions for JAXTPC simulation.

Both production and differentiable paths call these with identical signatures.
No @jax.jit decorators — called from within the simulator's outer JIT.
"""

import jax
import jax.numpy as jnp

from tools.config import (
    VolumeDeposits, SimParams, VolumeGeometry, SimConfig,
    VolumeIntermediates, PlaneIntermediates, PixelIntermediates, SCEOutputs,
)
from tools.drift import compute_drift_to_plane, correct_drift_for_plane, apply_drift_corrections
from tools.wires import (
    compute_wire_distances,
    prepare_deposit_for_response,
    accumulate_response_signals,
    scatter_contributions_to_buckets_batched,
    build_bucket_mapping,
    digitize_pixel_positions,
    prepare_pixel_deposit_for_response,
    scatter_contributions_to_pixel_buckets_batched,
    build_bucket_mapping_3d,
)


# ============================================================================
# E-field and angle computation
# ============================================================================

def compute_phi_drift(efield_correction, theta, phi, field_strength_Vcm):
    """Compute angle between track direction and local E-field.

    Parameters
    ----------
    efield_correction : (N, 3) dimensionless correction vector (E_local / |E_nominal|)
    theta : (N,) polar angle of track direction
    phi : (N,) azimuthal angle of track direction
    field_strength_Vcm : scalar, nominal E-field magnitude (V/cm)

    Returns
    -------
    phi_drift : (N,) angle between track and E-field in radians
    E_mag : (N,) or scalar, local E-field magnitude in V/cm
    """
    correction_mag = jnp.sqrt(jnp.sum(efield_correction ** 2, axis=-1))
    E_mag = field_strength_Vcm * correction_mag

    E_hat = efield_correction / jnp.maximum(correction_mag, 1e-10)[:, None]

    track_x = jnp.sin(theta) * jnp.cos(phi)
    track_y = jnp.sin(theta) * jnp.sin(phi)
    track_z = jnp.cos(theta)

    cos_phi = jnp.abs(
        track_x * E_hat[:, 0] + track_y * E_hat[:, 1] + track_z * E_hat[:, 2])
    phi_drift = jnp.arccos(jnp.clip(cos_phi, 0.0, 1.0))
    return phi_drift, E_mag


# ============================================================================
# Volume-level physics
# ============================================================================

def compute_volume_physics(
    deposits, sim_params, vol_geom, sce_fn, recomb_fn,
):
    """Volume-level physics: recombination + drift + SCE corrections.

    Parameters
    ----------
    deposits : DepositData
        Input deposits (masking via valid_mask).
    sim_params : SimParams
        Physics parameters.
    vol_geom : VolumeGeometry
        Static geometry for this volume.
    sce_fn : callable
        (positions_cm) -> SCEOutputs(efield_correction, drift_corr_cm).
    recomb_fn : callable
        (de, dx_cm, phi_drift, e_field_Vcm, recomb_params) -> (charges, photons).

    Returns
    -------
    VolumeIntermediates
        charges, photons (zeroed for invalid deposits), drift, positions.
    """
    positions_cm = deposits.positions_mm / 10.0
    dx_cm = deposits.dx / 10.0

    # Query SCE map once — returns both E-field correction and drift corrections
    sce = sce_fn(positions_cm)

    # Process normalized E-field correction for recombination
    phi_drift, E_mag = compute_phi_drift(
        sce.efield_correction, deposits.theta, deposits.phi,
        sim_params.recomb_params.field_strength_Vcm,
    )
    charges, photons = recomb_fn(
        deposits.de, dx_cm, phi_drift, E_mag, sim_params.recomb_params
    )

    # Zero out padding entries. n_actual is the count of real deposits;
    # everything beyond that is padding and must not contribute signal.
    # This is the single masking point — all downstream code trusts charges=0 for padding.
    padding_mask = jnp.arange(deposits.de.shape[0]) < deposits.n_actual
    charges = charges * padding_mask
    photons = photons * padding_mask

    # Drift to furthest plane (local frame: anode at x=0, drift toward -x)
    drift_dist, drift_time, yz = compute_drift_to_plane(
        positions_cm, 0.0, -1,
        sim_params.velocity_cm_us, vol_geom.furthest_plane_dist_cm
    )

    # Apply SCE drift corrections (velocity explicit for gradient flow)
    drift_dist, drift_time, yz = apply_drift_corrections(
        drift_dist, drift_time, yz,
        sce.drift_corr_cm[:, 0], sce.drift_corr_cm[:, 1], sce.drift_corr_cm[:, 2],
        sim_params.velocity_cm_us,
    )

    return VolumeIntermediates(
        charges=charges,
        photons=photons,
        drift_distance_cm=drift_dist,
        drift_time_us=drift_time,
        positions_cm=positions_cm,
        positions_yz_cm=yz,
        t0_us=deposits.t0_us,
        interaction_ids=deposits.interaction_ids,
        ancestor_track_ids=deposits.ancestor_track_ids,
    )


# ============================================================================
# Plane-level physics
# ============================================================================

def compute_plane_physics(vol_int, sim_params, vol_geom, plane_idx,
                          pre_window_us, readout_window_us):
    """Plane-level physics: drift correction + attenuation + wire geometry.

    Parameters
    ----------
    vol_int : VolumeIntermediates
        From compute_volume_physics.
    sim_params : SimParams
        Physics parameters.
    vol_geom : VolumeGeometry
        Static geometry for this volume.
    plane_idx : int
        Plane index (0, 1, or 2).
    pre_window_us : float
        Readout window extension before drift t=0 (μs).
    readout_window_us : float
        Total readout window duration (μs) = num_time_steps * time_step_us.

    Returns
    -------
    PlaneIntermediates
        Per-plane physics results for downstream response computation.
    """
    plane_dist_diff = vol_geom.furthest_plane_dist_cm - vol_geom.plane_distances_cm[plane_idx]
    drift_dist, drift_time = correct_drift_for_plane(
        vol_int.drift_distance_cm, vol_int.drift_time_us,
        sim_params.velocity_cm_us, plane_dist_diff
    )

    drift_time_safe = jnp.where(jnp.isnan(drift_time), 0.0, drift_time)
    attenuation = jnp.exp(-drift_time_safe / sim_params.lifetime_us)

    # Readout tick = drift time + initial deposit time + pre-window offset
    tick_us = drift_time + vol_int.t0_us + pre_window_us

    # Zero charges for deposits outside the readout window
    # (same pattern as padding mask in compute_volume_physics)
    in_window = (tick_us >= 0.0) & (tick_us < readout_window_us)
    charges = vol_int.charges * in_window

    closest_idx, closest_dist = compute_wire_distances(
        vol_int.positions_yz_cm,
        vol_geom.angles_rad[plane_idx],
        vol_geom.wire_spacings_cm[plane_idx],
        vol_geom.max_wire_indices[plane_idx],
        vol_geom.index_offsets[plane_idx],
    )

    return PlaneIntermediates(
        drift_distance_cm=drift_dist,
        drift_time_us=drift_time,
        tick_us=tick_us,
        attenuation=attenuation,
        closest_wire_idx=closest_idx,
        closest_wire_dist=closest_dist,
        charges=charges,
        photons=vol_int.photons,
        positions_cm=vol_int.positions_cm,
    )


# ============================================================================
# Per-chunk response computation (shared by dense and bucketed paths)
# ============================================================================

def compute_chunk_response(plane_int, response_fn, start, chunk_size,
                           cfg, vol_geom, plane_idx):
    """Slice one chunk, prepare deposits, compute response contributions.

    Shared by dense, bucketed, and diff paths. Same code always.

    Parameters
    ----------
    plane_int : PlaneIntermediates
        Per-plane physics results (full padded array).
    response_fn : callable
        (positions_cm, drift_distance_cm, wire_offsets, time_offsets) -> (N, kW, kH)
    start : int (traced)
        Start index for this chunk.
    chunk_size : int (static)
        Number of deposits per chunk.
    cfg : SimConfig
        Static simulation config.
    vol_geom : VolumeGeometry
        Volume geometry.
    plane_idx : int
        Plane index (0, 1, or 2).

    Returns
    -------
    wire_idx : jnp.ndarray (chunk_size,)
        Relative wire indices.
    time_idx : jnp.ndarray (chunk_size,)
        Time bin indices.
    intensities : jnp.ndarray (chunk_size,)
        Charge * attenuation per deposit.
    contributions : jnp.ndarray (chunk_size, kW, kH)
        Response kernel contributions per deposit.
    """
    b_charges    = jax.lax.dynamic_slice(plane_int.charges, (start,), (chunk_size,))
    b_tick       = jax.lax.dynamic_slice(plane_int.tick_us, (start,), (chunk_size,))
    b_wire_idx   = jax.lax.dynamic_slice(plane_int.closest_wire_idx, (start,), (chunk_size,))
    b_wire_dist  = jax.lax.dynamic_slice(plane_int.closest_wire_dist, (start,), (chunk_size,))
    b_atten      = jax.lax.dynamic_slice(plane_int.attenuation, (start,), (chunk_size,))
    b_pos_cm     = jax.lax.dynamic_slice(plane_int.positions_cm, (start, 0), (chunk_size, 3))
    b_drift_dist = jax.lax.dynamic_slice(plane_int.drift_distance_cm, (start,), (chunk_size,))

    # Prepare deposit data (vmapped scalar function)
    # Pass valid_hit=True always — charges already zeroed for invalid deposits
    # in compute_volume_physics. The function's wire bounds check still runs.
    deposit_data = jax.vmap(
        prepare_deposit_for_response,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None)
    )(
        b_charges, b_tick, b_wire_idx, b_wire_dist, b_atten,
        True,  # valid_hit — always True (charges handle masking)
        vol_geom.wire_spacings_cm[plane_idx],
        cfg.time_step_us,
        vol_geom.num_wires[plane_idx],
    )
    wire_idx, wire_offsets, time_idx, time_offsets, intensities = deposit_data

    # Response contributions (unified signature, backend-specific)
    contributions = response_fn(b_pos_cm, b_drift_dist, wire_offsets, time_offsets)

    return wire_idx, time_idx, intensities, contributions


# ============================================================================
# Dense signal accumulation loop
# ============================================================================

def compute_plane_signal(plane_int, response_fn, n_actual, chunk_size,
                         cfg, vol_geom, plane_idx, plane_kernel):
    """Dense accumulation: fori_loop(compute_chunk_response → accumulate).

    Parameters
    ----------
    plane_int : PlaneIntermediates
    response_fn : callable
    n_actual : int
        Actual number of valid deposits. Can be Python int (static, diff path)
        or traced JAX int (dynamic, production path).
    chunk_size : int (static)
        Deposits per fori_loop iteration.
    cfg : SimConfig
        Static simulation config.
    vol_geom : VolumeGeometry
        Volume geometry.
    plane_idx : int
        Plane index (0, 1, or 2).
    plane_kernel : dict
        Response kernel metadata for this plane.
    """
    max_safe_batches = plane_int.charges.shape[0] // chunk_size
    # Use Python min() for static values (diff path — supports reverse-mode grad)
    # Use jnp.minimum() for traced values (production path — no grad needed)
    if isinstance(n_actual, int):
        n_batches = min((n_actual + chunk_size - 1) // chunk_size, max_safe_batches)
    else:
        n_batches = jnp.minimum(
            (n_actual + chunk_size - 1) // chunk_size, max_safe_batches)

    num_wires = vol_geom.num_wires[plane_idx]
    num_time_steps = cfg.num_time_steps

    def body(i, signal_accum):
        start = i * chunk_size
        wire_idx, time_idx, intensities, contributions = \
            compute_chunk_response(plane_int, response_fn, start, chunk_size,
                                   cfg, vol_geom, plane_idx)
        batch = accumulate_response_signals(
            wire_idx, time_idx, intensities, contributions,
            num_wires, num_time_steps,
            plane_kernel.num_wires, plane_kernel.kernel_height,
            plane_kernel.wire_zero_bin, plane_kernel.time_zero_bin)
        return signal_accum + batch

    return jax.lax.fori_loop(0, n_batches, body,
                              jnp.zeros((num_wires, num_time_steps)))


# ============================================================================
# Bucketed signal accumulation loop
# ============================================================================

def compute_plane_signal_bucketed(plane_int, response_fn, n_actual, chunk_size,
                                   point_to_compact, max_buckets, B1, B2,
                                   cfg, vol_geom, plane_idx, plane_kernel):
    """Bucketed accumulation: fori_loop(compute_chunk_response → scatter).

    Supports both production (traced n_actual) and differentiable (static n_actual)
    paths via the same isinstance check as compute_plane_signal.
    """
    max_safe_batches = plane_int.charges.shape[0] // chunk_size
    if isinstance(n_actual, int):
        n_batches = min((n_actual + chunk_size - 1) // chunk_size, max_safe_batches)
    else:
        n_batches = jnp.minimum(
            (n_actual + chunk_size - 1) // chunk_size, max_safe_batches)

    num_wires = vol_geom.num_wires[plane_idx]
    num_time_steps = cfg.num_time_steps

    def body(i, carry_buckets):
        start = i * chunk_size
        wire_idx, time_idx, intensities, contributions = \
            compute_chunk_response(plane_int, response_fn, start, chunk_size,
                                   cfg, vol_geom, plane_idx)
        b_point_to_compact = jax.lax.dynamic_slice(point_to_compact, (start, 0), (chunk_size, 4))
        batch_buckets = scatter_contributions_to_buckets_batched(
            wire_idx, time_idx, intensities, contributions,
            b_point_to_compact, max_buckets,
            plane_kernel.num_wires, plane_kernel.kernel_height,
            B1, B2,
            plane_kernel.wire_zero_bin, plane_kernel.time_zero_bin,
            batch_size=chunk_size,
            num_wires=num_wires,
            num_time_steps=num_time_steps)
        return carry_buckets + batch_buckets

    return jax.lax.fori_loop(0, n_batches, body,
                              jnp.zeros((max_buckets, B1, B2)))


def compute_bucket_maps(plane_int, vol_geom, plane_idx, cfg, plane_kernel):
    """Compute wire/time maps and bucket mapping for bucketed accumulation.

    Parameters
    ----------
    plane_int : PlaneIntermediates
        Per-plane physics outputs.
    vol_geom : VolumeGeometry
        Volume geometry.
    plane_idx : int
        Plane index.
    cfg : SimConfig
        Static simulation config.
    plane_kernel : dict
        Response kernel parameters for this plane type.

    Returns
    -------
    point_to_compact : jnp.ndarray
    num_active : jnp.ndarray
    compact_to_key : jnp.ndarray
    B1 : int
    B2 : int
    """
    B1 = 2 * plane_kernel.num_wires
    B2 = 2 * plane_kernel.kernel_height
    # No valid_mask needed — padding entries have charges=0 (masked after recombination),
    # so they produce zero contributions regardless of bucket assignment.
    wire_map = jnp.clip(plane_int.closest_wire_idx,
                        0, vol_geom.num_wires[plane_idx] - 1)
    time_map = jnp.clip(jnp.floor(
        plane_int.tick_us / cfg.time_step_us
    ).astype(jnp.int32), 0, cfg.num_time_steps - 1)
    point_to_compact, num_active, compact_to_key = build_bucket_mapping(
        wire_map, time_map, B1, B2,
        vol_geom.num_wires[plane_idx], cfg.num_time_steps,
        cfg.max_active_buckets, plane_kernel.wire_zero_bin, plane_kernel.time_zero_bin)
    return point_to_compact, num_active, compact_to_key, B1, B2


# ============================================================================
# Pixel readout: physics, chunk response, and bucketed accumulation
# ============================================================================

def compute_pixel_physics(vol_int, sim_params, vol_geom,
                          pre_window_us, readout_window_us,
                          pixel_pitch_cm, pixel_origins_cm,
                          num_py, num_pz):
    """Pixel-level physics: attenuation + window cut + pixel digitization.

    Pixel equivalent of compute_plane_physics. No plane correction needed
    (pixels sit at the anode, no multi-plane geometry).

    Parameters
    ----------
    vol_int : VolumeIntermediates
        From compute_volume_physics.
    sim_params : SimParams
    vol_geom : VolumeGeometry
    pre_window_us : float
    readout_window_us : float
    pixel_pitch_cm : float
        Pixel pitch in cm.
    pixel_origins_cm : jnp.ndarray, shape (2,)
        Pixel grid origin [y_min, z_min] in cm.
    num_py, num_pz : int
        Pixel grid dimensions.

    Returns
    -------
    PixelIntermediates
    """
    drift_dist = vol_int.drift_distance_cm
    drift_time = vol_int.drift_time_us

    drift_time_safe = jnp.where(jnp.isnan(drift_time), 0.0, drift_time)
    attenuation = jnp.exp(-drift_time_safe / sim_params.lifetime_us)

    tick_us = drift_time + vol_int.t0_us + pre_window_us

    in_window = (tick_us >= 0.0) & (tick_us < readout_window_us)
    charges = vol_int.charges * in_window

    pixel_y_idx, pixel_z_idx, pixel_y_offset, pixel_z_offset = \
        digitize_pixel_positions(
            vol_int.positions_yz_cm, pixel_pitch_cm, pixel_origins_cm)

    return PixelIntermediates(
        drift_distance_cm=drift_dist,
        drift_time_us=drift_time,
        tick_us=tick_us,
        attenuation=attenuation,
        pixel_y_idx=pixel_y_idx,
        pixel_z_idx=pixel_z_idx,
        pixel_y_offset=pixel_y_offset,
        pixel_z_offset=pixel_z_offset,
        charges=charges,
        positions_cm=vol_int.positions_cm,
    )


def compute_chunk_pixel_response(pixel_int, response_fn, start, chunk_size,
                                 time_step_us, num_py, num_pz):
    """Slice one chunk of pixel deposits, prepare, compute response.

    Pixel equivalent of compute_chunk_response.

    Parameters
    ----------
    pixel_int : PixelIntermediates
        Per-pixel physics results (full padded array).
    response_fn : callable
        (positions_cm, drift_distance_cm, py_offsets, pz_offsets, time_offsets)
        -> (N, K_py, K_pz, K_t)
    start : int (traced)
        Start index for this chunk.
    chunk_size : int (static)
        Number of deposits per chunk.
    time_step_us : float
    num_py, num_pz : int
        Pixel grid dimensions.

    Returns
    -------
    pixel_y_idx : jnp.ndarray (chunk_size,)
    pixel_z_idx : jnp.ndarray (chunk_size,)
    time_idx : jnp.ndarray (chunk_size,)
    intensities : jnp.ndarray (chunk_size,)
    contributions : jnp.ndarray (chunk_size, K_py, K_pz, K_t)
    """
    b_charges = jax.lax.dynamic_slice(pixel_int.charges, (start,), (chunk_size,))
    b_tick = jax.lax.dynamic_slice(pixel_int.tick_us, (start,), (chunk_size,))
    b_py_idx = jax.lax.dynamic_slice(pixel_int.pixel_y_idx, (start,), (chunk_size,))
    b_pz_idx = jax.lax.dynamic_slice(pixel_int.pixel_z_idx, (start,), (chunk_size,))
    b_py_off = jax.lax.dynamic_slice(pixel_int.pixel_y_offset, (start,), (chunk_size,))
    b_pz_off = jax.lax.dynamic_slice(pixel_int.pixel_z_offset, (start,), (chunk_size,))
    b_atten = jax.lax.dynamic_slice(pixel_int.attenuation, (start,), (chunk_size,))
    b_pos_cm = jax.lax.dynamic_slice(pixel_int.positions_cm, (start, 0), (chunk_size, 3))
    b_drift_dist = jax.lax.dynamic_slice(pixel_int.drift_distance_cm, (start,), (chunk_size,))

    deposit_data = jax.vmap(
        prepare_pixel_deposit_for_response,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None)
    )(
        b_charges, b_tick, b_py_idx, b_pz_idx, b_py_off, b_pz_off,
        b_atten, True, time_step_us, num_py, num_pz,
    )
    pixel_y_idx, pixel_z_idx, py_offsets, pz_offsets, time_idx, time_offsets, intensities = deposit_data

    contributions = response_fn(b_pos_cm, b_drift_dist, py_offsets, pz_offsets, time_offsets)

    return pixel_y_idx, pixel_z_idx, time_idx, intensities, contributions


def compute_pixel_bucket_maps(pixel_int, num_py, num_pz, num_time_steps,
                              time_step_us, max_buckets,
                              kernel_py, kernel_pz, kernel_time,
                              py_zero_bin, pz_zero_bin, time_zero_bin):
    """Compute 3D bucket mapping for pixel bucketed accumulation.

    Pixel equivalent of compute_bucket_maps. Runs ONCE before the fori_loop.

    Parameters
    ----------
    pixel_int : PixelIntermediates
    num_py, num_pz, num_time_steps : int
    time_step_us : float
    max_buckets : int
    kernel_py, kernel_pz, kernel_time : int
        Pixel response kernel dimensions.
    py_zero_bin, pz_zero_bin, time_zero_bin : int
        Kernel center offsets.

    Returns
    -------
    point_to_compact : jnp.ndarray, shape (N, 8)
    num_active : jnp.ndarray
    compact_to_key : jnp.ndarray, shape (max_buckets,)
    B1, B2, B3 : int
        Tile sizes.
    """
    B1 = 2 * kernel_py
    B2 = 2 * kernel_pz
    B3 = 2 * kernel_time

    py_map = jnp.clip(pixel_int.pixel_y_idx, 0, num_py - 1)
    pz_map = jnp.clip(pixel_int.pixel_z_idx, 0, num_pz - 1)
    time_map = jnp.clip(jnp.floor(
        pixel_int.tick_us / time_step_us
    ).astype(jnp.int32), 0, num_time_steps - 1)

    point_to_compact, num_active, compact_to_key = build_bucket_mapping_3d(
        py_map, pz_map, time_map,
        B1, B2, B3, num_py, num_pz, num_time_steps,
        max_buckets, py_zero_bin, pz_zero_bin, time_zero_bin)

    return point_to_compact, num_active, compact_to_key, B1, B2, B3


def compute_pixel_signal_bucketed(pixel_int, response_fn, n_actual, chunk_size,
                                  point_to_compact, max_buckets,
                                  B1, B2, B3,
                                  time_step_us, num_py, num_pz, num_time_steps,
                                  kernel_py, kernel_pz, kernel_time,
                                  py_zero_bin, pz_zero_bin, time_zero_bin):
    """Bucketed pixel accumulation: fori_loop(compute_chunk → scatter).

    Pixel equivalent of compute_plane_signal_bucketed.

    Parameters
    ----------
    pixel_int : PixelIntermediates
    response_fn : callable
    n_actual : jnp.ndarray
        Number of valid deposits (traced).
    chunk_size : int (static)
    point_to_compact : jnp.ndarray, shape (N, 8)
    max_buckets : int
    B1, B2, B3 : int
        Tile sizes.
    time_step_us : float
    num_py, num_pz, num_time_steps : int
    kernel_py, kernel_pz, kernel_time : int
    py_zero_bin, pz_zero_bin, time_zero_bin : int

    Returns
    -------
    jnp.ndarray, shape (max_buckets, B1, B2, B3)
    """
    max_safe_batches = pixel_int.charges.shape[0] // chunk_size
    if isinstance(n_actual, int):
        n_batches = min((n_actual + chunk_size - 1) // chunk_size, max_safe_batches)
    else:
        n_batches = jnp.minimum(
            (n_actual + chunk_size - 1) // chunk_size, max_safe_batches)

    def body(i, carry_buckets):
        start = i * chunk_size
        py_idx, pz_idx, time_idx, intensities, contributions = \
            compute_chunk_pixel_response(
                pixel_int, response_fn, start, chunk_size,
                time_step_us, num_py, num_pz)
        b_p2c = jax.lax.dynamic_slice(point_to_compact, (start, 0), (chunk_size, 8))
        batch_buckets = scatter_contributions_to_pixel_buckets_batched(
            py_idx, pz_idx, time_idx, intensities, contributions,
            b_p2c, max_buckets,
            kernel_py, kernel_pz, kernel_time,
            B1, B2, B3,
            py_zero_bin, pz_zero_bin, time_zero_bin,
            batch_size=chunk_size,
            num_py=num_py, num_pz=num_pz, num_time_steps=num_time_steps)
        return carry_buckets + batch_buckets

    return jax.lax.fori_loop(0, n_batches, body,
                              jnp.zeros((max_buckets, B1, B2, B3)))
