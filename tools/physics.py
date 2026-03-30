"""
Shared physics pipeline functions for JAXTPC simulation.

Both production and differentiable paths call these with identical signatures.
No @jax.jit decorators — called from within the simulator's outer JIT.
"""

import jax
import jax.numpy as jnp

from tools.config import (
    DepositData, SimParams, SideGeometry, SimConfig,
    SideIntermediates, PlaneIntermediates, SCEOutputs,
)
from tools.drift import compute_drift_to_plane, correct_drift_for_plane, apply_drift_corrections
from tools.wires import (
    compute_wire_distances,
    prepare_deposit_for_response,
    accumulate_response_signals,
    scatter_contributions_to_buckets_batched,
    build_bucket_mapping,
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
# Side-level physics
# ============================================================================

def compute_side_physics(
    deposits, sim_params, side_geom, sce_fn, recomb_fn,
):
    """Side-level physics: recombination + drift + SCE corrections.

    Parameters
    ----------
    deposits : DepositData
        Input deposits (masking via valid_mask).
    sim_params : SimParams
        Physics parameters.
    side_geom : SideGeometry
        Static geometry for this side.
    sce_fn : callable
        (positions_cm) -> SCEOutputs(efield_correction, drift_corr_cm).
    recomb_fn : callable
        (de, dx_cm, phi_drift, e_field_Vcm, recomb_params) -> charges.

    Returns
    -------
    SideIntermediates
        charges (zeroed for invalid deposits), drift, positions.
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
    charges = recomb_fn(
        deposits.de, dx_cm, phi_drift, E_mag, sim_params.recomb_params
    )

    # Apply valid_mask once here — charges=0 for invalid deposits.
    # No need to propagate valid_mask further.
    charges = charges * deposits.valid_mask

    # Drift to furthest plane
    drift_dist, drift_time, yz = compute_drift_to_plane(
        positions_cm, side_geom.half_width_cm,
        sim_params.velocity_cm_us, side_geom.furthest_plane_dist_cm
    )

    # Apply SCE drift corrections (velocity explicit for gradient flow)
    drift_dist, drift_time, yz = apply_drift_corrections(
        drift_dist, drift_time, yz,
        sce.drift_corr_cm[:, 0], sce.drift_corr_cm[:, 1], sce.drift_corr_cm[:, 2],
        sim_params.velocity_cm_us,
    )

    return SideIntermediates(
        charges=charges,
        drift_distance_cm=drift_dist,
        drift_time_us=drift_time,
        positions_cm=positions_cm,
        positions_yz_cm=yz,
    )


# ============================================================================
# Plane-level physics
# ============================================================================

def compute_plane_physics(side_int, sim_params, side_geom, plane_idx):
    """Plane-level physics: drift correction + attenuation + wire geometry.

    Parameters
    ----------
    side_int : SideIntermediates
        From compute_side_physics.
    sim_params : SimParams
        Physics parameters.
    side_geom : SideGeometry
        Static geometry for this side.
    plane_idx : int
        Plane index (0, 1, or 2).

    Returns
    -------
    PlaneIntermediates
        Per-plane physics results for downstream response computation.
    """
    plane_dist_diff = side_geom.furthest_plane_dist_cm - side_geom.plane_distances_cm[plane_idx]
    drift_dist, drift_time = correct_drift_for_plane(
        side_int.drift_distance_cm, side_int.drift_time_us,
        sim_params.velocity_cm_us, plane_dist_diff
    )

    drift_time_safe = jnp.where(jnp.isnan(drift_time), 0.0, drift_time)
    attenuation = jnp.exp(-drift_time_safe / sim_params.lifetime_us)

    closest_idx, closest_dist = compute_wire_distances(
        side_int.positions_yz_cm,
        side_geom.angles_rad[plane_idx],
        side_geom.wire_spacings_cm[plane_idx],
        side_geom.max_wire_indices[plane_idx],
        side_geom.index_offsets[plane_idx],
    )

    return PlaneIntermediates(
        drift_distance_cm=drift_dist,
        drift_time_us=drift_time,
        attenuation=attenuation,
        closest_wire_idx=closest_idx,
        closest_wire_dist=closest_dist,
        charges=side_int.charges,
        positions_cm=side_int.positions_cm,
    )


# ============================================================================
# Per-chunk response computation (shared by dense and bucketed paths)
# ============================================================================

def compute_chunk_response(plane_int, response_fn, start, chunk_size,
                           cfg, side_geom, plane_idx):
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
    side_geom : SideGeometry
        Side geometry.
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
    b_drift_time = jax.lax.dynamic_slice(plane_int.drift_time_us, (start,), (chunk_size,))
    b_wire_idx   = jax.lax.dynamic_slice(plane_int.closest_wire_idx, (start,), (chunk_size,))
    b_wire_dist  = jax.lax.dynamic_slice(plane_int.closest_wire_dist, (start,), (chunk_size,))
    b_atten      = jax.lax.dynamic_slice(plane_int.attenuation, (start,), (chunk_size,))
    b_pos_cm     = jax.lax.dynamic_slice(plane_int.positions_cm, (start, 0), (chunk_size, 3))
    b_drift_dist = jax.lax.dynamic_slice(plane_int.drift_distance_cm, (start,), (chunk_size,))

    # Prepare deposit data (vmapped scalar function)
    # Pass valid_hit=True always — charges already zeroed for invalid deposits
    # in compute_side_physics. The function's wire bounds check still runs.
    deposit_data = jax.vmap(
        prepare_deposit_for_response,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None)
    )(
        b_charges, b_drift_time, b_wire_idx, b_wire_dist, b_atten,
        True,  # valid_hit — always True (charges handle masking)
        side_geom.wire_spacings_cm[plane_idx],
        cfg.time_step_us,
        side_geom.num_wires[plane_idx],
    )
    wire_idx, wire_offsets, time_idx, time_offsets, intensities = deposit_data

    # Response contributions (unified signature, backend-specific)
    contributions = response_fn(b_pos_cm, b_drift_dist, wire_offsets, time_offsets)

    return wire_idx, time_idx, intensities, contributions


# ============================================================================
# Dense signal accumulation loop
# ============================================================================

def compute_plane_signal(plane_int, response_fn, n_actual, chunk_size,
                         cfg, side_geom, plane_idx, plane_kernel):
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
    side_geom : SideGeometry
        Side geometry.
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

    num_wires = side_geom.num_wires[plane_idx]
    num_time_steps = cfg.num_time_steps

    def body(i, signal_accum):
        start = i * chunk_size
        wire_idx, time_idx, intensities, contributions = \
            compute_chunk_response(plane_int, response_fn, start, chunk_size,
                                   cfg, side_geom, plane_idx)
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
                                   cfg, side_geom, plane_idx, plane_kernel):
    """Bucketed accumulation: fori_loop(compute_chunk_response → scatter).

    Production-only (never diff path).
    """
    max_safe_batches = plane_int.charges.shape[0] // chunk_size
    n_batches = jnp.minimum(
        (n_actual + chunk_size - 1) // chunk_size,
        max_safe_batches)

    num_wires = side_geom.num_wires[plane_idx]
    num_time_steps = cfg.num_time_steps

    def body(i, carry_buckets):
        start = i * chunk_size
        wire_idx, time_idx, intensities, contributions = \
            compute_chunk_response(plane_int, response_fn, start, chunk_size,
                                   cfg, side_geom, plane_idx)
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


def compute_bucket_maps(deposits, plane_int, side_geom, plane_idx, cfg, plane_kernel):
    """Compute wire/time maps and bucket mapping for bucketed accumulation.

    Parameters
    ----------
    deposits : DepositData
        Input deposits (for valid_mask).
    plane_int : PlaneIntermediates
        Per-plane physics outputs.
    side_geom : SideGeometry
        Side geometry.
    plane_idx : int
        Plane index.
    cfg : SimConfig
        Static simulation config.
    plane_kernel : dict
        Response kernel parameters for this plane type.

    Returns
    -------
    point_to_compact : jnp.ndarray
        Point-to-compact mapping.
    num_active : jnp.ndarray
        Number of active buckets.
    compact_to_key : jnp.ndarray
        Compact-to-key mapping.
    B1 : int
        Bucket wire dimension.
    B2 : int
        Bucket time dimension.
    """
    B1 = 2 * plane_kernel.num_wires
    B2 = 2 * plane_kernel.kernel_height
    wire_map = jnp.where(
        deposits.valid_mask,
        jnp.clip(plane_int.closest_wire_idx,
                 0, side_geom.num_wires[plane_idx] - 1),
        jnp.int32(0))
    time_map = jnp.where(
        deposits.valid_mask,
        jnp.clip(jnp.floor(
            plane_int.drift_time_us / cfg.time_step_us
        ).astype(jnp.int32), 0, cfg.num_time_steps - 1),
        jnp.int32(0))
    point_to_compact, num_active, compact_to_key = build_bucket_mapping(
        wire_map, time_map, B1, B2,
        side_geom.num_wires[plane_idx], cfg.num_time_steps,
        cfg.max_active_buckets, plane_kernel.wire_zero_bin, plane_kernel.time_zero_bin)
    return point_to_compact, num_active, compact_to_key, B1, B2
