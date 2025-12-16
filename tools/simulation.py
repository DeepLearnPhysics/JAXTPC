"""
Detector simulation module with dual output paths and per-side padding.

This module provides the DetectorSimulator class that runs TWO parallel physics
simulation paths simultaneously:

Response Path:
    Generates detector response signals using pre-computed kernels
    convolved with diffusion. Output: response_signals

Hit Path:
    Calculates diffused charge without response convolution.
    Tracks which particle contributed to each location.
    Output: track_hits

V2 Changes:
    - Per-side padding with tier system (east/west split before JIT)
    - Automatic data splitting and padding inside process_event
    - Pre-simulation and post-simulation validation
    - No more manual padding in notebooks
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import traceback
import os
from functools import partial

# Config classes
from tools.config import (
    DepositData, DriftParams, TimeParams, PlaneGeometry,
    DiffusionParams, TrackHitsConfig,
    create_diffusion_params, create_drift_params, create_time_params,
    create_plane_geometry, create_track_hits_config
)

# Core physics modules
from tools.drift import (
    compute_drift_to_plane,
    correct_drift_for_plane,
    compute_lifetime_attenuation
)

# Wire calculation functions
from tools.wires import (
    # Shared wire calculations
    compute_wire_distances,
    compute_angular_scaling, compute_angular_scaling_vmap,
    compute_deposit_wire_angles, compute_deposit_wire_angles_vmap,

    # Response path functions
    prepare_deposit_for_response,
    accumulate_response_signals,

    # Bucketed accumulation (sparse)
    accumulate_response_signals_sparse_bucketed,
    sparse_buckets_to_dense,

    # Hit path functions (with K_wire x K_time diffusion)
    compute_gaussian_diffusion,
    prepare_deposit_with_diffusion,
    accumulate_signals
)

# Response kernel system (for response path)
from tools.kernels import load_response_kernels, apply_diffusion_response

# Track labeling system (for hit path)
from tools.track_hits import group_hits_by_track, label_hits

# Core modules
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import recombine_steps


# =============================================================================
# PADDING TIER SYSTEM
# =============================================================================

# Default padding tiers - array sizes that trigger JIT recompilation
# Using fixed tiers limits the number of JIT versions cached
PADDING_TIERS = (50_000, 75_000, 150_000, 250_000_000, 350_000, 450_000, 550_000, 650_000,
                 750_000, 850_000, 950_000, 1_000_000)


def pick_padding_tier(n_hits, tiers=PADDING_TIERS):
    """
    Pick the smallest tier that fits the data.

    Parameters
    ----------
    n_hits : int
        Actual number of hits to fit.
    tiers : tuple of int
        Available tier sizes.

    Returns
    -------
    int
        Selected tier size.
    """
    for tier in tiers:
        if n_hits <= tier:
            return tier
    print(f"ERROR: n_hits ({n_hits:,}) exceeds largest tier ({tiers[-1]:,}). Data will be truncated!")
    return tiers[-1]


# =============================================================================
# DETECTOR SIMULATOR CLASS
# =============================================================================

class DetectorSimulator:
    """
    LArTPC detector simulation with dual output paths and per-side padding.

    Produces:
    - response_signals: Full detector simulation with response kernels
    - track_hits: Track labeling information for analysis

    V2 Features:
    - Automatic splitting of data by detector side (east x<0, west x>=0)
    - Per-side padding using tier system to minimize JIT recompilations
    - Built-in pre/post simulation validation
    - No manual padding required from notebooks

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().
    response_path : str
        Path to response kernel files.
    track_config : TrackHitsConfig, optional
        Configuration for track labeling. If None, uses defaults.
    diffusion_params : DiffusionParams, optional
        Diffusion parameters. If None, computed from config.
    padding_tiers : tuple of int, optional
        Available padding tier sizes. Default: (50_000, 150_000, 350_000, 750_000)
    use_bucketed : bool
        If True, use sparse bucketed accumulation for very large detectors.
    max_active_buckets : int
        Max active buckets for sparse mode. Default 1000.
    """

    def __init__(
        self,
        detector_config,
        response_path="tools/responses/",
        track_config=None,
        diffusion_params=None,
        padding_tiers=PADDING_TIERS,
        use_bucketed=False,
        max_active_buckets=1000
    ):
        print("--- Creating DetectorSimulator (V2 - Per-Side Padding) ---")

        # Store config
        self.detector_config = detector_config
        self.padding_tiers = padding_tiers
        self.use_bucketed = use_bucketed
        self.sparse_output = use_bucketed  # sparse_output is True when use_bucketed is True
        self.max_active_buckets = max_active_buckets

        # Extract parameter bundles
        print("   Extracting parameters...")
        self.drift_params = create_drift_params(detector_config)
        self.time_params = create_time_params(detector_config)

        # Create plane geometry for all planes
        self.plane_names = [
            ['U', 'V', 'Y'],  # Side 0 (East, x < 0)
            ['U', 'V', 'Y']   # Side 1 (West, x >= 0)
        ]
        self.plane_geometry = {}
        for side_idx in range(2):
            for plane_idx in range(3):
                plane_type = self.plane_names[side_idx][plane_idx]
                self.plane_geometry[(side_idx, plane_idx)] = create_plane_geometry(
                    detector_config, side_idx, plane_idx, plane_type
                )

        # Create or use provided diffusion params
        if diffusion_params is None:
            self.diffusion_params = create_diffusion_params(
                max_sigma_trans_unitless=detector_config['max_sigma_trans_unitless'],
                max_sigma_long_unitless=detector_config['max_sigma_long_unitless'],
                num_s=16,
                n_sigma=3.0
            )
        else:
            self.diffusion_params = diffusion_params

        # Create or use provided track config
        if track_config is None:
            max_wires = max(
                np.max(detector_config['max_wire_indices_abs']) -
                np.min(detector_config['min_wire_indices_abs']) + 1,
                2000
            )
            self.track_config = create_track_hits_config(
                threshold=1.0,
                max_tracks=10000,
                max_keys=500000
            )
            self._max_wires = int(max_wires)
            self._max_time = self.time_params.num_steps
        else:
            self.track_config = track_config
            self._max_wires = 2000  # Default
            self._max_time = self.time_params.num_steps

        # Load response kernels
        print("   Loading response kernels...")
        self.response_kernels = load_response_kernels(
            response_path=response_path,
            num_s=self.diffusion_params.num_s,
            wire_spacing=0.1,  # Fixed kernel bin spacing
            time_spacing=float(self.time_params.step_size_us),
            max_sigma_trans_unitless=self.diffusion_params.max_sigma_trans_unitless,
            max_sigma_long_unitless=self.diffusion_params.max_sigma_long_unitless
        )

        # Extract static arrays for JIT
        self._prepare_static_args()

        # Build the JIT-compiled calculator
        self._build_calculator()

        print(f"   Config: tiers={padding_tiers}, num_s={self.diffusion_params.num_s}, "
              f"K_wire={self.diffusion_params.K_wire}, K_time={self.diffusion_params.K_time}")
        if use_bucketed:
            print(f"   Using BUCKETED accumulation (B=2*kernel, max_buckets={max_active_buckets})")
        print("--- DetectorSimulator Ready ---")

    def _prepare_static_args(self):
        """Prepare static arguments for JIT compilation."""
        # Convert arrays to hashable tuples for static args
        index_offsets = np.array(self.detector_config['index_offsets'])
        max_indices = np.array(self.detector_config['max_wire_indices_abs'])
        min_indices = np.array(self.detector_config['min_wire_indices_abs'])
        num_wires = np.array(self.detector_config['num_wires_actual'])

        self._index_offsets_tuple = tuple(tuple(int(x) for x in row) for row in index_offsets)
        self._max_indices_tuple = tuple(tuple(int(x) for x in row) for row in max_indices)
        self._min_indices_tuple = tuple(tuple(int(x) for x in row) for row in min_indices)
        self._num_wires_tuple = tuple(tuple(int(x) for x in row) for row in num_wires)

    def _split_and_pad_data(self, deposit_data: DepositData):
        """
        Split deposit data by side and pad to appropriate tiers.

        Parameters
        ----------
        deposit_data : DepositData
            Input data (can be any size, will be split and padded).

        Returns
        -------
        east_data : DepositData
            Padded data for east side (x < 0).
        west_data : DepositData
            Padded data for west side (x >= 0).
        counts : dict
            Actual counts: n_east, n_west, n_tracks.
        """
        positions_mm = deposit_data.positions_mm
        charges = deposit_data.charges
        valid_mask = deposit_data.valid_mask
        theta = deposit_data.theta
        phi = deposit_data.phi
        track_ids = deposit_data.track_ids

        # Only consider valid entries
        x_mm = positions_mm[:, 0]
        east_mask = valid_mask & (x_mm < 0)
        west_mask = valid_mask & (x_mm >= 0)

        n_east = int(jnp.sum(east_mask))
        n_west = int(jnp.sum(west_mask))
        n_tracks = int(len(jnp.unique(track_ids[valid_mask])))

        # Pick tiers
        east_tier = pick_padding_tier(n_east, self.padding_tiers)
        west_tier = pick_padding_tier(n_west, self.padding_tiers)

        print(f"   East side: {n_east:,} hits -> tier {east_tier:,}")
        print(f"   West side: {n_west:,} hits -> tier {west_tier:,}")

        # Extract and pad each side
        east_data = self._extract_and_pad(
            positions_mm, charges, theta, phi, track_ids,
            east_mask, n_east, east_tier
        )
        west_data = self._extract_and_pad(
            positions_mm, charges, theta, phi, track_ids,
            west_mask, n_west, west_tier
        )

        counts = {'n_east': n_east, 'n_west': n_west, 'n_tracks': n_tracks}
        return east_data, west_data, counts

    def _extract_and_pad(self, positions_mm, charges, theta, phi, track_ids,
                         mask, n_valid, pad_size):
        """Extract masked data and pad to target size."""
        # Extract valid entries
        pos = positions_mm[mask]
        chg = charges[mask]
        th = theta[mask]
        ph = phi[mask]
        tid = track_ids[mask]

        # Pad to tier size
        pad_width = pad_size - n_valid

        if pad_width > 0:
            valid_out = jnp.arange(pad_size) < n_valid
            pos = jnp.pad(pos, ((0, pad_width), (0, 0)), constant_values=0.0)
            chg = jnp.pad(chg, (0, pad_width), constant_values=0.0)
            th = jnp.pad(th, (0, pad_width), constant_values=0.0)
            ph = jnp.pad(ph, (0, pad_width), constant_values=0.0)
            tid = jnp.pad(tid, (0, pad_width), constant_values=0)
        else:
            # Truncate if needed (shouldn't happen if tier picked correctly)
            valid_out = jnp.ones(pad_size, dtype=bool)
            pos = pos[:pad_size]
            chg = chg[:pad_size]
            th = th[:pad_size]
            ph = ph[:pad_size]
            tid = tid[:pad_size]

        return DepositData(
            positions_mm=pos,
            charges=chg,
            valid_mask=valid_out,
            theta=th,
            phi=ph,
            track_ids=tid
        )

    def _validate_pre_simulation(self, counts):
        """Check inputs before running simulation."""
        if self.track_config.max_tracks < counts['n_tracks']:
            print(f"ERROR: max_tracks ({self.track_config.max_tracks:,}) < unique tracks ({counts['n_tracks']:,}). Tracks may be lost!")

    def _validate_post_simulation(self, track_hits, response_signals):
        """Check for truncation after simulation."""
        for plane_key, th in track_hits.items():
            actual_hits = int(th['num_hits'])
            if actual_hits >= self.track_config.max_keys:
                print(f"ERROR: Plane {plane_key}: num_hits ({actual_hits:,}) >= max_keys ({self.track_config.max_keys:,}). Data truncated!")

        if self.use_bucketed:
            for plane_key, resp in response_signals.items():
                buckets, num_active, compact_to_key, B1, B2 = resp
                if int(num_active) >= self.max_active_buckets:
                    print(f"ERROR: Plane {plane_key}: num_active ({int(num_active):,}) >= max_active_buckets ({self.max_active_buckets:,}). Data truncated!")

    def _build_calculator(self):
        """Build the JIT-compiled signal calculator."""
        # Extract all needed parameters
        num_time_steps = self.time_params.num_steps
        time_step_size_us = self.time_params.step_size_us
        detector_half_width_cm = self.drift_params.detector_half_width_cm
        velocity_cm_us = self.drift_params.velocity_cm_us
        lifetime_us = self.drift_params.lifetime_us
        diffusion_long_cm2_us = self.drift_params.diffusion_long_cm2_us
        diffusion_trans_cm2_us = self.drift_params.diffusion_trans_cm2_us

        K_wire = self.diffusion_params.K_wire
        K_time = self.diffusion_params.K_time

        all_plane_distances_cm = self.detector_config['all_plane_distances_cm']
        furthest_plane_indices = self.detector_config['furthest_plane_indices']
        angles_rad = self.detector_config['angles_rad']
        wire_spacings_cm = self.detector_config['wire_spacings_cm']

        plane_names = self.plane_names
        response_kernels = self.response_kernels
        track_threshold = self.track_config.threshold

        max_tracks = self.track_config.max_tracks
        max_wires = self._max_wires
        max_time = self._max_time
        max_keys = self.track_config.max_keys

        # Static tuples
        index_offsets_tuple = self._index_offsets_tuple
        max_indices_tuple = self._max_indices_tuple
        min_indices_tuple = self._min_indices_tuple
        num_wires_tuple = self._num_wires_tuple

        # Choose accumulation function based on mode
        if self.use_bucketed:
            max_buckets = self.max_active_buckets
            sparse_output = self.sparse_output

            def accumulate_fn(wire_idx, time_idx, intensities, contributions,
                              num_wires_plane, num_time_steps, kernel_num_wires, kernel_height):
                # Use sparse bucketing with two-phase algorithm
                buckets, num_active, compact_to_key = accumulate_response_signals_sparse_bucketed(
                    wire_idx, time_idx, intensities, contributions,
                    num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                    max_buckets
                )

                if sparse_output:
                    B1 = 2 * kernel_num_wires
                    B2 = 2 * kernel_height
                    return (buckets, num_active, compact_to_key, B1, B2)
                else:
                    B1 = 2 * kernel_num_wires
                    B2 = 2 * kernel_height
                    return sparse_buckets_to_dense(
                        buckets, compact_to_key, num_active,
                        B1, B2, num_wires_plane, num_time_steps, max_buckets
                    )

        else:
            def accumulate_fn(wire_idx, time_idx, intensities, contributions,
                              num_wires_plane, num_time_steps, kernel_num_wires, kernel_height):
                return accumulate_response_signals(
                    wire_idx, time_idx, intensities, contributions,
                    num_wires_plane, num_time_steps, kernel_num_wires, kernel_height
                )

        @partial(jax.jit, static_argnames=(
            'max_hits_east', 'max_hits_west',  # Per-side padding sizes
            'max_wire_indices_tuple', 'min_wire_indices_tuple',
            'index_offsets_tuple', 'num_wires_tuple',
            'max_tracks', 'max_wires', 'max_time', 'max_keys'
        ))
        def _calculate_signals_jit(
            # East side inputs (side 0, x < 0)
            east_positions_mm, east_charges, east_valid_mask,
            east_theta, east_phi, east_track_ids,
            # West side inputs (side 1, x >= 0)
            west_positions_mm, west_charges, west_valid_mask,
            west_theta, west_phi, west_track_ids,
            # Static args
            max_hits_east, max_hits_west,
            max_wire_indices_tuple, min_wire_indices_tuple,
            index_offsets_tuple, num_wires_tuple,
            max_tracks, max_wires, max_time, max_keys
        ):
            """JIT-compiled calculator with separate per-side data."""

            # Results lists
            response_signals_list = []
            track_hits_list = []

            for side_idx in range(2):
                # Select data for this side - no masking needed, already split!
                if side_idx == 0:  # East (x < 0)
                    positions_mm = east_positions_mm
                    charges = east_charges
                    valid_mask = east_valid_mask
                    theta = east_theta
                    phi = east_phi
                    track_ids = east_track_ids
                else:  # West (x >= 0)
                    positions_mm = west_positions_mm
                    charges = west_charges
                    valid_mask = west_valid_mask
                    theta = west_theta
                    phi = west_phi
                    track_ids = west_track_ids

                # Convert positions to cm
                positions_cm = positions_mm / 10.0

                # Calculate drift for the furthest plane on this side
                furthest_plane_idx = furthest_plane_indices[side_idx]
                furthest_plane_dist_cm = all_plane_distances_cm[side_idx, furthest_plane_idx]

                # Drift to furthest plane
                furthest_drift_distance_cm, furthest_drift_time_us, positions_yz_cm = compute_drift_to_plane(
                    positions_cm,
                    detector_half_width_cm,
                    velocity_cm_us,
                    furthest_plane_dist_cm
                )

                for plane_idx in range(3):
                    # Get plane parameters
                    plane_dist_cm = all_plane_distances_cm[side_idx, plane_idx]
                    angle_rad = angles_rad[side_idx, plane_idx]
                    spacing_cm = wire_spacings_cm[side_idx, plane_idx]
                    offset = index_offsets_tuple[side_idx][plane_idx]
                    max_idx_abs = max_wire_indices_tuple[side_idx][plane_idx]
                    min_idx_abs = min_wire_indices_tuple[side_idx][plane_idx]
                    num_wires_plane = num_wires_tuple[side_idx][plane_idx]
                    plane_type = plane_names[side_idx][plane_idx]

                    # --- Shared Preprocessing ---
                    # Correct drift for this plane
                    plane_dist_difference_cm = furthest_plane_dist_cm - plane_dist_cm
                    drift_distance_cm, drift_time_us = correct_drift_for_plane(
                        furthest_drift_distance_cm,
                        furthest_drift_time_us,
                        velocity_cm_us,
                        plane_dist_difference_cm
                    )

                    # Calculate attenuation
                    electron_lifetime_us = lifetime_us
                    drift_time_us_safe = jnp.where(jnp.isnan(drift_time_us), 0.0, drift_time_us)
                    attenuation_factors = jnp.exp(-drift_time_us_safe / electron_lifetime_us)

                    # Calculate closest wire distances
                    closest_wire_idx, closest_wire_distances = compute_wire_distances(
                        positions_yz_cm, angle_rad, spacing_cm,
                        max_idx_abs, offset
                    )

                    # Calculate angles
                    theta_xz, theta_y = compute_deposit_wire_angles_vmap(
                        theta, phi, angle_rad
                    )

                    # --- HIT PATH ---
                    # Angular scaling factor
                    angular_scaling_factor = compute_angular_scaling_vmap(theta_xz, theta_y)

                    # Process deposits with diffusion
                    prepare_deposit_vmap_hit = jax.vmap(
                        prepare_deposit_with_diffusion,
                        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None,
                                None, None, None, None, None, None),
                    )

                    hit_deposit_data = prepare_deposit_vmap_hit(
                        charges,  # No more side_charges - data already split
                        drift_time_us,
                        drift_distance_cm,
                        closest_wire_idx,
                        closest_wire_distances,
                        attenuation_factors,
                        theta_xz,
                        theta_y,
                        angular_scaling_factor,
                        valid_mask,  # No more valid_side_mask - data already split
                        K_wire,
                        K_time,
                        spacing_cm,
                        time_step_size_us,
                        diffusion_long_cm2_us,
                        diffusion_trans_cm2_us,
                        velocity_cm_us,
                        min_idx_abs,
                        num_wires_plane,
                        num_time_steps
                    )

                    # Track labeling
                    wire_indices_rel, time_indices, signal_values = hit_deposit_data

                    # Convert relative wire indices back to absolute
                    wire_indices_abs = wire_indices_rel + min_idx_abs

                    # Expand track IDs to match the deposit data structure
                    K_total = (2 * K_wire + 1) * (2 * K_time + 1)
                    track_ids_expanded = jnp.repeat(track_ids[:, jnp.newaxis], K_total, axis=1)

                    # Flatten arrays for track labeling
                    wire_indices_flat = wire_indices_abs.flatten()
                    time_indices_flat = time_indices.flatten()
                    track_ids_flat = track_ids_expanded.flatten()
                    charges_flat = signal_values.flatten()

                    # Create wire_time_indices array
                    wire_time_indices = jnp.stack([
                        wire_indices_flat,
                        time_indices_flat
                    ], axis=1)

                    # Group hits by track
                    (hits_by_track, num_hits,
                     track_boundaries, num_tracks,
                     track_ids_arr) = group_hits_by_track(
                        wire_time_indices,
                        track_ids_flat,
                        charges_flat,
                        min_charge_threshold=track_threshold,
                        max_tracks=max_tracks,
                        max_wires=max_wires,
                        max_time=max_time,
                        max_keys=max_keys
                    )

                    # Label hits (dominant track per wire,time)
                    num_stored = jnp.minimum(num_hits, max_keys)
                    labeled_hits, num_labeled = label_hits(
                        hits_by_track, num_stored,
                        track_ids_arr,
                        track_boundaries, num_tracks,
                        max_keys=max_keys, max_time=max_time
                    )

                    track_hit_result = {
                        'labeled_hits': labeled_hits,
                        'num_labeled': num_labeled,
                        'hits_by_track': hits_by_track,
                        'track_boundaries': track_boundaries,
                        'num_hits': num_hits,
                        'num_tracks': num_tracks,
                        'track_ids': track_ids_arr,
                    }

                    # --- RESPONSE PATH ---
                    # Calculate s parameter for diffusion
                    total_travel_distance_cm = detector_half_width_cm
                    s_values = drift_distance_cm / total_travel_distance_cm
                    s_values = jnp.clip(s_values, 0.0, 1.0)

                    # Process deposits for response path
                    prepare_deposit_vmap_response = jax.vmap(
                        prepare_deposit_for_response,
                        in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None),
                    )

                    deposit_data_kernel = prepare_deposit_vmap_response(
                        charges,  # No more side_charges
                        drift_time_us,
                        closest_wire_idx,
                        closest_wire_distances,
                        attenuation_factors,
                        valid_mask,  # No more valid_side_mask
                        spacing_cm,
                        time_step_size_us,
                        min_idx_abs,
                        num_wires_plane
                    )

                    wire_indices_rel_kernel, wire_offsets, time_indices_kernel, time_offsets, intensities = deposit_data_kernel

                    # Apply diffusion response kernels
                    plane_kernel = response_kernels[plane_type]
                    DKernel = plane_kernel['DKernel']
                    kernel_num_wires = plane_kernel['num_wires']
                    kernel_height = plane_kernel['kernel_height']
                    kernel_wire_stride = plane_kernel['wire_stride']
                    kernel_wire_spacing = plane_kernel['wire_spacing']
                    kernel_time_spacing = plane_kernel['time_spacing']

                    # Get kernel contributions
                    kernel_contributions = apply_diffusion_response(
                        DKernel, s_values, wire_offsets, time_offsets,
                        kernel_wire_stride, kernel_wire_spacing, kernel_time_spacing, kernel_num_wires
                    )

                    # Accumulate response signals
                    response_signal = accumulate_fn(
                        wire_indices_rel_kernel, time_indices_kernel, intensities, kernel_contributions,
                        num_wires_plane, num_time_steps, kernel_num_wires, kernel_height
                    )

                    # Store results
                    response_signals_list.append(response_signal)
                    track_hits_list.append(track_hit_result)

            return tuple(response_signals_list), tuple(track_hits_list)

        # Store raw JIT function (static args bound at call time based on tier)
        self._calculator_jit_raw = _calculate_signals_jit

    def __call__(self, deposit_data: DepositData):
        """Process deposits through both paths."""
        return self.process_event(deposit_data)

    def process_event(self, deposit_data: DepositData):
        """
        Process a single event through both simulation paths.

        Handles splitting by side and padding internally.

        Parameters
        ----------
        deposit_data : DepositData
            Input data containing positions, charges, and track IDs.
            Can be any size - will be split and padded to tiers internally.

        Returns
        -------
        response_signals : dict[(side, plane)] -> jnp.ndarray or sparse tuple
            Full detector response (W × T) for each plane.
        track_hits : dict[(side, plane)] -> dict
            Track labeling information for each plane.
        """
        # Split and pad data by side
        print("   Splitting and padding data by side...")
        east_data, west_data, counts = self._split_and_pad_data(deposit_data)

        # Pre-simulation validation
        self._validate_pre_simulation(counts)

        # Get tier sizes (become static args)
        max_hits_east = east_data.positions_mm.shape[0]
        max_hits_west = west_data.positions_mm.shape[0]

        # Call JIT-compiled calculator
        response_tuple, track_hits_tuple = self._calculator_jit_raw(
            # East data
            east_data.positions_mm, east_data.charges, east_data.valid_mask,
            east_data.theta, east_data.phi, east_data.track_ids,
            # West data
            west_data.positions_mm, west_data.charges, west_data.valid_mask,
            west_data.theta, west_data.phi, west_data.track_ids,
            # Static args
            max_hits_east=max_hits_east,
            max_hits_west=max_hits_west,
            max_wire_indices_tuple=self._max_indices_tuple,
            min_wire_indices_tuple=self._min_indices_tuple,
            index_offsets_tuple=self._index_offsets_tuple,
            num_wires_tuple=self._num_wires_tuple,
            max_tracks=self.track_config.max_tracks,
            max_wires=self._max_wires,
            max_time=self._max_time,
            max_keys=self.track_config.max_keys
        )

        # Convert tuples to dictionaries
        response_signals = {}
        track_hits = {}

        idx = 0
        for side_idx in range(2):
            for plane_idx in range(3):
                response_signals[(side_idx, plane_idx)] = response_tuple[idx]
                track_hits[(side_idx, plane_idx)] = track_hits_tuple[idx]
                idx += 1

        # Post-process track hits with proper boundary extraction
        for plane_key, track_result in track_hits.items():
            num_hits = track_result['num_hits']
            num_labeled = track_result['num_labeled']
            num_tracks = track_result['num_tracks']

            # Extract valid portions
            track_result['hits_by_track'] = track_result['hits_by_track'][:num_hits]
            track_result['labeled_hits'] = track_result['labeled_hits'][:num_labeled]
            track_result['track_boundaries'] = track_result['track_boundaries'][:num_tracks]

        # Post-simulation validation
        self._validate_post_simulation(track_hits, response_signals)

        return response_signals, track_hits

    def warm_up(self):
        """Trigger JIT compilation with dummy data for smallest tier."""
        print("Triggering JIT compilation...")

        # Warm up with smallest tier (fast)
        min_tier = self.padding_tiers[0]

        dummy_east = DepositData(
            positions_mm=jnp.zeros((min_tier, 3), dtype=jnp.float32),
            charges=jnp.zeros((min_tier,), dtype=jnp.float32),
            valid_mask=jnp.zeros((min_tier,), dtype=bool),
            theta=jnp.zeros((min_tier,), dtype=jnp.float32),
            phi=jnp.zeros((min_tier,), dtype=jnp.float32),
            track_ids=jnp.zeros((min_tier,), dtype=jnp.int32)
        )
        dummy_west = DepositData(
            positions_mm=jnp.zeros((min_tier, 3), dtype=jnp.float32),
            charges=jnp.zeros((min_tier,), dtype=jnp.float32),
            valid_mask=jnp.zeros((min_tier,), dtype=bool),
            theta=jnp.zeros((min_tier,), dtype=jnp.float32),
            phi=jnp.zeros((min_tier,), dtype=jnp.float32),
            track_ids=jnp.zeros((min_tier,), dtype=jnp.int32)
        )

        # Call directly to avoid split/pad logic
        _ = self._calculator_jit_raw(
            dummy_east.positions_mm, dummy_east.charges, dummy_east.valid_mask,
            dummy_east.theta, dummy_east.phi, dummy_east.track_ids,
            dummy_west.positions_mm, dummy_west.charges, dummy_west.valid_mask,
            dummy_west.theta, dummy_west.phi, dummy_west.track_ids,
            max_hits_east=min_tier,
            max_hits_west=min_tier,
            max_wire_indices_tuple=self._max_indices_tuple,
            min_wire_indices_tuple=self._min_indices_tuple,
            index_offsets_tuple=self._index_offsets_tuple,
            num_wires_tuple=self._num_wires_tuple,
            max_tracks=self.track_config.max_tracks,
            max_wires=self._max_wires,
            max_time=self._max_time,
            max_keys=self.track_config.max_keys
        )
        print(f"JIT compilation finished (tier {min_tier:,}). Other tiers compile on first use.")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_simulation(config_path, data_path, event_idx=0,
                   num_s=16,
                   response_path="tools/responses/",
                   track_threshold=1.0,
                   padding_tiers=PADDING_TIERS):
    """
    Run the detector simulation for a specific event.

    Convenience function that creates a DetectorSimulator and processes one event.
    Uses automatic per-side padding.

    Parameters
    ----------
    config_path : str
        Path to detector configuration YAML file.
    data_path : str
        Path to particle step data HDF5 file.
    event_idx : int, optional
        Index of event to process, by default 0.
    num_s : int, optional
        Number of diffusion levels in kernel interpolation, by default 16.
    response_path : str, optional
        Path to wire response kernel data.
    track_threshold : float, optional
        Minimum charge threshold for track labeling, by default 1.0.
    padding_tiers : tuple of int, optional
        Available padding tier sizes.

    Returns
    -------
    response_signals : dict
        Wire signals with detector response, keyed by (side_idx, plane_idx).
    track_hits : dict
        Track labeling results, keyed by (side_idx, plane_idx).
    simulator : DetectorSimulator
        The simulator instance (for reuse with additional events).
    """
    print("="*60)
    print(" LArTPC Detector Simulation (V2 - Per-Side Padding)")
    print("="*60)
    print(f"Config: {config_path}, Data: {data_path} (Event {event_idx})")

    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        detector_config = generate_detector(config_path)
        if detector_config is None:
            raise ValueError(f"Error loading detector configuration from {config_path}")
        print(f"Successfully loaded detector config from {config_path}")
    except Exception as e:
        print(f"ERROR loading config file '{config_path}': {e}")
        traceback.print_exc()
        raise

    # Create diffusion params
    diffusion_params = create_diffusion_params(
        max_sigma_trans_unitless=detector_config['max_sigma_trans_unitless'],
        max_sigma_long_unitless=detector_config['max_sigma_long_unitless'],
        num_s=num_s,
        n_sigma=3.0
    )

    # Create track config
    track_config = create_track_hits_config(threshold=track_threshold)

    # Create simulator
    try:
        simulator = DetectorSimulator(
            detector_config,
            response_path=response_path,
            track_config=track_config,
            diffusion_params=diffusion_params,
            padding_tiers=padding_tiers
        )
        simulator.warm_up()
    except Exception as e:
        print(f"\n--- Error during simulator creation: ---")
        traceback.print_exc()
        raise

    # Process event
    try:
        print(f"\n--- Processing Event {event_idx} ---")
        step_data = load_particle_step_data(data_path, event_idx)
        event_positions_mm = jnp.asarray(step_data.get('position', jnp.empty((0, 3))), dtype=jnp.float32)

        n_hits = event_positions_mm.shape[0]
        print(f"Loaded {n_hits:,} steps from event {event_idx}.")

        if n_hits == 0:
            print("WARNING: Event contains no particle steps.")
            return {}, {}, simulator

        # Get recombined charge
        recomb_charge = recombine_steps(step_data, detector_config)
        recomb_charge = jnp.asarray(recomb_charge, dtype=jnp.float32)

        # Extract angles and track IDs
        event_theta = jnp.asarray(step_data.get('theta', jnp.zeros((n_hits,))), dtype=jnp.float32)
        event_phi = jnp.asarray(step_data.get('phi', jnp.zeros((n_hits,))), dtype=jnp.float32)
        event_track_ids = jnp.asarray(step_data.get('track_id', jnp.ones((n_hits,))), dtype=jnp.int32)

        print(f"Unique tracks: {len(jnp.unique(event_track_ids)):,}")

        # Create deposit data (no manual padding needed!)
        deposit_data = DepositData(
            positions_mm=event_positions_mm,
            charges=recomb_charge,
            valid_mask=jnp.ones(n_hits, dtype=bool),
            theta=event_theta,
            phi=event_phi,
            track_ids=event_track_ids
        )

        # Run simulation (splitting and padding handled internally)
        print("Running simulation...")
        response_signals, track_hits = simulator(deposit_data)

        # Ensure calculations complete
        for key, arr in response_signals.items():
            if arr is not None:
                if isinstance(arr, tuple):
                    jax.block_until_ready(arr[0])
                else:
                    jax.block_until_ready(arr)

        print("\nSimulation complete.")
        print("Track labeling results:")
        for plane_key, results in track_hits.items():
            num_labeled = results['num_labeled']
            num_hits = results['num_hits']
            print(f"  Plane {plane_key}: {num_labeled} labeled hits, {num_hits} total hits")

        return response_signals, track_hits, simulator

    except Exception as e:
        print(f"\n--- Error during event processing: ---")
        traceback.print_exc()
        raise
