"""
Detector simulation module for LArTPC with per-side padding.

Provides the DetectorSimulator class with two output paths:

Response Path (always active):
    Generates detector response signals using pre-computed kernels
    convolved with diffusion. Output: response_signals

Hit Path (optional, controlled by include_track_hits):
    Calculates diffused charge without response convolution.
    Tracks which particle contributed to each location.
    Output: track_hits

Features:
    - Per-side padding with tier system (east/west split before JIT)
    - Automatic data splitting and padding inside process_event
    - Optional noise, electronics response, and track labeling
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
    prepare_deposit_with_diffusion,
)

# Response kernel system (for response path)
from tools.kernels import load_response_kernels, apply_diffusion_response

# Track labeling system (for hit path)
from tools.track_hits import group_hits_by_track, label_hits

# Core modules
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import create_recombination_fn

# Noise generation (for JIT-integrated noise)
from tools.noise import (
    _noise_core,
    _generate_noise_for_buckets,
    _get_noise_spectrum_shape,
    _NOISE_X, _NOISE_Y, _NOISE_Z
)


# =============================================================================
# PADDING TIER SYSTEM
# =============================================================================

# Default padding tiers - array sizes that trigger JIT recompilation.
# Using fixed tiers limits the number of JIT versions cached.
PADDING_TIERS = (100_000, 200_000)

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
    LArTPC detector simulation with per-side padding.

    Produces:
    - response_signals: Full detector simulation with response kernels (always)
    - track_hits: Track labeling information for analysis (optional)

    Features:
    - Automatic splitting of data by detector side (east x<0, west x>=0)
    - Per-side padding using tier system to minimize JIT recompilations
    - Optional noise, electronics response, and track labeling

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().
    response_path : str
        Path to response kernel files.
    track_config : TrackHitsConfig, optional
        Configuration for track labeling. If None, uses defaults.
        Ignored when include_track_hits=False.
    diffusion_params : DiffusionParams, optional
        Diffusion parameters. If None, computed from config.
    padding_tiers : tuple of int, optional
        Available padding tier sizes. Default: (100_000,).
    use_bucketed : bool
        If True, use sparse bucketed accumulation for very large detectors.
    max_active_buckets : int
        Max active buckets for sparse mode. Default 1000.
    include_noise : bool
        If True, add intrinsic noise inside JIT. Default False.
    include_electronics : bool
        If True, apply electronics response convolution. Default False.
    include_track_hits : bool
        If True, run the hit path for track labeling. Default True.
    recombination_model : str, optional
        Charge recombination model to use. Options:

        - ``'modified_box'`` : Modified Box model (ArgoNeuT 2013, arXiv:1306.1712).
          Angle-independent. Default params: α=0.93, β=0.212.
        - ``'emb'`` : Ellipsoid Modified Box model (ICARUS 2024, arXiv:2407.12969).
          Angle-dependent via track-to-drift-field angle φ.
          Default params: α=0.904, β_90=0.204, R=1.25.

        If None, reads from ``simulation.charge_recombination.model`` in the
        detector config, falling back to ``'modified_box'`` if not specified.
        See ``tools.recombination`` for full model documentation.
    """

    def __init__(
        self,
        detector_config,
        response_path="tools/responses/",
        track_config=None,
        diffusion_params=None,
        padding_tiers=PADDING_TIERS,
        use_bucketed=False,
        max_active_buckets=1000,
        include_noise=False,
        include_electronics=False,
        include_track_hits=True,
        electronics_chunk_size=None,
        electronics_threshold=0.0,
        recombination_model=None,
    ):
        print("--- Creating DetectorSimulator ---")

        # Store config
        self.detector_config = detector_config
        self.padding_tiers = padding_tiers
        self.use_bucketed = use_bucketed
        self.sparse_output = use_bucketed  # sparse_output is True when use_bucketed is True
        self.max_active_buckets = max_active_buckets

        # Store ADC conversion factor (response output is in electrons)
        self.electrons_per_adc = detector_config['electrons_per_adc']

        # Create recombination function (captured by JIT closure)
        self.recomb_fn, self.recomb_model = create_recombination_fn(
            detector_config, model=recombination_model
        )

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

        # Track labeling configuration
        self.include_track_hits = include_track_hits
        if include_track_hits:
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
                self._max_wires = 2000
                self._max_time = self.time_params.num_steps
        else:
            self.track_config = None
            self._max_wires = None
            self._max_time = None

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

        # Pre-compute noise parameters for JIT integration
        self.include_noise = include_noise
        if include_noise:
            num_time_steps = int(detector_config['num_time_steps'])
            wire_lengths_m = detector_config['wire_lengths_m']

            # Spectrum for dense mode (full time resolution)
            self._noise_spectrum_dense = jnp.array(_get_noise_spectrum_shape(num_time_steps))

            # Spectrum and bucket sizes for bucketed mode (B2 resolution per plane)
            self._noise_spectrum_bucketed = {}
            self._bucket_dims = {}  # Store (B1, B2) per plane type
            for plane_type in ['U', 'V', 'Y']:
                B1 = 2 * self.response_kernels[plane_type]['num_wires']
                B2 = 2 * self.response_kernels[plane_type]['kernel_height']
                self._bucket_dims[plane_type] = (B1, B2)
                self._noise_spectrum_bucketed[plane_type] = jnp.array(_get_noise_spectrum_shape(B2))

            # Wire lengths as JAX arrays (for indexing per bucket)
            self._wire_lengths_jax = {
                (s, p): jnp.array(wire_lengths_m[(s, p)], dtype=jnp.float32)
                for s in range(2) for p in range(3)
            }

            # Pre-computed series RMS for dense mode
            self._noise_series_rms = {
                (s, p): jnp.array(_NOISE_Y + _NOISE_Z * wire_lengths_m[(s, p)], dtype=jnp.float32)
                for s in range(2) for p in range(3)
            }

            self._noise_white_rms = _NOISE_X

        # Pre-compute electronics response parameters
        self.include_electronics = include_electronics
        self.electronics_threshold = electronics_threshold

        if include_electronics:
            from tools.electronics import load_electronics_response, compute_fft_size

            _TAU_US = 1000.0
            _N_TAU = 3.0

            self.electronics_kernels = load_electronics_response(
                time_step_us=float(self.time_params.step_size_us),
                tau_us=_TAU_US, n_tau=_N_TAU
            )

            R = len(self.electronics_kernels['U'])
            num_time = int(detector_config['num_time_steps'])
            self._electronics_fft_size = compute_fft_size(num_time, R)

            if electronics_chunk_size is None:
                self.electronics_chunk_size = int(np.max(detector_config['num_wires_actual']))
            else:
                self.electronics_chunk_size = electronics_chunk_size

            # Ensure _bucket_dims available for bucketed+electronics even without noise
            if use_bucketed and not hasattr(self, '_bucket_dims'):
                self._bucket_dims = {}
                for plane_type in ['U', 'V', 'Y']:
                    B1 = 2 * self.response_kernels[plane_type]['num_wires']
                    B2 = 2 * self.response_kernels[plane_type]['kernel_height']
                    self._bucket_dims[plane_type] = (B1, B2)

        # Output format flag
        if use_bucketed and include_electronics:
            self._output_format = 'wire_sparse'
        elif use_bucketed:
            self._output_format = 'bucketed'
        else:
            self._output_format = 'dense'

        # Extract static arrays for JIT
        self._prepare_static_args()

        # Build the JIT-compiled calculator
        self._build_calculator()

        print(f"   Recombination model: {self.recomb_model}")
        print(f"   Config: tiers={padding_tiers}, num_s={self.diffusion_params.num_s}, "
              f"K_wire={self.diffusion_params.K_wire}, K_time={self.diffusion_params.K_time}")
        if use_bucketed:
            print(f"   Using BUCKETED accumulation (B=2*kernel, max_buckets={max_active_buckets})")
        if include_noise:
            print(f"   Noise integration: ENABLED (added inside JIT)")
        if include_electronics:
            print(f"   Electronics response: ENABLED (FFT size={self._electronics_fft_size}, "
                  f"chunk={self.electronics_chunk_size}, output={self._output_format})")
        if include_track_hits:
            print(f"   Track labeling: ENABLED")
        else:
            print(f"   Track labeling: DISABLED")
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
        de = deposit_data.de
        dx = deposit_data.dx
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
            positions_mm, de, dx, theta, phi, track_ids,
            east_mask, n_east, east_tier
        )
        west_data = self._extract_and_pad(
            positions_mm, de, dx, theta, phi, track_ids,
            west_mask, n_west, west_tier
        )

        counts = {'n_east': n_east, 'n_west': n_west, 'n_tracks': n_tracks}
        return east_data, west_data, counts

    def _extract_and_pad(self, positions_mm, de, dx, theta, phi, track_ids,
                         mask, n_valid, pad_size):
        """Extract masked data and pad to target size."""
        # Extract valid entries
        pos = positions_mm[mask]
        de_arr = de[mask]
        dx_arr = dx[mask]
        th = theta[mask]
        ph = phi[mask]
        tid = track_ids[mask]

        # Pad to tier size
        pad_width = pad_size - n_valid

        if pad_width > 0:
            valid_out = jnp.arange(pad_size) < n_valid
            pos = jnp.pad(pos, ((0, pad_width), (0, 0)), constant_values=0.0)
            de_arr = jnp.pad(de_arr, (0, pad_width), constant_values=0.0)
            dx_arr = jnp.pad(dx_arr, (0, pad_width), constant_values=0.0)
            th = jnp.pad(th, (0, pad_width), constant_values=0.0)
            ph = jnp.pad(ph, (0, pad_width), constant_values=0.0)
            tid = jnp.pad(tid, (0, pad_width), constant_values=0)
        else:
            # Truncate if needed (shouldn't happen if tier picked correctly)
            valid_out = jnp.ones(pad_size, dtype=bool)
            pos = pos[:pad_size]
            de_arr = de_arr[:pad_size]
            dx_arr = dx_arr[:pad_size]
            th = th[:pad_size]
            ph = ph[:pad_size]
            tid = tid[:pad_size]

        return DepositData(
            positions_mm=pos,
            de=de_arr,
            dx=dx_arr,
            valid_mask=valid_out,
            theta=th,
            phi=ph,
            track_ids=tid
        )

    def _validate_pre_simulation(self, counts):
        """Check inputs before running simulation."""
        if self.include_track_hits and self.track_config.max_tracks < counts['n_tracks']:
            print(f"ERROR: max_tracks ({self.track_config.max_tracks:,}) < unique tracks ({counts['n_tracks']:,}). Tracks may be lost!")

    def _validate_post_simulation(self, track_hits, response_signals):
        """Check for truncation after simulation."""
        for plane_key, th in track_hits.items():
            actual_hits = int(th['num_hits'])
            if actual_hits >= self.track_config.max_keys:
                print(f"ERROR: Plane {plane_key}: num_hits ({actual_hits:,}) >= max_keys ({self.track_config.max_keys:,}). Data truncated!")

        if self._output_format == 'bucketed':
            for plane_key, resp in response_signals.items():
                buckets, num_active, compact_to_key, B1, B2 = resp
                if int(num_active) >= self.max_active_buckets:
                    print(f"ERROR: Plane {plane_key}: num_active ({int(num_active):,}) >= max_active_buckets ({self.max_active_buckets:,}). Data truncated!")

        elif self._output_format == 'wire_sparse':
            for plane_key, resp in response_signals.items():
                _, _, n_active_wires = resp
                if int(n_active_wires) >= self.electronics_chunk_size:
                    print(f"WARNING: Plane {plane_key}: active wires ({int(n_active_wires):,}) >= "
                          f"electronics_chunk_size ({self.electronics_chunk_size:,}).")

        elif self._output_format == 'dense' and self.include_electronics:
            for plane_key, resp in response_signals.items():
                n_active = int(jnp.sum(jnp.any(resp != 0, axis=1)))
                if n_active > self.electronics_chunk_size:
                    print(f"WARNING: Plane {plane_key}: active wires ({n_active:,}) > "
                          f"electronics_chunk_size ({self.electronics_chunk_size:,}).")

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
        electrons_per_adc = self.electrons_per_adc
        include_track_hits_flag = self.include_track_hits

        if include_track_hits_flag:
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

        # Recombination function (captured in closure)
        recomb_fn = self.recomb_fn

        # Choose accumulation function based on mode
        if self.use_bucketed:
            max_buckets = self.max_active_buckets
            sparse_output = self.sparse_output

            def accumulate_fn(wire_idx, time_idx, intensities, contributions,
                              num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                              wire_zero_bin, time_zero_bin):
                # Use sparse bucketing with two-phase algorithm
                buckets, num_active, compact_to_key = accumulate_response_signals_sparse_bucketed(
                    wire_idx, time_idx, intensities, contributions,
                    num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                    max_buckets, wire_zero_bin, time_zero_bin
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
                              num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                              wire_zero_bin, time_zero_bin):
                signal = accumulate_response_signals(
                    wire_idx, time_idx, intensities, contributions,
                    num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                    wire_zero_bin, time_zero_bin
                )
                return signal

        # Define electronics response function based on mode
        if self.include_electronics:
            from tools.electronics import (
                electronics_response_core, electronics_convolve_active,
                buckets_to_active_wires
            )
            e_kernels = {t: jnp.array(self.electronics_kernels[t]) for t in ['U', 'V', 'Y']}
            e_chunk = self.electronics_chunk_size
            e_fft = self._electronics_fft_size
            e_threshold = self.electronics_threshold

            if self.use_bucketed:
                max_buckets_e = self.max_active_buckets

                def make_elec_fn_bucketed(plane_type):
                    kernel = e_kernels[plane_type]
                    B1, B2 = self._bucket_dims[plane_type]

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

                elec_fns = {
                    (s, p): make_elec_fn_bucketed(plane_names[s][p])
                    for s in range(2) for p in range(3)
                }

                def electronics_fn(sig, side_idx, plane_idx, nw, nt):
                    return elec_fns[(side_idx, plane_idx)](sig, nw, nt)
            else:
                def make_elec_fn_dense(plane_type):
                    kernel = e_kernels[plane_type]

                    def fn(signal, nw, nt):
                        return electronics_response_core(signal, kernel, e_threshold, e_chunk, e_fft, nt)
                    return fn

                elec_fns = {
                    (s, p): make_elec_fn_dense(plane_names[s][p])
                    for s in range(2) for p in range(3)
                }

                def electronics_fn(sig, side_idx, plane_idx, nw, nt):
                    return elec_fns[(side_idx, plane_idx)](sig, nw, nt)
        else:
            def electronics_fn(sig, side_idx, plane_idx, nw, nt):
                return sig

        # Define noise function based on mode
        if self.include_noise:
            if self.use_bucketed and self.include_electronics:
                # Wire-sparse noise: reuse _noise_core on active wire rows
                spectrum_dense = self._noise_spectrum_dense
                wire_lengths_jax = self._wire_lengths_jax
                noise_white_rms = self._noise_white_rms
                e_chunk_noise = self.electronics_chunk_size

                def make_noise_fn_wire_sparse(side_idx, plane_idx):
                    lengths = wire_lengths_jax[(side_idx, plane_idx)]

                    def fn(key, signal_tuple, si, pi, nw, nt):
                        active_signals, wire_indices, n_active = signal_tuple
                        active_series_rms = _NOISE_Y + _NOISE_Z * lengths[wire_indices]
                        noise = _noise_core(key, e_chunk_noise, nt, spectrum_dense,
                                            active_series_rms, noise_white_rms)
                        valid = jnp.arange(e_chunk_noise) < n_active
                        return (active_signals + noise * valid[:, None], wire_indices, n_active)
                    return fn

                noise_fns_ws = {
                    (s, p): make_noise_fn_wire_sparse(s, p)
                    for s in range(2) for p in range(3)
                }

                def noise_fn(key, sig, side_idx, plane_idx, num_wires_plane, num_time_steps_plane):
                    return noise_fns_ws[(side_idx, plane_idx)](
                        key, sig, side_idx, plane_idx, num_wires_plane, num_time_steps_plane
                    )

            elif self.use_bucketed:
                wire_lengths_jax = self._wire_lengths_jax
                spectrum_bucketed = self._noise_spectrum_bucketed
                bucket_dims = self._bucket_dims
                max_buckets = self.max_active_buckets
                noise_white_rms = self._noise_white_rms

                # Create separate noise functions for each plane type to preserve static B1/B2
                def make_noise_fn_bucketed(plane_type):
                    B1, B2 = bucket_dims[plane_type]
                    spectrum = spectrum_bucketed[plane_type]

                    def noise_fn_plane(key, signal_tuple, side_idx, plane_idx, num_wires_plane, num_time_steps_plane):
                        buckets, num_active, compact_to_key, _, _ = signal_tuple

                        # Map buckets to wire ranges
                        NUM_BUCKETS_T = (num_time_steps_plane + B2 - 1) // B2
                        wire_starts = (compact_to_key // NUM_BUCKETS_T) * B1
                        wire_indices = wire_starts[:, None] + jnp.arange(B1)
                        wire_indices = jnp.clip(wire_indices, 0, num_wires_plane - 1)

                        # Get series RMS using actual wire lengths
                        lengths = wire_lengths_jax[(side_idx, plane_idx)]
                        bucket_series_rms = _NOISE_Y + _NOISE_Z * lengths[wire_indices]

                        # Generate noise for all buckets
                        noise = _generate_noise_for_buckets(
                            key, max_buckets, B1, B2, spectrum, bucket_series_rms, noise_white_rms
                        )

                        # Only add noise to active buckets
                        active_mask = jnp.arange(max_buckets) < num_active
                        noise = noise * active_mask[:, None, None]

                        return (buckets + noise, num_active, compact_to_key, B1, B2)
                    return noise_fn_plane

                # Create dict of noise functions per plane type
                noise_fns_bucketed = {
                    (s, p): make_noise_fn_bucketed(plane_names[s][p])
                    for s in range(2) for p in range(3)
                }

                def noise_fn(key, signal_tuple, side_idx, plane_idx, num_wires_plane, num_time_steps_plane):
                    return noise_fns_bucketed[(side_idx, plane_idx)](
                        key, signal_tuple, side_idx, plane_idx, num_wires_plane, num_time_steps_plane
                    )

            else:  # Dense mode
                spectrum_dense = self._noise_spectrum_dense
                noise_series_rms = self._noise_series_rms
                noise_white_rms = self._noise_white_rms

                def noise_fn(key, signal, side_idx, plane_idx, num_wires_plane, num_time_steps_plane):
                    series_rms = noise_series_rms[(side_idx, plane_idx)]
                    noise = _noise_core(key, num_wires_plane, num_time_steps_plane, spectrum_dense, series_rms, noise_white_rms)
                    return signal + noise

        else:  # No noise - identity functions
            if self.use_bucketed and self.include_electronics:
                def noise_fn(key, sig, side_idx, plane_idx, num_wires_plane, num_time_steps_plane):
                    return sig  # wire-sparse tuple passthrough
            elif self.use_bucketed:
                def noise_fn(key, signal_tuple, side_idx, plane_idx, num_wires_plane, num_time_steps_plane):
                    return signal_tuple
            else:
                def noise_fn(key, signal, side_idx, plane_idx, num_wires_plane, num_time_steps_plane):
                    return signal

        # Define track hits function based on mode
        if include_track_hits_flag:
            def track_hits_fn(track_hits_list,
                              charges, drift_time_us, drift_distance_cm,
                              closest_wire_idx, closest_wire_distances,
                              attenuation_factors, valid_mask, track_ids,
                              theta, phi, angle_rad,
                              spacing_cm, min_idx_abs, num_wires_plane):
                theta_xz, theta_y = compute_deposit_wire_angles_vmap(
                    theta, phi, angle_rad
                )
                angular_scaling_factor = compute_angular_scaling_vmap(theta_xz, theta_y)

                prepare_deposit_vmap_hit = jax.vmap(
                    prepare_deposit_with_diffusion,
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None,
                            None, None, None, None, None, None),
                )

                hit_deposit_data = prepare_deposit_vmap_hit(
                    charges, drift_time_us, drift_distance_cm,
                    closest_wire_idx, closest_wire_distances,
                    attenuation_factors, theta_xz, theta_y,
                    angular_scaling_factor, valid_mask,
                    K_wire, K_time, spacing_cm, time_step_size_us,
                    diffusion_long_cm2_us, diffusion_trans_cm2_us,
                    velocity_cm_us, min_idx_abs, num_wires_plane,
                    num_time_steps
                )

                wire_indices_rel, time_indices, signal_values = hit_deposit_data
                wire_indices_abs = wire_indices_rel + min_idx_abs

                K_total = (2 * K_wire + 1) * (2 * K_time + 1)
                track_ids_expanded = jnp.repeat(track_ids[:, jnp.newaxis], K_total, axis=1)

                wire_indices_flat = wire_indices_abs.flatten()
                time_indices_flat = time_indices.flatten()
                track_ids_flat = track_ids_expanded.flatten()
                charges_flat = signal_values.flatten()

                wire_time_indices = jnp.stack([
                    wire_indices_flat,
                    time_indices_flat
                ], axis=1)

                (hits_by_track, num_hits,
                 track_boundaries, num_tracks,
                 track_ids_arr) = group_hits_by_track(
                    wire_time_indices, track_ids_flat, charges_flat,
                    min_charge_threshold=track_threshold,
                    max_tracks=max_tracks, max_wires=max_wires,
                    max_time=max_time, max_keys=max_keys
                )

                num_stored = jnp.minimum(num_hits, max_keys)
                labeled_hits, num_labeled = label_hits(
                    hits_by_track, num_stored, track_ids_arr,
                    track_boundaries, num_tracks,
                    max_keys=max_keys, max_time=max_time
                )

                track_hits_list.append({
                    'labeled_hits': labeled_hits,
                    'num_labeled': num_labeled,
                    'hits_by_track': hits_by_track,
                    'track_boundaries': track_boundaries,
                    'num_hits': num_hits,
                    'num_tracks': num_tracks,
                    'track_ids': track_ids_arr,
                })
        else:
            def track_hits_fn(track_hits_list,
                              charges, drift_time_us, drift_distance_cm,
                              closest_wire_idx, closest_wire_distances,
                              attenuation_factors, valid_mask, track_ids,
                              theta, phi, angle_rad,
                              spacing_cm, min_idx_abs, num_wires_plane):
                pass

        @partial(jax.jit, static_argnames=(
            'max_hits_east', 'max_hits_west',  # Per-side padding sizes
            'max_wire_indices_tuple', 'min_wire_indices_tuple',
            'index_offsets_tuple', 'num_wires_tuple',
            'max_tracks', 'max_wires', 'max_time', 'max_keys'
        ))
        def _calculate_signals_jit(
            # East side inputs (side 0, x < 0)
            east_positions_mm, east_de, east_dx, east_valid_mask,
            east_theta, east_phi, east_track_ids,
            # West side inputs (side 1, x >= 0)
            west_positions_mm, west_de, west_dx, west_valid_mask,
            west_theta, west_phi, west_track_ids,
            # Noise key
            noise_key,
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
                    de = east_de
                    dx = east_dx
                    valid_mask = east_valid_mask
                    theta = east_theta
                    phi = east_phi
                    track_ids = east_track_ids
                else:  # West (x >= 0)
                    positions_mm = west_positions_mm
                    de = west_de
                    dx = west_dx
                    valid_mask = west_valid_mask
                    theta = west_theta
                    phi = west_phi
                    track_ids = west_track_ids

                # Convert dx from mm to cm (HDF5 data stores lengths in mm)
                dx_cm = dx / 10.0

                # Compute angle between track direction and drift field (x-axis)
                # for angular-dependent recombination models (EMB)
                dx_dir = jnp.sin(theta) * jnp.cos(phi)
                phi_drift = jnp.arccos(jnp.clip(jnp.abs(dx_dir), 0.0, 1.0))

                # Apply charge recombination
                charges = recomb_fn(de, dx_cm, phi_drift)

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

                    # Track labeling (no-op when disabled)
                    track_hits_fn(
                        track_hits_list,
                        charges, drift_time_us, drift_distance_cm,
                        closest_wire_idx, closest_wire_distances,
                        attenuation_factors, valid_mask, track_ids,
                        theta, phi, angle_rad,
                        spacing_cm, min_idx_abs, num_wires_plane
                    )

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
                    kernel_height = plane_kernel['kernel_height']      # kernel_height - 1 due to interpolation
                    kernel_wire_stride = plane_kernel['wire_stride']
                    kernel_wire_spacing = plane_kernel['wire_spacing']
                    wire_zero_bin = plane_kernel['wire_zero_bin']      # Where wire=0 is in output wires
                    time_zero_bin = plane_kernel['time_zero_bin']      # Where t=0 is in output time bins

                    # Get kernel contributions using linear interpolation
                    kernel_contributions = apply_diffusion_response(
                        DKernel, s_values, wire_offsets, time_offsets,
                        kernel_wire_stride, kernel_wire_spacing, kernel_num_wires
                    )

                    # Accumulate response signals (output is in ADC)
                    response_signal = accumulate_fn(
                        wire_indices_rel_kernel, time_indices_kernel, intensities, kernel_contributions,
                        num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                        wire_zero_bin, time_zero_bin
                    )

                    # Apply electronics response convolution
                    response_signal = electronics_fn(
                        response_signal, side_idx, plane_idx,
                        num_wires_plane, num_time_steps
                    )

                    # Apply noise (uses pre-split key for this plane)
                    plane_keys = jax.random.split(noise_key, 6)
                    plane_key = plane_keys[side_idx * 3 + plane_idx]
                    response_signal = noise_fn(plane_key, response_signal, side_idx, plane_idx, num_wires_plane, num_time_steps)

                    response_signals_list.append(response_signal)

            return tuple(response_signals_list), tuple(track_hits_list)

        # Store raw JIT function (static args bound at call time based on tier)
        self._calculator_jit_raw = _calculate_signals_jit

    def __call__(self, deposit_data: DepositData, key=None):
        """Process deposits through both paths."""
        return self.process_event(deposit_data, key=key)

    def process_event(self, deposit_data: DepositData, key=None):
        """
        Process a single event through both simulation paths.

        Handles splitting by side and padding internally.

        Parameters
        ----------
        deposit_data : DepositData
            Input data containing positions, charges, and track IDs.
            Can be any size - will be split and padded to tiers internally.
        key : jax.Array, optional
            JAX PRNGKey for noise generation. Default is PRNGKey(0).

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

        # Noise key (used even when noise is disabled - identity function ignores it)
        noise_key = key if key is not None else jax.random.PRNGKey(0)

        # Call JIT-compiled calculator
        response_tuple, track_hits_tuple = self._calculator_jit_raw(
            # East data
            east_data.positions_mm, east_data.de, east_data.dx, east_data.valid_mask,
            east_data.theta, east_data.phi, east_data.track_ids,
            # West data
            west_data.positions_mm, west_data.de, west_data.dx, west_data.valid_mask,
            west_data.theta, west_data.phi, west_data.track_ids,
            # Noise key
            noise_key,
            # Static args
            max_hits_east=max_hits_east,
            max_hits_west=max_hits_west,
            max_wire_indices_tuple=self._max_indices_tuple,
            min_wire_indices_tuple=self._min_indices_tuple,
            index_offsets_tuple=self._index_offsets_tuple,
            num_wires_tuple=self._num_wires_tuple,
            max_tracks=self.track_config.max_tracks if self.include_track_hits else 1,
            max_wires=self._max_wires if self.include_track_hits else 1,
            max_time=self._max_time if self.include_track_hits else 1,
            max_keys=self.track_config.max_keys if self.include_track_hits else 1
        )

        # Convert tuples to dictionaries
        response_signals = {}
        track_hits = {}

        idx = 0
        for side_idx in range(2):
            for plane_idx in range(3):
                response_signals[(side_idx, plane_idx)] = response_tuple[idx]
                if self.include_track_hits:
                    track_hits[(side_idx, plane_idx)] = track_hits_tuple[idx]
                idx += 1

        # Return raw results — call finalize_track_hits() after freeing
        # response_signals from GPU to avoid device memory pressure.
        return response_signals, track_hits

    def finalize_track_hits(self, track_hits):
        """
        Post-process track hits: extract valid portions and validate.

        Call this after moving response_signals off GPU (e.g. via np.asarray)
        to avoid device memory pressure from the int() sync calls.

        Parameters
        ----------
        track_hits : dict
            Raw track hits from process_event().

        Returns
        -------
        track_hits : dict
            Track hits with arrays sliced to valid lengths.
        """
        for plane_key, track_result in track_hits.items():
            num_hits = track_result['num_hits']
            num_labeled = track_result['num_labeled']
            num_tracks = track_result['num_tracks']

            track_result['hits_by_track'] = track_result['hits_by_track'][:num_hits]
            track_result['labeled_hits'] = track_result['labeled_hits'][:num_labeled]
            track_result['track_boundaries'] = track_result['track_boundaries'][:num_tracks]
            track_result['track_ids'] = track_result['track_ids'][:num_tracks]

        # Validate after slicing (int() calls already happened above)
        if self.track_config is not None:
            for plane_key, th in track_hits.items():
                actual_hits = int(th['num_hits'])
                if actual_hits >= self.track_config.max_keys:
                    print(f"ERROR: Plane {plane_key}: num_hits ({actual_hits:,}) >= max_keys ({self.track_config.max_keys:,}). Data truncated!")

        return track_hits

    def warm_up(self):
        """Trigger JIT compilation with dummy data for smallest tier."""
        print("Triggering JIT compilation...")

        # Warm up with smallest tier (fast)
        min_tier = self.padding_tiers[0]

        dummy_east = DepositData(
            positions_mm=jnp.zeros((min_tier, 3), dtype=jnp.float32),
            de=jnp.zeros((min_tier,), dtype=jnp.float32),
            dx=jnp.zeros((min_tier,), dtype=jnp.float32),
            valid_mask=jnp.zeros((min_tier,), dtype=bool),
            theta=jnp.zeros((min_tier,), dtype=jnp.float32),
            phi=jnp.zeros((min_tier,), dtype=jnp.float32),
            track_ids=jnp.zeros((min_tier,), dtype=jnp.int32)
        )
        dummy_west = DepositData(
            positions_mm=jnp.zeros((min_tier, 3), dtype=jnp.float32),
            de=jnp.zeros((min_tier,), dtype=jnp.float32),
            dx=jnp.zeros((min_tier,), dtype=jnp.float32),
            valid_mask=jnp.zeros((min_tier,), dtype=bool),
            theta=jnp.zeros((min_tier,), dtype=jnp.float32),
            phi=jnp.zeros((min_tier,), dtype=jnp.float32),
            track_ids=jnp.zeros((min_tier,), dtype=jnp.int32)
        )

        # Call directly to avoid split/pad logic
        _ = self._calculator_jit_raw(
            dummy_east.positions_mm, dummy_east.de, dummy_east.dx, dummy_east.valid_mask,
            dummy_east.theta, dummy_east.phi, dummy_east.track_ids,
            dummy_west.positions_mm, dummy_west.de, dummy_west.dx, dummy_west.valid_mask,
            dummy_west.theta, dummy_west.phi, dummy_west.track_ids,
            jax.random.PRNGKey(0),  # noise_key (dummy for warmup)
            max_hits_east=min_tier,
            max_hits_west=min_tier,
            max_wire_indices_tuple=self._max_indices_tuple,
            min_wire_indices_tuple=self._min_indices_tuple,
            index_offsets_tuple=self._index_offsets_tuple,
            num_wires_tuple=self._num_wires_tuple,
            max_tracks=self.track_config.max_tracks if self.include_track_hits else 1,
            max_wires=self._max_wires if self.include_track_hits else 1,
            max_time=self._max_time if self.include_track_hits else 1,
            max_keys=self.track_config.max_keys if self.include_track_hits else 1
        )
        print(f"JIT compilation finished (tier {min_tier:,}). Other tiers compile on first use.")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_simulation(config_path, data_path, event_idx=0,
                   num_s=16,
                   response_path="tools/responses/",
                   track_threshold=1.0,
                   padding_tiers=PADDING_TIERS,
                   include_track_hits=True):
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
    include_track_hits : bool, optional
        If True, run the hit path for track labeling. Default True.

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
    print(" LArTPC Detector Simulation")
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
    track_config = create_track_hits_config(threshold=track_threshold) if include_track_hits else None

    # Create simulator
    try:
        simulator = DetectorSimulator(
            detector_config,
            response_path=response_path,
            track_config=track_config,
            diffusion_params=diffusion_params,
            padding_tiers=padding_tiers,
            include_track_hits=include_track_hits
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

        # Extract de and dx (recombination is now done inside the simulator)
        event_de = jnp.asarray(step_data.get('de', jnp.zeros((n_hits,))), dtype=jnp.float32)
        event_dx = jnp.asarray(step_data.get('dx', jnp.zeros((n_hits,))), dtype=jnp.float32)

        # Extract angles and track IDs
        event_theta = jnp.asarray(step_data.get('theta', jnp.zeros((n_hits,))), dtype=jnp.float32)
        event_phi = jnp.asarray(step_data.get('phi', jnp.zeros((n_hits,))), dtype=jnp.float32)
        event_track_ids = jnp.asarray(step_data.get('track_id', jnp.ones((n_hits,))), dtype=jnp.int32)

        print(f"Unique tracks: {len(jnp.unique(event_track_ids)):,}")

        # Create deposit data (no manual padding needed!)
        # Recombination (de/dx -> electrons) is now done inside the simulator
        deposit_data = DepositData(
            positions_mm=event_positions_mm,
            de=event_de,
            dx=event_dx,
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
        if track_hits:
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
