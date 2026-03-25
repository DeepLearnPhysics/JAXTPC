"""
Detector simulation module for LArTPC with fixed-size padding.

Provides the DetectorSimulator class with two output paths:

Response Path (always active):
    Generates detector response signals using pre-computed kernels
    convolved with diffusion. Output: response_signals

Hit Path (optional, controlled by include_track_hits):
    Calculates diffused charge without response convolution.
    Tracks which particle contributed to each location.
    Output: track_hits

Features:
    - Fixed total_pad per side (single JIT compilation, no recompilation)
    - Batched response path via fori_loop (bounded peak memory)
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
    DiffusionParams, TrackHitsConfig, SegmentData, DigitizationConfig,
    create_diffusion_params, create_drift_params, create_time_params,
    create_plane_geometry, create_track_hits_config, create_digitization_config
)

# Core physics modules
from tools.drift import (
    compute_drift_to_plane,
    correct_drift_for_plane,
    compute_lifetime_attenuation,
    apply_drift_corrections,
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
    build_bucket_mapping,
    scatter_contributions_to_buckets_batched,
    sparse_buckets_to_dense,

    # Hit path functions (with K_wire x K_time diffusion)
    prepare_deposit_with_diffusion,
)

# Response kernel system (for response path)
from tools.kernels import load_response_kernels, apply_diffusion_response

# Track labeling system (for hit path)
from tools.track_hits import group_hits_by_track, label_hits, merge_chunk_hits, label_merged_hits, label_from_groups

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
# GROUP ID ASSIGNMENT
# =============================================================================

def compute_group_ids(positions_mm, track_ids, valid_mask,
                      group_size=5, gap_threshold_mm=5.0):
    """
    Assign group IDs for segment correspondence: N consecutive deposits per track.

    Groups are split on large spatial gaps (neutrons/gammas) to avoid
    grouping physically distant deposits. Uses vectorized numpy with
    stable sort to preserve trajectory ordering within each track.

    Parameters
    ----------
    positions_mm : np.ndarray, shape (N, 3)
        Deposit positions in mm.
    track_ids : np.ndarray, shape (N,), int32
        Track ID per deposit.
    valid_mask : np.ndarray, shape (N,), bool
        True for real deposits.
    group_size : int
        Number of consecutive deposits per group.
    gap_threshold_mm : float
        Start a new group if consecutive deposits in the same track
        are farther than this (handles neutrons/gammas).

    Returns
    -------
    group_ids : np.ndarray, shape (N,), int32
        Group ID for each deposit. Invalid/padded deposits get 0.
    group_to_track : np.ndarray, shape (n_groups,), int32
        Track ID for each group. Index 0 = invalid group.
    n_groups : int
        Total number of groups (including the invalid group 0).
    """
    n = len(track_ids)
    group_ids = np.zeros(n, dtype=np.int32)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return group_ids, np.array([0], dtype=np.int32), 1

    v_tids = track_ids[valid_idx]
    v_pos = positions_mm[valid_idx]

    # Stable sort by track_id preserves array (trajectory) order within each track
    sort_order = np.argsort(v_tids, kind='stable')
    sorted_idx = valid_idx[sort_order]
    sorted_tids = v_tids[sort_order]
    sorted_pos = v_pos[sort_order]
    nv = len(sorted_idx)

    # Track boundaries
    track_change = np.zeros(nv, dtype=bool)
    track_change[1:] = sorted_tids[1:] != sorted_tids[:-1]

    # Spatial gap boundaries (within same track)
    gaps = np.zeros(nv)
    gaps[1:] = np.linalg.norm(sorted_pos[1:] - sorted_pos[:-1], axis=1)
    gap_break = gaps > gap_threshold_mm

    # Contiguous segment starts: track change or spatial gap
    seg_start = track_change | gap_break
    seg_start[0] = True

    # Within-segment position via forward-filled segment start indices
    seg_start_positions = np.where(seg_start, np.arange(nv), 0)
    seg_start_positions = np.maximum.accumulate(seg_start_positions)
    within_seg = np.arange(nv) - seg_start_positions

    # Group boundaries: segment start or every N deposits within a segment
    group_start = seg_start.copy()
    group_start |= (within_seg % group_size == 0) & (within_seg > 0)

    # Consecutive group IDs (1-based; 0 reserved for invalid)
    group_labels = np.cumsum(group_start)

    # Write back to original deposit positions
    group_ids[sorted_idx] = group_labels

    # Build group_to_track lookup
    n_groups = int(group_labels.max()) + 1
    g2t = np.zeros(n_groups, dtype=np.int32)
    g2t[group_labels] = sorted_tids

    return group_ids, g2t, n_groups


# =============================================================================
# DETECTOR SIMULATOR CLASS
# =============================================================================

class DetectorSimulator:
    """
    LArTPC detector simulation with fixed-size padding.

    Produces:
    - response_signals: Full detector simulation with response kernels (always)
    - track_hits: Track labeling information for analysis (optional)

    Features:
    - Automatic splitting of data by detector side (east x<0, west x>=0)
    - Fixed total_pad per side — single JIT compilation for all events
    - Batched response path via fori_loop (bounded peak memory)
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
    total_pad : int
        Fixed pad size per side. Default 200_000.
    response_chunk_size : int
        Deposits per fori_loop batch in the response path. Default 50_000.
        Must divide total_pad evenly.
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
    include_electric_dist : bool
        If True, load space charge effect (SCE) maps from the default
        HDF5 file (``config/sce_jaxtpc.h5``) and apply E-field distortions
        and drift corrections during simulation.  Default False.
    electric_dist_path : str, optional
        Path to the per-side SCE HDF5 file.  Only used when
        ``include_electric_dist=True``.  Defaults to
        ``config/sce_jaxtpc.h5``.
    """

    def __init__(
        self,
        detector_config,
        response_path="tools/responses/",
        track_config=None,
        diffusion_params=None,
        total_pad=200_000,
        response_chunk_size=50_000,
        use_bucketed=False,
        max_active_buckets=1000,
        include_noise=False,
        include_electronics=False,
        include_track_hits=True,
        electronics_chunk_size=None,
        electronics_threshold=0.0,
        recombination_model=None,
        include_electric_dist=False,
        electric_dist_path=None,
        include_digitize=False,
        digitization_config=None,
        differentiable=False,
        n_segments=None,
        group_size=5,
        gap_threshold_mm=5.0,
    ):
        print("--- Creating DetectorSimulator ---")

        # Differentiable mode: force compatible flags
        self.differentiable = differentiable
        self.n_segments = n_segments
        self.group_size = group_size
        self.gap_threshold_mm = gap_threshold_mm
        if differentiable:
            if n_segments is None:
                raise ValueError("differentiable=True requires n_segments to be set")
            # Set total_pad and response_chunk_size to n_segments internally
            # (no padding, no fori_loop batching in differentiable mode)
            total_pad = n_segments
            response_chunk_size = n_segments
            use_bucketed = False
            include_noise = False
            include_electronics = False
            include_track_hits = False
            include_digitize = False

        # Store config
        self.detector_config = detector_config
        self.total_pad = total_pad
        self.response_chunk_size = response_chunk_size
        self.use_bucketed = use_bucketed
        self.sparse_output = use_bucketed  # sparse_output is True when use_bucketed is True
        self.max_active_buckets = max_active_buckets

        # Validate chunk alignment
        if total_pad % response_chunk_size != 0:
            raise ValueError(
                f"total_pad ({total_pad:,}) must be divisible by "
                f"response_chunk_size ({response_chunk_size:,})."
            )

        # ADC conversion factor (used by noise threshold; response kernels already output ADC)
        self.electrons_per_adc = detector_config['electrons_per_adc']

        # Create recombination function (captured by JIT closure)
        self.recomb_fn, self.recomb_model = create_recombination_fn(
            detector_config, model=recombination_model
        )

        # Space charge effect (electric field distortions)
        self.include_electric_dist = include_electric_dist
        if include_electric_dist:
            from tools.efield_distortions import load_sce_interpolation_fns
            if electric_dist_path is None:
                electric_dist_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 'config', 'sce_jaxtpc.h5'
                )
            print(f"   Loading SCE maps from {electric_dist_path}...")
            sce_efield_fn, sce_drift_correction_fn = load_sce_interpolation_fns(
                electric_dist_path
            )
            self._sce_efield_fn = sce_efield_fn
            self._sce_drift_correction_fn = sce_drift_correction_fn
        else:
            self._sce_efield_fn = None
            self._sce_drift_correction_fn = None

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
            # Validate hits_chunk_size alignment with total_pad
            hits_chunk = (track_config.hits_chunk_size if track_config is not None
                          else create_track_hits_config().hits_chunk_size)
            if total_pad % hits_chunk != 0:
                raise ValueError(
                    f"total_pad ({total_pad:,}) must be divisible by "
                    f"hits_chunk_size ({hits_chunk:,})."
                )
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

        # Digitization config
        self.include_digitize = include_digitize
        if include_digitize:
            if digitization_config is not None:
                self.digitization_config = digitization_config
            else:
                dig = detector_config.get('digitization', {})
                self.digitization_config = create_digitization_config(
                    n_bits=int(dig.get('n_bits', 12)),
                    pedestal_collection=int(dig.get('pedestal_collection', 410)),
                    pedestal_induction=int(dig.get('pedestal_induction', 1843)),
                    gain_scale=float(dig.get('gain_scale', 1.0)),
                )
        else:
            self.digitization_config = None

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

        sce_parts = []
        if self._sce_efield_fn is not None:
            sce_parts.append("E-field distortions")
        if self._sce_drift_correction_fn is not None:
            sce_parts.append("drift corrections")
        if sce_parts:
            print(f"   Space charge effects: ENABLED ({', '.join(sce_parts)})")
        else:
            print(f"   Space charge effects: DISABLED")
        print(f"   Recombination model: {self.recomb_model}")
        print(f"   Config: total_pad={total_pad:,}, response_chunk={response_chunk_size:,}, "
              f"num_s={self.diffusion_params.num_s}, "
              f"K_wire={self.diffusion_params.K_wire}, K_time={self.diffusion_params.K_time}")
        if use_bucketed:
            print(f"   Using BUCKETED accumulation (B=2*kernel, max_buckets={max_active_buckets})")
        if include_noise:
            print(f"   Noise integration: ENABLED (added inside JIT)")
        if include_electronics:
            print(f"   Electronics response: ENABLED (FFT size={self._electronics_fft_size}, "
                  f"chunk={self.electronics_chunk_size}, output={self._output_format})")
        if include_digitize:
            dc = self.digitization_config
            print(f"   Digitization: ENABLED ({dc.n_bits}-bit, "
                  f"pedestal Y={dc.pedestal_collection} U/V={dc.pedestal_induction}, "
                  f"gain={dc.gain_scale})")
        if include_track_hits:
            print(f"   Track labeling: ENABLED (group_size={self.group_size}, "
                  f"gap_threshold={self.gap_threshold_mm}mm)")
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
        Split deposit data by side and pad to total_pad.

        All operations use numpy to avoid XLA recompilation on variable-length
        intermediates. The final padded arrays are converted to JAX at the end
        (fixed total_pad shape → single JIT compilation for all events).

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
        positions_mm = np.asarray(deposit_data.positions_mm)
        de = np.asarray(deposit_data.de)
        dx = np.asarray(deposit_data.dx)
        valid_mask = np.asarray(deposit_data.valid_mask)
        theta = np.asarray(deposit_data.theta)
        phi = np.asarray(deposit_data.phi)
        track_ids = np.asarray(deposit_data.track_ids)

        # Only consider valid entries
        x_mm = positions_mm[:, 0]
        east_mask = valid_mask & (x_mm < 0)
        west_mask = valid_mask & (x_mm >= 0)

        n_east = int(np.sum(east_mask))
        n_west = int(np.sum(west_mask))
        n_tracks = int(len(np.unique(track_ids[valid_mask])))

        if n_east > self.total_pad:
            print(f"ERROR: n_east ({n_east:,}) > total_pad ({self.total_pad:,}). Truncating!")
        if n_west > self.total_pad:
            print(f"ERROR: n_west ({n_west:,}) > total_pad ({self.total_pad:,}). Truncating!")

        print(f"   East side: {n_east:,} hits (pad {self.total_pad:,})")
        print(f"   West side: {n_west:,} hits (pad {self.total_pad:,})")

        # Compute group IDs for segment correspondence (before split)
        if self.include_track_hits:
            all_group_ids, group_to_track, n_groups = compute_group_ids(
                positions_mm, track_ids, valid_mask,
                group_size=self.group_size,
                gap_threshold_mm=self.gap_threshold_mm,
            )
        else:
            all_group_ids = np.zeros(len(track_ids), dtype=np.int32)
            group_to_track = np.array([0], dtype=np.int32)
            n_groups = 1

        # Extract and pad each side (numpy), convert to JAX at the end
        east_data = self._extract_and_pad(
            positions_mm, de, dx, theta, phi, track_ids,
            east_mask, min(n_east, self.total_pad), self.total_pad
        )
        west_data = self._extract_and_pad(
            positions_mm, de, dx, theta, phi, track_ids,
            west_mask, min(n_west, self.total_pad), self.total_pad
        )

        # Extract and pad group_ids per side
        east_gids = self._extract_and_pad_array(
            all_group_ids, east_mask, min(n_east, self.total_pad), self.total_pad, pad_val=0
        )
        west_gids = self._extract_and_pad_array(
            all_group_ids, west_mask, min(n_west, self.total_pad), self.total_pad, pad_val=0
        )

        counts = {'n_east': min(n_east, self.total_pad),
                  'n_west': min(n_west, self.total_pad),
                  'n_tracks': n_tracks}
        group_data = {
            'east_group_ids': jnp.asarray(east_gids),
            'west_group_ids': jnp.asarray(west_gids),
            'group_to_track': group_to_track,
            'n_groups': n_groups,
        }
        return east_data, west_data, counts, group_data

    def _extract_and_pad(self, positions_mm, de, dx, theta, phi, track_ids,
                         mask, n_valid, pad_size):
        """Extract masked data and pad to target size.

        All operations in numpy. Returns DepositData with JAX arrays
        at the fixed tier shape (no XLA recompilation).
        """
        # Extract valid entries (numpy — variable length, no compilation)
        pos = positions_mm[mask]
        de_arr = de[mask]
        dx_arr = dx[mask]
        th = theta[mask]
        ph = phi[mask]
        tid = track_ids[mask]

        # Pad to tier size (numpy)
        pad_width = pad_size - n_valid

        if pad_width > 0:
            valid_out = np.arange(pad_size) < n_valid
            pos = np.pad(pos, ((0, pad_width), (0, 0)), constant_values=0.0)
            de_arr = np.pad(de_arr, (0, pad_width), constant_values=0.0)
            dx_arr = np.pad(dx_arr, (0, pad_width), constant_values=0.0)
            th = np.pad(th, (0, pad_width), constant_values=0.0)
            ph = np.pad(ph, (0, pad_width), constant_values=0.0)
            tid = np.pad(tid, (0, pad_width), constant_values=0)
        else:
            # Truncate if needed (shouldn't happen if tier picked correctly)
            valid_out = np.ones(pad_size, dtype=bool)
            pos = pos[:pad_size]
            de_arr = de_arr[:pad_size]
            dx_arr = dx_arr[:pad_size]
            th = th[:pad_size]
            ph = ph[:pad_size]
            tid = tid[:pad_size]

        # Convert to JAX at fixed tier shape (no recompilation)
        return DepositData(
            positions_mm=jnp.asarray(pos),
            de=jnp.asarray(de_arr),
            dx=jnp.asarray(dx_arr),
            valid_mask=jnp.asarray(valid_out),
            theta=jnp.asarray(th),
            phi=jnp.asarray(ph),
            track_ids=jnp.asarray(tid),
        )

    @staticmethod
    def _extract_and_pad_array(arr, mask, n_valid, pad_size, pad_val=0):
        """Extract masked 1D array and pad to target size (numpy)."""
        extracted = arr[mask][:n_valid]
        pad_width = pad_size - n_valid
        if pad_width > 0:
            return np.pad(extracted, (0, pad_width), constant_values=pad_val)
        return extracted[:pad_size]

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
            track_inter_thresh = self.track_config.inter_thresh
            max_tracks = self.track_config.max_tracks
            max_wires = self._max_wires
            max_time = self._max_time
            max_keys = self.track_config.max_keys
            hits_chunk_size = self.track_config.hits_chunk_size

        # Static tuples
        index_offsets_tuple = self._index_offsets_tuple
        max_indices_tuple = self._max_indices_tuple
        min_indices_tuple = self._min_indices_tuple
        num_wires_tuple = self._num_wires_tuple

        # Recombination function (captured in closure)
        recomb_fn = self.recomb_fn
        nominal_field_Vcm = float(
            self.detector_config['electric_field']['field_strength']
        )

        # Space charge effect functions (Python branch resolves at trace time)
        _sce_efield_fn = self._sce_efield_fn
        _sce_drift_correction_fn = self._sce_drift_correction_fn

        if _sce_efield_fn is not None:
            def sce_recomb_inputs_fn(positions_cm, theta, phi):
                """Compute phi_drift and |E| from local E-field map."""
                E_local = _sce_efield_fn(positions_cm)
                E_mag = jnp.sqrt(jnp.sum(E_local ** 2, axis=-1))
                E_mag_safe = jnp.maximum(E_mag, 1e-10)

                track_x = jnp.sin(theta) * jnp.cos(phi)
                track_y = jnp.sin(theta) * jnp.sin(phi)
                track_z = jnp.cos(theta)

                cos_phi_drift = jnp.abs(
                    track_x * (E_local[:, 0] / E_mag_safe)
                    + track_y * (E_local[:, 1] / E_mag_safe)
                    + track_z * (E_local[:, 2] / E_mag_safe)
                )
                phi_drift = jnp.arccos(jnp.clip(cos_phi_drift, 0.0, 1.0))
                return phi_drift, E_mag
        else:
            def sce_recomb_inputs_fn(positions_cm, theta, phi):
                """Nominal: phi_drift from x-axis, constant E-field."""
                dx_dir = jnp.sin(theta) * jnp.cos(phi)
                phi_drift = jnp.arccos(jnp.clip(jnp.abs(dx_dir), 0.0, 1.0))
                return phi_drift, nominal_field_Vcm

        if _sce_drift_correction_fn is not None:
            def sce_drift_fn(positions_cm, dist, time_us, yz):
                """Apply drift corrections from SCE map."""
                corr = _sce_drift_correction_fn(positions_cm)
                return apply_drift_corrections(
                    dist, time_us, yz,
                    corr[:, 0], corr[:, 1], corr[:, 2],
                    velocity_cm_us,
                )
        else:
            def sce_drift_fn(positions_cm, dist, time_us, yz):
                """No-op: return inputs unchanged."""
                return dist, time_us, yz

        # Response path batching parameters (captured in closure)
        response_chunk_size = self.response_chunk_size
        use_bucketed = self.use_bucketed
        sparse_output = self.sparse_output
        differentiable_flag = self.differentiable
        if use_bucketed:
            max_buckets = self.max_active_buckets

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

        # Define digitization function based on mode
        if self.include_digitize:
            dc = self.digitization_config
            _dig_gain = float(dc.gain_scale)
            _dig_adc_max = float((1 << dc.n_bits) - 1)
            _dig_ped_collection = float(dc.pedestal_collection)
            _dig_ped_induction = float(dc.pedestal_induction)

            def _digitize_signal(signal, gain_scale, pedestal, adc_max):
                scaled = signal * gain_scale
                unsigned = scaled + pedestal
                unsigned = jnp.round(unsigned)
                unsigned = jnp.clip(unsigned, 0.0, adc_max)
                return unsigned - pedestal

            if self.use_bucketed and self.include_electronics:
                # Wire-sparse: apply to active_signals array in 3-tuple
                def make_dig_fn_ws(plane_type):
                    ped = _dig_ped_collection if plane_type == 'Y' else _dig_ped_induction
                    def fn(signal_tuple):
                        active_signals, wire_indices, n_active = signal_tuple
                        return (_digitize_signal(active_signals, _dig_gain, ped, _dig_adc_max),
                                wire_indices, n_active)
                    return fn

                dig_fns = {
                    (s, p): make_dig_fn_ws(plane_names[s][p])
                    for s in range(2) for p in range(3)
                }
                def digitize_fn(sig, side_idx, plane_idx):
                    return dig_fns[(side_idx, plane_idx)](sig)

            elif self.use_bucketed:
                # Bucketed: apply to buckets array in 5-tuple
                def make_dig_fn_bucketed(plane_type):
                    ped = _dig_ped_collection if plane_type == 'Y' else _dig_ped_induction
                    def fn(signal_tuple):
                        buckets, num_active, compact_to_key, b1, b2 = signal_tuple
                        return (_digitize_signal(buckets, _dig_gain, ped, _dig_adc_max),
                                num_active, compact_to_key, b1, b2)
                    return fn

                dig_fns = {
                    (s, p): make_dig_fn_bucketed(plane_names[s][p])
                    for s in range(2) for p in range(3)
                }
                def digitize_fn(sig, side_idx, plane_idx):
                    return dig_fns[(side_idx, plane_idx)](sig)

            else:
                # Dense: apply directly to (W, T) array
                def make_dig_fn_dense(plane_type):
                    ped = _dig_ped_collection if plane_type == 'Y' else _dig_ped_induction
                    def fn(signal):
                        return _digitize_signal(signal, _dig_gain, ped, _dig_adc_max)
                    return fn

                dig_fns = {
                    (s, p): make_dig_fn_dense(plane_names[s][p])
                    for s in range(2) for p in range(3)
                }
                def digitize_fn(sig, side_idx, plane_idx):
                    return dig_fns[(side_idx, plane_idx)](sig)
        else:
            def digitize_fn(sig, side_idx, plane_idx):
                return sig

        # Define track hits function based on mode
        if include_track_hits_flag:
            def track_hits_fn(track_hits_list,
                              charges, drift_time_us, drift_distance_cm,
                              closest_wire_idx, closest_wire_distances,
                              attenuation_factors, valid_mask, group_ids,
                              theta, phi, angle_rad,
                              spacing_cm, min_idx_abs, num_wires_plane,
                              n_actual):
                theta_xz, theta_y = compute_deposit_wire_angles_vmap(
                    theta, phi, angle_rad
                )
                angular_scaling_factor = compute_angular_scaling_vmap(theta_xz, theta_y)

                SENTINEL_PK = jnp.int32(2**30)
                K_total = (2 * K_wire + 1) * (2 * K_time + 1)
                exp_size = hits_chunk_size * K_total
                max_safe_chunks = charges.shape[0] // hits_chunk_size
                num_chunks = jnp.minimum(
                    (n_actual + hits_chunk_size - 1) // hits_chunk_size,
                    max_safe_chunks
                )

                prepare_deposit_vmap_hit = jax.vmap(
                    prepare_deposit_with_diffusion,
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None,
                            None, None, None, None, None, None),
                )

                def body(i, state):
                    s_pk, s_gid, s_ch, s_count, s_rowsums = state
                    start = i * hits_chunk_size

                    # Slice chunk from pre-computed arrays
                    c_charges = jax.lax.dynamic_slice(charges, (start,), (hits_chunk_size,))
                    c_drift_time = jax.lax.dynamic_slice(drift_time_us, (start,), (hits_chunk_size,))
                    c_drift_dist = jax.lax.dynamic_slice(drift_distance_cm, (start,), (hits_chunk_size,))
                    c_wire_idx = jax.lax.dynamic_slice(closest_wire_idx, (start,), (hits_chunk_size,))
                    c_wire_dist = jax.lax.dynamic_slice(closest_wire_distances, (start,), (hits_chunk_size,))
                    c_atten = jax.lax.dynamic_slice(attenuation_factors, (start,), (hits_chunk_size,))
                    c_theta_xz = jax.lax.dynamic_slice(theta_xz, (start,), (hits_chunk_size,))
                    c_theta_y = jax.lax.dynamic_slice(theta_y, (start,), (hits_chunk_size,))
                    c_ang_scale = jax.lax.dynamic_slice(angular_scaling_factor, (start,), (hits_chunk_size,))
                    c_valid = jax.lax.dynamic_slice(valid_mask, (start,), (hits_chunk_size,))
                    c_gids = jax.lax.dynamic_slice(group_ids, (start,), (hits_chunk_size,))

                    # Diffusion expansion via vmap
                    wire_rel, time_idx, sig_val = prepare_deposit_vmap_hit(
                        c_charges, c_drift_time, c_drift_dist,
                        c_wire_idx, c_wire_dist, c_atten,
                        c_theta_xz, c_theta_y, c_ang_scale, c_valid,
                        K_wire, K_time, spacing_cm, time_step_size_us,
                        diffusion_long_cm2_us, diffusion_trans_cm2_us,
                        velocity_cm_us, min_idx_abs, num_wires_plane,
                        num_time_steps
                    )

                    # Per-deposit row_sum: total diffused charge above threshold
                    chunk_rowsums = jnp.sum(
                        jnp.where(sig_val > track_inter_thresh, sig_val, 0.0),
                        axis=1,
                    )
                    s_rowsums = jax.lax.dynamic_update_slice(
                        s_rowsums, chunk_rowsums, (start,)
                    )

                    # Flatten + encode pixel_key
                    wire_abs = wire_rel + min_idx_abs
                    gid_exp = jnp.repeat(c_gids[:, jnp.newaxis], K_total, axis=1)

                    w_flat = wire_abs.reshape(exp_size).astype(jnp.int32)
                    t_flat = time_idx.reshape(exp_size).astype(jnp.int32)
                    gid_flat = gid_exp.reshape(exp_size).astype(jnp.int32)
                    ch_flat = sig_val.reshape(exp_size)

                    chunk_pk = w_flat * max_time + t_flat
                    chunk_valid = ch_flat > 0.0
                    chunk_pk = jnp.where(chunk_valid, chunk_pk, SENTINEL_PK)
                    chunk_gid = jnp.where(chunk_valid, gid_flat, jnp.int32(0))
                    chunk_ch = jnp.where(chunk_valid, ch_flat, 0.0).astype(jnp.float32)

                    # Merge with running state (groups by (pixel, group_id))
                    new_pk, new_gid, new_ch, new_count = merge_chunk_hits(
                        s_pk, s_gid, s_ch,
                        chunk_pk, chunk_gid, chunk_ch,
                        track_inter_thresh
                    )

                    return (new_pk, new_gid, new_ch, new_count, s_rowsums)

                init_state = (
                    jnp.full(max_keys, SENTINEL_PK, dtype=jnp.int32),
                    jnp.zeros(max_keys, dtype=jnp.int32),
                    jnp.zeros(max_keys, dtype=jnp.float32),
                    jnp.int32(0),
                    jnp.zeros(charges.shape[0], dtype=jnp.float32),
                )

                final_pk, final_gid, final_ch, final_count, final_rowsums = jax.lax.fori_loop(
                    0, num_chunks, body, init_state
                )

                # Return raw merge state — label_from_groups runs outside JIT
                track_hits_list.append(
                    (final_pk, final_gid, final_ch, final_count, final_rowsums)
                )
        else:
            def track_hits_fn(track_hits_list,
                              charges, drift_time_us, drift_distance_cm,
                              closest_wire_idx, closest_wire_distances,
                              attenuation_factors, valid_mask, group_ids,
                              theta, phi, angle_rad,
                              spacing_cm, min_idx_abs, num_wires_plane,
                              n_actual):
                pass

        @partial(jax.jit, static_argnames=(
            'max_wire_indices_tuple', 'min_wire_indices_tuple',
            'index_offsets_tuple', 'num_wires_tuple',
            'max_tracks', 'max_wires', 'max_time', 'max_keys'
        ))
        def _calculate_signals_jit(
            # East side inputs (side 0, x < 0)
            east_positions_mm, east_de, east_dx, east_valid_mask,
            east_theta, east_phi, east_track_ids, east_group_ids,
            # West side inputs (side 1, x >= 0)
            west_positions_mm, west_de, west_dx, west_valid_mask,
            west_theta, west_phi, west_track_ids, west_group_ids,
            # Noise key
            noise_key,
            # Actual deposit counts (traced, not static)
            n_east, n_west,
            # Static args (geometry only — no per-side sizes)
            max_wire_indices_tuple, min_wire_indices_tuple,
            index_offsets_tuple, num_wires_tuple,
            max_tracks, max_wires, max_time, max_keys
        ):
            """JIT-compiled calculator with separate per-side data."""

            # Results lists
            response_signals_list = []
            track_hits_list = []

            # vmap for response path (shared across all planes)
            prepare_deposit_vmap_response = jax.vmap(
                prepare_deposit_for_response,
                in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None),
            )

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
                    group_ids = east_group_ids
                    n_actual = n_east
                else:  # West (x >= 0)
                    positions_mm = west_positions_mm
                    de = west_de
                    dx = west_dx
                    valid_mask = west_valid_mask
                    theta = west_theta
                    phi = west_phi
                    track_ids = west_track_ids
                    group_ids = west_group_ids
                    n_actual = n_west

                # Convert units: mm → cm
                positions_cm = positions_mm / 10.0
                dx_cm = dx / 10.0

                # Recombination inputs (SCE-aware or nominal)
                phi_drift, E_field_Vcm = sce_recomb_inputs_fn(positions_cm, theta, phi)

                # Apply charge recombination
                charges = recomb_fn(de, dx_cm, phi_drift, E_field_Vcm)

                # Calculate drift for the furthest plane on this side
                furthest_plane_idx = furthest_plane_indices[side_idx]
                furthest_plane_dist_cm = all_plane_distances_cm[side_idx, furthest_plane_idx]

                # Drift to furthest plane (nominal geometry)
                furthest_drift_distance_cm, furthest_drift_time_us, positions_yz_cm = compute_drift_to_plane(
                    positions_cm,
                    detector_half_width_cm,
                    velocity_cm_us,
                    furthest_plane_dist_cm
                )

                # Apply SCE drift corrections (identity when disabled)
                furthest_drift_distance_cm, furthest_drift_time_us, positions_yz_cm = sce_drift_fn(
                    positions_cm,
                    furthest_drift_distance_cm,
                    furthest_drift_time_us,
                    positions_yz_cm,
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
                        attenuation_factors, valid_mask, group_ids,
                        theta, phi, angle_rad,
                        spacing_cm, min_idx_abs, num_wires_plane,
                        n_actual
                    )

                    # --- RESPONSE PATH (batched via fori_loop) ---
                    # Calculate s parameter for diffusion (guard NaN for padded entries)
                    total_travel_distance_cm = detector_half_width_cm
                    s_values = drift_distance_cm / total_travel_distance_cm
                    s_values = jnp.clip(s_values, 0.0, 1.0)
                    s_values = jnp.where(valid_mask, s_values, 0.0)

                    # Response kernel parameters
                    plane_kernel = response_kernels[plane_type]
                    DKernel = plane_kernel['DKernel']
                    kernel_num_wires = plane_kernel['num_wires']
                    kernel_height = plane_kernel['kernel_height']
                    kernel_wire_stride = plane_kernel['wire_stride']
                    kernel_wire_spacing = plane_kernel['wire_spacing']
                    wire_zero_bin = plane_kernel['wire_zero_bin']
                    time_zero_bin = plane_kernel['time_zero_bin']

                    if differentiable_flag:
                        # --- DIFFERENTIABLE PATH (no fori_loop) ---
                        # Wrapped in jax.remat so only one plane's backward
                        # intermediates are live at a time (6x memory savings).
                        @jax.remat
                        def _plane_response(charges, drift_time_us, closest_wire_idx,
                                            closest_wire_distances, attenuation_factors,
                                            valid_mask, s_values):
                            deposit_data_full = prepare_deposit_vmap_response(
                                charges, drift_time_us, closest_wire_idx,
                                closest_wire_distances, attenuation_factors, valid_mask,
                                spacing_cm, time_step_size_us, min_idx_abs, num_wires_plane
                            )
                            wire_idx_rel, wire_offsets, time_idx, time_offsets, intensities = deposit_data_full

                            kernel_contributions = apply_diffusion_response(
                                DKernel, s_values, wire_offsets, time_offsets,
                                kernel_wire_stride, kernel_wire_spacing, kernel_num_wires
                            )

                            return accumulate_response_signals(
                                wire_idx_rel, time_idx, intensities, kernel_contributions,
                                num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                                wire_zero_bin, time_zero_bin
                            )

                        response_signal = _plane_response(
                            charges, drift_time_us, closest_wire_idx,
                            closest_wire_distances, attenuation_factors,
                            valid_mask, s_values
                        )

                    elif use_bucketed:
                        # --- BUCKETED RESPONSE PATH ---
                        B1 = 2 * kernel_num_wires
                        B2 = 2 * kernel_height

                        # Batch count for fori_loop
                        max_safe_batches = charges.shape[0] // response_chunk_size
                        n_batches = jnp.minimum(
                            (n_actual + response_chunk_size - 1) // response_chunk_size,
                            max_safe_batches
                        )

                        # Pre-compute wire/time indices for bucket mapping (element-wise, full array)
                        wire_idx_for_mapping = jnp.where(
                            valid_mask,
                            jnp.clip(closest_wire_idx - min_idx_abs, 0, num_wires_plane - 1),
                            jnp.int32(0)
                        )
                        time_idx_for_mapping = jnp.where(
                            valid_mask,
                            jnp.clip(
                                jnp.floor(drift_time_us / time_step_size_us).astype(jnp.int32),
                                0, num_time_steps - 1
                            ),
                            jnp.int32(0)
                        )

                        # Phase 1: build_bucket_mapping on full array (sort + dedup, no vmap needed)
                        point_to_compact, num_active, compact_to_key = build_bucket_mapping(
                            wire_idx_for_mapping, time_idx_for_mapping,
                            B1, B2, num_wires_plane, num_time_steps, max_buckets,
                            wire_zero_bin, time_zero_bin
                        )

                        def response_body_bucketed(i, carry_buckets):
                            start = i * response_chunk_size
                            b_charges    = jax.lax.dynamic_slice(charges,                (start,), (response_chunk_size,))
                            b_drift_time = jax.lax.dynamic_slice(drift_time_us,          (start,), (response_chunk_size,))
                            b_wire_idx   = jax.lax.dynamic_slice(closest_wire_idx,       (start,), (response_chunk_size,))
                            b_wire_dist  = jax.lax.dynamic_slice(closest_wire_distances, (start,), (response_chunk_size,))
                            b_atten      = jax.lax.dynamic_slice(attenuation_factors,    (start,), (response_chunk_size,))
                            b_valid      = jax.lax.dynamic_slice(valid_mask,             (start,), (response_chunk_size,))
                            b_s          = jax.lax.dynamic_slice(s_values,               (start,), (response_chunk_size,))
                            b_ptc        = jax.lax.dynamic_slice(point_to_compact,       (start, 0), (response_chunk_size, 4))

                            deposit_data_b = prepare_deposit_vmap_response(
                                b_charges, b_drift_time, b_wire_idx, b_wire_dist,
                                b_atten, b_valid,
                                spacing_cm, time_step_size_us, min_idx_abs, num_wires_plane
                            )
                            wire_idx_rel, wire_offsets_b, time_idx_b, time_offsets_b, intensities_b = deposit_data_b

                            kernel_contributions_b = apply_diffusion_response(
                                DKernel, b_s, wire_offsets_b, time_offsets_b,
                                kernel_wire_stride, kernel_wire_spacing, kernel_num_wires
                            )

                            batch_buckets = scatter_contributions_to_buckets_batched(
                                wire_idx_rel, time_idx_b, intensities_b, kernel_contributions_b,
                                b_ptc, max_buckets, kernel_num_wires, kernel_height, B1, B2,
                                wire_zero_bin, time_zero_bin,
                                batch_size=response_chunk_size,
                                num_wires=num_wires_plane, num_time_steps=num_time_steps
                            )
                            return carry_buckets + batch_buckets

                        response_buckets = jax.lax.fori_loop(
                            0, n_batches, response_body_bucketed,
                            jnp.zeros((max_buckets, B1, B2))
                        )

                        if sparse_output:
                            response_signal = (response_buckets, num_active, compact_to_key, B1, B2)
                        else:
                            response_signal = sparse_buckets_to_dense(
                                response_buckets, compact_to_key, num_active,
                                B1, B2, num_wires_plane, num_time_steps, max_buckets
                            )

                    else:
                        # --- DENSE RESPONSE PATH (batched via fori_loop) ---
                        max_safe_batches = charges.shape[0] // response_chunk_size
                        n_batches = jnp.minimum(
                            (n_actual + response_chunk_size - 1) // response_chunk_size,
                            max_safe_batches
                        )

                        def response_body_dense(i, signal_accum):
                            start = i * response_chunk_size
                            b_charges    = jax.lax.dynamic_slice(charges,                (start,), (response_chunk_size,))
                            b_drift_time = jax.lax.dynamic_slice(drift_time_us,          (start,), (response_chunk_size,))
                            b_wire_idx   = jax.lax.dynamic_slice(closest_wire_idx,       (start,), (response_chunk_size,))
                            b_wire_dist  = jax.lax.dynamic_slice(closest_wire_distances, (start,), (response_chunk_size,))
                            b_atten      = jax.lax.dynamic_slice(attenuation_factors,    (start,), (response_chunk_size,))
                            b_valid      = jax.lax.dynamic_slice(valid_mask,             (start,), (response_chunk_size,))
                            b_s          = jax.lax.dynamic_slice(s_values,               (start,), (response_chunk_size,))

                            deposit_data_b = prepare_deposit_vmap_response(
                                b_charges, b_drift_time, b_wire_idx, b_wire_dist,
                                b_atten, b_valid,
                                spacing_cm, time_step_size_us, min_idx_abs, num_wires_plane
                            )
                            wire_idx_rel, wire_offsets_b, time_idx_b, time_offsets_b, intensities_b = deposit_data_b

                            kernel_contributions_b = apply_diffusion_response(
                                DKernel, b_s, wire_offsets_b, time_offsets_b,
                                kernel_wire_stride, kernel_wire_spacing, kernel_num_wires
                            )

                            batch_signal = accumulate_response_signals(
                                wire_idx_rel, time_idx_b, intensities_b, kernel_contributions_b,
                                num_wires_plane, num_time_steps, kernel_num_wires, kernel_height,
                                wire_zero_bin, time_zero_bin
                            )
                            return signal_accum + batch_signal

                        response_signal = jax.lax.fori_loop(
                            0, n_batches, response_body_dense,
                            jnp.zeros((num_wires_plane, num_time_steps))
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

                    # Apply digitization (quantize + clamp)
                    response_signal = digitize_fn(response_signal, side_idx, plane_idx)

                    response_signals_list.append(response_signal)

            return tuple(response_signals_list), tuple(track_hits_list)

        # Store raw JIT function (single compilation for all events)
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
            Can be any size - will be split and padded internally.
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
        east_data, west_data, counts, group_data = self._split_and_pad_data(deposit_data)

        # Pre-simulation validation
        self._validate_pre_simulation(counts)

        # Noise key (used even when noise is disabled - identity function ignores it)
        noise_key = key if key is not None else jax.random.PRNGKey(0)

        # Group IDs (zeros when track_hits disabled)
        east_group_ids = group_data['east_group_ids']
        west_group_ids = group_data['west_group_ids']

        # Call JIT-compiled calculator (single compilation for all events)
        response_tuple, track_hits_tuple = self._calculator_jit_raw(
            # East data (always shape (total_pad, ...))
            east_data.positions_mm, east_data.de, east_data.dx, east_data.valid_mask,
            east_data.theta, east_data.phi, east_data.track_ids, east_group_ids,
            # West data (always shape (total_pad, ...))
            west_data.positions_mm, west_data.de, west_data.dx, west_data.valid_mask,
            west_data.theta, west_data.phi, west_data.track_ids, west_group_ids,
            # Noise key
            noise_key,
            # Actual deposit counts (traced — dynamic batch count derived from these)
            n_east=counts['n_east'],
            n_west=counts['n_west'],
            # Static args (geometry only — no per-side sizes)
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
        g2t = group_data['group_to_track']
        for side_idx in range(2):
            for plane_idx in range(3):
                response_signals[(side_idx, plane_idx)] = response_tuple[idx]
                if self.include_track_hits:
                    # Raw state from JIT: (pk, gid, ch, count, row_sums)
                    raw = track_hits_tuple[idx]
                    track_hits[(side_idx, plane_idx)] = raw
                idx += 1

        # Store group_data for postprocessing access
        self._last_group_data = group_data

        # Return raw results — call finalize_track_hits() after freeing
        # response_signals from GPU to avoid device memory pressure.
        return response_signals, track_hits

    def build_forward(self, dx_per_de=None, dx_mm=0.5):
        """Return a differentiable forward function: SegmentData -> tuple of 6 response arrays.

        The returned callable accepts a SegmentData(positions_mm=(N,3), de=(N,))
        and returns a tuple of 6 arrays (east_U, east_V, east_Y, west_U, west_V, west_Y),
        each of shape (num_wires, num_time_steps) for the corresponding plane.

        Parameters
        ----------
        dx_per_de : array (N,) or None
            Per-segment ratio dx_truth / dE_truth.  When provided, dx is
            computed dynamically as ``segments.de * dx_per_de``, preserving
            each segment's truth dE/dx through recombination regardless of
            how its learned dE evolves.  When None, falls back to dx_mm.
        dx_mm : float
            Fixed segment length in mm for all segments (default 0.5 mm).
            Only used when dx_per_de is None.

        No padding — arrays are passed at exactly (N, ...) size. The JIT
        compiles once for a given N and reuses the compilation on subsequent
        calls with the same N.

        The function is not JIT-compiled — compose ``jax.jit(jax.grad(loss_fn))``
        yourself to control compilation scope.

        Requires ``differentiable=True`` at construction time.
        """
        assert self.differentiable, "build_forward() requires differentiable=True"

        calculator = self._calculator_jit_raw
        N = self.n_segments

        # Fixed arrays at N size (captured in closure)
        if dx_per_de is not None:
            dx_per_de = jnp.asarray(dx_per_de)  # (N,) captured in closure
            dx_fixed = None
        else:
            dx_fixed = jnp.full(N, dx_mm)
        theta = jnp.zeros(N)                               # Not used by modified_box
        phi = jnp.zeros(N)                                 # Not used by modified_box
        track_ids = jnp.zeros(N, dtype=jnp.int32)
        group_ids_dummy = jnp.zeros(N, dtype=jnp.int32)   # Dummy, track_hits disabled
        noise_key = jax.random.PRNGKey(0)                  # Dummy, noise disabled
        valid = jnp.ones(N, dtype=bool)                    # All entries are real

        # Static geometry args (captured in closure)
        max_indices = self._max_indices_tuple
        min_indices = self._min_indices_tuple
        offsets = self._index_offsets_tuple
        num_wires = self._num_wires_tuple

        def forward(segments):
            positions_mm = segments.positions_mm  # (N, 3)
            de = segments.de                      # (N,)

            # Dynamic dx: preserves per-segment dE/dx from truth
            if dx_fixed is not None:
                dx = dx_fixed
            else:
                dx = jnp.clip(de * dx_per_de, 0.01, 10.0)  # mm, clamped

            # Side masks directly on the input (no padding)
            east_valid = valid & (positions_mm[:, 0] < 0)
            west_valid = valid & (positions_mm[:, 0] >= 0)

            response_tuple, _ = calculator(
                # East
                positions_mm, de, dx, east_valid, theta, phi, track_ids, group_ids_dummy,
                # West
                positions_mm, de, dx, west_valid, theta, phi, track_ids, group_ids_dummy,
                noise_key,
                n_east=N, n_west=N,
                max_wire_indices_tuple=max_indices,
                min_wire_indices_tuple=min_indices,
                index_offsets_tuple=offsets,
                num_wires_tuple=num_wires,
                max_tracks=1, max_wires=1, max_time=1, max_keys=1
            )
            return response_tuple  # (east_U, east_V, east_Y, west_U, west_V, west_Y)

        return forward

    def finalize_track_hits(self, track_hits):
        """
        Post-process track hits: derive track labels from group merge state.

        Applies label_from_groups to each plane's raw (pk, gid, ch, count, row_sums)
        tuple, producing the standard track_hits dict format with labeled_hits,
        hits_by_track, and group_correspondence.

        Call this after moving response_signals off GPU (e.g. via np.asarray)
        to avoid device memory pressure from the int() sync calls.

        Parameters
        ----------
        track_hits : dict
            Raw track hits from process_event(). Each value is a 5-tuple
            (pk, gid, ch, count, row_sums) from the JIT fori_loop.

        Returns
        -------
        track_hits : dict
            Track hits with labeled_hits, hits_by_track, group_correspondence.
        """
        g2t = self._last_group_data['group_to_track']
        max_time = self._max_time

        for plane_key, raw in track_hits.items():
            pk, gid, ch, count, row_sums = raw
            result = label_from_groups(
                pk, gid, ch, count, g2t, max_time
            )
            result['row_sums'] = row_sums
            track_hits[plane_key] = result

        # Validate: check for group merge overflow
        if self.track_config is not None:
            for plane_key, th in track_hits.items():
                gp = th.get('group_correspondence')
                if gp is not None:
                    count_val = int(gp[-1])
                    if count_val >= self.track_config.max_keys:
                        print(f"ERROR: Plane {plane_key}: group merge count ({count_val:,}) >= "
                              f"max_keys ({self.track_config.max_keys:,}). "
                              f"Segment correspondence data TRUNCATED! "
                              f"Increase max_keys or reduce event size.")

        return track_hits

    def warm_up(self):
        """Trigger JIT compilation with dummy data."""
        print("Triggering JIT compilation...")

        pad = self.total_pad

        dummy = DepositData(
            positions_mm=jnp.zeros((pad, 3), dtype=jnp.float32),
            de=jnp.zeros((pad,), dtype=jnp.float32),
            dx=jnp.zeros((pad,), dtype=jnp.float32),
            valid_mask=jnp.zeros((pad,), dtype=bool),
            theta=jnp.zeros((pad,), dtype=jnp.float32),
            phi=jnp.zeros((pad,), dtype=jnp.float32),
            track_ids=jnp.zeros((pad,), dtype=jnp.int32)
        )

        dummy_group_ids = jnp.zeros((pad,), dtype=jnp.int32)

        # Call directly to avoid split/pad logic
        _ = self._calculator_jit_raw(
            dummy.positions_mm, dummy.de, dummy.dx, dummy.valid_mask,
            dummy.theta, dummy.phi, dummy.track_ids, dummy_group_ids,
            dummy.positions_mm, dummy.de, dummy.dx, dummy.valid_mask,
            dummy.theta, dummy.phi, dummy.track_ids, dummy_group_ids,
            jax.random.PRNGKey(0),
            n_east=0,
            n_west=0,
            max_wire_indices_tuple=self._max_indices_tuple,
            min_wire_indices_tuple=self._min_indices_tuple,
            index_offsets_tuple=self._index_offsets_tuple,
            num_wires_tuple=self._num_wires_tuple,
            max_tracks=self.track_config.max_tracks if self.include_track_hits else 1,
            max_wires=self._max_wires if self.include_track_hits else 1,
            max_time=self._max_time if self.include_track_hits else 1,
            max_keys=self.track_config.max_keys if self.include_track_hits else 1
        )
        print(f"JIT compilation finished (total_pad={pad:,}). Single compilation for all events.")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_simulation(config_path, data_path, event_idx=0,
                   num_s=16,
                   response_path="tools/responses/",
                   track_threshold=1.0,
                   total_pad=200_000,
                   response_chunk_size=50_000,
                   include_track_hits=True):
    """
    Run the detector simulation for a specific event.

    Convenience function that creates a DetectorSimulator and processes one event.

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
    total_pad : int, optional
        Fixed pad size per side. Default 200_000.
    response_chunk_size : int, optional
        Deposits per fori_loop batch. Default 50_000.
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
            total_pad=total_pad,
            response_chunk_size=response_chunk_size,
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
        event_positions_mm = np.asarray(step_data.get('position', np.empty((0, 3))), dtype=np.float32)

        n_hits = event_positions_mm.shape[0]
        print(f"Loaded {n_hits:,} steps from event {event_idx}.")

        if n_hits == 0:
            print("WARNING: Event contains no particle steps.")
            return {}, {}, simulator

        # Extract de and dx (recombination is now done inside the simulator)
        event_de = np.asarray(step_data.get('de', np.zeros((n_hits,))), dtype=np.float32)
        event_dx = np.asarray(step_data.get('dx', np.zeros((n_hits,))), dtype=np.float32)

        # Extract angles and track IDs
        event_theta = np.asarray(step_data.get('theta', np.zeros((n_hits,))), dtype=np.float32)
        event_phi = np.asarray(step_data.get('phi', np.zeros((n_hits,))), dtype=np.float32)
        event_track_ids = np.asarray(step_data.get('track_id', np.ones((n_hits,))), dtype=np.int32)

        print(f"Unique tracks: {len(np.unique(event_track_ids)):,}")

        # Create deposit data as numpy (split_and_pad handles numpy->JAX conversion)
        deposit_data = DepositData(
            positions_mm=event_positions_mm,
            de=event_de,
            dx=event_dx,
            valid_mask=np.ones(n_hits, dtype=bool),
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
