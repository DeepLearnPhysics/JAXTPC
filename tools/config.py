"""
Configuration classes for JAXTPC detector simulation.

This module defines all NamedTuple parameter bundles used throughout
the simulation code for clean parameter passing and type hints.
"""

from typing import NamedTuple, Any
from pathlib import Path
import numpy as np
import jax.numpy as jnp


class DepositData(NamedTuple):
    """Padded input data from particle simulation steps."""
    positions_mm: jnp.ndarray    # (N_pad, 3) - hit positions
    de: jnp.ndarray              # (N_pad,) - energy deposits in MeV
    dx: jnp.ndarray              # (N_pad,) - step lengths in mm (converted to cm in simulation)
    valid_mask: jnp.ndarray      # (N_pad,) - True for real hits
    theta: jnp.ndarray           # (N_pad,) - polar angle of step direction
    phi: jnp.ndarray             # (N_pad,) - azimuthal angle of step direction
    track_ids: jnp.ndarray       # (N_pad,) - particle track ID
    group_ids: jnp.ndarray       # (N_pad,) - segment group ID for correspondence


class DiffusionConfig(NamedTuple):
    """Diffusion parameters for track_hits path (production with DKernel)."""
    long_cm2_us: float      # Longitudinal diffusion coefficient
    trans_cm2_us: float     # Transverse diffusion coefficient
    K_wire: int             # Half-width in wire direction (computed from sigmas)
    K_time: int             # Half-width in time direction (computed from sigmas)
    velocity_cm_us: float   # Drift velocity (for diffusion spread calculation)
    num_s: int              # Number of diffusion levels for kernel interpolation
    max_sigma_trans_unitless: float  # Max transverse sigma in wire-pitch units
    max_sigma_long_unitless: float   # Max longitudinal sigma in time-bin units


class TrackHitsConfig(NamedTuple):
    """Configuration for track hit labeling."""
    threshold: float        # Minimum charge to keep
    max_tracks: int         # Max tracks for array pre-allocation
    max_keys: int           # Max unique (track, wire, time) combinations
    hits_chunk_size: int    # Deposits per fori_loop chunk (must divide padding tiers)
    inter_thresh: float     # Intermediate pruning threshold per merge iteration


class SimConfig(NamedTuple):
    """Static simulation configuration (closure-captured, not differentiable).

    All static values in one place. Closure-captured by both JIT builders.
    Changing any value requires new closure → JIT recompilation (correct for static config).
    """
    # Time grid
    num_time_steps: int
    time_step_us: float

    # Array dimensions / batching
    total_pad: int
    response_chunk_size: int
    max_wires: int                      # max wire count across all planes

    # Mode flags
    use_bucketed: bool
    include_track_hits: bool
    include_noise: bool
    include_electronics: bool
    include_digitize: bool

    # Bucketed mode
    max_active_buckets: int

    # Geometry (per-side)
    side_geom: tuple                    # (SideGeometry, SideGeometry)

    # Plane names
    plane_names: tuple                  # (('U','V','Y'), ('U','V','Y'))

    # Output format: 'dense', 'bucketed', or 'wire_sparse'
    output_format: str

    # Readout
    electrons_per_adc: float            # Conversion factor (e.g., 182 for MicroBooNE)
    noise_spectrum_path: str            # Path to noise_spectrum.npz

    # Optional (None when disabled/not needed)
    diffusion: Any                      # DiffusionConfig or None
    track_hits: Any                     # TrackHitsConfig or None


class ResponseKernel(NamedTuple):
    """Pre-computed wire response kernel for a single plane type (U, V, or Y).

    The DKernel table is generated from base_kernel via DCT-domain Gaussian
    blurring at each s_level. The freq_w/freq_t arrays and base_kernel are
    stored so the table can be regenerated inside JIT with different diffusion
    constants (enabling differentiable diffusion parameters).
    """
    DKernel: jnp.ndarray     # (num_s, kernel_height, kernel_width) diffusion kernel table
    num_wires: int            # Number of output wires the kernel spans
    kernel_height: int        # Output time bins (kernel_height - 1 due to interpolation)
    wire_spacing: float       # Wire spacing in cm (read from kernel file)
    time_spacing: float       # Simulation time spacing in us
    wire_zero_bin: int        # Wire=0 position in output wire units
    time_zero_bin: int        # t=0 position in output time bins
    base_kernel: jnp.ndarray  # (H, W) raw kernel before diffusion
    freq_w: jnp.ndarray       # (W,) DCT-II frequency grid, wire axis
    freq_t: jnp.ndarray       # (H,) DCT-II frequency grid, time axis
    s_levels: jnp.ndarray     # (num_s,) diffusion levels from 0 to 1


class DigitizationConfig(NamedTuple):
    """ADC digitization parameters for realistic detector output."""
    n_bits: int              # ADC resolution (e.g., 12)
    pedestal_collection: int # Baseline ADC for Y (collection) planes
    pedestal_induction: int  # Baseline ADC for U/V (induction) planes
    gain_scale: float        # Gain rescale factor (1.0 = MicroBooNE/SBND)


def create_digitization_config(
    n_bits: int = 12,
    pedestal_collection: int = 410,
    pedestal_induction: int = 1843,
    gain_scale: float = 1.0,
) -> DigitizationConfig:
    """
    Create DigitizationConfig with specified parameters.

    Parameters
    ----------
    n_bits : int, optional
        ADC resolution in bits, by default 12.
    pedestal_collection : int, optional
        Baseline ADC value for collection (Y) planes, by default 410.
    pedestal_induction : int, optional
        Baseline ADC value for induction (U/V) planes, by default 1843.
    gain_scale : float, optional
        Gain rescale factor applied before digitization, by default 1.0.

    Returns
    -------
    DigitizationConfig
        Configured digitization parameters.
    """
    return DigitizationConfig(
        n_bits=n_bits,
        pedestal_collection=pedestal_collection,
        pedestal_induction=pedestal_induction,
        gain_scale=gain_scale,
    )


def create_track_hits_config(
    threshold: float = 1.0,
    max_tracks: int = 10000,
    max_keys: int = 4000000,
    hits_chunk_size: int = 25000,
    inter_thresh: float = 1.0,
) -> TrackHitsConfig:
    """
    Create TrackHitsConfig with specified parameters.

    Parameters
    ----------
    threshold : float, optional
        Minimum charge threshold for keeping hits, by default 1.0.
    max_tracks : int, optional
        Maximum number of tracks for array pre-allocation, by default 10000.
    max_keys : int, optional
        Maximum number of unique (track, wire, time) combinations, by default 1000000.
    hits_chunk_size : int, optional
        Number of deposits per fori_loop iteration, by default 25000.
        Must evenly divide all padding tiers (e.g. 100000, 200000).
    inter_thresh : float, optional
        Intermediate pruning threshold applied each merge iteration, by default 1.0.

    Returns
    -------
    TrackHitsConfig
        Configured track hits parameters.
    """
    return TrackHitsConfig(
        threshold=threshold,
        max_tracks=max_tracks,
        max_keys=max_keys,
        hits_chunk_size=hits_chunk_size,
        inter_thresh=inter_thresh,
    )


def create_sim_config(detector_config, total_pad=200_000, response_chunk_size=50_000,
                      use_bucketed=False, max_active_buckets=1000,
                      include_noise=False, include_electronics=False,
                      include_track_hits=False, include_digitize=False,
                      track_config=None, include_diffusion=True, num_s=16):
    """Create SimConfig from raw parsed YAML detector configuration.

    Calls geometry functions directly to compute all derived parameters.

    Parameters
    ----------
    detector_config : dict
        Raw parsed YAML from generate_detector().
    total_pad : int
        Fixed pad size per side. Default 200,000.
    response_chunk_size : int
        Deposits per fori_loop batch. Must divide total_pad. Default 50,000.
    use_bucketed : bool
        Use sparse bucketed accumulation. Default False.
    max_active_buckets : int
        Max active buckets for sparse mode. Default 1000.
    include_noise : bool
        Add intrinsic noise inside JIT. Default False.
    include_electronics : bool
        Apply RC-RC electronics convolution. Default False.
    include_track_hits : bool
        Run track labeling path. Default False.
    include_digitize : bool
        Apply ADC digitization. Default False.
    track_config : TrackHitsConfig, optional
        Track labeling parameters. Built with defaults if None.
    include_diffusion : bool
        If True, compute DiffusionConfig (for DKernel/track_hits).
        False for NN-only paths where diffusion is implicitly encoded.
    num_s : int
        Number of diffusion levels for kernel interpolation. Default 16.
    """
    from tools.geometry import (
        get_detector_dimensions, get_drift_params, get_plane_geometry,
        calculate_time_params, pre_calculate_all_wire_params,
        calculate_max_diffusion_sigmas, _calculate_wire_lengths,
    )

    # Compute detector geometry from raw YAML
    dims_cm = get_detector_dimensions(detector_config)
    half_width, velocity = get_drift_params(detector_config, dims_cm)
    num_time_steps, time_step_us, _ = calculate_time_params(
        detector_config, dims_cm, velocity)
    plane_distances, furthest_indices = get_plane_geometry(detector_config)
    (angles_rad, wire_spacings_cm, index_offsets,
     num_wires_actual, max_wire_indices,
    ) = pre_calculate_all_wire_params(detector_config, dims_cm)
    wire_lengths_m = _calculate_wire_lengths(
        dims_cm, angles_rad, wire_spacings_cm, index_offsets,
        num_wires_actual)

    # Build SideGeometry for each side
    n_sides = len(detector_config['wire_planes']['sides'])
    n_planes = len(detector_config['wire_planes']['sides'][0]['planes'])
    side_geom = tuple(
        SideGeometry(
            half_width_cm=half_width,
            furthest_plane_dist_cm=float(plane_distances[s, int(furthest_indices[s])]),
            plane_distances_cm=tuple(float(plane_distances[s, p]) for p in range(n_planes)),
            angles_rad=tuple(float(angles_rad[s, p]) for p in range(n_planes)),
            wire_spacings_cm=tuple(float(wire_spacings_cm[s, p]) for p in range(n_planes)),
            index_offsets=tuple(int(index_offsets[s, p]) for p in range(n_planes)),
            max_wire_indices=tuple(int(max_wire_indices[s, p]) for p in range(n_planes)),
            num_wires=tuple(int(num_wires_actual[s, p]) for p in range(n_planes)),
            wire_lengths_m=tuple(wire_lengths_m[(s, p)] for p in range(n_planes)),
        )
        for s in range(n_sides)
    )

    # Plane type labels used as kernel keys ('U', 'V', 'Y')
    _PLANE_LABELS = ('U', 'V', 'Y')
    plane_names = tuple(_PLANE_LABELS[:n_planes] for _ in range(n_sides))

    # Diffusion config (optional)
    if include_diffusion:
        long_diff = float(detector_config['simulation']['drift']['longitudinal_diffusion']) / 1e6
        trans_diff = float(detector_config['simulation']['drift']['transverse_diffusion']) / 1e6
        wire_spacing_ref = float(wire_spacings_cm[0, 0])

        _, _, max_sigma_trans, max_sigma_long = calculate_max_diffusion_sigmas(
            half_width, velocity, trans_diff, long_diff, wire_spacing_ref, time_step_us)

        diffusion = DiffusionConfig(
            long_cm2_us=long_diff,
            trans_cm2_us=trans_diff,
            K_wire=max(1, int(np.ceil(3.0 * max_sigma_trans))),
            K_time=max(1, int(np.ceil(3.0 * max_sigma_long))),
            velocity_cm_us=velocity,
            num_s=num_s,
            max_sigma_trans_unitless=max_sigma_trans,
            max_sigma_long_unitless=max_sigma_long,
        )
    else:
        diffusion = None

    # Track hits config (optional)
    if include_track_hits:
        if track_config is None:
            track_config = create_track_hits_config()
        max_wires = max(int(np.max(max_wire_indices) + 1), 2000)
    else:
        track_config = None
        max_wires = 0

    # Derive output format from mode flags
    if use_bucketed and include_electronics:
        output_format = 'wire_sparse'
    elif use_bucketed:
        output_format = 'bucketed'
    else:
        output_format = 'dense'

    return SimConfig(
        num_time_steps=num_time_steps,
        time_step_us=time_step_us,
        total_pad=total_pad,
        response_chunk_size=response_chunk_size,
        max_wires=max_wires,
        use_bucketed=use_bucketed,
        include_track_hits=include_track_hits,
        include_noise=include_noise,
        include_electronics=include_electronics,
        include_digitize=include_digitize,
        max_active_buckets=max_active_buckets,
        side_geom=side_geom,
        plane_names=plane_names,
        output_format=output_format,
        electrons_per_adc=float(detector_config['readout'].get('electrons_per_adc', 182)),
        noise_spectrum_path=str(
            Path(__file__).parent.parent / 'config' / 'noise_spectrum.npz'),
        diffusion=diffusion,
        track_hits=track_config,
    )


class ModifiedBoxParams(NamedTuple):
    """ArgoNeuT 2013 Modified Box recombination parameters."""
    density: jnp.ndarray
    w_value: jnp.ndarray
    excitation_ratio: jnp.ndarray
    field_strength_Vcm: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray


class EMBParams(NamedTuple):
    """ICARUS 2024 Ellipsoid Modified Box recombination parameters."""
    density: jnp.ndarray
    w_value: jnp.ndarray
    excitation_ratio: jnp.ndarray
    field_strength_Vcm: jnp.ndarray
    alpha: jnp.ndarray
    beta_90: jnp.ndarray
    R: jnp.ndarray


class SimParams(NamedTuple):
    """All tunable simulation parameters — physics scalars and NN models."""
    velocity_cm_us: jnp.ndarray
    lifetime_us: jnp.ndarray
    diffusion_trans_cm2_us: jnp.ndarray   # Transverse diffusion coefficient
    diffusion_long_cm2_us: jnp.ndarray    # Longitudinal diffusion coefficient
    recomb_params: Any              # ModifiedBoxParams or EMBParams
    response_models: Any            # dict {(side_idx, plane_type): eqx.Module} or None
    sce_models: Any                 # tuple (east_siren, west_siren) or None


class SCEOutputs(NamedTuple):
    """Raw outputs from SCE correction map query."""
    efield_correction: jnp.ndarray  # (N, 3) dimensionless, E_local / |E_nominal|
    drift_corr_cm: jnp.ndarray     # (N, 3) drift corrections [dx, dy, dz] in cm


class SideGeometry(NamedTuple):
    """Static geometry for one detector side."""
    half_width_cm: float
    furthest_plane_dist_cm: float
    plane_distances_cm: tuple       # (n_planes,) per plane
    angles_rad: tuple               # (n_planes,)
    wire_spacings_cm: tuple         # (n_planes,)
    index_offsets: tuple            # (n_planes,) int
    max_wire_indices: tuple         # (n_planes,) int
    num_wires: tuple                # (n_planes,) int
    wire_lengths_m: tuple           # (n_planes,) of np.ndarray, wire lengths in meters


class SideIntermediates(NamedTuple):
    """Output of compute_side_physics, input to compute_plane_physics."""
    charges: jnp.ndarray            # (N,) zeroed for invalid deposits (valid_mask applied)
    photons: jnp.ndarray            # (N,) scintillation photons (valid_mask applied)
    drift_distance_cm: jnp.ndarray  # (N,)
    drift_time_us: jnp.ndarray     # (N,)
    positions_cm: jnp.ndarray      # (N, 3) original positions (for NN response)
    positions_yz_cm: jnp.ndarray   # (N, 2) projected (for wire distances)


class PlaneIntermediates(NamedTuple):
    """Output of compute_plane_physics, input to response computation."""
    drift_distance_cm: jnp.ndarray  # (N,) SCE-corrected, plane-corrected
    drift_time_us: jnp.ndarray     # (N,)
    attenuation: jnp.ndarray       # (N,)
    closest_wire_idx: jnp.ndarray  # (N,) int
    closest_wire_dist: jnp.ndarray # (N,)
    charges: jnp.ndarray           # (N,) zeroed for invalid deposits
    photons: jnp.ndarray           # (N,) scintillation photons
    positions_cm: jnp.ndarray      # (N, 3) carried through for response_fn


def create_sim_params(detector_config, recombination_model='modified_box',
                      response_models=None, sce_models=None):
    """Create SimParams from raw parsed YAML detector config.

    Parameters
    ----------
    detector_config : dict
        Raw parsed YAML from generate_detector().
    recombination_model : str
        'modified_box' or 'emb'.
    response_models : dict or None
        {(side_idx, plane_type): eqx.Module} for NN response, or None for DKernel.
    sce_models : tuple or None
        (east_siren, west_siren) for SIREN SCE, or None for HDF5/nominal.
    """
    from tools.geometry import get_drift_params
    _, velocity_cm_us = get_drift_params(detector_config)
    lifetime_ms = float(detector_config['simulation']['drift']['electron_lifetime'])

    velocity = jnp.array(velocity_cm_us)
    lifetime = jnp.array(lifetime_ms * 1000.0)

    density = float(detector_config['medium']['properties']['density'])
    w_value = float(detector_config['medium']['properties']['ionization_energy'])
    excitation_ratio = float(detector_config['medium']['properties']['excitation_ratio'])
    field_strength = float(detector_config['electric_field']['field_strength'])

    sim_cfg = detector_config.get('simulation', {})
    recomb_cfg = sim_cfg.get('charge_recombination', {})
    params_cfg = recomb_cfg.get('recomb_parameters', {})

    def _get_recomb_param(key):
        if key not in params_cfg:
            raise KeyError(
                f"Recombination parameter '{key}' not found in config "
                f"(simulation.charge_recombination.recomb_parameters.{key}). "
                f"Required for model='{recombination_model}'.")
        return jnp.array(float(params_cfg[key]))

    if recombination_model == 'modified_box':
        recomb = ModifiedBoxParams(
            density=jnp.array(density),
            w_value=jnp.array(w_value),
            excitation_ratio=jnp.array(excitation_ratio),
            field_strength_Vcm=jnp.array(field_strength),
            alpha=_get_recomb_param('alpha'),
            beta=_get_recomb_param('beta'),
        )
    elif recombination_model == 'emb':
        recomb = EMBParams(
            density=jnp.array(density),
            w_value=jnp.array(w_value),
            excitation_ratio=jnp.array(excitation_ratio),
            field_strength_Vcm=jnp.array(field_strength),
            alpha=_get_recomb_param('alpha_emb'),
            beta_90=_get_recomb_param('beta_90'),
            R=_get_recomb_param('R_anisotropy'),
        )
    else:
        raise ValueError(f"Unknown recombination model: {recombination_model}")

    drift_cfg = sim_cfg.get('drift', {})
    # YAML stores cm²/s, convert to cm²/us
    diffusion_trans = jnp.array(float(drift_cfg.get('transverse_diffusion', 12.0)) / 1e6)
    diffusion_long = jnp.array(float(drift_cfg.get('longitudinal_diffusion', 7.2)) / 1e6)

    return SimParams(
        velocity_cm_us=velocity,
        lifetime_us=lifetime,
        diffusion_trans_cm2_us=diffusion_trans,
        diffusion_long_cm2_us=diffusion_long,
        recomb_params=recomb,
        response_models=response_models,
        sce_models=sce_models,
    )


def create_deposit_data(positions_mm, de, dx, theta=None, phi=None,
                        track_ids=None, group_ids=None):
    """Convenience constructor for DepositData with sensible defaults.

    Required: positions_mm (N,3), de (N,), dx (N,) or scalar.
    Optional: theta, phi (default zeros), track_ids, group_ids (default zeros).
    valid_mask is always ones — caller creates unpadded deposits.
    forward() handles padding to total_pad internally.
    """
    N = positions_mm.shape[0]
    return DepositData(
        positions_mm=positions_mm,
        de=de,
        dx=jnp.full(N, dx) if jnp.ndim(dx) == 0 else dx,
        valid_mask=jnp.ones(N, bool),
        theta=theta if theta is not None else jnp.zeros(N),
        phi=phi if phi is not None else jnp.zeros(N),
        track_ids=track_ids if track_ids is not None else jnp.zeros(N, jnp.int32),
        group_ids=group_ids if group_ids is not None else jnp.zeros(N, jnp.int32),
    )



def pad_deposit_data(deposits, target_size):
    """Pad DepositData to target_size with valid_mask=False entries.

    Used by forward() to pad to total_pad (next multiple of chunk_size).
    Padding entries have de=0, dx=1 (avoids division by zero in recombination),
    valid_mask=False (zeroes charges in compute_side_physics).
    """
    N = deposits.positions_mm.shape[0]
    pad_size = target_size - N
    if pad_size <= 0:
        return deposits
    return DepositData(
        positions_mm=jnp.pad(deposits.positions_mm, ((0, pad_size), (0, 0))),
        de=jnp.pad(deposits.de, (0, pad_size)),
        dx=jnp.pad(deposits.dx, (0, pad_size), constant_values=1.0),
        valid_mask=jnp.pad(deposits.valid_mask, (0, pad_size)),  # False for padding
        theta=jnp.pad(deposits.theta, (0, pad_size)),
        phi=jnp.pad(deposits.phi, (0, pad_size)),
        track_ids=jnp.pad(deposits.track_ids, (0, pad_size)),
        group_ids=jnp.pad(deposits.group_ids, (0, pad_size)),
    )
