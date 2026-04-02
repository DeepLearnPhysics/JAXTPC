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
    """Multi-volume padded input data for simulation.

    Holds a tuple of VolumeDeposits (one per volume) plus metadata
    used outside JIT for post-processing.

    Constructed by build_deposit_data() or load_event().
    """
    volumes: tuple          # (VolumeDeposits_0, VolumeDeposits_1, ...) — passed to JIT
    group_to_track: Any     # tuple of np.ndarray lookups per volume (outside JIT only)
    original_indices: Any   # tuple of np.ndarray per volume (outside JIT only)


class VolumeDeposits(NamedTuple):
    """Single-volume deposit arrays, extracted from DepositData for physics functions.

    All fields are single JAX arrays (not tuples). Constructed by
    get_volume_deposits(deposits, vol_idx) inside the JIT volume loop.

    No valid_mask field — padding is zeroed via n_actual:
        mask = jnp.arange(total_pad) < n_actual
    Applied once after recombination in compute_volume_physics.
    """
    positions_mm: jnp.ndarray    # (total_pad, 3)
    de: jnp.ndarray              # (total_pad,)
    dx: jnp.ndarray              # (total_pad,)
    theta: jnp.ndarray           # (total_pad,)
    phi: jnp.ndarray             # (total_pad,)
    track_ids: jnp.ndarray       # (total_pad,)
    group_ids: jnp.ndarray       # (total_pad,)
    t0_us: jnp.ndarray           # (total_pad,) initial deposit time in μs
    interaction_ids: jnp.ndarray # (total_pad,) vertex/interaction label (int16)
    ancestor_track_ids: jnp.ndarray  # (total_pad,) primary shower ancestor (int32)
    pdg: jnp.ndarray             # (total_pad,) particle species at step (int32)
    charge: jnp.ndarray          # (total_pad,) recombined electrons (zeros on input, filled after sim)
    photons: jnp.ndarray         # (total_pad,) scintillation photons (zeros on input, filled after sim)
    qs_fractions: jnp.ndarray    # (total_pad,) group charge fraction (zeros on input, filled after sim)
    n_actual: int                 # number of real deposits (rest is padding)


def get_volume_deposits(deposits, vol_idx):
    """Extract single-volume VolumeDeposits from DepositData.

    Called inside the JIT volume loop.
    """
    return deposits.volumes[vol_idx]


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
    pre_window_us: float                    # readout window extension before drift t=0
    post_window_us: float                   # readout window extension after max drift

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

    # Volumes
    n_volumes: int                      # number of detector volumes
    volumes: tuple                      # (VolumeGeometry, ...) per volume

    # Plane names
    plane_names: tuple                  # per-volume: (('U','V','Y'), ('U','V','Y'))

    # Output format: 'dense', 'bucketed', or 'wire_sparse'
    output_format: str

    # Readout
    electrons_per_adc: float            # Conversion factor (e.g., 182 for MicroBooNE)
    noise_spectrum_path: str            # Path to noise_spectrum.npz

    # Optional (None when disabled/not needed)
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
    """Create DigitizationConfig with specified parameters."""
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
    """Create TrackHitsConfig with specified parameters."""
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

    Reads per-volume geometry from the 'volumes' array in the config.
    Each volume gets its own VolumeGeometry with per-volume DiffusionConfig.

    Parameters
    ----------
    detector_config : dict
        Raw parsed YAML from generate_detector().
    total_pad : int
        Fixed pad size per volume. Default 200,000.
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
        If True, compute DiffusionConfig per volume.
    num_s : int
        Number of diffusion levels for kernel interpolation. Default 16.
    """
    from tools.geometry import (
        get_drift_velocity, get_plane_geometry_for_volume,
        get_single_plane_wire_params, calculate_max_diffusion_sigmas,
        _calculate_wire_lengths_for_volume,
    )

    volumes_cfg = detector_config['volumes']
    n_volumes = len(volumes_cfg)

    # Global parameters
    velocity = get_drift_velocity(detector_config)
    sampling_rate = float(detector_config['readout']['sampling_rate'])
    time_step_us = 1.0 / sampling_rate

    # Diffusion coefficients (global, used per-volume with per-volume max_drift)
    long_diff = float(detector_config['simulation']['drift']['longitudinal_diffusion']) / 1e6
    trans_diff = float(detector_config['simulation']['drift']['transverse_diffusion']) / 1e6

    # Build VolumeGeometry for each volume
    all_volumes = []
    _PLANE_LABELS = ('U', 'V', 'Y')

    for v, vol_cfg in enumerate(volumes_cfg):
        geo = vol_cfg['geometry']
        ranges = geo['ranges']
        drift_dir = geo['drift_direction']
        planes_cfg = vol_cfg['planes']
        n_planes = len(planes_cfg)

        x_min, x_max = ranges[0]
        # Max drift distance = full x-extent of this volume
        max_drift = x_max - x_min
        # Anode: where electrons arrive
        #   drift_direction == +1 → electrons drift toward +x → anode at x_max
        #   drift_direction == -1 → electrons drift toward -x → anode at x_min
        x_anode = x_max if drift_dir == 1 else x_min

        # Per-volume dimensions (for wire geometry calculation)
        dims_cm = {
            'y': ranges[1][1] - ranges[1][0],
            'z': ranges[2][1] - ranges[2][0],
            'x': x_max - x_min,
        }

        # Plane geometry
        plane_distances, furthest_idx = get_plane_geometry_for_volume(planes_cfg)

        # Wire parameters per plane
        angles = []
        spacings = []
        offsets = []
        n_wires_list = []
        max_wire_list = []
        for p, plane_cfg in enumerate(planes_cfg):
            angle, spacing, offset, n_wires, max_wire = get_single_plane_wire_params(
                plane_cfg, dims_cm)
            angles.append(angle)
            spacings.append(spacing)
            offsets.append(offset)
            n_wires_list.append(n_wires)
            max_wire_list.append(max_wire)

        # Wire lengths
        wire_lengths = _calculate_wire_lengths_for_volume(
            dims_cm,
            angles, spacings, offsets, n_wires_list)

        # Per-volume DiffusionConfig
        if include_diffusion:
            wire_spacing_ref = spacings[0]
            _, _, max_sigma_trans, max_sigma_long = calculate_max_diffusion_sigmas(
                max_drift, velocity, trans_diff, long_diff,
                wire_spacing_ref, time_step_us)
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

        all_volumes.append(VolumeGeometry(
            volume_id=v,
            ranges_cm=tuple(tuple(r) for r in ranges),
            drift_direction=drift_dir,
            x_anode_cm=x_anode,
            max_drift_cm=max_drift,
            n_planes=n_planes,
            furthest_plane_dist_cm=float(plane_distances[furthest_idx]),
            plane_distances_cm=tuple(float(d) for d in plane_distances),
            angles_rad=tuple(angles),
            wire_spacings_cm=tuple(spacings),
            index_offsets=tuple(offsets),
            max_wire_indices=tuple(max_wire_list),
            num_wires=tuple(n_wires_list),
            wire_lengths_m=tuple(wire_lengths),
            diffusion=diffusion,
        ))

    volumes = tuple(all_volumes)

    # Global time params from longest drift across all volumes
    longest_drift = max(vol.max_drift_cm for vol in volumes)
    max_drift_time = longest_drift / velocity

    # Readout window extensions (fraction of max drift time)
    readout_cfg = detector_config.get('readout', {})
    pre_window_frac = float(readout_cfg.get('pre_window_fraction', 0.0))
    post_window_frac = float(readout_cfg.get('post_window_fraction', 0.0))
    pre_window_us = pre_window_frac * max_drift_time
    post_window_us = post_window_frac * max_drift_time
    total_window_us = pre_window_us + max_drift_time + post_window_us

    num_time_steps = int(np.ceil(total_window_us / time_step_us)) + 1
    num_time_steps = max(1, num_time_steps)

    # Plane names per volume
    plane_names = tuple(
        tuple(_PLANE_LABELS[:vol.n_planes]) for vol in volumes)

    # Track hits config
    if include_track_hits:
        if track_config is None:
            track_config = create_track_hits_config()
        max_wires = max(
            w for vol in volumes for w in vol.max_wire_indices) + 1
        max_wires = max(max_wires, 2000)
    else:
        track_config = None
        max_wires = 0

    # Output format
    if use_bucketed and include_electronics:
        output_format = 'wire_sparse'
    elif use_bucketed:
        output_format = 'bucketed'
    else:
        output_format = 'dense'

    return SimConfig(
        num_time_steps=num_time_steps,
        time_step_us=time_step_us,
        pre_window_us=pre_window_us,
        post_window_us=post_window_us,
        total_pad=total_pad,
        response_chunk_size=response_chunk_size,
        max_wires=max_wires,
        use_bucketed=use_bucketed,
        include_track_hits=include_track_hits,
        include_noise=include_noise,
        include_electronics=include_electronics,
        include_digitize=include_digitize,
        max_active_buckets=max_active_buckets,
        n_volumes=n_volumes,
        volumes=volumes,
        plane_names=plane_names,
        output_format=output_format,
        electrons_per_adc=float(detector_config['readout'].get('electrons_per_adc', 182)),
        noise_spectrum_path=str(
            Path(__file__).parent.parent / 'config' / 'noise_spectrum.npz'),
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
    response_models: Any            # dict {(vol_idx, plane_type): eqx.Module} or None
    sce_models: Any                 # tuple of per-volume models, or None


class SCEOutputs(NamedTuple):
    """Raw outputs from SCE correction map query."""
    efield_correction: jnp.ndarray  # (N, 3) dimensionless, E_local / |E_nominal|
    drift_corr_cm: jnp.ndarray     # (N, 3) drift corrections [dx, dy, dz] in cm


class VolumeGeometry(NamedTuple):
    """Static geometry for one detector volume."""
    volume_id: int                  # 0, 1, ...
    ranges_cm: tuple                # ((x_min,x_max), (y_min,y_max), (z_min,z_max))
    drift_direction: int            # +1 or -1
    x_anode_cm: float               # derived from ranges + drift_direction
    max_drift_cm: float            # x_max - x_min (max drift distance in this volume)
    n_planes: int                   # number of readout planes
    furthest_plane_dist_cm: float
    plane_distances_cm: tuple       # (n_planes,) per plane
    angles_rad: tuple               # (n_planes,)
    wire_spacings_cm: tuple         # (n_planes,)
    index_offsets: tuple            # (n_planes,) int
    max_wire_indices: tuple         # (n_planes,) int
    num_wires: tuple                # (n_planes,) int
    wire_lengths_m: tuple           # (n_planes,) of np.ndarray, wire lengths in meters
    diffusion: Any                  # DiffusionConfig or None


class VolumeIntermediates(NamedTuple):
    """Output of compute_volume_physics, input to compute_plane_physics."""
    charges: jnp.ndarray            # (N,) zeroed for invalid deposits (valid_mask applied)
    photons: jnp.ndarray            # (N,) scintillation photons (valid_mask applied)
    drift_distance_cm: jnp.ndarray  # (N,)
    drift_time_us: jnp.ndarray     # (N,)
    positions_cm: jnp.ndarray      # (N, 3) original positions (for NN response)
    positions_yz_cm: jnp.ndarray   # (N, 2) projected (for wire distances)
    t0_us: jnp.ndarray             # (N,) initial deposit time in μs


class PlaneIntermediates(NamedTuple):
    """Output of compute_plane_physics, input to response computation."""
    drift_distance_cm: jnp.ndarray  # (N,) SCE-corrected, plane-corrected
    drift_time_us: jnp.ndarray     # (N,) pure drift time (for diffusion sigma)
    tick_us: jnp.ndarray           # (N,) readout tick time = drift + t0 + pre_window
    attenuation: jnp.ndarray       # (N,)
    closest_wire_idx: jnp.ndarray  # (N,) int
    closest_wire_dist: jnp.ndarray # (N,)
    charges: jnp.ndarray           # (N,) zeroed for invalid/out-of-window deposits
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
        {(vol_idx, plane_type): eqx.Module} for NN response, or None for DKernel.
    sce_models : tuple or None
        Per-volume tuple of SCE models, or None for HDF5/nominal.
    """
    from tools.geometry import get_drift_velocity
    velocity_cm_us = get_drift_velocity(detector_config)
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
                        track_ids=None, group_ids=None, t0_us=None,
                        interaction_ids=None, ancestor_track_ids=None,
                        pdg=None, n_volumes=1):
    """Convenience constructor for single-volume DepositData.

    Wraps arrays into 1-element tuples (single volume). For multi-volume
    data, use prepare_event() from loader.py instead.

    Required: positions_mm (N,3), de (N,), dx (N,) or scalar.
    Optional: theta, phi (default zeros), track_ids, group_ids, t0_us (default zeros).
    """
    N = positions_mm.shape[0]
    dx_arr = jnp.full(N, dx) if jnp.ndim(dx) == 0 else dx
    th = theta if theta is not None else jnp.zeros(N)
    ph = phi if phi is not None else jnp.zeros(N)
    tids = track_ids if track_ids is not None else jnp.full(N, -1, jnp.int32)
    gids = group_ids if group_ids is not None else jnp.zeros(N, jnp.int32)
    t0 = t0_us if t0_us is not None else jnp.zeros(N)
    iids = interaction_ids if interaction_ids is not None else jnp.full(N, -1, jnp.int16)
    atids = ancestor_track_ids if ancestor_track_ids is not None else jnp.full(N, -1, jnp.int32)
    pdg_arr = pdg if pdg is not None else jnp.zeros(N, jnp.int32)

    vol = VolumeDeposits(
        positions_mm=positions_mm, de=de, dx=dx_arr,
        theta=th, phi=ph, track_ids=tids, group_ids=gids, t0_us=t0,
        interaction_ids=iids, ancestor_track_ids=atids, pdg=pdg_arr,
        charge=jnp.zeros(N), photons=jnp.zeros(N), qs_fractions=jnp.zeros(N),
        n_actual=N,
    )
    vols = (vol,) * n_volumes
    return DepositData(
        volumes=vols,
        group_to_track=(None,) * n_volumes,
        original_indices=(None,) * n_volumes,
    )



def pad_deposit_data(deposits, target_size):
    """Pad each volume's VolumeDeposits arrays to target_size.

    Used by forward() to pad to total_pad (next multiple of chunk_size).
    Padding entries have de=0, dx=1 (avoids division by zero in recombination).
    Padding is zeroed after recombination via n_actual mask in compute_volume_physics.
    """
    def _pad(arr, pad_val=0):
        N = arr.shape[0]
        pad_size = target_size - N
        if pad_size <= 0:
            return arr
        if arr.ndim == 2:
            return jnp.pad(arr, ((0, pad_size), (0, 0)), constant_values=pad_val)
        return jnp.pad(arr, (0, pad_size), constant_values=pad_val)

    def _pad_vol(vol):
        return VolumeDeposits(
            positions_mm=_pad(vol.positions_mm),
            de=_pad(vol.de),
            dx=_pad(vol.dx, pad_val=1.0),
            theta=_pad(vol.theta),
            phi=_pad(vol.phi),
            track_ids=_pad(vol.track_ids, pad_val=-1),
            group_ids=_pad(vol.group_ids),
            t0_us=_pad(vol.t0_us),
            interaction_ids=_pad(vol.interaction_ids, pad_val=-1),
            ancestor_track_ids=_pad(vol.ancestor_track_ids, pad_val=-1),
            pdg=_pad(vol.pdg),
            charge=_pad(vol.charge),
            photons=_pad(vol.photons),
            qs_fractions=_pad(vol.qs_fractions),
            n_actual=vol.n_actual,
        )

    return DepositData(
        volumes=tuple(_pad_vol(v) for v in deposits.volumes),
        group_to_track=deposits.group_to_track,
        original_indices=deposits.original_indices,
    )
