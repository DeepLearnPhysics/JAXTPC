"""
LArTPC detector simulation with modular physics pipeline.

Deposits are transformed to volume-local coordinates by the loader:
    x_local = drift_dir * (x_anode - x_global), y/z centered on volume.
All volumes are geometrically identical in local frame — one scan/vmap
body handles any number of volumes.

Execution paths:
    - process_event(deposits, sim_params): Production path with fori_loop
      batching, optional noise/electronics/track_hits/digitization.
    - forward(params, deposits): Differentiable path with remat, gradients
      through SimParams fields (velocity, lifetime, diffusion, recomb).

SimParams is a JIT argument — changing physics values does NOT trigger
recompilation. Volume iteration uses lax.scan (default) or vmap.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp

# Config types and factories
from tools.config import (
    DepositData, VolumeDeposits, SCEOutputs,
    create_sim_params, create_sim_config,
    create_deposit_data, pad_deposit_data,
)

# Data construction
from tools.loader import build_deposit_data, load_event

# Response kernels (loaded at __init__, used by shared factory)
from tools.kernels import (
    load_response_kernels, apply_diffusion_response, generate_dkernel_table,
    load_pixel_response_kernel, apply_pixel_diffusion_response,
)

# Track labeling factory + post-processing
from tools.track_hits import create_track_hits_fn_for_volume, compute_qs_fractions

# Recombination
from tools.recombination import RECOMB_MODELS

# Post-processing factories
from tools.noise import create_noise_fn_for_volume
from tools.electronics import create_electronics_fn_for_volume, create_digitize_fn_for_volume


# =============================================================================
# VOLUME ITERATION STRATEGIES
# =============================================================================

def scan_over(fn, xs):
    """Iterate fn over leading-axis arrays using lax.scan (sequential)."""
    def body(carry, inputs):
        return carry, fn(*inputs)
    _, outputs = jax.lax.scan(body, None, xs)
    return outputs

def vmap_over(fn, xs):
    """Iterate fn over leading-axis arrays using vmap (parallel)."""
    return jax.vmap(lambda *args: fn(*args))(*xs)


# =============================================================================
# DETECTOR SIMULATOR CLASS
# =============================================================================

class DetectorSimulator:
    """
    LArTPC detector simulation with fixed-size padding.

    All volumes share identical structural geometry (validated at init).
    Deposits arrive in local coordinates (anode at x=0, yz centered).
    Volume iteration via lax.scan (default) or vmap.
    """

    def __init__(
        self,
        detector_config,
        response_path=None,
        track_config=None,
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
        iterate_mode='scan',
    ):
        print("--- Creating DetectorSimulator ---")

        # Differentiable mode: force compatible flags
        if n_segments is not None and not differentiable:
            print("   WARNING: n_segments ignored without differentiable=True")
        self.n_segments = n_segments if differentiable else None
        if differentiable:
            if n_segments is None:
                raise ValueError("differentiable=True requires n_segments to be set")
            if total_pad != 200_000:
                print(f"   NOTE: differentiable=True overrides total_pad={total_pad:,} → {n_segments:,}")
            total_pad = n_segments
            response_chunk_size = n_segments
            use_bucketed = False
            include_noise = False
            include_electronics = False
            include_track_hits = False
            include_digitize = False

        # Validate chunk alignment
        if total_pad % response_chunk_size != 0:
            raise ValueError(
                f"total_pad ({total_pad:,}) must be divisible by "
                f"response_chunk_size ({response_chunk_size:,}).")

        # Resolve recombination model name
        self.recomb_model = (recombination_model
                             or detector_config['simulation']['charge_recombination']['model'])
        if self.recomb_model not in RECOMB_MODELS:
            raise ValueError(
                f"Unknown recombination model: '{self.recomb_model}'. "
                f"Must be one of: {RECOMB_MODELS}")

        print("   Extracting parameters...")

        # Build SimConfig
        self._sim_config = create_sim_config(
            detector_config,
            total_pad=total_pad,
            response_chunk_size=response_chunk_size,
            use_bucketed=use_bucketed,
            max_active_buckets=max_active_buckets,
            include_noise=include_noise,
            include_electronics=include_electronics,
            include_track_hits=include_track_hits,
            include_digitize=include_digitize,
            track_config=track_config,
        )
        cfg = self._sim_config

        # Validate chunk alignment
        if include_track_hits and cfg.track_hits is not None:
            if total_pad % cfg.track_hits.hits_chunk_size != 0:
                raise ValueError(
                    f"total_pad ({total_pad:,}) must be divisible by "
                    f"hits_chunk_size ({cfg.track_hits.hits_chunk_size:,}).")

        # Default SimParams (tunable physics — JIT argument)
        self._default_sim_params = create_sim_params(
            detector_config,
            recombination_model=self.recomb_model,
        )

        # Load SCE maps (per-volume, converted to local frame)
        sce_per_volume = self._load_sce(include_electric_dist, electric_dist_path)
        self._include_sce = sce_per_volume is not None

        # Volume iteration mode
        self._iterate = scan_over if iterate_mode == 'scan' else vmap_over
        self._volume_mode = iterate_mode

        # Readout type (all volumes must match)
        vol_geom = cfg.volumes[0]
        self._readout_type = vol_geom.readout_type
        for v in cfg.volumes[1:]:
            if v.readout_type != self._readout_type:
                raise ValueError(
                    f"Mixed readout types not supported: volume 0 is "
                    f"'{self._readout_type}', volume {v.volume_id} is '{v.readout_type}'")

        # Pixel readout requires bucketed accumulation (dense is too large)
        if self._readout_type == 'pixel' and not cfg.use_bucketed:
            print("   NOTE: pixel readout forces bucketed accumulation")
            buckets = cfg.max_active_buckets if cfg.max_active_buckets else 1000
            self._sim_config = cfg._replace(use_bucketed=True,
                                            max_active_buckets=buckets)
            cfg = self._sim_config

        # Load response kernels (once, shared across all volumes)
        print("   Loading response kernels...")
        from tools.geometry import calculate_max_diffusion_sigmas
        d = vol_geom.diffusion
        global_max_drift = max(v.max_drift_cm for v in cfg.volumes)
        self._global_max_drift = global_max_drift

        if self._readout_type == 'pixel':
            pixel_response_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'pixel_response.npz')
            if response_path is not None:
                candidate = os.path.join(response_path, 'pixel_response.npz')
                if os.path.exists(candidate):
                    pixel_response_path = candidate
            pixlar_path = '/tmp/pixlar-detsim-jax/pixlar/data/pixel_response_shielded.npz'
            if not os.path.exists(pixel_response_path) and os.path.exists(pixlar_path):
                pixel_response_path = pixlar_path
            self.response_kernels = load_pixel_response_kernel(
                pixel_response_path,
                num_s=d.num_s,
                time_spacing=cfg.time_step_us,
                pixel_pitch_cm=vol_geom.pixel_pitch_cm,
                max_sigma_trans_unitless=d.max_sigma_trans_unitless,
                max_sigma_long_unitless=d.max_sigma_long_unitless,
            )
        else:
            _, _, global_sigma_trans, global_sigma_long = calculate_max_diffusion_sigmas(
                global_max_drift, d.velocity_cm_us, d.trans_cm2_us, d.long_cm2_us,
                vol_geom.wire_spacings_cm[0], cfg.time_step_us)
            self.response_kernels = load_response_kernels(
                response_path=response_path,
                num_s=d.num_s,
                time_spacing=cfg.time_step_us,
                max_sigma_trans_unitless=global_sigma_trans,
                max_sigma_long_unitless=global_sigma_long,
            )

        # Build shared factories
        sce_factory, _build_response_fn, _build_response_fn_diff, _recomb_fn = \
            self._setup_shared_factories(sce_per_volume)

        # Build post-processing factories (once, shared)
        self.electronics_chunk_size = None
        self._electronics_fft_size = None

        e_fn, e_meta = create_electronics_fn_for_volume(
            cfg, vol_geom, self.response_kernels if self._readout_type == 'wire' else None,
            electronics_chunk_size=electronics_chunk_size,
            electronics_threshold=electronics_threshold)
        if e_meta.get('e_chunk'):
            self.electronics_chunk_size = e_meta['e_chunk']
            self._electronics_fft_size = e_meta.get('e_fft')

        noise_fn = create_noise_fn_for_volume(
            cfg, vol_geom,
            self.response_kernels if self._readout_type == 'wire' else None,
            e_chunk=self.electronics_chunk_size)

        d_fn, dig_cfg = create_digitize_fn_for_volume(cfg, vol_geom, digitization_config)
        if dig_cfg:
            self.digitization_config = dig_cfg

        th_fn, th_zero, th_decode = create_track_hits_fn_for_volume(cfg, vol_geom)

        # Populate spatial decode fns for finalize_track_hits
        self._spatial_decode_fns = {}
        n_readouts = vol_geom.n_planes if self._readout_type == 'wire' else 1
        for vi in range(cfg.n_volumes):
            for pi in range(n_readouts):
                self._spatial_decode_fns[(vi, pi)] = th_decode

        # Build JIT-compiled calculators
        self._build_jit(
            _recomb_fn, sce_factory, _build_response_fn, _build_response_fn_diff,
            e_fn, noise_fn, d_fn, th_fn,
        )

        self._print_summary()

    def _load_sce(self, include_electric_dist, electric_dist_path):
        """Load per-volume SCE maps. Returns list of (efield_fn, corr_fn) or None."""
        if not include_electric_dist:
            return None
        from tools.efield_distortions import load_sce_per_volume
        if electric_dist_path is None:
            electric_dist_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'config', 'sce_jaxtpc.h5')
        print(f"   Loading SCE maps from {electric_dist_path}...")
        return load_sce_per_volume(electric_dist_path, volumes=self._sim_config.volumes)

    def _print_summary(self):
        """Print configuration summary after initialization."""
        cfg = self._sim_config
        if self._include_sce:
            print("   Space charge effects: ENABLED")
        else:
            print("   Space charge effects: DISABLED")
        print(f"   Recombination model: {self.recomb_model}")
        d0 = cfg.volumes[0].diffusion
        print(f"   Config: total_pad={cfg.total_pad:,}, response_chunk={cfg.response_chunk_size:,}, "
              f"num_s={d0.num_s if d0 else 'N/A'}, "
              f"K_wire={d0.K_wire if d0 else 'N/A'}, "
              f"K_time={d0.K_time if d0 else 'N/A'}")
        if cfg.use_bucketed:
            print(f"   Using BUCKETED accumulation (max_buckets={cfg.max_active_buckets})")
        if cfg.include_noise:
            print("   Noise integration: ENABLED")
        if cfg.include_electronics:
            print(f"   Electronics response: ENABLED (FFT={self._electronics_fft_size}, "
                  f"chunk={self.electronics_chunk_size}, output={cfg.output_format})")
        if cfg.include_digitize and hasattr(self, 'digitization_config') and self.digitization_config:
            dig_cfg = self.digitization_config
            print(f"   Digitization: ENABLED ({dig_cfg.n_bits}-bit)")
        print(f"   Track labeling: {'ENABLED' if cfg.include_track_hits else 'DISABLED'}")
        print(f"   Readout: {self._readout_type}")
        print(f"   Volumes: {cfg.n_volumes} (iterate={self._volume_mode})")
        print("--- DetectorSimulator Ready ---")

    def _setup_shared_factories(self, sce_per_volume):
        """Build SCE, response, and recombination factories (shared across volumes)."""
        from tools.recombination import compute_quanta, XI_FN

        cfg = self._sim_config
        _nominal_field = float(self._default_sim_params.recomb_params.field_strength_Vcm)

        # ── SCE factory (local frame) ──
        if sce_per_volume is not None:
            # Real SCE — maps already in local frame from load_sce_per_volume.
            # For scan, all volumes must share one factory. Use volume 0's maps.
            # (Different per-volume SCE requires stacking maps — future work.)
            efield_fn, corr_fn = sce_per_volume[0]
            def sce_factory(ef=efield_fn, cf=corr_fn, nf=_nominal_field):
                def _sce(positions_cm):
                    E_local = ef(positions_cm)
                    E_normalized = E_local / nf
                    drift_corr = cf(positions_cm)
                    return SCEOutputs(efield_correction=E_normalized, drift_corr_cm=drift_corr)
                return _sce
        else:
            # Nominal SCE — identical for all volumes in local frame
            def sce_factory():
                def _sce(pos):
                    N = pos.shape[0]
                    corr = jnp.broadcast_to(
                        jnp.array([1.0, 0.0, 0.0]), (N, 3))
                    return SCEOutputs(
                        efield_correction=corr,
                        drift_corr_cm=jnp.zeros((N, 3)))
                return _sce

        # ── Response factories ──
        kernels = self.response_kernels
        _global_max_drift = self._global_max_drift

        if self._readout_type == 'pixel':
            pk = kernels  # PixelResponseKernel
            def _build_response_fn(sim_params):
                dkernel = pk.DKernel
                def response_fn(positions_cm, drift_distance_cm,
                                py_offsets, pz_offsets, time_offsets):
                    s_values = jnp.clip(jnp.sqrt(drift_distance_cm / _global_max_drift), 0.0, 1.0)
                    return apply_pixel_diffusion_response(
                        dkernel, s_values, py_offsets, pz_offsets, time_offsets,
                        pk.pixel_spacing, pk.kernel_py, pk.kernel_pz, pk.rebin_factor)
                return response_fn
            _build_response_fn_diff = _build_response_fn  # TODO: pixel diff response
        else:
            def _build_response_fn(sim_params, plane_type):
                kernel = kernels[plane_type]
                dkernel = kernel.DKernel
                def response_fn(positions_cm, drift_distance_cm, wire_offsets, time_offsets):
                    s_values = jnp.clip(jnp.sqrt(drift_distance_cm / _global_max_drift), 0.0, 1.0)
                    return apply_diffusion_response(
                        dkernel, s_values, wire_offsets, time_offsets,
                        kernel.wire_spacing, kernel.num_wires)
                return response_fn

            def _build_response_fn_diff(sim_params, plane_type):
                """Diff response — recomputes DKernel from SimParams diffusion.
                Conv filter sizes (ks_w, ks_t) are static from ResponseKernel."""
                kernel = kernels[plane_type]
                max_drift_time = _global_max_drift / sim_params.velocity_cm_us
                sigma_trans_max_cm = jnp.sqrt(
                    2.0 * sim_params.diffusion_trans_cm2_us * max_drift_time)
                sigma_long_max_us = jnp.sqrt(
                    2.0 * (sim_params.diffusion_long_cm2_us
                           / sim_params.velocity_cm_us**2) * max_drift_time)
                dkernel = generate_dkernel_table(
                    sigma_trans_max_cm, sigma_long_max_us,
                    kernel.base_kernel, kernel.kernel_dx, kernel.kernel_dy,
                    kernel.s_levels, ks_w=kernel.ks_w, ks_t=kernel.ks_t)
                def response_fn(positions_cm, drift_distance_cm, wire_offsets, time_offsets):
                    s_values = jnp.clip(jnp.sqrt(drift_distance_cm / _global_max_drift), 0.0, 1.0)
                    return apply_diffusion_response(
                        dkernel, s_values, wire_offsets, time_offsets,
                        kernel.wire_spacing, kernel.num_wires)
                return response_fn

        # ── Recombination ──
        _xi_fn = XI_FN[self.recomb_model]
        def _recomb_fn(de, dx, phi_drift, e_field_Vcm, params):
            return compute_quanta(de, dx, phi_drift, e_field_Vcm, params, _xi_fn)

        return sce_factory, _build_response_fn, _build_response_fn_diff, _recomb_fn

    def _build_jit(self, _recomb_fn, sce_factory, _build_response_fn,
                   _build_response_fn_diff,
                   electronics_fn, noise_fn, digitize_fn, track_hits_fn):
        """Build all JIT-compiled calculators using scan/vmap."""
        from tools.physics import (
            compute_volume_physics, compute_plane_physics,
            compute_plane_signal, compute_plane_signal_bucketed,
            compute_bucket_maps,
            compute_pixel_physics, compute_pixel_bucket_maps,
            compute_pixel_signal_bucketed,
        )

        cfg = self._sim_config
        vol_geom = cfg.volumes[0]
        kernels = self.response_kernels
        total_pad = cfg.total_pad
        n_volumes = cfg.n_volumes
        iterate = self._iterate
        include_track_hits = cfg.include_track_hits

        # ── Build process_one_volume body ──
        if self._readout_type == 'pixel':
            pk = kernels  # PixelResponseKernel

            def process_one_volume(vol_deps, vol_key, sim_params):
                sce_fn = sce_factory()
                vol_int = compute_volume_physics(
                    vol_deps, sim_params, vol_geom, sce_fn, _recomb_fn)

                readout_window_us = cfg.num_time_steps * cfg.time_step_us
                pixel_response_fn = _build_response_fn(sim_params)

                pixel_int = compute_pixel_physics(
                    vol_int, sim_params, vol_geom,
                    cfg.pre_window_us, readout_window_us,
                    vol_geom.pixel_pitch_cm,
                    jnp.array(vol_geom.pixel_origins_cm),
                    vol_geom.pixel_shape[0], vol_geom.pixel_shape[1])

                ptc, num_active, ctk, B1, B2, B3 = compute_pixel_bucket_maps(
                    pixel_int, vol_geom.pixel_shape[0], vol_geom.pixel_shape[1],
                    cfg.num_time_steps, cfg.time_step_us,
                    cfg.max_active_buckets,
                    pk.kernel_py, pk.kernel_pz, pk.kernel_time,
                    pk.py_zero_bin, pk.pz_zero_bin, pk.time_zero_bin)

                response_buckets = compute_pixel_signal_bucketed(
                    pixel_int, pixel_response_fn, vol_deps.n_actual,
                    cfg.response_chunk_size,
                    ptc, cfg.max_active_buckets, B1, B2, B3,
                    cfg.time_step_us,
                    vol_geom.pixel_shape[0], vol_geom.pixel_shape[1],
                    cfg.num_time_steps,
                    pk.kernel_py, pk.kernel_pz, pk.kernel_time,
                    pk.py_zero_bin, pk.pz_zero_bin, pk.time_zero_bin)

                # Single readout plane
                stacked_signal = (response_buckets[None], num_active[None], ctk[None])

                if include_track_hits:
                    vol_qs = compute_qs_fractions(
                        vol_int.charges, vol_deps.group_ids, total_pad)
                    hits = track_hits_fn(
                        pixel_int, vol_deps, vol_geom, 0, vol_deps.n_actual)
                    stacked_hits = tuple(h[None] for h in hits)
                else:
                    vol_qs = jnp.zeros(total_pad, dtype=jnp.float32)
                    stacked_hits = ()

                return stacked_signal, stacked_hits, vol_qs, vol_int.charges, vol_int.photons

        else:
            # Wire readout
            n_planes = vol_geom.n_planes
            _PLANE_LABELS = tuple(cfg.plane_names[0])

            def process_one_volume(vol_deps, vol_key, sim_params):
                sce_fn = sce_factory()
                vol_int = compute_volume_physics(
                    vol_deps, sim_params, vol_geom, sce_fn, _recomb_fn)

                readout_window_us = cfg.num_time_steps * cfg.time_step_us
                plane_keys = jax.random.split(vol_key, n_planes)

                plane_signals = []
                plane_intermediates = []
                for plane_idx in range(n_planes):
                    plane_type = _PLANE_LABELS[plane_idx]
                    plane_int = compute_plane_physics(
                        vol_int, sim_params, vol_geom, plane_idx,
                        cfg.pre_window_us, readout_window_us)

                    if include_track_hits:
                        plane_intermediates.append(plane_int)

                    response_fn = _build_response_fn(sim_params, plane_type)
                    plane_kernel = kernels[plane_type]

                    if cfg.use_bucketed:
                        ptc, num_active, ctk, B1, B2 = compute_bucket_maps(
                            plane_int, vol_geom, plane_idx, cfg, plane_kernel)
                        response_buckets = compute_plane_signal_bucketed(
                            plane_int, response_fn, vol_deps.n_actual,
                            cfg.response_chunk_size,
                            ptc, cfg.max_active_buckets, B1, B2,
                            cfg, vol_geom, plane_idx, plane_kernel)
                        response_signal = (response_buckets, num_active, ctk, B1, B2)
                    else:
                        response_signal = compute_plane_signal(
                            plane_int, response_fn, vol_deps.n_actual,
                            cfg.response_chunk_size,
                            cfg, vol_geom, plane_idx, plane_kernel)

                    response_signal = electronics_fn(
                        response_signal, plane_idx,
                        vol_geom.num_wires[plane_idx], cfg.num_time_steps)
                    response_signal = noise_fn(
                        plane_keys[plane_idx], response_signal, plane_idx,
                        vol_geom.num_wires[plane_idx], cfg.num_time_steps)
                    response_signal = digitize_fn(response_signal, plane_idx)
                    plane_signals.append(response_signal)

                # Stack plane signals
                if cfg.use_bucketed:
                    stacked_signal = (
                        jnp.stack([s[0] for s in plane_signals]),
                        jnp.stack([s[1] for s in plane_signals]),
                        jnp.stack([s[2] for s in plane_signals]))
                else:
                    # Pad to max wires so planes can be stacked
                    max_w = max(vol_geom.num_wires)
                    padded = [jnp.pad(s, ((0, max_w - s.shape[0]), (0, 0)))
                              for s in plane_signals]
                    stacked_signal = (jnp.stack(padded),)

                # Track hits
                if include_track_hits:
                    vol_qs = compute_qs_fractions(
                        vol_int.charges, vol_deps.group_ids, total_pad)
                    hits_list = []
                    for plane_idx in range(n_planes):
                        hits = track_hits_fn(
                            plane_intermediates[plane_idx], vol_deps, vol_geom,
                            plane_idx, vol_deps.n_actual)
                        hits_list.append(hits)
                    stacked_hits = tuple(
                        jnp.stack([h[i] for h in hits_list])
                        for i in range(len(hits_list[0])))
                else:
                    vol_qs = jnp.zeros(total_pad, dtype=jnp.float32)
                    stacked_hits = ()

                return stacked_signal, stacked_hits, vol_qs, vol_int.charges, vol_int.photons

        # ── Production JIT ──
        @jax.jit
        def _calculator_jit(sim_params, stacked_deps, noise_key):
            vol_keys = jax.random.split(noise_key, n_volumes)
            fn = lambda deps, key: process_one_volume(deps, key, sim_params)
            return iterate(fn, (stacked_deps, vol_keys))

        self._calculator_jit = _calculator_jit

        # ── Light-only JIT ──
        def light_one_volume(vol_deps, sim_params):
            sce_fn = sce_factory()
            vol_int = compute_volume_physics(
                vol_deps, sim_params, vol_geom, sce_fn, _recomb_fn)
            return (vol_int.charges, vol_int.photons, vol_int.positions_cm,
                    vol_int.interaction_ids, vol_int.ancestor_track_ids)

        @jax.jit
        def _light_calculator_jit(sim_params, stacked_deps):
            fn = lambda deps: light_one_volume(deps, sim_params)
            return iterate(fn, (stacked_deps,))

        self._light_calculator_jit = _light_calculator_jit

        # ── Differentiable path ──
        if self.n_segments is not None and self._readout_type == 'wire':
            n_segments = self.n_segments

            def diff_one_volume(vol_deps, sim_params):
                sce_fn = sce_factory()
                vol_int = compute_volume_physics(
                    vol_deps, sim_params, vol_geom, sce_fn, _recomb_fn)

                readout_window_us = cfg.num_time_steps * cfg.time_step_us
                plane_signals = []
                for plane_idx in range(vol_geom.n_planes):
                    plane_type = _PLANE_LABELS[plane_idx]
                    plane_int = compute_plane_physics(
                        vol_int, sim_params, vol_geom, plane_idx,
                        cfg.pre_window_us, readout_window_us)
                    response_fn = _build_response_fn_diff(sim_params, plane_type)
                    plane_kernel = kernels[plane_type]
                    signal = compute_plane_signal(
                        plane_int, response_fn, n_segments,
                        cfg.response_chunk_size,
                        cfg, vol_geom, plane_idx, plane_kernel)
                    plane_signals.append(signal)
                max_w = max(vol_geom.num_wires)
                padded = [jnp.pad(s, ((0, max_w - s.shape[0]), (0, 0)))
                          for s in plane_signals]
                return jnp.stack(padded)

            @jax.remat
            def _forward_diff(params, stacked_deps):
                fn = lambda deps: diff_one_volume(deps, params)
                return iterate(fn, (stacked_deps,))

            self._forward_diff = _forward_diff

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def __call__(self, deposits: DepositData, key=None):
        return self.process_event(deposits, key=key)

    def process_event(self, deposits: DepositData, sim_params=None, key=None):
        """Run production simulation.

        Parameters
        ----------
        deposits : DepositData
            Pre-split, padded, grouped deposits from build_deposit_data or load_event.
            Positions must be in local coordinates.
        sim_params : SimParams, optional
        key : jax PRNGKey, optional

        Returns
        -------
        response_signals : dict
            Keyed by (vol_idx, plane_idx).
        track_hits : dict
            Keyed by (vol_idx, plane_idx). Contains raw merge state.
        deposits : DepositData
            Input deposits with charge, photons, qs_fractions filled.
        """
        if sim_params is None:
            sim_params = self._default_sim_params

        noise_key = key if key is not None else jax.random.PRNGKey(0)
        cfg = self._sim_config
        n_volumes = cfg.n_volumes
        n_readouts = cfg.volumes[0].n_planes if self._readout_type == 'wire' else 1

        # Stack and run
        stacked_deps = jax.tree.map(lambda *xs: jnp.stack(xs), *deposits.volumes)
        raw_out = self._calculator_jit(sim_params, stacked_deps, noise_key)
        stacked_signal, stacked_hits, all_qs, all_charges, all_photons = raw_out

        # Unstack signals
        response_signals = {}
        if cfg.use_bucketed:
            out_buckets, out_num_active, out_ctk = stacked_signal
            if self._readout_type == 'wire':
                pk = self.response_kernels[cfg.plane_names[0][0]]
                B1 = 2 * pk.num_wires
                B2 = 2 * pk.kernel_height
                for v in range(n_volumes):
                    for p in range(n_readouts):
                        response_signals[(v, p)] = (
                            out_buckets[v, p], out_num_active[v, p],
                            out_ctk[v, p], B1, B2)
            else:
                pk = self.response_kernels
                B1 = 2 * pk.kernel_py
                B2 = 2 * pk.kernel_pz
                B3 = 2 * pk.kernel_time
                for v in range(n_volumes):
                    response_signals[(v, 0)] = (
                        out_buckets[v, 0], out_num_active[v, 0],
                        out_ctk[v, 0], B1, B2, B3)

            # Check for max_active_buckets overflow
            for (v, p), sig in response_signals.items():
                na = int(sig[1])
                if na >= cfg.max_active_buckets:
                    raise RuntimeError(
                        f"Bucket overflow vol {v} plane {p}: "
                        f"num_active={na:,} >= max_active_buckets={cfg.max_active_buckets:,}. "
                        f"Increase --max-buckets.")
        else:
            (out_dense,) = stacked_signal
            vol_geom = cfg.volumes[0]
            for v in range(n_volumes):
                for p in range(n_readouts):
                    n_wires = vol_geom.num_wires[p] if self._readout_type == 'wire' else vol_geom.pixel_shape[0]
                    response_signals[(v, p)] = out_dense[v, p, :n_wires]

        # Unstack track hits
        track_hits = {}
        if cfg.include_track_hits and len(stacked_hits) > 0:
            for v in range(n_volumes):
                for p in range(n_readouts):
                    track_hits[(v, p)] = tuple(
                        stacked_hits[i][v, p] for i in range(len(stacked_hits)))
            track_hits['group_to_track'] = deposits.group_to_track

            # Check for max_keys overflow
            if cfg.track_hits is not None:
                max_keys = cfg.track_hits.max_keys
                for key, raw in track_hits.items():
                    if not isinstance(key, tuple):
                        continue
                    v, p = key
                    count = int(raw[4])
                    if count >= max_keys:
                        raise RuntimeError(
                            f"track_hits overflow vol {v} plane {p}: "
                            f"count={count:,} >= max_keys={max_keys:,}. "
                            f"Increase --max-keys or run profiler.setup_production.")

        # Rebuild filled deposits
        filled_volumes = tuple(
            vol._replace(
                charge=all_charges[v],
                photons=all_photons[v],
                qs_fractions=all_qs[v],
            )
            for v, vol in enumerate(deposits.volumes)
        )
        filled_deposits = deposits._replace(volumes=filled_volumes)

        return response_signals, track_hits, filled_deposits

    def process_event_light(self, deposits: DepositData, sim_params=None):
        """Compute per-segment charge and scintillation light only.

        Returns DepositData with charge and photons filled (no wire response).
        """
        if sim_params is None:
            sim_params = self._default_sim_params

        stacked_deps = jax.tree.map(lambda *xs: jnp.stack(xs), *deposits.volumes)
        all_charges, all_photons, all_positions, all_interaction_ids, all_ancestor_track_ids = \
            self._light_calculator_jit(sim_params, stacked_deps)

        filled_volumes = tuple(
            vol._replace(
                charge=all_charges[v],
                photons=all_photons[v],
                interaction_ids=all_interaction_ids[v],
                ancestor_track_ids=all_ancestor_track_ids[v],
            )
            for v, vol in enumerate(deposits.volumes)
        )
        return deposits._replace(volumes=filled_volumes)

    def finalize_track_hits(self, track_hits):
        """Derive track labels from raw group merge state."""
        group_to_track_per_vol = track_hits.pop('group_to_track')
        result = {}
        for (vol_idx, plane_idx), raw in track_hits.items():
            g2t = group_to_track_per_vol[vol_idx]
            from tools.track_hits import label_from_groups
            state_sk, state_tk, state_gk, state_ch, state_count, row_sums = raw
            decode_fn = self._spatial_decode_fns.get((vol_idx, plane_idx))
            labeled = label_from_groups(
                state_sk, state_tk, state_gk, state_ch, state_count,
                g2t, decode_spatial_fn=decode_fn)
            labeled['row_sums'] = row_sums
            result[(vol_idx, plane_idx)] = labeled
        return result

    @property
    def config(self):
        return self._sim_config

    @property
    def default_sim_params(self):
        return self._default_sim_params

    def to_dense(self, response_signals):
        """Convert response signals to dense (W, T) arrays per plane."""
        from tools.output import to_dense
        return to_dense(response_signals, self._sim_config)

    def to_sparse(self, response_signals, threshold_enc=0):
        """Convert response signals to sparse (wire, time, values) per plane."""
        from tools.output import to_sparse
        threshold_adc = threshold_enc / self._sim_config.electrons_per_adc
        return to_sparse(response_signals, self._sim_config, threshold_adc)

    def warm_up(self):
        """Trigger JIT compilation with dummy data."""
        print("Triggering JIT compilation...")
        pad = self._sim_config.total_pad
        n_vol = self._sim_config.n_volumes
        dummy_vol = VolumeDeposits(
            positions_mm=jnp.zeros((pad, 3), dtype=jnp.float32),
            de=jnp.zeros(pad, dtype=jnp.float32),
            dx=jnp.ones(pad, dtype=jnp.float32),
            theta=jnp.zeros(pad, dtype=jnp.float32),
            phi=jnp.zeros(pad, dtype=jnp.float32),
            track_ids=jnp.full(pad, -1, dtype=jnp.int32),
            group_ids=jnp.zeros(pad, dtype=jnp.int32),
            t0_us=jnp.zeros(pad, dtype=jnp.float32),
            interaction_ids=jnp.full(pad, -1, dtype=jnp.int16),
            ancestor_track_ids=jnp.full(pad, -1, dtype=jnp.int32),
            pdg=jnp.zeros(pad, dtype=jnp.int32),
            charge=jnp.zeros(pad, dtype=jnp.float32),
            photons=jnp.zeros(pad, dtype=jnp.float32),
            qs_fractions=jnp.zeros(pad, dtype=jnp.float32),
            n_actual=0,
        )
        stacked_dummy = jax.tree.map(
            lambda x: jnp.broadcast_to(
                jnp.asarray(x)[None],
                (n_vol,) + jnp.asarray(x).shape),
            dummy_vol)
        _ = self._calculator_jit(
            self._default_sim_params, stacked_dummy, jax.random.PRNGKey(0))
        print(f"JIT compilation finished (total_pad={pad:,}, iterate={self._volume_mode}).")

    def forward(self, params, deposits):
        """Differentiable forward pass.

        Parameters
        ----------
        params : SimParams
        deposits : DepositData
            Must be in local coordinates and padded to total_pad.

        Returns
        -------
        tuple of signal arrays, one per (vol_idx, plane_idx).
        """
        cfg = self._sim_config
        deposits = pad_deposit_data(deposits, cfg.total_pad)
        stacked_deps = jax.tree.map(lambda *xs: jnp.stack(xs), *deposits.volumes)
        all_signals = self._forward_diff(params, stacked_deps)
        # all_signals shape: (n_volumes, n_planes, num_wires, num_time)
        return tuple(
            all_signals[v, p]
            for v in range(cfg.n_volumes)
            for p in range(cfg.volumes[v].n_planes))

    def forward_segments(self, params, positions_mm, de, dx):
        """Lightweight differentiable forward for segment-like data.

        Positions are in GLOBAL coordinates — transformed to local per volume
        internally. Can be called inside jax.grad.
        """
        cfg = self._sim_config
        total_pad = cfg.total_pad
        N = positions_mm.shape[0]

        dx_arr = jnp.full(N, dx) if jnp.ndim(dx) == 0 else dx

        def _pad(arr, pad_val=0):
            pad_size = total_pad - N
            if arr.ndim == 2:
                return jnp.pad(arr, ((0, pad_size), (0, 0)), constant_values=pad_val)
            return jnp.pad(arr, (0, pad_size), constant_values=pad_val)

        padded_pos = _pad(positions_mm)
        padded_de = _pad(de)
        padded_dx = _pad(dx_arr, pad_val=1.0)

        # Build per-volume deposits with local coordinate transform
        x_cm = padded_pos[:, 0] / 10.0
        volumes = []
        for vol_idx in range(cfg.n_volumes):
            vol = cfg.volumes[vol_idx]
            x_min, x_max = vol.ranges_cm[0]
            vol_mask = (x_cm >= x_min) & (x_cm < x_max)
            masked_de = padded_de * vol_mask

            # Transform to local coordinates
            x_local = vol.drift_direction * (vol.x_anode_cm * 10.0 - padded_pos[:, 0])
            y_local = padded_pos[:, 1] - vol.yz_center_cm[0] * 10.0
            z_local = padded_pos[:, 2] - vol.yz_center_cm[1] * 10.0
            local_pos = jnp.stack([x_local, y_local, z_local], axis=1)

            volumes.append(VolumeDeposits(
                positions_mm=local_pos,
                de=masked_de,
                dx=padded_dx,
                theta=jnp.zeros(total_pad),
                phi=jnp.zeros(total_pad),
                track_ids=jnp.full(total_pad, -1, dtype=jnp.int32),
                group_ids=jnp.zeros(total_pad, dtype=jnp.int32),
                t0_us=jnp.zeros(total_pad),
                interaction_ids=jnp.full(total_pad, -1, dtype=jnp.int16),
                ancestor_track_ids=jnp.full(total_pad, -1, dtype=jnp.int32),
                pdg=jnp.zeros(total_pad, dtype=jnp.int32),
                charge=jnp.zeros(total_pad),
                photons=jnp.zeros(total_pad),
                qs_fractions=jnp.zeros(total_pad),
                n_actual=total_pad,
            ))

        stacked_deps = jax.tree.map(lambda *xs: jnp.stack(xs), *volumes)
        all_signals = self._forward_diff(params, stacked_deps)
        return tuple(
            all_signals[v, p]
            for v in range(cfg.n_volumes)
            for p in range(cfg.volumes[v].n_planes))
