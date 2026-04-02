"""
LArTPC detector simulation with modular physics pipeline.

Two execution paths:
    - process_event(deposits, sim_params): Production path with fori_loop
      batching, optional noise/electronics/track_hits/digitization.
    - forward(params, deposits): Differentiable path with remat, gradients
      through all SimParams fields (velocity, lifetime, recomb, NN models).

Both paths share the same physics functions (compute_volume_physics,
compute_plane_physics) and response function (response_fn with unified
signature). SimParams is a JIT argument — changing physics values does
NOT trigger recompilation.
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
    get_volume_deposits,
)

# Data construction
from tools.loader import build_deposit_data, load_event

# Response kernels (loaded at __init__, used by shared factory)
from tools.kernels import (
    load_response_kernels, apply_diffusion_response, generate_dkernel_table,
)

# Track labeling factory + post-processing
from tools.track_hits import create_track_hits_fn_for_volume, finalize_track_hits, compute_qs_fractions

# Recombination
from tools.recombination import RECOMB_MODELS

# Post-processing factories
from tools.noise import create_noise_fn_for_volume
from tools.electronics import create_electronics_fn_for_volume, create_digitize_fn_for_volume


# =============================================================================
# DETECTOR SIMULATOR CLASS
# =============================================================================

class DetectorSimulator:
    """
    LArTPC detector simulation with fixed-size padding.

    Two execution paths:
        - process_event(): Production path (fori_loop batching, post-processing).
        - forward(): Differentiable path (remat, gradients through SimParams).
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

        # Load SCE maps (per-volume)
        sce_per_volume = self._load_sce(include_electric_dist, electric_dist_path)
        self._include_sce = sce_per_volume is not None

        # Load response kernels per volume
        print("   Loading response kernels...")
        self.response_kernels = []
        for vol in cfg.volumes:
            d = vol.diffusion
            self.response_kernels.append(load_response_kernels(
                response_path=response_path,
                num_s=d.num_s,
                time_spacing=cfg.time_step_us,
                max_sigma_trans_unitless=d.max_sigma_trans_unitless,
                max_sigma_long_unitless=d.max_sigma_long_unitless,
            ))

        # Build shared factories
        sce_factories, _build_response_fn, _build_response_fn_diff, _recomb_fn = \
            self._setup_shared_factories(sce_per_volume)

        # Build per-volume post-processing factories
        vol_electronics_fns = []
        vol_noise_fns = []
        vol_digitize_fns = []
        vol_track_hits_fns = []
        self.electronics_chunk_size = None
        self._electronics_fft_size = None

        for vol_idx, vol in enumerate(cfg.volumes):
            vol_kernels = self.response_kernels[vol_idx]

            e_fn, e_meta = create_electronics_fn_for_volume(
                cfg, vol, vol_kernels,
                electronics_chunk_size=electronics_chunk_size,
                electronics_threshold=electronics_threshold)
            if e_meta.get('e_chunk'):
                self.electronics_chunk_size = e_meta['e_chunk']
                self._electronics_fft_size = e_meta.get('e_fft')
            vol_electronics_fns.append(e_fn)

            vol_noise_fns.append(create_noise_fn_for_volume(
                cfg, vol, vol_kernels, e_chunk=self.electronics_chunk_size))

            d_fn, dig_cfg = create_digitize_fn_for_volume(cfg, vol, digitization_config)
            if dig_cfg:
                self.digitization_config = dig_cfg
            vol_digitize_fns.append(d_fn)

            vol_track_hits_fns.append(create_track_hits_fn_for_volume(cfg, vol))

        # Build JIT-compiled calculators
        self._build_jit_functions(
            sce_factories, _build_response_fn, _build_response_fn_diff,
            _recomb_fn, vol_electronics_fns, vol_noise_fns,
            vol_digitize_fns, vol_track_hits_fns,
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
        return load_sce_per_volume(electric_dist_path)

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
        print(f"   Volumes: {cfg.n_volumes}")
        print("--- DetectorSimulator Ready ---")

    def _setup_shared_factories(self, sce_per_volume):
        """Build per-volume SCE, response, and recombination factories."""
        from tools.recombination import compute_quanta, XI_FN

        cfg = self._sim_config
        _nominal_field = float(self._default_sim_params.recomb_params.field_strength_Vcm)

        # ── Per-volume SCE factories ──
        sce_factories = []
        for vol_idx in range(cfg.n_volumes):
            vol_geom = cfg.volumes[vol_idx]
            if sce_per_volume is not None:
                efield_fn, corr_fn = sce_per_volume[vol_idx]
                def _make_sce(ef=efield_fn, cf=corr_fn, nf=_nominal_field):
                    def _sce(positions_cm):
                        E_local = ef(positions_cm)
                        E_normalized = E_local / nf
                        drift_corr = cf(positions_cm)
                        return SCEOutputs(efield_correction=E_normalized, drift_corr_cm=drift_corr)
                    return _sce
                sce_factories.append(_make_sce)
            else:
                drift_dir = float(vol_geom.drift_direction)
                def _make_nominal(d=drift_dir):
                    def _sce(pos):
                        N = pos.shape[0]
                        corr = jnp.broadcast_to(
                            jnp.array([-d, 0.0, 0.0]), (N, 3))
                        return SCEOutputs(
                            efield_correction=corr,
                            drift_corr_cm=jnp.zeros((N, 3)))
                    return _sce
                sce_factories.append(_make_nominal)

        # ── Response factories ──
        response_kernels = self.response_kernels
        volumes = cfg.volumes

        def _build_response_fn(sim_params, vol_idx, plane_type):
            """Production response — uses pre-computed DKernel table."""
            kernel = response_kernels[vol_idx][plane_type]
            max_drift = volumes[vol_idx].max_drift_cm
            dkernel = kernel.DKernel
            def response_fn(positions_cm, drift_distance_cm, wire_offsets, time_offsets):
                s_values = jnp.clip(drift_distance_cm / max_drift, 0.0, 1.0)
                return apply_diffusion_response(
                    dkernel, s_values, wire_offsets, time_offsets,
                    kernel.wire_spacing, kernel.num_wires)
            return response_fn

        def _build_response_fn_diff(sim_params, vol_idx, plane_type):
            """Diff response — recomputes DKernel from SimParams diffusion."""
            kernel = response_kernels[vol_idx][plane_type]
            max_drift = volumes[vol_idx].max_drift_cm
            max_drift_time = max_drift / sim_params.velocity_cm_us
            sigma_trans_max_cm = jnp.sqrt(
                2.0 * sim_params.diffusion_trans_cm2_us * max_drift_time)
            sigma_long_max_us = jnp.sqrt(
                2.0 * (sim_params.diffusion_long_cm2_us
                       / sim_params.velocity_cm_us**2) * max_drift_time)
            dkernel = generate_dkernel_table(
                sigma_trans_max_cm, sigma_long_max_us,
                kernel.base_kernel, kernel.freq_w, kernel.freq_t,
                kernel.s_levels)
            def response_fn(positions_cm, drift_distance_cm, wire_offsets, time_offsets):
                s_values = jnp.clip(drift_distance_cm / max_drift, 0.0, 1.0)
                return apply_diffusion_response(
                    dkernel, s_values, wire_offsets, time_offsets,
                    kernel.wire_spacing, kernel.num_wires)
            return response_fn

        # ── Recombination ──
        _xi_fn = XI_FN[self.recomb_model]
        def _recomb_fn(de, dx, phi_drift, e_field_Vcm, params):
            return compute_quanta(de, dx, phi_drift, e_field_Vcm, params, _xi_fn)

        return sce_factories, _build_response_fn, _build_response_fn_diff, _recomb_fn

    def _build_jit_functions(self, sce_factories, _build_response_fn,
                             _build_response_fn_diff, _recomb_fn,
                             vol_electronics_fns, vol_noise_fns,
                             vol_digitize_fns, vol_track_hits_fns):
        """Build JIT-compiled calculators from per-volume factories."""
        from tools.physics import (
            compute_volume_physics, compute_plane_physics,
            compute_plane_signal, compute_plane_signal_bucketed,
            compute_bucket_maps,
        )

        cfg = self._sim_config
        response_kernels = self.response_kernels
        n_volumes = cfg.n_volumes
        total_pad = cfg.total_pad

        # ── Production JIT ──
        # Define _process_volume and _skip_volume at closure-definition time.
        # Track_hits on/off is resolved here — no if-guards inside traced functions.

        if cfg.include_track_hits:
            def _make_process_volume(vol_idx):
                vol_geom = cfg.volumes[vol_idx]
                sce_fn_factory = sce_factories[vol_idx]
                electronics_fn = vol_electronics_fns[vol_idx]
                noise_fn = vol_noise_fns[vol_idx]
                digitize_fn = vol_digitize_fns[vol_idx]
                track_hits_fn = vol_track_hits_fns[vol_idx]
                vol_kernels = response_kernels[vol_idx]

                def _process(vol_deps, sim_params, vol_key):
                    vol_signals = {}
                    vol_hits = {}
                    plane_keys = jax.random.split(vol_key, vol_geom.n_planes)

                    sce_fn = sce_fn_factory()
                    vol_int = compute_volume_physics(
                        vol_deps, sim_params, vol_geom, sce_fn, _recomb_fn)

                    vol_qs = compute_qs_fractions(
                        vol_int.charges, vol_deps.group_ids, total_pad)

                    readout_window_us = cfg.num_time_steps * cfg.time_step_us
                    for plane_idx in range(vol_geom.n_planes):
                        plane_type = cfg.plane_names[vol_idx][plane_idx]
                        plane_int = compute_plane_physics(
                            vol_int, sim_params, vol_geom, plane_idx,
                            cfg.pre_window_us, readout_window_us)

                        vol_hits[plane_idx] = track_hits_fn(
                            plane_int, vol_deps, vol_geom, plane_idx, vol_deps.n_actual)

                        response_fn = _build_response_fn(sim_params, vol_idx, plane_type)
                        plane_kernel = vol_kernels[plane_type]

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
                        vol_signals[plane_idx] = response_signal

                    return vol_signals, vol_hits, vol_qs, vol_int.charges, vol_int.photons
                return _process

            def _make_skip_volume(vol_idx):
                vol_geom = cfg.volumes[vol_idx]
                vol_kernels = response_kernels[vol_idx]

                # Pre-build zero pytrees matching _process output
                zero_signals = {}
                zero_hits = {}
                for p in range(vol_geom.n_planes):
                    plane_type = cfg.plane_names[vol_idx][p]
                    pk = vol_kernels[plane_type]
                    if cfg.use_bucketed and cfg.include_electronics:
                        e_chunk = max(vol_geom.num_wires)
                        zero_signals[p] = (
                            jnp.zeros((e_chunk, cfg.num_time_steps)),
                            jnp.zeros(e_chunk, dtype=jnp.int32),
                            jnp.int32(0))
                    elif cfg.use_bucketed:
                        B1, B2 = 2 * pk.num_wires, 2 * pk.kernel_height
                        zero_signals[p] = (
                            jnp.zeros((cfg.max_active_buckets, B1, B2)),
                            jnp.int32(0),
                            jnp.zeros(cfg.max_active_buckets, dtype=jnp.int32),
                            B1, B2)
                    else:
                        zero_signals[p] = jnp.zeros(
                            (vol_geom.num_wires[p], cfg.num_time_steps))

                    # Track hits zero tuple (5-tuple matching track_hits_fn output)
                    SENTINEL_PK = jnp.int32(2**30)
                    zero_hits[p] = (
                        jnp.full(cfg.track_hits.max_keys, SENTINEL_PK, dtype=jnp.int32),
                        jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.int32),
                        jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.float32),
                        jnp.int32(0),
                        jnp.zeros(total_pad, dtype=jnp.float32))

                zero_qs = jnp.zeros(total_pad, dtype=jnp.float32)
                zero_charges = jnp.zeros(total_pad, dtype=jnp.float32)
                zero_photons = jnp.zeros(total_pad, dtype=jnp.float32)

                def _skip(vol_deps, sim_params, vol_key):
                    return zero_signals, zero_hits, zero_qs, zero_charges, zero_photons
                return _skip

        else:
            # Track hits disabled — simpler process/skip without track_hits or qs
            def _make_process_volume(vol_idx):
                vol_geom = cfg.volumes[vol_idx]
                sce_fn_factory = sce_factories[vol_idx]
                electronics_fn = vol_electronics_fns[vol_idx]
                noise_fn = vol_noise_fns[vol_idx]
                digitize_fn = vol_digitize_fns[vol_idx]
                vol_kernels = response_kernels[vol_idx]

                def _process(vol_deps, sim_params, vol_key):
                    vol_signals = {}
                    plane_keys = jax.random.split(vol_key, vol_geom.n_planes)

                    sce_fn = sce_fn_factory()
                    vol_int = compute_volume_physics(
                        vol_deps, sim_params, vol_geom, sce_fn, _recomb_fn)

                    readout_window_us = cfg.num_time_steps * cfg.time_step_us
                    for plane_idx in range(vol_geom.n_planes):
                        plane_type = cfg.plane_names[vol_idx][plane_idx]
                        plane_int = compute_plane_physics(
                            vol_int, sim_params, vol_geom, plane_idx,
                            cfg.pre_window_us, readout_window_us)

                        response_fn = _build_response_fn(sim_params, vol_idx, plane_type)
                        plane_kernel = vol_kernels[plane_type]

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
                        vol_signals[plane_idx] = response_signal

                    return vol_signals, {}, jnp.zeros(total_pad), vol_int.charges, vol_int.photons
                return _process

            def _make_skip_volume(vol_idx):
                vol_geom = cfg.volumes[vol_idx]
                vol_kernels = response_kernels[vol_idx]

                zero_signals = {}
                for p in range(vol_geom.n_planes):
                    plane_type = cfg.plane_names[vol_idx][p]
                    pk = vol_kernels[plane_type]
                    if cfg.use_bucketed and cfg.include_electronics:
                        e_chunk = max(vol_geom.num_wires)
                        zero_signals[p] = (
                            jnp.zeros((e_chunk, cfg.num_time_steps)),
                            jnp.zeros(e_chunk, dtype=jnp.int32),
                            jnp.int32(0))
                    elif cfg.use_bucketed:
                        B1, B2 = 2 * pk.num_wires, 2 * pk.kernel_height
                        zero_signals[p] = (
                            jnp.zeros((cfg.max_active_buckets, B1, B2)),
                            jnp.int32(0),
                            jnp.zeros(cfg.max_active_buckets, dtype=jnp.int32),
                            B1, B2)
                    else:
                        zero_signals[p] = jnp.zeros(
                            (vol_geom.num_wires[p], cfg.num_time_steps))

                def _skip(vol_deps, sim_params, vol_key):
                    return zero_signals, {}, jnp.zeros(total_pad), jnp.zeros(total_pad), jnp.zeros(total_pad)
                return _skip

        # Build per-volume process/skip closures
        process_fns = [_make_process_volume(v) for v in range(n_volumes)]
        skip_fns = [_make_skip_volume(v) for v in range(n_volumes)]

        @jax.jit
        def _calculate_signals_jit(sim_params, all_vol_deposits, noise_key):
            """Production JIT. Receives tuple of VolumeDeposits."""
            response_signals = {}
            track_hits = {}
            qs_per_volume = []
            charges_per_volume = []
            photons_per_volume = []
            vol_keys = jax.random.split(noise_key, n_volumes)

            for vol_idx in range(n_volumes):
                vol_deps = all_vol_deposits[vol_idx]
                vol_key = vol_keys[vol_idx]

                vol_signals, vol_hits, vol_qs, vol_charges, vol_photons = jax.lax.cond(
                    vol_deps.n_actual > 0,
                    process_fns[vol_idx],
                    skip_fns[vol_idx],
                    vol_deps, sim_params, vol_key)

                for p in vol_signals:
                    response_signals[(vol_idx, p)] = vol_signals[p]
                for p in vol_hits:
                    track_hits[(vol_idx, p)] = vol_hits[p]
                qs_per_volume.append(vol_qs)
                charges_per_volume.append(vol_charges)
                photons_per_volume.append(vol_photons)

            return response_signals, track_hits, qs_per_volume, charges_per_volume, photons_per_volume

        self._calculator_jit_raw = _calculate_signals_jit

        # ── Light-only JIT ──
        @jax.jit
        def _calculate_light_jit(sim_params, all_vol_deposits):
            results = []
            for vol_idx in range(n_volumes):
                vol_deps = all_vol_deposits[vol_idx]
                sce_fn = sce_factories[vol_idx]()
                vol_int = compute_volume_physics(
                    vol_deps, sim_params, cfg.volumes[vol_idx], sce_fn, _recomb_fn)
                results.append((vol_int.charges, vol_int.photons, vol_int.positions_cm))
            return results

        self._light_calculator_jit = _calculate_light_jit

        # ── Differentiable path ──
        if self.n_segments is not None:
            n_segments = self.n_segments

            @jax.remat
            def _forward_diff(params, all_vol_deposits):
                """Same pipeline as production but with static fori_loop bounds.

                n_segments is closure-captured Python int → static fori_loop
                bound → reverse-mode differentiable.
                """
                response_signals = {}

                for vol_idx in range(n_volumes):
                    vol_deps = all_vol_deposits[vol_idx]
                    vol_geom = cfg.volumes[vol_idx]

                    sce_fn = sce_factories[vol_idx]()
                    vol_int = compute_volume_physics(
                        vol_deps, params, vol_geom, sce_fn, _recomb_fn)

                    readout_window_us = cfg.num_time_steps * cfg.time_step_us
                    for plane_idx in range(vol_geom.n_planes):
                        plane_type = cfg.plane_names[vol_idx][plane_idx]
                        plane_int = compute_plane_physics(
                            vol_int, params, vol_geom, plane_idx,
                            cfg.pre_window_us, readout_window_us)

                        response_fn = _build_response_fn_diff(params, vol_idx, plane_type)
                        plane_kernel = response_kernels[vol_idx][plane_type]

                        response_signals[(vol_idx, plane_idx)] = compute_plane_signal(
                            plane_int, response_fn, n_segments,
                            cfg.response_chunk_size,
                            cfg, vol_geom, plane_idx, plane_kernel)

                return response_signals

            self._forward_diff = _forward_diff

    def __call__(self, deposits: DepositData, key=None):
        return self.process_event(deposits, key=key)

    def process_event(self, deposits: DepositData, sim_params=None, key=None):
        """Run production simulation.

        Parameters
        ----------
        deposits : DepositData
            Pre-split, padded, grouped deposits from build_deposit_data or load_event.
        sim_params : SimParams, optional
        key : jax PRNGKey, optional

        Returns
        -------
        response_signals : dict
            Keyed by (vol_idx, plane_idx).
        track_hits : dict
            Keyed by (vol_idx, plane_idx). Contains raw merge state.
        deposits : DepositData
            Input deposits with charge, photons, qs_fractions filled from simulation.
        """
        if sim_params is None:
            sim_params = self._default_sim_params

        noise_key = key if key is not None else jax.random.PRNGKey(0)

        response_signals, track_hits, qs_per_vol, charges_per_vol, photons_per_vol = \
            self._calculator_jit_raw(sim_params, deposits.volumes, noise_key)

        # Stitch Q, L, qs back into VolumeDeposits
        filled_volumes = tuple(
            vol._replace(
                charge=charges_per_vol[v],
                photons=photons_per_vol[v],
                qs_fractions=qs_per_vol[v],
            )
            for v, vol in enumerate(deposits.volumes)
        )
        filled_deposits = deposits._replace(volumes=filled_volumes)

        # Bundle per-volume group_to_track for finalize_track_hits
        if self._sim_config.include_track_hits:
            track_hits['group_to_track'] = deposits.group_to_track

        return response_signals, track_hits, filled_deposits

    def process_event_light(self, deposits: DepositData, sim_params=None):
        """Compute per-segment charge and scintillation light only.

        Returns DepositData with charge and photons filled (no wire response).
        """
        if sim_params is None:
            sim_params = self._default_sim_params

        results = self._light_calculator_jit(sim_params, deposits.volumes)
        filled_volumes = tuple(
            vol._replace(charge=results[v][0], photons=results[v][1])
            for v, vol in enumerate(deposits.volumes)
        )
        return deposits._replace(volumes=filled_volumes)

    def finalize_track_hits(self, track_hits):
        """Derive track labels from raw group merge state.

        Uses per-volume group_to_track from DepositData (stored in track_hits
        by process_event).
        """
        group_to_track_per_vol = track_hits.pop('group_to_track')
        result = {}
        for (vol_idx, plane_idx), raw in track_hits.items():
            g2t = group_to_track_per_vol[vol_idx]
            from tools.track_hits import label_from_groups
            state_pk, state_gid, state_ch, state_count, row_sums = raw
            labeled = label_from_groups(
                state_pk, state_gid, state_ch, state_count,
                g2t, self._sim_config.num_time_steps)
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
        dummy_volumes = tuple(dummy_vol for _ in range(n_vol))
        _ = self._calculator_jit_raw(
            self._default_sim_params, dummy_volumes, jax.random.PRNGKey(0))
        print(f"JIT compilation finished (total_pad={pad:,}).")

    def forward(self, params, deposits):
        """Differentiable forward pass.

        Parameters
        ----------
        params : SimParams
        deposits : DepositData
            From build_deposit_data or create_deposit_data. Must be padded
            to total_pad.

        Returns
        -------
        tuple of signal arrays, one per (vol_idx, plane_idx).
        """
        deposits = pad_deposit_data(deposits, self._sim_config.total_pad)
        response_signals = self._forward_diff(params, deposits.volumes)
        return tuple(
            response_signals[(v, p)]
            for v in range(self._sim_config.n_volumes)
            for p in range(self._sim_config.volumes[v].n_planes))

    def forward_segments(self, params, positions_mm, de, dx):
        """Lightweight differentiable forward for segment-like data.

        Can be called inside jax.grad — uses JAX ops for volume assignment
        (no numpy splitting). Each volume sees all deposits but masks by
        position range. Padding via n_actual mask after recombination.
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
        padded_theta = jnp.zeros(total_pad)
        padded_phi = jnp.zeros(total_pad)
        padded_tids = jnp.full(total_pad, -1, dtype=jnp.int32)
        padded_gids = jnp.zeros(total_pad, dtype=jnp.int32)
        padded_t0 = jnp.zeros(total_pad)

        # Build per-volume VolumeDeposits using position-based masking
        # Each volume sees ALL deposits but masks de to zero for out-of-range positions
        x_cm = padded_pos[:, 0] / 10.0
        volumes = []
        for vol_idx in range(cfg.n_volumes):
            vol = cfg.volumes[vol_idx]
            x_min, x_max = vol.ranges_cm[0]
            vol_mask = (x_cm >= x_min) & (x_cm < x_max)
            # Mask de — out-of-range deposits get de=0, producing zero charges
            masked_de = padded_de * vol_mask
            volumes.append(VolumeDeposits(
                positions_mm=padded_pos,
                de=masked_de,
                dx=padded_dx,
                theta=padded_theta,
                phi=padded_phi,
                track_ids=padded_tids,
                group_ids=padded_gids,
                t0_us=padded_t0,
                interaction_ids=jnp.full(total_pad, -1, dtype=jnp.int16),
                ancestor_track_ids=jnp.full(total_pad, -1, dtype=jnp.int32),
                pdg=jnp.zeros(total_pad, dtype=jnp.int32),
                charge=jnp.zeros(total_pad),
                photons=jnp.zeros(total_pad),
                qs_fractions=jnp.zeros(total_pad),
                n_actual=total_pad,  # static — process all (masking handles zeros)
            ))

        response_signals = self._forward_diff(params, tuple(volumes))
        return tuple(
            response_signals[(v, p)]
            for v in range(cfg.n_volumes)
            for p in range(cfg.volumes[v].n_planes))
