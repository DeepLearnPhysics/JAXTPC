"""
LArTPC detector simulation with modular physics pipeline.

Two execution paths:
    - process_event(deposit_data, sim_params): Production path with fori_loop
      batching, optional noise/electronics/track_hits/digitization.
    - forward(params, deposits): Differentiable path with remat, gradients
      through all SimParams fields (velocity, lifetime, recomb, NN models).

Both paths share the same physics functions (compute_side_physics,
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
    DepositData, SCEOutputs,
    create_sim_params, create_sim_config,
    pad_deposit_data,
)


# Data splitting/padding (host-side numpy)
from tools.loader import split_and_pad_data

# Response kernels (loaded at __init__, used by shared factory)
from tools.kernels import (
    load_response_kernels, apply_diffusion_response, generate_dkernel_table,
)

# Track labeling factory + post-processing
from tools.track_hits import create_track_hits_fn, finalize_track_hits, compute_qs_fractions

# Recombination
from tools.recombination import RECOMB_MODELS

# Post-processing factories
from tools.noise import create_noise_fn
from tools.electronics import create_electronics_fn, create_digitize_fn


# =============================================================================
# DETECTOR SIMULATOR CLASS
# =============================================================================

class DetectorSimulator:
    """
    LArTPC detector simulation with fixed-size padding.

    Two execution paths:
        - process_event(): Production path (fori_loop batching, post-processing).
        - forward(): Differentiable path (remat, gradients through SimParams).

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().
    response_path : str
        Path to response kernel files.
    track_config : TrackHitsConfig, optional
        Configuration for track labeling. If None, uses defaults.
    total_pad : int
        Fixed pad size per side. Default 200_000.
    response_chunk_size : int
        Deposits per fori_loop batch. Default 50_000. Must divide total_pad.
    use_bucketed : bool
        If True, use sparse bucketed accumulation. Default False.
    max_active_buckets : int
        Max active buckets for sparse mode. Default 1000.
    include_noise : bool
        If True, add intrinsic noise inside JIT. Default False.
    include_electronics : bool
        If True, apply RC⊗RC electronics convolution. Default False.
    include_track_hits : bool
        If True, run track labeling path. Default True.
    electronics_chunk_size : int, optional
        Max active wires for electronics. Default: max num_wires_actual.
    electronics_threshold : float
        Active wire detection threshold. Default 0.0.
    recombination_model : str, optional
        ``'modified_box'`` or ``'emb'``. Default from config.
    include_electric_dist : bool
        If True, load SCE maps (E-field + drift corrections). Default False.
    electric_dist_path : str, optional
        Path to SCE HDF5 file. Default ``config/sce_jaxtpc.h5``.
    include_digitize : bool
        If True, apply ADC digitization (round + clip). Default False.
    digitization_config : DigitizationConfig, optional
        Digitization parameters. If None, built from detector_config.
    differentiable : bool
        If True, configure for differentiable forward() path. Default False.
    n_segments : int, optional
        Number of segments for differentiable mode. Required when
        differentiable=True.
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
                f"response_chunk_size ({response_chunk_size:,})."
            )

        # Resolve recombination model name
        self.recomb_model = (recombination_model
                             or detector_config['simulation']['charge_recombination']['model'])
        if self.recomb_model not in RECOMB_MODELS:
            raise ValueError(
                f"Unknown recombination model: '{self.recomb_model}'. "
                f"Must be one of: {RECOMB_MODELS}")

        # Space charge effect (electric field distortions)
        sce_efield_fn, sce_drift_correction_fn = self._load_sce(
            include_electric_dist, electric_dist_path)
        self._include_sce = sce_efield_fn is not None

        print("   Extracting parameters...")

        # Build SimConfig — ALL static config in one place
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

        # Create default SimParams (tunable physics — JIT argument)
        self._default_sim_params = create_sim_params(
            detector_config,
            recombination_model=self.recomb_model,
        )

        # Load response kernels
        print("   Loading response kernels...")
        self.response_kernels = load_response_kernels(
            response_path=response_path,
            num_s=cfg.diffusion.num_s,
            time_spacing=cfg.time_step_us,
            max_sigma_trans_unitless=cfg.diffusion.max_sigma_trans_unitless,
            max_sigma_long_unitless=cfg.diffusion.max_sigma_long_unitless,
        )

        # Shared factories (used by BOTH production and diff paths)
        build_sce_fn, build_response_fn, build_response_fn_diff, recomb_fn = \
            self._setup_shared_factories(sce_efield_fn, sce_drift_correction_fn)

        # Post-processing factories (each pre-computes its own data internally)
        electronics_fn, e_meta = create_electronics_fn(
            cfg, self.response_kernels,
            electronics_chunk_size=electronics_chunk_size,
            electronics_threshold=electronics_threshold,
        )
        self.electronics_chunk_size = e_meta.get('e_chunk')
        self._electronics_fft_size = e_meta.get('e_fft')

        noise_fn = create_noise_fn(
            cfg, self.response_kernels,
            e_chunk=self.electronics_chunk_size,
        )

        digitize_fn, self.digitization_config = create_digitize_fn(
            cfg, digitization_config,
        )

        track_hits_fn = create_track_hits_fn(cfg)

        # Build JIT-compiled calculators
        self._build_jit_functions(
            build_sce_fn, build_response_fn, build_response_fn_diff,
            recomb_fn, electronics_fn, noise_fn, digitize_fn, track_hits_fn,
        )

        self._print_summary()

    def _validate_pre_simulation(self, counts):
        """Check inputs before running simulation."""
        pass

    def _validate_post_simulation(self, track_hits, response_signals):
        """Check for truncation after simulation."""
        if self._sim_config.include_track_hits:
            for plane_key, th in track_hits.items():
                if plane_key == 'group_to_track':
                    continue
                # th is (final_pk, final_gid, final_ch, final_count, final_rowsums)
                actual_count = int(th[3])
                if actual_count >= self._sim_config.track_hits.max_keys:
                    print(f"ERROR: Plane {plane_key}: merge count ({actual_count:,}) >= "
                          f"max_keys ({self._sim_config.track_hits.max_keys:,}). Data truncated!")

        if self._sim_config.output_format == 'bucketed':
            for plane_key, resp in response_signals.items():
                buckets, num_active, compact_to_key, B1, B2 = resp
                if int(num_active) >= self._sim_config.max_active_buckets:
                    print(f"ERROR: Plane {plane_key}: num_active ({int(num_active):,}) >= max_active_buckets ({self._sim_config.max_active_buckets:,}). Data truncated!")

        elif self._sim_config.output_format == 'wire_sparse':
            for plane_key, resp in response_signals.items():
                _, _, n_active_wires = resp
                if int(n_active_wires) >= self.electronics_chunk_size:
                    print(f"WARNING: Plane {plane_key}: active wires ({int(n_active_wires):,}) >= "
                          f"electronics_chunk_size ({self.electronics_chunk_size:,}).")

        elif self._sim_config.output_format == 'dense' and self._sim_config.include_electronics:
            for plane_key, resp in response_signals.items():
                n_active = int(jnp.sum(jnp.any(resp != 0, axis=1)))
                if n_active > self.electronics_chunk_size:
                    print(f"WARNING: Plane {plane_key}: active wires ({n_active:,}) > "
                          f"electronics_chunk_size ({self.electronics_chunk_size:,}).")

    def _load_sce(self, include_electric_dist, electric_dist_path):
        """Load space charge effect maps if enabled.

        Returns (sce_efield_fn, sce_drift_correction_fn) — both None when disabled.
        """
        if include_electric_dist:
            from tools.efield_distortions import load_sce_interpolation_fns
            if electric_dist_path is None:
                electric_dist_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 'config', 'sce_jaxtpc.h5'
                )
            print(f"   Loading SCE maps from {electric_dist_path}...")
            return load_sce_interpolation_fns(electric_dist_path)
        return None, None

    def _print_summary(self):
        """Print configuration summary after initialization."""
        cfg = self._sim_config
        if self._include_sce:
            print("   Space charge effects: ENABLED (E-field distortions, drift corrections)")
        else:
            print("   Space charge effects: DISABLED")
        print(f"   Recombination model: {self.recomb_model}")
        print(f"   Config: total_pad={cfg.total_pad:,}, response_chunk={cfg.response_chunk_size:,}, "
              f"num_s={cfg.diffusion.num_s if cfg.diffusion else 'N/A'}, "
              f"K_wire={cfg.diffusion.K_wire if cfg.diffusion else 'N/A'}, "
              f"K_time={cfg.diffusion.K_time if cfg.diffusion else 'N/A'}")
        if cfg.use_bucketed:
            print(f"   Using BUCKETED accumulation (B=2*kernel, max_buckets={cfg.max_active_buckets})")
        if cfg.include_noise:
            print("   Noise integration: ENABLED (added inside JIT)")
        if cfg.include_electronics:
            print(f"   Electronics response: ENABLED (FFT size={self._electronics_fft_size}, "
                  f"chunk={self.electronics_chunk_size}, output={cfg.output_format})")
        if cfg.include_digitize:
            dig_cfg = self.digitization_config
            print(f"   Digitization: ENABLED ({dig_cfg.n_bits}-bit, "
                  f"pedestal Y={dig_cfg.pedestal_collection} U/V={dig_cfg.pedestal_induction}, "
                  f"gain={dig_cfg.gain_scale})")
        if cfg.include_track_hits:
            print("   Track labeling: ENABLED")
        else:
            print("   Track labeling: DISABLED")
        print("--- DetectorSimulator Ready ---")

    def _setup_shared_factories(self, sce_efield_fn, sce_drift_correction_fn):
        """Define shared factories used by both production and diff JIT builders.

        Returns (build_sce_fn, build_response_fn, recomb_fn).
        These are factory functions defined at __init__ time (outside JIT),
        called inside JIT within unrolled side/plane loops. No if inside JIT.
        """
        from tools.recombination import compute_quanta, XI_FN

        # ── SCE factory ──
        if sce_efield_fn is not None:
            # HDF5 SCE — wrap existing functions into SCEOutputs interface
            _nominal_field = float(self._default_sim_params.recomb_params.field_strength_Vcm)
            def _build_sce_fn(sce_models):
                def _sce(positions_cm):
                    E_local = sce_efield_fn(positions_cm)
                    E_normalized = E_local / _nominal_field
                    drift_corr = sce_drift_correction_fn(positions_cm)
                    return SCEOutputs(efield_correction=E_normalized, drift_corr_cm=drift_corr)
                return _sce
        else:
            # Nominal SCE — no corrections, just E-field direction
            def _build_sce_fn(sce_models):
                def _sce(pos):
                    N = pos.shape[0]
                    sign = jnp.where(pos[:, 0] < 0, 1.0, -1.0)
                    corr = jnp.stack([sign, jnp.zeros(N), jnp.zeros(N)], axis=-1)
                    return SCEOutputs(efield_correction=corr, drift_corr_cm=jnp.zeros((N, 3)))
                return _sce
        # ── Response factories ──
        # Production: use pre-computed DKernel (no DCT recomputation)
        # Diff: recompute DKernel inside JIT from SimParams diffusion values
        response_kernels = self.response_kernels
        side_geom = self._sim_config.side_geom

        def _build_response_fn(sim_params, side_idx, plane_type):
            """Production response — uses pre-computed DKernel table."""
            kernel = response_kernels[plane_type]
            half_width = side_geom[side_idx].half_width_cm
            dkernel = kernel.DKernel
            def response_fn(positions_cm, drift_distance_cm, wire_offsets, time_offsets):
                s_values = jnp.clip(drift_distance_cm / half_width, 0.0, 1.0)
                return apply_diffusion_response(
                    dkernel, s_values, wire_offsets, time_offsets,
                    kernel.wire_spacing, kernel.num_wires)
            return response_fn

        def _build_response_fn_diff(sim_params, side_idx, plane_type):
            """Diff response — recomputes DKernel from SimParams diffusion."""
            kernel = response_kernels[plane_type]
            half_width = side_geom[side_idx].half_width_cm
            max_drift_time = half_width / sim_params.velocity_cm_us
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
                s_values = jnp.clip(drift_distance_cm / half_width, 0.0, 1.0)
                return apply_diffusion_response(
                    dkernel, s_values, wire_offsets, time_offsets,
                    kernel.wire_spacing, kernel.num_wires)
            return response_fn

        # ── Recombination function — returns (Q, L) for any model ──
        _xi_fn = XI_FN[self.recomb_model]
        def _recomb_fn(de, dx, phi_drift, e_field_Vcm, params):
            return compute_quanta(de, dx, phi_drift, e_field_Vcm, params, _xi_fn)

        return _build_sce_fn, _build_response_fn, _build_response_fn_diff, _recomb_fn

    def _build_jit_functions(self, _build_sce_fn, _build_response_fn,
                             _build_response_fn_diff, _recomb_fn,
                             electronics_fn, noise_fn, digitize_fn, track_hits_fn):
        """Build JIT-compiled calculators from shared factories.

        Creates self._calculator_jit_raw (production, always).
        The diff path uses a separate un-decorated function with
        n_segments as a static closure value (required for jax.grad
        through fori_loop).
        """
        from tools.physics import (
            compute_side_physics, compute_plane_physics,
            compute_plane_signal, compute_plane_signal_bucketed,
            compute_bucket_maps,
        )

        # ── Shared closure captures ──
        cfg = self._sim_config
        response_kernels = self.response_kernels

        # ── Production JIT ──
        total_pad = cfg.total_pad
        _include_track_hits = cfg.include_track_hits

        @jax.jit
        def _calculate_signals_jit(sim_params, deposits_east, deposits_west,
                                    noise_key, n_east, n_west):
            response_signals = {}
            track_hits = {}
            qs_per_side = [jnp.zeros(total_pad), jnp.zeros(total_pad)]
            side_deposits_list = [deposits_east, deposits_west]
            n_actual_list = [n_east, n_west]
            plane_keys = jax.random.split(noise_key, 6)

            for side_idx in range(2):
                deposits = side_deposits_list[side_idx]
                n_actual = n_actual_list[side_idx]
                side_geom = cfg.side_geom[side_idx]

                sce_fn = _build_sce_fn(sim_params.sce_models)
                side_int = compute_side_physics(
                    deposits, sim_params, side_geom, sce_fn, _recomb_fn)

                if _include_track_hits:
                    qs_per_side[side_idx] = compute_qs_fractions(
                        side_int.charges, deposits.group_ids, total_pad)

                for plane_idx in range(3):
                    plane_type = cfg.plane_names[side_idx][plane_idx]
                    plane_int = compute_plane_physics(
                        side_int, sim_params, side_geom, plane_idx)

                    track_hits_fn(track_hits, plane_int, deposits,
                                  side_geom, side_idx, plane_idx, n_actual)

                    response_fn = _build_response_fn(sim_params, side_idx, plane_type)
                    plane_kernel = response_kernels[plane_type]

                    if cfg.use_bucketed:
                        point_to_compact, num_active, compact_to_key, B1, B2 = compute_bucket_maps(
                            deposits, plane_int, side_geom, plane_idx,
                            cfg, plane_kernel)
                        response_buckets = compute_plane_signal_bucketed(
                            plane_int, response_fn, n_actual, cfg.response_chunk_size,
                            point_to_compact, cfg.max_active_buckets, B1, B2,
                            cfg, side_geom, plane_idx, plane_kernel)
                        response_signal = (response_buckets, num_active, compact_to_key, B1, B2)
                    else:
                        response_signal = compute_plane_signal(
                            plane_int, response_fn, n_actual,
                            cfg.response_chunk_size,
                            cfg, side_geom, plane_idx, plane_kernel)

                    response_signal = electronics_fn(
                        response_signal, side_idx, plane_idx,
                        side_geom.num_wires[plane_idx], cfg.num_time_steps)

                    plane_key = plane_keys[side_idx * 3 + plane_idx]
                    response_signal = noise_fn(
                        plane_key, response_signal, side_idx, plane_idx,
                        side_geom.num_wires[plane_idx], cfg.num_time_steps)

                    response_signal = digitize_fn(response_signal, side_idx, plane_idx)
                    response_signals[(side_idx, plane_idx)] = response_signal

            return response_signals, track_hits, qs_per_side

        self._calculator_jit_raw = _calculate_signals_jit

        # ── Light-only JIT (SCE + recombination, no wire signals) ──
        _sce_fn = _build_sce_fn(None)

        @jax.jit
        def _calculate_light_jit(sim_params, deposits_east, deposits_west,
                                  n_east, n_west):
            side_int_east = compute_side_physics(
                deposits_east, sim_params, cfg.side_geom[0], _sce_fn, _recomb_fn)
            side_int_west = compute_side_physics(
                deposits_west, sim_params, cfg.side_geom[1], _sce_fn, _recomb_fn)
            return (side_int_east.charges, side_int_east.photons,
                    side_int_west.charges, side_int_west.photons)

        self._light_calculator_jit = _calculate_light_jit

        # ── Differentiable path (no @jax.jit — caller wraps in jax.grad) ──
        if self.n_segments is not None:
            n_segments = self.n_segments

            @jax.remat
            def _forward_diff(params, east_deposits, west_deposits):
                """Same pipeline as production but with static n_segments.

                No @jax.jit — called inside user's jax.grad/jit context.
                n_segments is a closure-captured Python int → static fori_loop
                bound → reverse-mode differentiable. @jax.remat bounds
                backward memory to one plane's intermediates at a time.
                """
                response_signals = {}
                side_deposits_list = [east_deposits, west_deposits]

                for side_idx in range(2):
                    deposits = side_deposits_list[side_idx]
                    side_geom = cfg.side_geom[side_idx]

                    sce_fn = _build_sce_fn(params.sce_models)
                    side_int = compute_side_physics(
                        deposits, params, side_geom, sce_fn, _recomb_fn)

                    for plane_idx in range(3):
                        plane_type = cfg.plane_names[side_idx][plane_idx]
                        plane_int = compute_plane_physics(
                            side_int, params, side_geom, plane_idx)

                        response_fn = _build_response_fn_diff(params, side_idx, plane_type)
                        plane_kernel = response_kernels[plane_type]

                        response_signals[(side_idx, plane_idx)] = compute_plane_signal(
                            plane_int, response_fn, n_segments,
                            cfg.response_chunk_size,
                            cfg, side_geom, plane_idx, plane_kernel)

                return response_signals

            self._forward_diff = _forward_diff

    def __call__(self, deposit_data: DepositData, key=None):
        return self.process_event(deposit_data, key=key)

    def process_event(self, deposit_data: DepositData, sim_params=None, key=None):
        """Run production simulation. Returns (response_signals, track_hits) dicts.

        track_hits is self-contained: contains raw per-plane data plus
        'group_to_track' metadata. Pass to finalize_track_hits() after
        moving signals off GPU.
        """
        if sim_params is None:
            sim_params = self._default_sim_params

        print("   Splitting and padding data by side...")
        east_data, west_data, counts = split_and_pad_data(
            deposit_data, self._sim_config.total_pad)

        self._validate_pre_simulation(counts)

        noise_key = key if key is not None else jax.random.PRNGKey(0)

        response_signals, track_hits, qs_per_side = self._calculator_jit_raw(
            sim_params, east_data, west_data, noise_key,
            counts['n_east'], counts['n_west'],
        )

        # Bundle group_to_track and unsplit Q_s fractions
        qs_fractions = None
        if self._sim_config.include_track_hits:
            gids = np.asarray(deposit_data.group_ids)
            tids = np.asarray(deposit_data.track_ids)
            valid = np.asarray(deposit_data.valid_mask)
            max_gid = int(gids[valid].max()) if valid.any() else 0
            group_to_track = np.zeros(max_gid + 1, dtype=np.int32)
            group_to_track[gids[valid]] = tids[valid]
            track_hits['group_to_track'] = group_to_track

            # Unsplit Q_s fractions back to original deposit order
            x_mm = np.asarray(deposit_data.positions_mm[:, 0])
            east_mask = valid & (x_mm < 0)
            west_mask = valid & (x_mm >= 0)
            n_e = min(int(east_mask.sum()), counts['n_east'])
            n_w = min(int(west_mask.sum()), counts['n_west'])
            qs_fractions = np.zeros(len(x_mm), dtype=np.float32)
            qs_fractions[np.where(east_mask)[0][:n_e]] = np.asarray(qs_per_side[0][:n_e])
            qs_fractions[np.where(west_mask)[0][:n_w]] = np.asarray(qs_per_side[1][:n_w])

        self._validate_post_simulation(track_hits, response_signals)

        return response_signals, track_hits, qs_fractions

    def process_event_light(self, deposit_data: DepositData, sim_params=None):
        """Compute per-segment charge and scintillation light.

        Runs only SCE + recombination — no wire signals, response,
        electronics, or noise. Returns arrays in original segment order.

        Parameters
        ----------
        deposit_data : DepositData
            Input deposits (any size).
        sim_params : SimParams, optional
            Override physics parameters. Uses defaults if None.

        Returns
        -------
        result : dict
            Per-side results with keys:
            - 'east': (charges, photons) each jnp.ndarray shape (total_pad,)
            - 'west': (charges, photons) each jnp.ndarray shape (total_pad,)
            - 'n_east': int, valid entries in east arrays
            - 'n_west': int, valid entries in west arrays
            - 'east_idx': np.ndarray, original segment indices for east
            - 'west_idx': np.ndarray, original segment indices for west
        """
        if sim_params is None:
            sim_params = self._default_sim_params

        east_data, west_data, counts = split_and_pad_data(
            deposit_data, self._sim_config.total_pad)

        Q_east, L_east, Q_west, L_west = self._light_calculator_jit(
            sim_params, east_data, west_data,
            counts['n_east'], counts['n_west'])

        x_mm = np.asarray(deposit_data.positions_mm[:, 0])
        valid = np.asarray(deposit_data.valid_mask)

        return {
            'east': (Q_east, L_east),
            'west': (Q_west, L_west),
            'n_east': counts['n_east'],
            'n_west': counts['n_west'],
            'east_idx': np.where(valid & (x_mm < 0))[0][:counts['n_east']],
            'west_idx': np.where(valid & (x_mm >= 0))[0][:counts['n_west']],
        }

    def finalize_track_hits(self, track_hits):
        """Derive track labels from raw group merge state."""
        return finalize_track_hits(track_hits, self._sim_config.num_time_steps)

    @property
    def config(self):
        """Read-only access to SimConfig (detector geometry, mode flags, etc.)."""
        return self._sim_config

    @property
    def default_sim_params(self):
        """Read-only access to default SimParams."""
        return self._default_sim_params

    def to_dense(self, response_signals):
        """Convert response signals to dense (W, T) arrays per plane."""
        from tools.output import to_dense
        return to_dense(response_signals, self._sim_config)

    def to_sparse(self, response_signals, threshold_enc=0):
        """Convert response signals to sparse (wire, time, values) per plane.

        Parameters
        ----------
        threshold_enc : float
            Threshold in electrons. Pixels below this are dropped.
        """
        from tools.output import to_sparse
        threshold_adc = threshold_enc / self._sim_config.electrons_per_adc
        return to_sparse(response_signals, self._sim_config, threshold_adc)

    def warm_up(self):
        """Trigger JIT compilation with dummy data."""
        print("Triggering JIT compilation...")
        pad = self._sim_config.total_pad
        dummy = DepositData(
            positions_mm=jnp.zeros((pad, 3), dtype=jnp.float32),
            de=jnp.zeros((pad,), dtype=jnp.float32),
            dx=jnp.zeros((pad,), dtype=jnp.float32),
            valid_mask=jnp.zeros((pad,), dtype=bool),
            theta=jnp.zeros((pad,), dtype=jnp.float32),
            phi=jnp.zeros((pad,), dtype=jnp.float32),
            track_ids=jnp.zeros((pad,), dtype=jnp.int32),
            group_ids=jnp.zeros((pad,), dtype=jnp.int32),
        )
        _ = self._calculator_jit_raw(
            self._default_sim_params, dummy, dummy,
            jax.random.PRNGKey(0), 0, 0,
        )
        print(f"JIT compilation finished (total_pad={pad:,}).")

    def forward(self, params, deposits):
        """Differentiable forward pass. Returns tuple of 6 signal arrays.

        Uses the same physics pipeline as process_event but without
        @jax.jit (caller wraps in jax.grad). n_segments is a static
        closure value so fori_loop bounds are compile-time constants.

        Parameters
        ----------
        params : SimParams
            All tunable state (physics + optional NN models).
        deposits : DepositData
            Input deposits (caller-constructed via create_deposit_data).
        """
        deposits = pad_deposit_data(deposits, self._sim_config.total_pad)
        east_mask = deposits.valid_mask & (deposits.positions_mm[:, 0] < 0)
        west_mask = deposits.valid_mask & (deposits.positions_mm[:, 0] >= 0)
        east_deposits = deposits._replace(valid_mask=east_mask)
        west_deposits = deposits._replace(valid_mask=west_mask)
        response_signals = self._forward_diff(params, east_deposits, west_deposits)
        return tuple(response_signals[(s, p)] for s in range(2) for p in range(3))

    def forward_segments(self, params, positions_mm, de, dx):
        """Lightweight differentiable forward for segment-like data.

        Constructs DepositData internally with dummy theta/phi/track_ids.
        Use this when angles are not needed (e.g., modified_box recombination).
        For EMB recombination (angle-dependent), use forward() with full
        DepositData including real theta/phi.

        Parameters
        ----------
        params : SimParams
        positions_mm : (N, 3) array
        de : (N,) array — energy deposits in MeV
        dx : float or (N,) array — step size in mm
        """
        N = positions_mm.shape[0]
        deposits = DepositData(
            positions_mm=positions_mm,
            de=de,
            dx=jnp.full(N, dx) if jnp.ndim(dx) == 0 else dx,
            valid_mask=jnp.ones(N, dtype=bool),
            theta=jnp.zeros(N, dtype=jnp.float32),
            phi=jnp.zeros(N, dtype=jnp.float32),
            track_ids=jnp.zeros(N, dtype=jnp.int32),
            group_ids=jnp.zeros(N, dtype=jnp.int32),
        )
        return self.forward(params, deposits)

