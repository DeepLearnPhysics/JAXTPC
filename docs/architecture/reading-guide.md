# Reading Guide — the code spine

This page is a guided read of JAXTPC in **execution order**. Open the source
alongside it: every step names a real function or class you can jump to. By the
end you should understand how a raw list of energy deposits becomes a set of
readout signals, and where to branch off into the physics deep-dives.

The whole framework hangs off one idea: **a detector is N geometrically
identical volumes in a local frame, and one JIT-compiled body handles any
number of them.** Static structure is baked into closures at construction time;
only tunable physics values (`SimParams`) cross the JIT boundary at call time.

!!! tip "Suggested files to keep open"
    `tools/geometry.py`, `tools/config.py`, `tools/simulation.py`,
    `tools/physics.py`. The call chain flows across exactly these four.

---

## 1. `generate_detector(yaml)` — `tools/geometry.py`

The entry point for any run. It reads and **validates** a detector YAML and
returns the raw parsed `dict` — nothing more. It checks that the required
top-level sections exist (`volumes`, `readout`, `simulation`, `medium`,
`electric_field`) and that each volume has a `geometry` block with `ranges` and
`drift_direction`, plus either `planes` (wire) or `readout.type: pixel` with a
`pixel_pitch`/`pixel_shape`.

```python
from tools.geometry import generate_detector
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
```

Derived quantities (anode position, wire counts, diffusion sigmas,
`num_time_steps`) are **not** computed here — they are the job of `config.py`,
which is the next stop.

---

## 2. `create_sim_config` + `create_sim_params` — `tools/config.py`

This is where the **static / dynamic split** is established, and it is the most
important concept in the codebase.

- **`create_sim_config(detector_config, …)`** builds a `SimConfig` NamedTuple:
  array dimensions, mode flags, and a tuple of per-volume `VolumeGeometry`
  objects (each carrying a `DiffusionConfig`). It computes `num_time_steps` from
  the **longest drift across all volumes** plus the pre/post readout-window
  extensions, and derives `output_format` (`'dense'`, `'bucketed'`, or
  `'wire_sparse'`). Everything in `SimConfig` is **static** — closure-captured
  at construction, and changing any of it forces a JIT recompile.

- **`create_sim_params(detector_config, …)`** builds a `SimParams` NamedTuple:
  the tunable physics scalars — `velocity_cm_us`, `lifetime_us`,
  `diffusion_trans_cm2_us`, `diffusion_long_cm2_us`, and a `recomb_params`
  bundle (`ModifiedBoxParams` or `EMBParams`). This is the **only JIT
  argument**; changing these values does **not** recompile.

See [config vs params](config-vs-params.md) for the full split.

---

## 3. `DetectorSimulator.__init__` — the closure factory · `tools/simulation.py`

This is the heart of the framework, and the reason the JIT body can stay simple.
`__init__` does all the one-time, data-independent work and freezes it into
closures. Read it top to bottom:

1. **Flag resolution.** Detects the readout type from volume 0, forces
   compatible flags for the differentiable and pixel paths, and validates chunk
   divisibility (`total_pad % response_chunk_size == 0`).
2. **Builds `SimConfig`** via `create_sim_config`, and stores the default
   `SimParams`.
3. **Loads response kernels once** — `load_response_kernels` (wire) or
   `load_pixel_response_kernel` (pixel) — shared across all volumes. These carry
   the `DKernel` diffusion table.
4. **Computes analytic box dims** for the group-as-bucket track-hits path
   (`compute_box_dims` in `tools/track_hits.py`) and freezes them into
   `cfg.track_hits`. These depend on the group definition + geometry + kernel
   window, **not** on the data.
5. **Builds the per-concern factories:**
    - `_setup_shared_factories` → SCE, response, and recombination closures.
    - `create_electronics_fn_for_volume`, `create_noise_fn_for_volume`,
      `create_digitize_fn_for_volume`, `create_track_hits_fn_for_volume`.

    Each factory captures the static `SimConfig`/`VolumeGeometry` and returns a
    closure whose signature depends on the enabled modes. (These signatures are
    invisible to introspection — hence the prose in the physics pages.)
6. **`_build_jit`** composes those closures into a single
   `process_one_volume(vol_deps, vol_key, sim_params)` and wraps it in
   `iterate(fn, …)`, where `iterate` is `scan_over` (default) or `vmap_over`.

!!! note "The plane loop is Python, unrolled at trace time"
    Inside `process_one_volume` (wire path), `for plane_idx in range(n_planes)`
    is an ordinary Python loop. It runs **once, during tracing**, emitting a
    fully unrolled graph — one straight-line copy of the per-plane physics per
    plane. `plane_idx` is a concrete int inside the trace, so it can index into
    static tuples like `vol_geom.wire_spacings_cm[plane_idx]`. The volume axis,
    by contrast, is mapped by `scan`/`vmap` and must have uniform shapes — which
    is why all volumes share reference geometry in the local frame.

See [simulator](simulator.md) for the factory/closure pattern in depth
(with diagram D3) and scan-vs-vmap iteration.

---

## 4. `process_event` — production entry point · `tools/simulation.py`

This is the host-side wrapper that runs one event through the production JIT.

```python
response_signals, track_hits, deposits = simulator.process_event(deposits, key=key)
```

Its job is orchestration, not physics:

1. **Stack volumes.** `jax.tree.map(lambda *xs: jnp.stack(xs), *deposits.volumes)`
   turns the tuple of per-volume `VolumeDeposits` into leading-axis-stacked
   arrays for `iterate`.
2. **Call `_calculator_jit(sim_params, stacked_deps, noise_key)`** — the single
   compiled function. Everything physics happens inside here.
3. **Host-side unstack + overflow checks.** Signals and track-hits are unstacked
   into `{(vol, plane): …}` dicts. This is where the capacity **overflow
   `RuntimeError`s** are raised — `max_active_buckets`, `max_keys`, and `maxg`
   are all checked here on the host (values known after the JIT returns, or from
   the deposits themselves). See [capacities](../concepts/capacities.md).
4. **Optional coherent noise** — `tools.coherent_noise.add_coherent_noise`,
   applied off-JIT in NumPy (it is per-wire-group, not per-deposit; wire + dense
   only, enforced at construction).
5. **Rebuild filled `DepositData`** — the returned deposits carry `charge`,
   `photons`, and `qs_fractions` from the physics pass.

The sibling `process_event_light` and `forward`/`forward_segments` call into the
same closures but skip most of this — see [execution paths](execution-paths.md).

---

## 5. The shared physics body — `tools/physics.py`

`process_one_volume` calls straight into `physics.py`. **Both the production and
differentiable paths call these functions identically** — no `@jax.jit`
decorators, because they run inside the simulator's outer JIT.

Read them in this order:

1. **`compute_volume_physics(deposits, sim_params, vol_geom, sce_fn, recomb_fn)`**
   → `VolumeIntermediates`. Runs recombination (`recomb_fn`), queries the SCE map
   (`sce_fn`), computes the base drift to the anode (local frame: anode at
   `x=0`, drift toward `−x`), and applies SCE drift corrections.

    !!! warning "The single padding mask lives here"
        `charges *= jnp.arange(N) < n_actual` (and the same for photons). This is
        the **only** place padding is masked. Every downstream stage trusts that
        padding entries have `charges = 0` and therefore contribute nothing. See
        [padding & masking](../concepts/padding-masking.md).

2. **`compute_plane_physics(vol_int, …, plane_idx, …)`** → `PlaneIntermediates`.
   Per-plane: subtracts the plane's distance-from-anode offset
   (`correct_drift_for_plane`), computes lifetime `attenuation = exp(−t/τ)`,
   zeroes charges **outside the readout window**, and projects `(y,z)` onto the
   closest wire (`compute_wire_distances`). (The pixel path uses
   `compute_pixel_physics` instead — no per-plane correction, plus pixel
   digitization.)

3. **`compute_chunk_response(plane_int, response_fn, start, chunk_size, …)`** —
   slices one chunk of deposits, prepares them (`prepare_deposit_for_response`,
   vmapped), and evaluates `response_fn` to get the `(chunk, kW, kH)` kernel
   contributions. `response_fn` interpolates the `DKernel` table at each
   deposit's diffusion level `s = sqrt(drift_distance / max_drift)`.

4. **`compute_plane_signal(plane_int, response_fn, n_actual, chunk_size, …)`** —
   the accumulation loop: `jax.lax.fori_loop` over chunks, each calling
   `compute_chunk_response` then scatter-adding into the dense `(num_wires,
   num_time_steps)` array via `accumulate_response_signals`.

    !!! note "The two execution paths diverge on exactly one line"
        `compute_plane_signal` branches on `isinstance(n_actual, int)`. A Python
        `int` (differentiable path, static loop bound) uses `min(...)`; a traced
        JAX value (production path) uses `jnp.minimum(...)`. That single check is
        the entire structural difference between the two paths at this level. See
        [execution paths](execution-paths.md).

The bucketed variants (`compute_bucket_maps`, `compute_plane_signal_bucketed`)
follow the same shape but scatter into sparse `(max_buckets, B1, B2)` tiles
instead of a dense array — a memory-saver for wire readout.

---

## Where to go next

You now have the spine. Branch into the depth pages:

- [Data model](data-model.md) — `DepositData` / `VolumeDeposits` and the three
  `*Intermediates` types that flow through the body above.
- [Simulator](simulator.md) — the closure/factory construction in depth.
- [Execution paths](execution-paths.md) — production vs differentiable vs light.
- [Pipeline overview](pipeline-overview.md) — the full response chain, stage by
  stage, with links into each physics module.
- Physics deep-dives: [recombination](../physics/recombination.md),
  [drift & diffusion](../physics/drift-diffusion.md),
  [response kernels](../physics/response-kernels.md),
  [electronics & noise](../physics/electronics-noise.md),
  [SCE](../physics/sce.md), [units](../physics/units.md).
- [Truth / track-hits](../truth/track-hits.md) — group→track correspondence.
