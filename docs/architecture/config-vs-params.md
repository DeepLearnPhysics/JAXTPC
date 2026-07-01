# Config vs Params

The simulation is governed by two `NamedTuple` bundles with opposite lifecycles.
Getting this split right is what keeps JAXTPC fast: **`SimConfig` is static**
(baked into the JIT closure — change anything and you recompile), while
**`SimParams` is dynamic** (the single traced argument — change it freely with no
recompilation).

## The two bundles

| | `SimConfig` | `SimParams` |
|---|---|---|
| **Role** | Static configuration | Tunable physics |
| **How it reaches JIT** | Captured in the closure at `DetectorSimulator.__init__` | Passed as the JIT argument each call |
| **Contents** | Array dims, mode flags, per-volume `VolumeGeometry`, plane names, output format | velocity, lifetime, diffusion coeffs, recombination params, optional NN / SCE models |
| **Change cost** | **New closure → JIT recompilation** | **No recompilation** |
| **Built by** | `create_sim_config(detector_config, ...)` | `create_sim_params(detector_config, ...)` |
| **Differentiable?** | No | Yes (velocity, lifetime, diffusion, recombination carry gradients) |

## `SimConfig` — static, closure-captured

```python
class SimConfig(NamedTuple):
    # time grid
    num_time_steps; time_step_us; pre_window_us; post_window_us
    # array dims / batching
    total_pad; response_chunk_size; max_wires
    # mode flags
    use_bucketed; include_track_hits
    include_intrinsic_noise; include_coherent_noise
    include_electronics; include_digitize
    max_active_buckets
    # volumes
    n_volumes; volumes            # (VolumeGeometry, ...) per volume
    plane_names                   # e.g. (('U','V','Y'), ('U','V','Y'))
    output_format                 # 'dense' | 'bucketed' | 'wire_sparse'
    # readout
    electrons_per_adc; noise_spectrum_path
    track_hits                    # TrackHitsConfig or None
```

`create_sim_config(detector_config, ...)` builds this from the parsed YAML. Two
derivations are worth calling out because they are not free parameters:

- **`num_time_steps`** is computed from the **longest drift across all volumes**
  plus the optional readout windows:

  ```python
  max_drift_time = longest_drift / velocity
  total_window   = pre_window_us + max_drift_time + post_window_us
  num_time_steps = int(ceil(total_window / time_step_us)) + 1
  ```

  `pre_window_us` / `post_window_us` come from `readout.pre_window_fraction` /
  `post_window_fraction` (fractions of the max drift time).

- **`output_format`** is derived from the mode flags, not set directly:

  ```python
  if use_bucketed and include_electronics: 'wire_sparse'
  elif use_bucketed:                       'bucketed'
  else:                                    'dense'
  ```

`create_sim_config` also builds one `VolumeGeometry` per volume (each with its own
`DiffusionConfig`, whose `K_wire`/`K_time` half-widths and max sigmas come from
the volume's max drift), assigns `plane_names` (`('Pixel',)` for pixel volumes,
`('U','V','Y')[:n_planes]` for wire), and constructs a default `TrackHitsConfig`
when `include_track_hits=True`.

!!! warning "What forces a recompilation"
    Because `SimConfig` is captured in the closure, changing **any** of these
    triggers a fresh JIT trace:

    - array dimensions — `total_pad`, `response_chunk_size`, `max_wires`,
      `max_active_buckets`, `num_time_steps`;
    - **any mode flag** — `include_electronics`, `include_intrinsic_noise`,
      `include_coherent_noise`, `include_digitize`, `include_track_hits`,
      `use_bucketed`;
    - **detector geometry** — number of volumes, wire counts/angles/spacings,
      pixel shape, anything in `volumes` / `plane_names`;
    - the derived `output_format` and the `track_hits` capacities.

    Since `num_time_steps` and `output_format` are *derived*, a change to
    sampling rate, drift velocity, geometry extent, or the mode flags recompiles
    indirectly. Keep `SimConfig` fixed across an event batch — that single JIT
    compilation is what makes the batch loop cheap.

## `SimParams` — dynamic, the JIT argument

```python
class SimParams(NamedTuple):
    velocity_cm_us
    lifetime_us
    diffusion_trans_cm2_us; diffusion_long_cm2_us
    recomb_params        # ModifiedBoxParams or EMBParams
    response_models      # {(vol_idx, plane_type): eqx.Module} or None
    sce_models           # tuple of per-volume models, or None
```

`create_sim_params(detector_config, recombination_model=..., response_models=...,
sce_models=...)` builds this from the same YAML. It selects the recombination
bundle by model name — `ModifiedBoxParams` for `'modified_box'`/`'passthrough'`,
`EMBParams` for `'emb'` — and converts YAML units (e.g. diffusion cm²/s → cm²/µs,
lifetime ms → µs).

Everything in `SimParams` can change between calls **without recompiling**: sweep
electron lifetime, retune recombination, or swap SCE/NN models and the same
compiled function runs. In the differentiable path
([`forward`](execution-paths.md)), these scalars are exactly the parameters that
carry gradients.

!!! note "Rule of thumb"
    If a value changes an **array shape or a code branch**, it belongs in
    `SimConfig` (and costs a recompile). If it is a **physics number** the model
    should be able to sweep or differentiate, it belongs in `SimParams`.

## Putting them together

```python
from tools.config import create_sim_config, create_sim_params
from tools.geometry import generate_detector

detector_config = generate_detector('config/cubic_wireplane_config.yaml')
sim_config = create_sim_config(detector_config, include_digitize=True)
sim_params = create_sim_params(detector_config, recombination_model='emb')
```

`sim_config` is handed to `DetectorSimulator(...)` (captured in the closure);
`sim_params` is passed per call to `process_event` / `forward` — or omitted, in
which case the simulator uses the params it was constructed with.
