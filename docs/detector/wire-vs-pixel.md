# Wire vs. pixel readout

JAXTPC supports two readout technologies from the same physics pipeline, chosen
per volume in the [config](config-schema.md). They share recombination, drift,
diffusion, and the DKernel response, but diverge at projection, accumulation, and
everything downstream. The differences are deliberate and affect units, output
shapes, and which post-processing stages run.

## The two readouts at a glance

| | **Wire** | **Pixel** |
|---|---|---|
| Sensor geometry | 3 angled 1-D wire planes (U/V/Y) | Single 2-D pixel grid (y×z) |
| Config | `planes[]` list (`angle`, `wire_spacing`, `distance_from_anode`) | `readout: {type: pixel, pixel_pitch, pixel_shape}` |
| Kernel | Dimensionless field-impulse fraction | Chip gain (ADC/e⁻) baked in |
| Hits-stage units | **ENC** (electrons) | **ADC** |
| Sensor-signal units | **ADC** (post-digitize) | **ADC** (same pass) |
| Post-chain | electronics → noise → digitize | **skipped** |
| Passes | response → post-chain | **single pass** (signal + truth together) |
| Output format | dense / bucketed / wire-sparse | dense pixel grid |

## Wire readout

Each wire volume has three angled planes — first induction (U, +60°), second
induction (V, −60°), and collection (Y, 0°) in the SBN convention (other presets
use different angles). For each deposit and plane, the (y, z) position is
projected onto the nearest wire index at that plane's `angle`/`wire_spacing`.

The wire response kernel is a **dimensionless** field-impulse fraction, so
`charge (electrons) × kernel → ENC` (equivalent noise charge, in electrons). The
wire signal then runs the full post-chain:

```
response (ENC) → electronics (RC⊗RC) → noise → digitize → sensor (ADC)
```

Electronics shaping, intrinsic + coherent noise, and 12-bit digitization all run
inside the same JIT function. The sensor output is in **ADC**; the hits stage is
in **ENC**. Wire supports three accumulation formats — `dense`, `bucketed`
(a memory-saver), and `wire_sparse` (post-electronics).

## Pixel readout

A pixel volume has a single 2-D grid (`pixel_shape = [num_py, num_pz]`) at
`pixel_pitch`. Each deposit maps to a center pixel plus a fractional offset, and
the pixel kernel spreads charge over neighboring pixels in (y, z, t).

The pixel kernel has the **chip gain baked in** (ADC per drift electron), so the
response is already in **ADC** — there is no separate electronics, noise, or
digitize step:

```
response → sensor (ADC)     # electronics / noise / digitize are skipped
```

### Single-pass: signal and truth from one response pass

The defining pixel difference (PLAN §8, row 9): the pixel branch produces **both
the sensor signal and the per-group truth decomposition from a single response
pass**. The wire path derives truth (track-hits) as a separate concern layered
on the sensor pipeline; the pixel path reads both out of the same accumulation,
because the readout is a plain 2-D grid with the gain already applied. This makes
pixel simulation cheaper per event and is why the post-chain stages are absent —
there is nothing to shape or digitize after the gain-baked response.

!!! note "Pixel bucketed mode exists but is not wired"
    A complete pixel analog of the wire bucketed accumulation exists in the code
    (`physics.compute_pixel_bucket_maps` / `compute_pixel_signal_bucketed`,
    `wires.scatter_contributions_to_pixel_buckets_batched`, and the matching
    decode in `output.py`), and its output is already decodable. But
    `simulation.py`'s pixel branch never dispatches it — pixel always uses the
    dense/box path. It is kept as a reference implementation (PLAN §8b), one
    dispatch site away from being live, not an end-to-end supported mode.

## Units by readout

The same threshold field names mean **different units** depending on readout —
this is the footgun that bites most often:

| Threshold | Wire | Pixel |
|---|---|---|
| `inter_thresh` | ENC | ADC |
| `corr_threshold` / `hits_threshold` | ENC | ADC |
| `threshold_adc` | ADC | ADC |

The wire hits stage is ENC while the pixel hits stage is ADC, so a threshold set
for one readout is not portable to the other. See [units](../physics/units.md)
for the canonical table and the full explanation of where each threshold is
applied.

## Output shapes

- **Wire** — one signal per plane; sparse form is
  `{(vol, plane): {'wire', 'time', 'values'}}`. Dense per plane is
  `(num_wires, num_time_steps)`. `num_wires` is derived from geometry and differs
  per plane and per angle.
- **Pixel** — one grid per volume, `(num_py, num_pz, num_time_steps)` dense; the
  plane name is simply `('Pixel',)`. There is no U/V/Y split.

## Try it

The runnable pixel walkthrough:
[`notebooks/readout/pixel_simulation.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/readout/pixel_simulation.ipynb).

## See also

- [Config schema](config-schema.md) — the `planes` vs. `readout: {type: pixel}` YAML.
- [Presets](presets.md) — `cubic_pixel_config.yaml` (2 volumes) and
  `dune_ndlar_config.yaml` (70 volumes) are the pixel presets.
- [Units](../physics/units.md) — the canonical ENC-vs-ADC / threshold-units reference.
- [Response kernels](../physics/response-kernels.md) — how the DKernel is built and interpolated.
