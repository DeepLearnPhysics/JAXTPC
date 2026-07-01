# Detector configuration schema

A JAXTPC detector is defined entirely in a YAML file. `generate_detector()`
([`tools/geometry.py`][geo]) parses and validates the file into a plain dict;
`create_sim_config()` and `create_sim_params()` ([`tools/config.py`][cfg]) then
derive the static geometry (`SimConfig`) and dynamic physics scalars
(`SimParams`) from it. This page is the authoritative reference for every
section the loader consumes.

[geo]: https://github.com/DeepLearnPhysics/JAXTPC/blob/main/tools/geometry.py
[cfg]: https://github.com/DeepLearnPhysics/JAXTPC/blob/main/tools/config.py

!!! info "What is validated vs. what is derived"
    `generate_detector()` only checks that the five required top-level sections
    (`volumes`, `readout`, `simulation`, `medium`, `electric_field`) exist and
    that each volume has `geometry.ranges`, `geometry.drift_direction`, and
    **either** a `planes` list (wire) **or** a `readout.type: pixel` block with
    `pixel_pitch` + `pixel_shape`. Everything else — wire counts, diffusion
    sigmas, number of time steps, output format — is *computed* downstream in
    `create_sim_config()`.

## Required top-level sections

| Section | Purpose |
|---|---|
| `volumes` | List of independent drift volumes (geometry + readout per volume). |
| `readout` | Global readout electronics: sampling rate, ADC conversion, window fractions, digitization. |
| `simulation` | Physics scalars: drift, diffusion, lifetime, recombination model + parameters. |
| `medium` | Liquid-argon properties (density, W_ion, excitation ratio, T, P). |
| `electric_field` | Drift-field strength (V/cm). |

Missing any of these raises `KeyError` at `generate_detector()` time.

## Annotated example (wire readout)

The default `config/cubic_wireplane_config.yaml` — a dual-TPC, SBND-scale
detector with U/V/Y wire planes:

```yaml
volumes:
  - id: 0                                     # (1)
    description: "TPC East (x < 0)"
    geometry:
      ranges: [[-216.0, 0.0], [-216.0, 216.0], [-216.0, 216.0]]  # cm [x, y, z] (2)
      drift_direction: -1                     # (3)
    planes:                                   # (4)
      - {plane_id: 0, type: first_induction,  angle:  60.0, wire_spacing: 0.3, distance_from_anode: 0.6, bias_voltage: -200.0}
      - {plane_id: 1, type: second_induction, angle: -60.0, wire_spacing: 0.3, distance_from_anode: 0.3, bias_voltage:    0.0}
      - {plane_id: 2, type: collection,       angle:   0.0, wire_spacing: 0.3, distance_from_anode: 0.0, bias_voltage:  500.0}
  - id: 1
    description: "TPC West (x > 0)"
    geometry:
      ranges: [[0.0, 216.0], [-216.0, 216.0], [-216.0, 216.0]]
      drift_direction: 1                      # opposite drift → shared central cathode
    planes:
      - {plane_id: 3, angle:  60.0, wire_spacing: 0.3, distance_from_anode: 0.6}
      - {plane_id: 4, angle: -60.0, wire_spacing: 0.3, distance_from_anode: 0.3}
      - {plane_id: 5, angle:   0.0, wire_spacing: 0.3, distance_from_anode: 0.0}

readout:
  sampling_rate: 2.0            # MHz → time_step_us = 1/sampling_rate         (5)
  electrons_per_adc: 182        # informational conversion factor
  pre_window_fraction: 0.3      # fraction of max drift time added before t=0  (6)
  post_window_fraction: 0.3     # fraction of max drift time added after max drift
  digitization:
    n_bits: 12                  # ADC resolution
    pedestal_collection: 410    # baseline ADC for Y (collection) planes
    pedestal_induction: 1843    # baseline ADC for U/V (induction) planes
    gain_scale: 1.0

simulation:
  drift:
    velocity: 1.6                 # mm/us (converted to cm/us internally)      (7)
    longitudinal_diffusion: 7.2   # cm^2/s
    transverse_diffusion: 12.0    # cm^2/s
    electron_lifetime: 10.0       # ms
  charge_recombination:
    model: emb                    # 'modified_box' | 'emb' | 'passthrough'     (8)
    recomb_parameters:
      alpha: 0.93                 # Modified Box (ArgoNeuT)
      beta: 0.212
      alpha_emb: 0.904            # Ellipsoid Modified Box (ICARUS 2024)
      beta_90: 0.204
      R_anisotropy: 1.25

medium:
  type: liquid_argon
  properties:
    density: 1.396                # g/cm^3
    ionization_energy: 23.6       # eV per ion pair (W_ion)
    excitation_ratio: 0.21        # N_ex / N_i
  temperature: 87.0              # K
  pressure: 1.0                  # atm

electric_field:
  field_strength: 500.0          # V/cm                                        (9)
```

1. `id` is informative; volume order in the list is what the loader uses.
2. `ranges` are **global** cm bounds `[[x_lo,x_hi],[y_lo,y_hi],[z_lo,z_hi]]`.
   Deposits are split by x into volumes and then transformed to a local,
   yz-centered, anode-at-0 frame (see [coordinates](../concepts/coordinates.md)).
3. `drift_direction` is `+1` (drift toward `+x`, anode at `x_hi`) or `-1`
   (drift toward `-x`, anode at `x_lo`). Two opposite-drift volumes model a
   central-cathode dual TPC.
4. Wire volumes carry a `planes` list; pixel volumes replace it with a
   per-volume `readout` block (see below).
5. `sampling_rate` (MHz) sets `time_step_us = 1 / sampling_rate`.
6. Window fractions extend the readout window; `num_time_steps` is derived from
   the longest drift time plus these windows (default 0.0 if omitted).
7. `velocity` is given in **mm/µs** in YAML and converted to cm/µs internally.
8. Recombination model + shared parameters — see
   [recombination](../physics/recombination.md).
9. Field strength enters recombination and (via `SimParams`) the physics.

## Field reference

### `volumes[]`

| Key | Required | Type | Meaning |
|---|---|---|---|
| `id` | no | int | Informative label; list order defines volume index. |
| `description` | no | str | Free text (printed in summaries). |
| `geometry.ranges` | **yes** | `[[x_lo,x_hi],[y_lo,y_hi],[z_lo,z_hi]]` | Global volume bounds in cm. |
| `geometry.drift_direction` | **yes** | `+1` / `-1` | Drift toward `+x` / `-x`; anode is the corresponding face. |
| `planes` | wire only | list | Wire-plane list (see below). Required unless `readout.type: pixel`. |
| `readout` | pixel only | dict | Per-volume pixel readout block (see below). |

### `volumes[].planes[]` (wire readout)

| Key | Required | Type | Meaning |
|---|---|---|---|
| `plane_id` | no | int | Informative global plane label. |
| `type` | no | str | `first_induction` / `second_induction` / `collection` (labels U/V/Y). |
| `angle` | **yes** | deg | Wire angle from vertical; sets the (y,z)→wire projection. |
| `wire_spacing` | **yes** | cm | Wire pitch; must be > 0. Sets `num_wires` (derived). |
| `distance_from_anode` | **yes** | cm | Plane distance from the anode; the furthest plane sets the drift target. |
| `bias_voltage` | no | V | Informational only. |

Wire counts, index offsets, and per-wire lengths are **computed** by
`create_sim_config()` from `ranges`, `angle`, and `wire_spacing` — they are not
specified in YAML.

### `volumes[].readout` (pixel readout)

A volume becomes a pixel volume when its `readout.type` is `pixel`. It then has
**no** `planes` list.

| Key | Required | Type | Meaning |
|---|---|---|---|
| `type` | **yes** | `pixel` | Selects pixel readout for this volume. |
| `pixel_pitch` | **yes** | cm | Square pixel pitch; used as the transverse spatial reference for diffusion. |
| `pixel_shape` | **yes** | `[num_py, num_pz]` | Pixel grid dimensions in (y, z). |

The pixel-grid origin is derived in the **local, yz-centered** frame
(`origin = range_min − center = −half_extent`), so pixel indexing is correct
for non-origin-centered volumes. See
[wire vs. pixel](wire-vs-pixel.md) for the readout differences.

### `readout` (global)

| Key | Required | Type | Meaning |
|---|---|---|---|
| `sampling_rate` | **yes** | MHz | Sets `time_step_us = 1/sampling_rate`. |
| `electrons_per_adc` | no (default 182) | float | Informational conversion factor (not consumed in the wire JIT path — see [units](../physics/units.md)). |
| `pre_window_fraction` | no (default 0.0) | float | Extra readout window before drift `t=0`, as a fraction of max drift time. |
| `post_window_fraction` | no (default 0.0) | float | Extra readout window after max drift. |
| `digitization.n_bits` | no | int | ADC resolution (wire digitize step). |
| `digitization.pedestal_collection` | no | int | Baseline ADC for collection (Y) planes. |
| `digitization.pedestal_induction` | no | int | Baseline ADC for induction (U/V) planes. |
| `digitization.gain_scale` | no | float | Gain rescale factor. |

!!! note "Global vs. per-volume `readout`"
    The **top-level** `readout` section holds electronics settings shared across
    the detector (sampling rate, digitization). The **per-volume** `readout`
    block (pixel volumes only) holds that volume's pixel geometry. They are
    distinct keys at different nesting levels.

### `simulation`

| Key | Type | Meaning |
|---|---|---|
| `drift.velocity` | mm/µs | Drift velocity (converted to cm/µs internally). |
| `drift.longitudinal_diffusion` | cm²/s | Longitudinal diffusion coefficient. |
| `drift.transverse_diffusion` | cm²/s | Transverse diffusion coefficient. |
| `drift.electron_lifetime` | ms | Attenuation lifetime. |
| `charge_recombination.model` | str | `modified_box`, `emb`, or `passthrough`. |
| `charge_recombination.recomb_parameters` | dict | `alpha`, `beta` (Modified Box) + `alpha_emb`, `beta_90`, `R_anisotropy` (EMB). |

Optional `simulation.coherent_noise` (see `cubic_wireplane_config.yaml`) carries
the coherent per-group noise parameters (`group_size`, `beta`, `rms_adc`,
`corner_freq_hz`, `spectral_slope`, `sampling_rate_hz`); see
[electronics & noise](../physics/electronics-noise.md).

### `medium`

| Key | Type | Meaning |
|---|---|---|
| `type` | str | `liquid_argon`. |
| `properties.density` | g/cm³ | LAr density (enters recombination ξ). |
| `properties.ionization_energy` | eV | W_ion, energy per ion pair. |
| `properties.excitation_ratio` | — | N_ex/N_i (α), universal LAr constant. |
| `temperature` | K | Operating temperature. |
| `pressure` | atm | Operating pressure. |

### `electric_field`

| Key | Type | Meaning |
|---|---|---|
| `field_strength` | V/cm | Drift-field magnitude; enters recombination and `SimParams`. |

## See also

- [Shipped presets](presets.md) — the seven bundled detector configs.
- [Wire vs. pixel readout](wire-vs-pixel.md) — the two readout paths.
- [Config vs. params](../architecture/config-vs-params.md) — static `SimConfig`
  (recompiles) vs. dynamic `SimParams`.
- [Coordinates](../concepts/coordinates.md) — the local-frame transform.
- [Adding a detector](../contributing/adding-a-detector.md) — authoring a new config.
