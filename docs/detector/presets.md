# Shipped detector presets

JAXTPC ships ready-to-run YAML configs for several real and reference detectors
under `config/`. All follow the [config schema](config-schema.md); pass one to
`generate_detector()` to build a detector. Every geometry uses the same
multi-volume machinery — the presets differ only in volume count, readout type,
and scale.

| Config file | Volumes | Readout | Scale / notes |
|---|---|---|---|
| `cubic_wireplane_config.yaml` | **2** | Wire U/V/Y (±60°, 0°) | Default. Dual-TPC, SBND-scale cube (216 cm/side per volume), central cathode. |
| `cubic_pixel_config.yaml` | **2** | Pixel (1000×1000) | Same geometry as cubic_wireplane, pixel readout at 4.32 mm pitch. |
| `sbnd_config.yaml` | **2** | Wire U/V/Y (±60°, 0°) | SBND (Fermilab SBN near detector). Dual-drift, 2 m drift, 4×4×5 m active. |
| `microboone_config.yaml` | **1** | Wire U/V/Y (±60°, 0°) | MicroBooNE. Single-drift TPC, 2.56 m drift, 2.33×10.37 m readout. |
| `icarus_config.yaml` | **2** | Wire (90°, ∓30°) | ICARUS T600, **one cryostat** = 2 opposite-drift TPCs. Non-SBN wire angles; 1.5 mm 1st-induction pitch. Full T600 = 2 cryostats = 4 volumes. |
| `dune_fd1_config.yaml` | **4** | Wire U/V/Y (±35.7°, 0°) | DUNE FD Module 1 (horizontal drift). A-C-A-C-A: 4 drift volumes, 3.594 m drift each, 4.669/4.79 mm pitch. |
| `dune_ndlar_config.yaml` | **70** | Pixel (800×256) | DUNE ND-LAr / ArgonCube. 35 modules × 2 drifts, LArPix 3.72 mm pitch, 46.8 cm drift per side. |

!!! note "Volume counting"
    A "volume" is one independent drift region (one anode-facing sub-detector),
    not a whole cryostat. A central-cathode module with drift in both directions
    is **two** volumes. The counts above are the number of `- id:` entries in
    each YAML.

## Wire-plane presets

**`cubic_wireplane_config.yaml`** — the default. Two 216 cm cubes sharing a
central cathode, U/V/Y planes at ±60° and 0° with 3 mm pitch. Wire-scale but
generic, good for tutorials and closure tests.

**`sbnd_config.yaml`** — SBND, the Short-Baseline Near Detector at Fermilab.
Dual-drift TPC with a central cathode and two symmetric 2 m drift volumes;
4×4×5 m active LAr, MicroBooNE-family cold electronics.

**`microboone_config.yaml`** — MicroBooNE, a single-drift TPC (cathode at
x = 256 cm, anode planes at x = 0). The only single-volume preset. Config uses
design 500 V/cm values (the detector operated at 273 V/cm).

**`icarus_config.yaml`** — ICARUS T600, modeling **one cryostat** as two
opposite-drift TPCs. Wire angles differ from the SBN convention (horizontal
1st-induction, ±30° from vertical for the others) and the 1st-induction pitch is
1.5 mm. Sampling is 2.5 MHz and `electrons_per_adc` is 75 (ICARUS calibration).
The full T600 is two cryostats — instantiate the config twice for 4 volumes.

**`dune_fd1_config.yaml`** — DUNE Far Detector Module 1 (horizontal drift). Four
drift volumes in an A-C-A-C-A arrangement, 3.594 m drift each, wire angles
±35.7° and 4.669/4.79 mm pitch. Induction wire counts are approximate because the
continuous rectangular model does not capture per-APA wire segmentation.

## Pixel presets

**`cubic_pixel_config.yaml`** — the same dual-cube geometry as
`cubic_wireplane_config`, but each volume uses pixel readout: 4.32 mm pitch,
1000×1000 pixels covering the 432 cm y/z span. The direct wire↔pixel comparison
config.

**`dune_ndlar_config.yaml`** — DUNE ND-LAr / ArgonCube, the largest preset. 35
modules in a 5×7 grid, each dual-drift → **70** pixel volumes. LArPix readout at
3.72 mm pitch, 800×256 pixels per volume, 46.8 cm drift per side. Demonstrates
the framework scaling to many volumes.

## See also

- [Config schema](config-schema.md) — the full YAML reference.
- [Wire vs. pixel readout](wire-vs-pixel.md) — how the two readout types differ.
- [Adding a detector](../contributing/adding-a-detector.md) — author your own.
