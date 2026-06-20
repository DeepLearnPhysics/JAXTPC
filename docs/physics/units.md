# Units convention: ENC vs ADC

The wire and pixel paths use **different unit conventions at the hits stage**.
This is intentional, and it is the single most common source of confusion.

| Readout | Kernel | Hits stage | Sensor signal | Pipeline |
|---|---|---|---|---|
| **Wire** | dimensionless e⁻-impulse fraction | **ENC** (electrons) | **ADC** (12-bit, post-digitize) | response (ENC) → electronics → noise → digitize → ADC |
| **Pixel** | ADC per drift-electron (gain baked in) | **ADC** | **ADC** (no separate digitize) | response → done (signal and hits from one pass) |

**Why they differ:** the wire path applies the field response *before*
electronics, so its kernel is a dimensionless field-impulse fraction and the
intermediate hits are in electrons (ENC). The pixel path bakes the chip's
ADC/electron gain into the kernel and skips the digitize step entirely, so its
hits and sensor signal are already in ADC.

## Thresholds mean different things per readout

The same threshold field carries different units depending on readout. This is
the part that bites:

| Threshold | Wire | Pixel | Where applied |
|---|---|---|---|
| `inter_thresh` | ENC | ADC | in-JIT box compaction / merge pruning (`track_hits.py`) |
| `corr_threshold` / `hits_threshold` | ENC | ADC | host, CSR encode (`production/save.py`) |
| `threshold_adc` | ADC | ADC | host, `to_sparse` (sensor only, `tools/output.py`) |

So when you set a threshold, ask **which readout** and **which stage** — an
`inter_thresh` of `1.0` means 1 electron for wire but 1 ADC count for pixel.

## Wire kernel NPZ caveat

`tools/responses/{U,V,Y}_plane_kernel.npz` carry `units = 'ADC_per_electron'`
metadata plus an `adc_per_electron ≈ 0.005` field. **In this pipeline the kernel
values are treated as a dimensionless field-impulse contribution**
(`intensity (electrons) × kernel → ENC`). The `adc_per_electron` /
`electrons_per_adc` metadata is *not* consumed in the JIT path — it's
informational, reflecting the kernel's first-principles calibration source. The
"ADC" produced by `_digitize_signal` is just the quantization step (round +
pedestal + clip) applied to values the electronics chain has already shaped to
ADC scale.

## Practical guidance

- Reading wire hits? They're **electrons**. Reading pixel hits? They're **ADC**.
- Converting electrons ↔ ADC for wire sensor output uses `cfg.electrons_per_adc`.
- When in doubt, print the magnitudes: wire ENC values are in the hundreds–
  thousands of electrons; pixel ADC values are small integers/counts.
