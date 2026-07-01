# Electronics, Noise & Digitization

After the wire response is accumulated, three optional stages shape it into a
realistic digitized waveform: **electronics shaping**, **noise injection**, and
**digitization**. Each is built as a per-volume, per-plane closure at simulator
construction and runs inside the JIT graph (coherent noise is the one host-side
exception). The **pixel readout skips this entire chain** — the pixel kernel
already produces ADC (see [wire vs pixel](../detector/wire-vs-pixel.md)).

Units flow through this chain: the wire response enters as **ENC** (electrons),
electronics + noise shape it into ADC-scale magnitudes, and digitize quantizes to
**ADC**. Full accounting in [units](units.md).

## Electronics: RC⊗RC shaping

`tools/electronics.py` convolves each wire waveform with an RC⊗RC (two-stage RC)
amplifier impulse response:

```
h(t) = δ(t) + (t/τ − 2)·(1/τ)·e^(−t/τ)·dt
```

with `τ = 1000 µs` and the kernel truncated at `n_tau = 3` time constants
(`create_rcrc_response`, `load_electronics_response`). The same kernel is used for
U/V/Y planes.

The convolution is done **sparsely, via FFT, on active wires only**
(`electronics_response_core`):

1. Find wires whose max abs value clears a threshold; gather up to
   `electronics_chunk_size` of them (`jnp.where(..., size=chunk_size)`).
2. FFT-convolve the gathered rows with the response
   (`rfft → multiply → irfft`), using a power-of-2 `fft_size` sized for linear
   convolution.
3. Trim to `num_time`, zero padding rows beyond the active count, and scatter back
   to the dense `(num_wires, num_time)` array.

In bucketed mode the analogous path (`buckets_to_active_wires` →
`electronics_convolve_active`) converts sparse buckets to wire-sparse rows first,
then applies the same FFT convolution. Enabled by `include_electronics`; a no-op
closure is substituted otherwise or for pixel volumes.

## Noise

JAXTPC has **two independent, separately-toggled** noise components. They model
different physics and live in different files.

| | Intrinsic (incoherent) | Coherent |
|---|---|---|
| File | `tools/noise.py` | `tools/coherent_noise.py` |
| Config flag | `include_intrinsic_noise` | `include_coherent_noise` |
| Correlation | **Per-wire** (each wire independent) | **Per-group** (wires in a group share one waveform) |
| Source model | MicroBooNE ENC vs wire length (arXiv:1705.07341 Eq. 3.6) | MicroBooNE coherent noise |
| Implementation | JAX, inside JIT (also host `add_noise`) | NumPy, host-side (off-JIT) |
| Runs where | GPU, in the sim graph | CPU, after sim, added to signals |

### Intrinsic (per-wire) ENC model

The RMS noise per wire follows the MicroBooNE parameterization

```
ENC ~ RMS_ADC = sqrt(x² + (y + z·L)²)
```

where `L` is the wire length (m), `x` is white/parallel noise, `y` is the series
noise baseline, and `z` is the wire-capacitance coupling (ADC/m); `x, y, z` are
loaded from `config/noise_spectrum.npz` (`load_noise_params`). Two physically
distinct components are generated (`_noise_core`):

- **Series noise** — shaped by an empirical spectral shape interpolated to the
  FFT frequency axis (`_get_noise_spectrum_shape`, keyed to the detector's actual
  `sampling_rate = 1e6 / time_step_us`), then rescaled so each wire's RMS equals
  `series_rms = y + z·L`. This is the length-dependent, per-wire term.
- **White / parallel noise** — a flat-spectrum Gaussian at RMS `x`, identical in
  character for every wire.

Both scale with wire length only through `series_rms`, so longer wires are
noisier. Dense, bucketed, and wire-sparse signal formats each have a matching
generator (`_generate_noise_for_plane`, `_generate_noise_for_buckets`, and the
wire-sparse branch in `create_noise_fn_for_volume`).

### Coherent (per-group) noise

Coherent noise is **correlated across a wire group**: all wires in a group share a
single waveform, and adjacent groups are anti-correlated. `generate_group_waveforms`:

1. Draws each group's waveform from the amplitude spectrum
   `A(f) = 1 / (1 + f/f_corner)^(slope/2)` with `A(0) = 0` (no DC).
2. Applies inter-group coupling `w'(g) = w(g) − β·(w(g−1) + w(g+1))`.
3. Rescales to the target per-group RMS **after** coupling (coupling inflates
   variance, so normalizing before would leave the RMS ~2% high).

`broadcast_to_wires` then copies each group waveform to all wires in the group.
Defaults (`_COHERENT_DEFAULTS`, mirroring the `simulation.coherent_noise` YAML
block): `group_size=64`, `beta=0.15`, `rms_adc=2.5`, `corner_freq_hz=20000`,
`spectral_slope=1.5`. Because it is per-group and NumPy-only, it runs on the host
after the GPU sim (`add_coherent_noise`, dispatched from `process_event` and from
the host `add_noise(..., coherent=True)`).

!!! note "`include_noise` is a deprecated alias"
    The live flags are `include_intrinsic_noise` and `include_coherent_noise`.
    `include_noise` remains only as a back-compat alias — prefer the explicit
    flags so the two components are toggled independently.

## Digitization

`_digitize_signal` (in `tools/electronics.py`) quantizes the shaped waveform:

```python
scaled   = signal * gain_scale
unsigned = round(scaled + pedestal)
unsigned = clip(unsigned, 0, adc_max)        # adc_max = 2**n_bits - 1
return unsigned - pedestal                    # pedestal-subtracted ADC
```

Defaults (`create_digitization_config`): `n_bits=12` (`adc_max = 4095`),
`gain_scale=1.0`, and **per-plane pedestals** — `pedestal_collection=410` for the
Y (collection) plane, `pedestal_induction=1843` for U/V (induction) planes. The
step is a pure round/offset/clip on values the electronics + noise chain has
already shaped into ADC-scale magnitudes; enabled by `include_digitize`, no-op for
pixel volumes.

!!! note "Pixel skips the whole chain"
    For pixel readout, electronics, noise, and digitize are all no-ops: the pixel
    kernel bakes in chip gain (ADC per drift-electron), so pixel signal and hits
    are already in ADC after the single response pass. See
    [wire vs pixel](../detector/wire-vs-pixel.md).
