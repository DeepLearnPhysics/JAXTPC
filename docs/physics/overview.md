# Physics overview

JAXTPC turns Geant4-style energy deposits into a simulated detector readout by
running each event through a fixed physics chain, once per volume and (for wire
readout) once per plane. This page is a navigational hub: it walks the chain
stage by stage in prose and links each stage to its deep-dive page.

Every stage below is implemented in the **shared physics body**
(`tools/physics.py`), which both execution paths — the production path
(`process_event`) and the differentiable path (`forward`) — call with identical
signatures. For the end-to-end data-flow figure, see the
[pipeline diagram](../architecture/pipeline-overview.md); this page does not
duplicate it.

!!! note "Units up front"
    Wire and pixel readout use **different unit conventions** at the hits stage
    (wire is in ENC electrons, pixel is in ADC). For anything unit-related,
    always defer to [units](units.md) — it is the canonical source.

## The chain

### 1. Recombination

Energy deposits `(de, dx, theta, phi)` become ionization electrons (charge `Q`)
and scintillation photons (light `L`). The charge–light split is anti-correlated
and fixed by energy conservation. Three models are available
(`modified_box`, `emb`, `passthrough`), selected in the YAML
`charge_recombination` block.

→ [recombination](recombination.md)

### 2. Drift and lifetime attenuation

Each deposit's charge drifts to the anode. JAXTPC computes an anode-referenced
drift distance/time, applies optional space-charge (SCE) corrections, then
corrects per plane. Charge is attenuated by `exp(−t_drift / lifetime)` from
electron capture on impurities.

→ [drift & diffusion](drift-diffusion.md)

### 3. Diffusion

The drifting charge cloud spreads longitudinally and transversely, with sigmas
that grow with drift distance. In the production path this spread is **not**
applied per deposit analytically — it is baked into the response-kernel table,
which is pre-blurred at a range of diffusion levels and interpolated at each
deposit's drift-derived level `s`.

→ [drift & diffusion](drift-diffusion.md#diffusion)

### 4. Response kernels

The (diffusion-convolved) field/readout response is a kernel table `DKernel`
indexed by the diffusion level `s`. At runtime the table is interpolated at each
deposit's `s` to produce per-deposit response contributions. (The wire kernel is
a dimensionless field-impulse fraction; the pixel kernel bakes in chip gain —
see [units](units.md).)

→ [response kernels](response-kernels.md)

### 5. Accumulation

Per-deposit contributions are scatter-added into the readout array over chunks
of deposits (`response_chunk_size`), in one of three modes: **dense**
`(num_wires, num_time)`, **bucketed** (a sparse memory-saver), or the pixel
**box** path. This produces the raw response signal per plane/volume.

→ [pipeline diagram](../architecture/pipeline-overview.md)

### 6. Post-processing — wire only

For **wire** readout the raw response (in ENC) is then shaped and digitized:

- **Electronics** — RC⊗RC field-to-ADC shaping via sparse FFT on active wires.
- **Noise** — intrinsic ENC noise (and optional coherent per-group noise).
- **Digitization** — ADC quantization with per-plane pedestals.

→ [electronics & noise](electronics-noise.md)

### 6′. Single pass — pixel only

**Pixel** readout bakes the chip gain into the kernel and derives both the
sensor signal and per-group truth from a **single response pass** — it skips the
electronics → noise → digitize sub-chain entirely, emitting ADC directly.

→ [units](units.md), [wire vs pixel](../detector/wire-vs-pixel.md)

## Where to go next

| You want to understand… | Page |
|---|---|
| Q/L split, the three recombination models | [recombination](recombination.md) |
| drift geometry, attenuation, diffusion sigmas | [drift & diffusion](drift-diffusion.md) |
| the `DKernel` table and `s`-interpolation | [response kernels](response-kernels.md) |
| electronics shaping, noise, digitization | [electronics & noise](electronics-noise.md) |
| space-charge corrections | [SCE](sce.md) |
| ENC vs ADC and threshold units | [units](units.md) |
| the whole chain as one figure | [pipeline diagram](../architecture/pipeline-overview.md) |
