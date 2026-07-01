# Glossary

A quick A–Z of the terms and gotchas you meet reading or running JAXTPC. Where a
concept has a canonical page, the entry links to it rather than restating it.

## A

**ADC** — Analog-to-digital converter counts, the digitized sensor output (12-bit
by default). Wire hits are in [ENC](#e); the wire *sensor* signal becomes ADC only
after the electronics → noise → digitize chain. Pixel hits **and** sensor are ADC
throughout (gain baked into the kernel). See [units](../physics/units.md).

**Anode-referenced** — In the volume-local frame the anode sits at `x_local = 0`
and drift runs toward `−x`, so `x_local = drift_direction * (x_anode − x_global) ≥ 0`
is the drift distance. See [coordinates](../concepts/coordinates.md).

**Accumulation mode** — How per-deposit response kernels are scattered into the
sensor image: **dense** (`(num_wires, num_time_steps)` array, default),
**bucketed** (sparse tiles, a wire-only memory saver), or **box** (per-group tile,
the default track-hits path). See [capacities](../concepts/capacities.md#bucketed-accumulation-mode-memory-saving-alternative).

## B

**Box (track-hits path)** — The default group-as-bucket track-hits mode: each
group gets a fixed footprint tile (the *box dims*) indexed by `group_id`. Bounds
`maxg` (groups/event) and `max_keys` (non-zero cell budget). Contrast **merge**
(non-default alternative). See [track-hits](../truth/track-hits.md).

**Box dims** — The per-group footprint tile size (`box_bpy/box_bpz/box_bt` for
pixel; `box_bw/box_btw` for wire). **Derived analytically** by `compute_box_dims`
from the grouping rule, not scanned from data — so there is no runtime box-dim
overflow. See [capacities](../concepts/capacities.md#box-dims-analytic-not-scanned).

**Bucketed accumulation** — Wire-only memory-saving sensor mode
(`use_bucketed=True` / `--bucketed`): scatters contributions into sparse tiles
`(max_buckets, B1, B2)` instead of a dense image. Incompatible with coherent
noise. See [capacities](../concepts/capacities.md#bucketed-accumulation-mode-memory-saving-alternative).

## D

**Deposit** — A single Geant4 energy-deposit step: a position, `de` (energy), `dx`
(step length), direction (`theta`/`phi`), and provenance ids. The atomic input
unit. Padding deposits have `de=0, dx=1, track_ids=−1`.

**Diffusion level `s`** — The index into the `DKernel` table for a deposit, set by
its drift distance: `s = clip(sqrt(drift / max_drift), 0, 1)`. `s=0` is the
sharpest kernel (at the anode), `s=1` the most diffused (furthest drift). The
runtime interpolates the table at each deposit's `s`.

**DKernel** — The pre-built table of response kernels indexed by diffusion level
`s`, shape `(num_s, …)`. Built by reflect-pad + separable Gaussian convolution
(**not** a DCT, despite older comments). See
[response-kernels](../physics/response-kernels.md).

**Dense accumulation** — The default wire sensor mode: scatter-add each response
kernel into a `(num_wires, num_time_steps)` array per plane.

## E

**ENC** — Equivalent Noise Charge, i.e. electrons. The **wire** hits stage is in
ENC (hundreds–thousands of electrons); pixel hits are in ADC instead. This is the
single most common units gotcha. See [units](../physics/units.md).

**EMB** — Ell-Modified-Box recombination model (ICARUS 2024): adds an angular
correction `β_eff(φ)` so tracks parallel to the E-field recombine more. See
[recombination](../physics/recombination.md).

## G

**Group** — A run of consecutive deposits of one track (split on spatial gaps),
the unit of the box track-hits path. **Group ids in production `hits/` files are
1-based** — entry `group_to_track[0]` is unused. Groups never span volumes.

**`group_to_track`** — Per-volume numpy lookup mapping a `group_id` back to its
Geant4 track id (kept on the host, outside JIT).

## L

**Local frame** — The per-volume coordinate frame the loader transforms deposits
into: anode at `x=0`, drift toward `−x`, y/z centered. All volumes share reference
geometry in this frame, so the JIT body uses fixed constants (the geometry-
uniformity invariant). See [coordinates](../concepts/coordinates.md).

## M

**`max_buckets`** — Capacity bounding the number of active sparse buckets in
bucketed (wire) mode. Overflow → **crash**. Inert for pixel. See
[capacities](../concepts/capacities.md#max_buckets-active-buckets-bucketed-mode).

**`max_keys`** — Capacity bounding the number of non-zero box cells the track-hits
path stores (charge-dependent, the trickiest to size). Overflow → **crash** (the
reported count is correct, so it tells you how high to set it). See
[capacities](../concepts/capacities.md#max_keys-track-hits-box-cell-budget).

**`maxg` / `maxg_medium`** — Capacity bounding groups per event (the box's first
dimension); `maxg_medium` is the medium-tier value for tiered routing. Overflow →
**log-skip & reprocess**. See
[capacities](../concepts/capacities.md#maxg-maxg_medium-groups-per-event).

**Merge (track-hits path)** — The non-default alternative to box: per-chunk
sensor-hit merging (`merge_chunk_sensor_hits`). Live but off by default. See
[track-hits](../truth/track-hits.md).

**Modified Box** — The ArgoNeuT recombination model: `ξ = β/(ρ·E)·dE/dx`, no
angular dependence. One of the two physical models. See
[recombination](../physics/recombination.md).

## P

**`passthrough`** — Recombination "model" that skips recombination and returns
charge directly — a debugging/idealized option alongside `modified_box` and `emb`.

**Pedestal** — The per-plane DC offset added during digitization; the ADC baseline
a signal rides on. Applied in `_digitize_signal` (round + pedestal + clip).

## Q

**`qs_fractions`** — Each deposit's fractional share of its group's recombined
charge — the weights that let group-level track-hits be attributed back to
individual deposits. See [track-hits](../truth/track-hits.md).

## R

**Recombination models** — `modified_box`, `emb`, or `passthrough` (set in the
YAML under `charge_recombination.model`). See
[recombination](../physics/recombination.md).

**`response_chunk` / `hits_chunk`** — `fori_loop` batch sizes (speed knobs, not
overflow capacities) for sensor accumulation and track-hits. **Each must divide
`total_pad` evenly** or construction raises a `ValueError`. See
[capacities](../concepts/capacities.md#response_chunk-hits_chunk-chunk-sizes).

## S

**SCE** — Space Charge Effect: field distortions that displace drifting electrons.
Applied via correction maps; channel-0 is a drift **time** (µs), not a distance.
See [sce](../physics/sce.md).

**`SimConfig`** — The **static**, closure-captured configuration (array
dimensions, mode flags, geometry, plane names). Changing any value forces JIT
recompilation. Contrast `SimParams`. See
[config vs params](../architecture/config-vs-params.md).

**`SimParams`** — The **dynamic** physics parameters passed as a JIT argument
(velocity, lifetime, diffusion, recombination scalars, optional NN/SCE models).
Changeable without recompiling. See
[config vs params](../architecture/config-vs-params.md).

## T

**`total_pad`** — The fixed length every per-volume deposit array is padded to
(bounds the leading array dimension). Overflow → **crash at load**. See
[capacities](../concepts/capacities.md#total_pad-deposits-per-volume).

**Track** — A Geant4 particle trajectory; deposits carry a `track_id`, and groups
map back to tracks via `group_to_track`. Track hits are the per-sensor
decomposition attributed back to tracks. See [track-hits](../truth/track-hits.md).
</content>
</invoke>
