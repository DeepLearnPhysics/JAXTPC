# JAXTPC

GPU-accelerated, **differentiable** liquid-argon Time Projection Chamber (TPC)
detector simulation in JAX. It models the full detector response chain — charge
recombination, electron drift with lifetime attenuation, diffusion-convolved
wire/pixel response, electronics shaping, noise, and ADC digitization — for
arbitrary multi-volume geometries (SBND, MicroBooNE, ICARUS, DUNE FD1, DUNE
ND-LAr) with both wire and pixel readout.

## Two execution paths

- **Production** — `DetectorSimulator.process_event(...)`: batched, JIT-compiled,
  bounded-memory; the path for generating datasets.
- **Differentiable** — `forward(...)` / `forward_segments(...)`: memory-efficient
  reverse-mode gradients through every physics parameter (velocity, lifetime,
  diffusion, recombination), for optimization and reconstruction.

## Start here

- **[Install](install.md)** — `pip install -e .`
- **[Quickstart](quickstart.md)** — run a simulation in ~30 lines
- **[Notebooks](https://github.com/DeepLearnPhysics/JAXTPC/tree/main/notebooks)** —
  themed, runnable examples (getting started, physics, readout, gradients, reco,
  calibration, production)

## Understand the code

New to the codebase? Read the spine in order — it explains how a list of energy
deposits becomes readout signals, using real function names:

- **[Reading guide](architecture/reading-guide.md)** — a guided walk through the
  call chain, source open alongside
- **[Pipeline overview](architecture/pipeline-overview.md)** — the full response
  chain in one diagram
- **[Data model](architecture/data-model.md)** — the types that flow through it
- **[Config vs params](architecture/config-vs-params.md)** ·
  **[Execution paths](architecture/execution-paths.md)** ·
  **[The simulator](architecture/simulator.md)**

## Read these before you get surprised

Two conventions account for most confusion:

- **[Units (ENC vs ADC)](physics/units.md)** — wire hits are in electrons (ENC),
  pixel hits are in ADC; thresholds mean different things per readout.
- **[Coordinates & frames](concepts/coordinates.md)** — positions are
  volume-local/centered; this is load-bearing for the whole pipeline.

## Project status

See the **[documentation roadmap](PLAN.md)** for what's written, what's planned,
and how the docs and notebooks are organized.
