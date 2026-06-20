# JAXTPC Documentation & Notebooks Plan

Roadmap for turning JAXTPC into a documented, usable library for four audiences:
external/new users, internal production users, ML/reco researchers, and
extenders/developers. This is a planning artifact — check items off as built.

Status legend: `[ ]` todo · `[~]` partial/exists-needs-work · `[x]` done

---

## 1. Goals

- A new user can **install, run a simulation, and understand the output** in minutes.
- The project's **non-obvious conventions** (units, coordinate frame, capacity
  sizing, the two execution paths) are written down where users will find them —
  not buried in `CLAUDE.md` (which ships to no one) or in tribal memory.
- Production, differentiable/ML, and extension workflows each have a clear path.
- Docs stay honest: API reference is generated from docstrings; notebooks are
  executed in CI; the existing nbstripout filter keeps notebook diffs clean.

## 2. Audiences and what each needs

| Audience | Primary needs | Entry points |
|---|---|---|
| **External / new users** | install, runnable quickstart, physics concepts, citation | Quickstart, NB 00–04, Concepts |
| **Internal production users** | batch pipeline, HDF5 schemas, profiler/capacity sizing, slurm, dataset specifics | Production docs, NB 07 |
| **ML / reco researchers** | differentiable path, gradients, truth correspondence, losses, point clouds | Differentiable docs, NB 06, 08 |
| **Extenders / devs** | config schema, adding detectors, frame/uniformity invariants, tests & CI | Config reference, Contributing, NB 01/09 |

## 3. Tooling decision

- **Site generator:** `mkdocs` + `mkdocs-material` (low-effort, standard for
  scientific Python; search, nav, dark mode out of the box).
- **API reference:** `mkdocstrings[python]` — auto-generated from the (now
  corrected) numpy-style docstrings in `tools/`. No hand-maintained API pages.
- **Notebook embedding:** `mkdocs-jupyter` renders the `examples/*.ipynb` as doc
  pages directly, so tutorials live in one place and are runnable.
- **Hosting:** GitHub Pages via a `docs` CI job (`mkdocs gh-deploy` or an Actions
  workflow). Add a `docs` extra in `pyproject` for the doc dependencies.
- **Honesty checks (CI):** execute notebooks (`jupyter nbconvert --execute`) on a
  small/synthetic path so stale APIs fail loudly; optional link-check.

## 4. Prerequisite: packaging (makes "install" real)

"For people to use" requires a real install story. Today everything is
`import tools` from the repo root (generic top-level names `tools`/`config`/
`scripts` will collide on install). This is the foundation the install/quickstart
pages rest on.

- `[ ]` **`pyproject.toml`** — PEP 621 metadata: name `jaxtpc`, version (sync to
  the existing `0.9.0` tag + a `__version__`), `requires-python>=3.10`,
  dependencies from `requirements.txt` (jax bounded `>=0.4,<0.7`, numpy, scipy,
  matplotlib, h5py, hdf5plugin, pyyaml, pillow), optional extras: `dev`
  (pytest, nbstripout), `docs` (mkdocs-material, mkdocstrings, mkdocs-jupyter),
  `gpu` (jax[cuda12]).
- `[ ]` **Package layout** — move the importable code under a single namespace
  (`src/jaxtpc/{tools,production,profiler,viewer}`) and rewrite imports
  (`tools.` → `jaxtpc.tools.`). Breaking; do as one focused PR. *Decision needed:*
  do this now vs ship as a repo-root project first. (See the production-readiness
  assessment for detail.)
- `[ ]` **`console_scripts`** — `jaxtpc-batch`, `jaxtpc-setup-production`,
  `jaxtpc-viewer`, `jaxtpc-export-gif`, `jaxtpc-make-labl`.
- `[ ]` **`LICENSE`**, `[ ]` **`CITATION.cff`** (this is a tool that will be cited),
  `[ ]` **`CONTRIBUTING.md`**.
- `[ ]` **Ship package data** — `tools/responses/*.npz`, `tools/data/*.csv`,
  `config/*.yaml`, and the runtime assets currently outside the package
  (`config/noise_spectrum.npz`, `config/pixel_response.npz`, the SCE map) via
  `package-data`/`importlib.resources`. Note: those three assets are loaded today
  via `tools/../config` and will break on install unless relocated/declared.

## 5. The "must-capture" tribal knowledge (dedicated treatment)

These are the things that bite people and currently live only in `CLAUDE.md` or
nowhere. Each gets a focused doc page **and** a notebook section.

1. **Units convention** — wire hits in **ENC** (electrons), pixel hits in **ADC**;
   `inter_thresh` / `threshold_adc` / `corr_threshold` mean different things per
   readout. The single biggest footgun. (Source: `CLAUDE.md` units table.)
2. **Local coordinate frame** — deposits are transformed to volume-local/centered
   coords (`y_local = y_global - y_center`; `x_local` anode-referenced via
   `drift_direction`). All per-volume geometry reduces to one local frame so the
   JIT body uses fixed constants. The recently-fixed pixel-origin bug lived
   exactly here — document the invariant (and the geometry-uniformity assumption
   across volumes).
3. **Capacity sizing** — `total_pad`, `maxg`/`maxg_medium`, `max_keys`,
   `response_chunk`/`hits_chunk`, `max_buckets`, analytic box dims: what each
   bounds, how the profiler sizes them, and overflow behavior (hard crash vs
   logged-and-reprocess). (Source: `profiler/README.md`.)
4. **Two execution paths** — `process_event` (production, batched, JIT) vs
   `forward`/`forward_segments` (differentiable, remat); when to use which.
5. **Output schema** — the sensor / step / hits (+ separate labl) HDF5 files,
   their delta/CSR encodings, codecs, and 1-based group ids.

## 6. Documentation site map (`docs/`)

Each page notes: **[audience]** · content · *source material to migrate*.

```
docs/
  index.md                      [all] what/why, capabilities, the two paths, links
  install.md                    [all] pip install, GPU JAX, verify; *needs pyproject*
  quickstart.md                 [external] 30-line runnable example -> NB 00

  concepts/
    architecture.md             [dev] DetectorSimulator, factory/closure pattern, scan/vmap over volumes  *from CLAUDE.md*
    coordinates.md              [dev] local frame, centering, anode refs, uniformity invariant
    config-vs-params.md         [all] SimConfig (static) vs SimParams (dynamic), recompilation
    paths.md                    [all] production vs differentiable

  physics/
    overview.md                 [physics] the full response chain, one diagram
    recombination.md            [physics] modified-box / EMB / passthrough; Q & L; angular dep
    drift-diffusion.md          [physics] drift, lifetime attenuation, DCT/spatial-conv diffusion kernels
    response-kernels.md         [physics] DKernel table, s-level interpolation, wire vs pixel kernels
    electronics-noise.md        [physics] RC*RC shaping, intrinsic + coherent noise, digitization
    sce.md                      [physics] space charge: maps, time-primary corrections, displacement
    units.md                    [all] **ENC vs ADC** convention + threshold-units table  *from CLAUDE.md*

  detector/
    config-schema.md            [dev] full YAML reference (volumes, geometry, planes/pixel, simulation, readout, electric_field)
    presets.md                  [all] SBND, MicroBooNE, ICARUS, DUNE FD1, ND-LAr, cubic wire/pixel
    wire-vs-pixel.md            [all] readout types, when each, output differences

  production/
    batch.md                    [internal] run_batch CLI, threaded save, workers  *from production/README.md*
    data-formats.md             [internal/ML] sensor/step/hits/labl HDF5 schema, encodings, codecs  *from production/README.md*
    profiler.md                 [internal] capacity sizing, setup_production, overflow/reprocess  *from profiler/README.md*
    slurm.md                    [internal] array drivers, resume, dataset layout  *from production/RUN_PRODUCTION.md*

  differentiable/
    overview.md                 [ML] forward / forward_segments, remat, what's differentiable
    losses.md                   [ML] multi-scale spectral MSE, point-cloud / OT losses
    optimization.md             [ML] gradient examples, particle_generator, a fit demo -> NB 08

  truth/
    track-hits.md               [ML/internal] group->track correspondence, qs_fractions, labeling

  viz/
    plotting.md                 [all] visualization + pixel_visualization helpers
    viewer.md                   [all] interactive 3D/2D viewer + GIF export  *from viewer/README.md*

  reference/
    api/                        [dev] auto-generated (mkdocstrings) per module
    glossary.md                 [all] terms + gotchas (units, overflow, frames)
    faq.md                      [all] common errors and fixes
    citation.md                 [external] how to cite

  contributing/
    development.md              [dev] env, tests (`pytest -m "not slow"`), CI gates  *from tests/TESTS.md*
    invariants.md               [dev] frame & geometry-uniformity invariants, adding-a-detector checklist
```

Most pages are **migration + correction** of existing `README`/`CLAUDE.md`
content, not greenfield writing. After migration, `CLAUDE.md` shrinks to
AI-assistant guidance + pointers into `docs/`.

## 7. Notebook series (`examples/`)

Numbered, progressive. Each: **[audience]** · content · status.

| # | Notebook | Audience | Content | Status |
|---|---|---|---|---|
| 00 | `00_quickstart.ipynb` | external | install check → synthetic event → wire sim → 4 plots, minimal | `[ ]` new (extract from `run_simulation`) |
| 01 | `01_detector_config.ipynb` | all/dev | YAML anatomy, multi-volume, visualize geometry, wire vs pixel | `[ ]` new |
| 02 | `02_physics_walkthrough.ipynb` | physics | one event, a plot at each pipeline stage (Q/L→drift→diffusion→kernel→electronics→noise→ADC) | `[ ]` new |
| 03 | `03_wire_vs_pixel_units.ipynb` | all | side-by-side; ENC vs ADC made concrete; thresholds | `[ ]` new |
| 04 | `04_recombination_models.ipynb` | physics | modified-box vs EMB vs passthrough; angular dependence | `[ ]` new |
| 05 | `05_space_charge.ipynb` | physics | generate/load SCE maps, apply, visualize distortion | `[ ]` new |
| 06 | `06_truth_and_track_hits.ipynb` | ML/internal | group→track correspondence; per-particle labels | `[ ]` new |
| 07 | `07_production_pipeline.ipynb` | internal | run_batch, the 3 HDF5 files, profiler sizing, load output | `[~]` fold in `production/view_production.ipynb` |
| 08 | `08_differentiable.ipynb` | ML | forward_segments, gradients wrt velocity/lifetime, gradient-descent fit demo | `[ ]` new |
| 09 | `09_custom_detector.ipynb` | dev | build a new geometry end-to-end | `[ ]` new |
| A1 | `response_kernels.ipynb` | physics/dev | kernel + diffusion visualization (appendix) | `[~]` existing `tools/responses/response_visualization.ipynb` |

Existing `run_simulation.ipynb` / `run_pixel_simulation.ipynb` become the basis
for 00/02/03 (they already run self-contained on a synthetic event). Keep all
notebooks runnable with **no external data** (synthetic-event helper) so they
work in CI and for external users; show the `load_event(...)` swap for real data.

## 8. Content migration map (existing → destination)

| Existing | Destination |
|---|---|
| `README.md` | trimmed landing page + `docs/index.md`, `install.md`, `quickstart.md` |
| `CLAUDE.md` (architecture, units, data types, pipeline) | `docs/concepts/*`, `docs/physics/*`, `docs/detector/config-schema.md`; CLAUDE.md → assistant-only + links |
| `production/README.md` | `docs/production/batch.md`, `data-formats.md` |
| `production/RUN_PRODUCTION.md` | `docs/production/slurm.md` (fix stale config filenames) |
| `profiler/README.md` | `docs/production/profiler.md` |
| `viewer/README.md` | `docs/viz/viewer.md` |
| `tests/TESTS.md` | `docs/contributing/development.md` |
| 4 working notebooks | `examples/` (renumbered) |

## 9. Phased execution plan

**Phase 0 — foundation (unblocks "install + run")**
- `[ ]` `pyproject.toml`, `LICENSE`, `CITATION.cff`
- `[ ]` decide packaging layout (src/jaxtpc namespace vs repo-root)
- `[ ]` relocate/declare the 3 out-of-package runtime assets

**Phase 1 — front door (external users)**
- `[ ]` mkdocs skeleton + CI deploy + API auto-gen
- `[ ]` `index`, `install`, `quickstart`; NB 00
- `[ ]` `physics/units.md` + `concepts/coordinates.md` (the top footguns)

**Phase 2 — physics & tutorials**
- `[ ]` NB 01–04; physics pages; detector config schema

**Phase 3 — production (internal)**
- `[ ]` production pages (migrate + fix stale names); NB 07

**Phase 4 — differentiable/ML**
- `[ ]` differentiable pages; losses/pointcloud; NB 06, 08

**Phase 5 — extension & polish**
- `[ ]` NB 09; contributing/invariants; glossary/FAQ; SCE page + NB 05

## 10. Keeping docs honest (maintenance)

- API reference is generated from docstrings — fix docstrings, not duplicate prose.
- Notebooks executed in CI (synthetic-event path) so a stale API fails the build.
- `nbstripout` filter + `notebook-output-check` CI keep notebook diffs clean.
- A short "docs touched?" reminder when public APIs change (PR checklist in
  `contributing/`).
- Single source of truth for the units table and capacity glossary — link to it,
  don't restate.

---

### Open decisions (need a call before/within Phase 0–1)
1. **Packaging namespace** — adopt `src/jaxtpc/` now (breaking, clean) or ship
   repo-root and defer? Affects every import and the install page.
2. **Docs hosting** — public GitHub Pages (external audience) vs internal only.
3. **Notebook execution in CI** — full synthetic run (slower, catches more) vs
   import/collect-only (fast).
