# JAXTPC Documentation Build Plan

Authoritative spec for building the JAXTPC documentation website. This is the
single source of truth the page-writing agents follow: it fixes the site
structure, maps every page to its source material, lists the corrections that
must be made instead of migrated, and defines the diagrams, notebooks, API
reference, and build order.

The organizing principle: **the site is centered on a code-reading spine**
(`architecture/`) that explains how the code actually fits together, in
execution order, using real function names — not a page-per-README migration.
A reader who understands `DepositData → *Intermediates → response_signals` and
the closure/scan structure can then read any physics module in isolation.

Status legend: `[ ]` todo · `[~]` partial/exists-needs-work · `[x]` done

---

## 1. Locked decisions

| Decision | Choice |
|---|---|
| **Audience scope** | All four audiences at **equal depth** (new users, production, ML/reco, extenders/devs). |
| **Install story** | `pip install -e .` — repo-root packaging already works (`pyproject.toml` packages `tools/`, `production/`, … as top-level; `package-data` ships `tools/responses/*.npz` + `tools/data/*`). The `src/jaxtpc/` namespace refactor is **deferred** (breaking; out of scope for docs). |
| **Notebooks** | **Embed the 7 existing** notebooks in-site via `mkdocs-jupyter`; **author 2 gap notebooks** (`physics_walkthrough`, `custom_detector`). All notebooks stay CI-runnable on **synthetic data** (no external files). |
| **Hosting** | **Public GitHub Pages.** Genericize/remove internal specifics (SDF paths, the doraemon dataset, cluster SLURM loops, specific run IDs). |
| **Site generator** | `mkdocs` + `mkdocs-material`. |
| **API reference** | `mkdocstrings[python]`, **grouped by concept** (see §7). Auto-generated from docstrings after the §6 docstring fixes. |
| **Honesty checks** | Notebooks executed in CI on the synthetic path; `mkdocs build --strict`; existing `nbstripout` filter keeps notebook diffs clean. |

## 2. Audiences

| Audience | Primary needs | Entry points |
|---|---|---|
| **New / external users** | install, runnable quickstart, mental model of the code, physics concepts, citation | Quickstart → Architecture spine → Physics |
| **Production users** | batch pipeline, HDF5 schemas, profiler/capacity sizing, scaling, viewer | Production section |
| **ML / reco researchers** | differentiable path, gradients, losses, particle generation, truth correspondence | Differentiable + Truth sections |
| **Extenders / devs** | config schema, adding detectors, frame/uniformity invariants, tests & CI, internal I/O | Architecture + Detector + Contributing |

## 3. Tooling

- **Site:** `mkdocs` + `mkdocs-material` (search, nav sections, dark mode). Config lives in `mkdocs.yml`; add a `docs` extra in `pyproject` for the build deps.
- **API reference:** `mkdocstrings[python]` — enable the plugin (currently commented out in `mkdocs.yml`) once the §6 docstring fixes land.
- **Notebooks:** `mkdocs-jupyter` renders `notebooks/**/*.ipynb` as pages so tutorials live in-site and stay runnable.
- **Hosting:** GitHub Pages via a `docs` CI job (`mkdocs gh-deploy` or an Actions workflow).
- **CI honesty:** execute notebooks (`jupyter nbconvert --execute`) on the synthetic path so a stale API fails the build; `mkdocs build --strict` catches broken links/refs; optional link-check.

---

## 4. Site map

Each page: **[audience]** · one-line purpose · *source material* · `[status]`.
`← code` means write from the source code (not migrated from a README).

```
docs/
  index.md                    [all]      what/why, the two paths, the 5 footguns, nav        *README overview + CLAUDE*  [~] refine
  install.md                  [all]      pip install -e ., GPU JAX wheel, warm_up() verify   *README install*           [~] refine
  quickstart.md               [all]      ~30-line synthetic wire run → plot → NB 00          *README quickstart*        [~] refine

  architecture/               ← THE CODE SPINE (new top-level section)
    reading-guide.md          [dev]      ordered walkthrough of the call chain, file by file  ← code                    [ ] new
    data-model.md             [all]      DepositData/VolumeDeposits + 3 Intermediates + outputs; diagram D2  ← code       [ ] new
    config-vs-params.md       [all]      SimConfig (static, closure-captured, recompiles) vs SimParams (JIT arg)  *CLAUDE*  [ ] new
    simulator.md              [dev]      DetectorSimulator.__init__ factory/closure pattern; scan/vmap; trace-time unroll; diagram D3  ← code  [ ] new
    execution-paths.md        [all]      process_event vs forward/forward_segments vs process_event_light; traced vs host  *CLAUDE*  [ ] new
    pipeline-overview.md      [physics]  the whole response chain, one big data-flow diagram D1, links into physics/*  ← code  [ ] new

  concepts/
    coordinates.md            [dev]      local frame, centering, anode ref, uniformity invariant; diagram D4  *exists*     [x] keep
    padding-masking.md        [dev]      total_pad, de=0/dx=1 padding, the single n_actual mask, chunk divisibility  ← code  [ ] new
    capacities.md             [all]      total_pad/maxg/max_keys/response_chunk/hits_chunk/max_buckets/box-dims + overflow; diagram D5  *profiler/README*  [ ] new

  physics/
    overview.md               [physics]  the response chain in words, links each stage        ← code                     [ ] new
    recombination.md          [physics]  modified_box / emb / passthrough; Q & L; W_ion vs W_ph; angular β_eff(φ)  *CLAUDE + code*  [ ] new
    drift-diffusion.md        [physics]  drift to anode + per-plane correction + lifetime attenuation; diffusion sigmas  ← code  [ ] new
    response-kernels.md       [physics]  DKernel table (num_s, s=d/max_drift) + s-interp; wire vs pixel kernel; NB response_kernels  ← code (CORRECT the "DCT" claim)  [ ] new
    electronics-noise.md      [physics]  RC⊗RC sparse-FFT shaping; intrinsic ENC; coherent (per-group) noise; digitization  ← code  [ ] new
    sce.md                    [physics]  SCE maps, channel-0 = Δt(µs) semantics, per-side maps, local-frame conversion  ← code  [ ] new
    units.md                  [all]      ENC (wire) vs ADC (pixel) + threshold-units table    *exists — canonical source*  [x] keep

  detector/
    config-schema.md          [dev]      full YAML reference: volumes/geometry/planes|pixel/simulation/readout/electric_field/medium  *config/*.yaml + geometry.py*  [ ] new
    presets.md                [all]      SBND / µBooNE / ICARUS / DUNE FD1 / ND-LAr / cubic wire+pixel (correct volume counts)  *config/*  [ ] new
    wire-vs-pixel.md          [all]      readout differences, single-pass pixel (signal+truth together), units, output shapes; NB pixel_simulation  ← code  [ ] new

  production/
    batch.md                  [production]  run_batch flow, threaded save (reader-prefetch / main / save-workers, per-file locks), tiered routing  *production/README — rewrite from code*  [ ] new
    data-formats.md           [production/ML]  sensor/step/hits/labl schemas: delta + CSR encodings, 1-based groups, codecs, decode recipes; NB view_production  *production/README + save.py*  [ ] new
    profiler.md               [production]  setup_production + each sizing script; charge-aware max_keys; calibration knobs  *profiler/README*  [ ] new
    scaling.md                [production]  sharding, resume (.done / --skip-existing), per-worker files — GENERICIZED (no SLURM/doraemon)  *RUN_PRODUCTION.md*  [ ] new

  differentiable/
    overview.md               [ML]       forward / forward_segments, remat, what params carry gradients, pixel unsupported  *CLAUDE + code*  [ ] new
    losses.md                 [ML]       Parseval multi-scale blur MSE; Sobolev H^-s; geomean/log1p plane rebalancing  *losses.py*  [ ] new
    particle-generation.md    [ML]       CSDA-range inversion for differentiable muons; trig parameterization; build_muon_forward  *particle_generator.py*  [ ] new
    optimization.md           [ML]       end-to-end gradient fit demo; NB optimization                *NB*                     [ ] new

  truth/
    track-hits.md             [ML/production]  group→track correspondence, qs_fractions, box vs merge, finalize_track_hits two-stage; NB segments_closure  *track_hits.py*  [ ] new

  viz/
    plotting.md               [all]      visualization + pixel_visualization; DeadbandNorm; sparse/dense contract  *code*     [ ] new
    viewer.md                 [all]      interactive 3D/2D viewer + GIF export                 *viewer/README*             [ ] new

  reference/
    api/                      [dev]      mkdocstrings, grouped by concept (see §7)             ← docstrings                [ ] new
    glossary.md               [all]      terms + gotchas (units, overflow, frames)             ← code                      [ ] new
    faq.md                    [all]      the overflow RuntimeErrors + fixes (verbatim messages users hit)  ← code           [ ] new
    citation.md               [external] how to cite                                            *CITATION.cff*             [ ] new

  contributing/
    development.md            [dev]      env, pytest -m "not slow", CI gates                    *tests/TESTS.md*           [ ] new
    adding-a-detector.md      [dev]      frame + uniformity invariants, config-authoring checklist  ← code                 [ ] new
    io-formats-internal.md    [dev]      tools/utils.py standalone HDF5 vs production format — when to use which  *utils.py*  [ ] new
```

**Ordering rationale:** architecture spine + data-model come immediately after
quickstart, *before* the physics deep-dives. The reverse order (physics first)
leaves a reader lost in modules with no mental model of how they connect.

---

## 5. Diagrams

Five figures anchor the site. (Author as Mermaid/SVG; keep source in `docs/assets/`.)

- **D1 — Pipeline data-flow (the money diagram).** Horizontal stages:
  `dE,dx → recombination → Q,L → drift + SCE → per-plane correct + attenuate →
  wire/pixel project → DKernel response (s-interp) → accumulate (dense/bucketed/box)
  → [electronics → noise → digitize] → sensor`. Annotate the **units transition**
  (dimensionless kernel × electrons = ENC; digitize → ADC) and mark where **pixel
  skips** the bracketed post-chain. → `architecture/pipeline-overview.md`.
- **D2 — Data-type flow.** `DepositData{VolumeDeposits×N} → (JIT) →
  VolumeIntermediates → PlaneIntermediates/PixelIntermediates → response_signals
  dict + track_hits dict + filled DepositData`. Mark traced vs host. → `architecture/data-model.md`.
- **D3 — Closure/factory structure.** `__init__` builds N factory closures
  capturing static `SimConfig`; `_build_jit` composes them into
  `process_one_volume`; `iterate` maps it over the volume axis; `sim_params` is
  the only JIT argument. → `architecture/simulator.md`.
- **D4 — Coordinate frames.** Global → local: anode-at-0, drift-toward-−x,
  yz-centered. → `concepts/coordinates.md`.
- **D5 — Capacity map.** Each capacity → the array shape it bounds → its overflow
  path (crash / log-skip / truncate). → `concepts/capacities.md`.

---

## 6. Corrections — rewrite from code, do NOT migrate

The source READMEs/CLAUDE.md contain stale or false claims. Writing agents must
verify against the current source and **correct these**, not propagate them.
(Where the claim also lives in a source docstring, fix the docstring too — it
feeds the API reference.)

1. **"DCT diffusion" is false.** `tools/kernels.py:generate_dkernel_table` does
   reflect-pad + separable Gaussian conv (`lax.conv_general_dilated`), not a DCT.
   The false claim is in `CLAUDE.md` **and** the `kernels.py` module docstring.
   Correct both; describe the real algorithm in `physics/response-kernels.md`.
2. **`production/README.md` is stale**: hardcoded 2-volume `(2,3)` shapes;
   `include_noise` (code uses `include_intrinsic_noise` / `include_coherent_noise`,
   with `include_noise` only a deprecated alias); a single-file-lock claim (code
   uses per-file locks `sen_lock`/`step_lock`/`hits_lock` + `--per-worker-files`);
   and `hits_threshold` as a YAML key (actual production-config key is
   `corr_threshold`). Rewrite `production/*` from code. (The "no resume yet" claim
   is in `RUN_PRODUCTION.md`, not this file — see §6.6.)
3. **`tools/utils.py` is not "misc utilities."** It is a standalone single-file
   HDF5 event/SCE I/O format that overlaps `production/save.py`, and only supports
   the 2-volume U/V/Y geometry (`_PLANE_NAMES` KeyErrors otherwise). Document in
   `contributing/io-formats-internal.md`.
4. **Drop `pointcloud.py` / `space_points.py`** — they do not exist; remove from
   every inventory (already fixed in README/CLAUDE trees; keep out of docs).
5. **Mark legacy/provisional code:** `track_hits.py`'s standalone path
   `group_hits_by_track` / `label_hits` / `sparse_hits_to_dense` (K_wire×K_time
   neighbor system) is legacy — kept for out-of-pipeline use and tests, **not**
   used by the simulator (per the module docstring). Separately, `label_merged_hits`
   is **uncalled** (effectively dead code) and belongs to the non-default *merge*
   path, not this trio — don't lump it in. `production/make_labl.py` is a
   self-described temporary stand-in. Say so wherever they appear so readers don't
   mistake them for the production API.
6. **Genericize `RUN_PRODUCTION.md`** into `production/scaling.md`: strip
   doraemon, `/sdf/...` paths, `CUDA_VISIBLE_DEVICES` SLURM loops, specific run
   IDs, and pre-tiered perf numbers. Also **correct its stale "no resume yet"
   claim** — resume exists (`--skip-existing` + `.done` markers). Document the
   *mechanism* (sharding, resume, per-worker files), not the site.
7. **Volume counts:** use the corrected table (icarus = 2, dune_fd1 = 4, ndlar = 70)
   in `detector/presets.md`.

---

## 7. API reference (mkdocstrings)

**Group by concept, one page per group** (not one-flat-page-per-module) so
cross-references between return types resolve. Suggested `reference/api/` pages,
each pulling `:::` directives from the relevant modules:

- **Core simulation** — `DetectorSimulator`, `process_event`, `forward`, `forward_segments`, `process_event_light`, `finalize_track_hits`, `warm_up`.
- **Data types** — all of `config.py` (the NamedTuples + factories) on one page (return types documented only here).
- **Loading & config** — `generate_detector`, `load_event`, `build_deposit_data`, `load_particle_step_data`, `create_sim_config`, `create_sim_params`.
- **Physics** — `recombination`, `drift`, the `physics` shared body.
- **Response & readout** — `kernels`, `wires`, `output`, `sparse_utils`.
- **Post-processing** — `electronics`, `noise`, `coherent_noise`, `efield_distortions`.
- **Truth** — `track_hits`.
- **Differentiable** — `losses`, `particle_generator`, `nn_utils`.
- **Visualization** — `visualization`, `pixel_visualization`.
- **Production** — `production/save.py`, `production/load.py` (format reference).

**Docstring prerequisites (do before enabling auto-gen, else pages render thin/misleading):**
- Fix the `kernels.py` **DCT** module wording (§6.1).
- Add prose for the **mode-dependent factory closures** (electronics/noise/digitize/track-hits) — signatures vary by dense/bucketed/wire-sparse mode and are invisible to mkdocstrings; document on the concept pages.
- **Mark legacy paths** in `track_hits.py` docstrings (§6.5).
- Document `run_batch.py`'s threaded `main()` (nested `save_worker`/`_read_loop`/`_build_sim` closures) **by hand** in `production/batch.md`; auto-gen surfaces none of it.
- `visualization.py` single-plane/waveform variants: thin one-line docstrings, no Params/Returns, none document the returned `Figure` — flesh out if they get API pages.
- Decide the mkdocstrings filter for `_`-prefixed functions: a few format-critical ones (`_save_wire_plane`, `_decode_plane_hits`) are private but load-bearing for the file format — include on the formats page or document the format by hand.

Docstring health is otherwise good: `recombination`, `drift`, `wires`, `noise`,
`coherent_noise`, `efield_distortions`, `losses`, `particle_generator`,
`pixel_visualization`, `sparse_utils`, `utils`, `save`, `load` are essentially
auto-gen-ready.

---

## 8. Tricky/non-obvious topics that must get first-class treatment

Ranked by how badly they bite, with code location and the page that owns each.

| # | Topic | Code location | Owner page |
|---|---|---|---|
| 1 | **Units: ENC (wire) vs ADC (pixel)**; `inter_thresh`/`corr_threshold`/`threshold_adc` differ per readout | `config.py:TrackHitsConfig`, `track_hits.py`, `noise.py`/`electronics.py`, `save.py` | `physics/units.md` (canonical), referenced everywhere |
| 2 | **Local coordinate frame + geometry-uniformity** (JIT body captures `cfg.volumes[0]` for all volumes) | `loader.py` transform, `simulation.py:forward_segments`, `config.py` pixel origin | `concepts/coordinates.md` |
| 3 | **Capacity sizing + overflow semantics** (crash vs log-skip vs truncate; divisibility constraints) | `process_event` host checks, `total_pad` load check, `max_buckets` RuntimeError | `concepts/capacities.md` + `reference/faq.md` (verbatim errors) |
| 4 | **Two execution paths + what's differentiable** (`fori_loop`/traced n_actual vs `remat`/static n_actual, wire-only) | `compute_plane_signal` `isinstance(n_actual, int)` branch | `architecture/execution-paths.md` |
| 5 | **Padding/masking single point** (`de=0,dx=1`; one mask; downstream trusts charges=0) | `compute_volume_physics` (`charges *= arange < n_actual`) | `concepts/padding-masking.md` |
| 6 | **SCE channel-0 is a drift TIME (µs), not a distance** (time primary, distance derived) | `efield_distortions.py`, `drift.apply_drift_corrections` | `physics/sce.md` |
| 7 | **The "DCT" lie** (real: reflect-pad + separable Gaussian conv) | `kernels.py:generate_dkernel_table` | `physics/response-kernels.md` |
| 8 | **Group→track correspondence, 1-based group ids, qs_fractions** | `loader.compute_group_ids`, `track_hits.compute_qs_fractions`, `finalize_track_hits` | `truth/track-hits.md` |
| 9 | **Pixel single-pass** (signal + per-group truth from one response pass; skips electronics/noise/digitize) | `simulation.py:process_one_volume` pixel branch | `detector/wire-vs-pixel.md` |
| 10 | **Factory closures have implicit, mode-dependent signatures** mkdocstrings can't infer | `create_*_fn_for_volume` in `simulation.py` | prose on electronics/noise pages |

---

## 8b. Alternate modes & example code — keep and document (do NOT delete)

A value-first audit (2026) found that most "uncalled" code is intentional:
alternate modes, reference implementations, public API, toy generators, and
planned scaffolding. **These are kept on purpose and are documentation
material** — showcase them as the framework's option menu. Only strictly-
superseded cruft was removed (commit `ed85862`: `compute_gaussian_diffusion` +
`prepare_pixel_deposit_with_diffusion`, `label_merged_hits`, `diff_dedx` +
`_softplus`, the orphan `load._plane_label`, `export_gif.golden_hash_colors`).

| Mode / surface | Code | Status | Document in |
|---|---|---|---|
| **Wire bucketed accumulation** (memory-saver) | `compute_plane_signal_bucketed`, `compute_bucket_maps`, `sparse_buckets_to_dense`, `generate_noise_bucketed`, `visualize_active_buckets`; compact reference `accumulate_response_signals_sparse_bucketed` | LIVE (`use_bucketed`/`--bucketed`) | `concepts/capacities.md` + `viz/plotting.md` + API |
| **`vmap` volume iteration** | `vmap_over`, `iterate_mode='vmap'` | LIVE (constructor flag) | `architecture/simulator.md` (scan vs vmap, D3) |
| **Merge track-hits path** | `merge_chunk_sensor_hits`, `box_enabled=False` branch | LIVE (non-default) | `truth/track-hits.md` (box vs merge) |
| **Legacy standalone track-hits** | `group_hits_by_track` / `label_hits` / `sparse_hits_to_dense` | test-only, marked legacy | `truth/track-hits.md` (mark legacy) |
| **Standalone sparse↔dense API** | `sparse_utils.py`; `sim.to_dense`/`to_sparse` | reference/API | `viz/plotting.md` + API |
| **Standalone single-file HDF5 I/O** | `utils.py` (`save_event`/`load_event`/…, SCE I/O) | utility/planned | `contributing/io-formats-internal.md` |
| **Toy generators** | `particle_generator` numpy muon gen; `efield_distortions.generate_toy_efield_map`, `compute_drift_corrections` | utility/planned (SCE port) | `differentiable/particle-generation.md`, `physics/sce.md` |
| **Extra viewers** | `visualize_single_plane`/`_waveforms`/`_wire_planes`; `visualize_pixel_3d`/`_buckets`/`_all_pixel_volumes` | utility | `viz/plotting.md`, `detector/wire-vs-pixel.md` |

**Needs-decision (documented, not yet resolved):**
- **Pixel bucketed mode** — `physics.compute_pixel_bucket_maps` / `compute_pixel_signal_bucketed` (+ `wires.scatter_contributions_to_pixel_buckets_batched`, `build_bucket_mapping_3d`, `pixel_visualization.visualize_pixel_buckets`) is a *complete* pixel analog of the wire bucketed mode whose output is already decoded (`output.py`), but `simulation.py`'s pixel branch never dispatches it. **Decision: keep, document as the pixel analog / reference** (not wired end-to-end yet). One dispatch site from working if ever wanted.
- **`nn_utils.py`** (`inv_symlog`/`unfold_kernel`/`normalize_positions`) — NN/SIREN response scaffolding with no in-repo consumer; keep as **planned** (tie to SCE-SIREN work), add a "not yet wired" marker.
- **Module-level `finalize_track_hits`** (`track_hits.py`) — a name-collision duplicate of the live `DetectorSimulator.finalize_track_hits` *method* (the method is the entry point; the module function is only stale-imported by one test). Resolve later: remove the duplicate, or de-dup by having the method call it.

---

## 9. Reading guide — the code spine (content spec for `architecture/reading-guide.md`)

Walk the reader through the code in **execution order**, naming real functions:

1. `generate_detector(yaml)` (`geometry.py`) → raw validated dict; derived params come later.
2. `create_sim_config` + `create_sim_params` (`config.py`) → the static/dynamic split. `create_sim_config` builds per-volume `VolumeGeometry` (with `DiffusionConfig`), computes `num_time_steps` from longest drift + pre/post windows, derives `output_format`.
3. `DetectorSimulator.__init__` (`simulation.py`) — **the closure factory**: loads kernels once, computes analytic box dims (`compute_box_dims`), builds per-concern factories (`_setup_shared_factories` for SCE/response/recomb; `create_electronics_fn_for_volume`, `create_noise_fn_for_volume`, `create_digitize_fn_for_volume`, `create_track_hits_fn_for_volume`), then `_build_jit` assembles `process_one_volume` and wraps it in `iterate(fn, …)` (scan or vmap). The plane `for plane_idx in range(n_planes)` loop is Python, unrolled into the traced graph.
4. `process_event` (`simulation.py`) — stack volumes, call `_calculator_jit`, then **host-side unstack + overflow checks** (max_keys/maxg/bucket overflow raise here), optional coherent noise (numpy, off-JIT), rebuild filled `DepositData`.
5. Shared physics body (`physics.py`): `compute_volume_physics` (recomb + drift + SCE + the single padding mask) → `compute_plane_physics` (per-plane drift correction, attenuation, window cut, wire projection) → `compute_chunk_response` → `compute_plane_signal` (`fori_loop` accumulate). **Both execution paths call these identically.**
6. Branch to depth pages: recombination, drift, kernels, wires, electronics, noise, track_hits.

---

## 10. Notebooks

**Embed the 7 existing** (via `mkdocs-jupyter`), mapped to pages:

| Notebook | Embedded in |
|---|---|
| `getting_started/00_quickstart.ipynb` | `quickstart.md` |
| `getting_started/wire_simulation.ipynb` | `architecture/pipeline-overview.md` / single-event usage |
| `physics/response_kernels.ipynb` | `physics/response-kernels.md` |
| `readout/pixel_simulation.ipynb` | `detector/wire-vs-pixel.md` |
| `gradients/optimization.ipynb` | `differentiable/optimization.md` |
| `reco/segments_closure.ipynb` | `truth/track-hits.md` |
| `production/view_production.ipynb` | `production/data-formats.md` (reconcile its `get_file_paths` with `run_batch`'s `run_*/` + `_NN` output layout before embedding) |

**Author 2 gap notebooks** (CI-runnable on synthetic data, no external files):

| # | Notebook | Content |
|---|---|---|
| G1 | `physics/physics_walkthrough.ipynb` | one event, a plot at each pipeline stage (Q/L → drift → diffusion → kernel → electronics → noise → ADC); realizes diagram D1 |
| G2 | `detector/custom_detector.ipynb` | build a new geometry end-to-end; realizes `contributing/adding-a-detector.md` |

Keep every notebook runnable with **no external data** (synthetic-event helper);
show the `load_event(...)` swap for real data.

---

## 11. Content migration map (source → destination)

| Source | Destination(s) | Action |
|---|---|---|
| `README.md` | `index.md`, `install.md`, `quickstart.md` | trim + refine |
| `CLAUDE.md` (architecture, units, data types, pipeline) | `architecture/*`, `physics/*`, `detector/config-schema.md` | migrate + correct; CLAUDE.md → assistant-only + links |
| `production/README.md` | `production/batch.md`, `data-formats.md` | **rewrite from code** (stale — §6.2) |
| `production/RUN_PRODUCTION.md` | `production/scaling.md` | genericize (§6.6) |
| `profiler/README.md` | `production/profiler.md` + `concepts/capacities.md` | split (what/why/overflow → capacities; which-script → profiler) |
| `viewer/README.md` | `viz/viewer.md` | migrate |
| `tests/TESTS.md` | `contributing/development.md` | migrate |
| `CITATION.cff` | `reference/citation.md` | render |
| (no source) | `architecture/reading-guide.md`, `data-model.md`, `simulator.md`; `concepts/padding-masking.md`; `differentiable/particle-generation.md`; `contributing/io-formats-internal.md`, `adding-a-detector.md` | **new, from code** |

After migration, `CLAUDE.md` shrinks to AI-assistant guidance + pointers into `docs/`.

---

## 12. Build order (execution phases)

Each phase = one dispatch of section-writing agents that read the real code +
source docs and write the `.md` pages, verifying every code snippet.

- **Phase 0 — prerequisites** `[ ]`
  - Docstring fixes (§7): DCT wording in `kernels.py`; legacy markers in `track_hits.py`.
  - Enable `mkdocstrings` + `mkdocs-jupyter` in `mkdocs.yml`; add `docs` extra to `pyproject`.
  - Author diagram D1 (pipeline) — unblocks the physics + architecture pages.
- **Phase 1 — the spine (highest value)** `[ ]`
  - `architecture/*` (reading-guide, data-model, config-vs-params, simulator, execution-paths, pipeline-overview); `concepts/padding-masking.md`, `concepts/capacities.md`; refine `index`/`install`/`quickstart`. Diagrams D2, D3, D5.
- **Phase 2 — physics & detector** `[ ]`
  - `physics/*` (overview, recombination, drift-diffusion, response-kernels, electronics-noise, sce); `detector/*` (config-schema, presets, wire-vs-pixel). Embed response_kernels + pixel_simulation NBs. Author NB G1.
- **Phase 3 — production** `[ ]`
  - `production/*` (batch, data-formats, profiler, scaling — rewrite/genericize per §6). Embed view_production NB (after path reconcile).
- **Phase 4 — differentiable / ML & truth** `[ ]`
  - `differentiable/*` (overview, losses, particle-generation, optimization); `truth/track-hits.md`. Embed optimization + segments_closure NBs.
- **Phase 5 — reference, viz, contributing, polish** `[ ]`
  - `reference/*` (api groups, glossary, faq, citation); `viz/*`; `contributing/*` (development, adding-a-detector, io-formats-internal). Author NB G2.
  - Final: wire full `nav`, `mkdocs build --strict`, notebook-exec CI job, GitHub Pages deploy.

All work on the `docs-site` branch; merge to `main` when the build is green.

## 13. Keeping docs honest (maintenance)

- API reference is generated from docstrings — fix docstrings, not duplicate prose.
- Notebooks executed in CI (synthetic path) so a stale API fails the build.
- `nbstripout` filter + notebook-output-check CI keep notebook diffs clean.
- Single source of truth for the units table (`physics/units.md`) and the
  capacity glossary (`concepts/capacities.md`) — link to them, don't restate.
- "Docs touched?" reminder in the PR checklist when public APIs change.

## 14. Remaining open decisions

1. **Packaging namespace** — `src/jaxtpc/` refactor is deferred; revisit if/when
   a public PyPI release is wanted (affects every import + the install page).
2. **Notebook execution in CI** — full synthetic run (slower, catches more) vs
   import/collect-only (fast). Recommend full synthetic run.
3. **`docs/PLAN.md` visibility** — this build plan currently sits in the public
   nav as "Roadmap." Decide before deploy whether to keep it public, trim it to a
   short roadmap, or move it out of `nav`.
