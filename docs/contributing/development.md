# Development setup

This page is for developers working *on* JAXTPC — running the test suite,
keeping notebook diffs clean, and understanding the CI gates. If you only want
to *use* the framework, start at [Install](../install.md) and
[Quickstart](../quickstart.md) instead.

## Editable install with the `dev` extra

Work from an editable install so your changes are picked up without reinstalling,
and pull in the developer tools (`pytest`, `nbstripout`) via the `dev` extra:

```bash
git clone https://github.com/DeepLearnPhysics/JAXTPC
cd JAXTPC
pip install -e ".[dev]"
```

The editable install keeps the repo root on `sys.path`, so `import tools`
(and `production`, `profiler`, `viewer`) resolve, and the runtime assets that
live *outside* the package — the config YAMLs, `config/noise_spectrum.npz` —
resolve via their current relative paths. The `src/jaxtpc/` namespace refactor
is deferred, so a non-editable wheel is not yet the supported path.

!!! note "GPU JAX"
    The default dependencies install **CPU** JAX. For GPU, install the CUDA
    wheel per [the JAX docs](https://docs.jax.dev) or use the `gpu` extra
    (`pip install -e ".[gpu]"`). The test suite is CPU-only and does not need a
    GPU.

## Running the tests

The suite is **264 tests**, all CPU-only on synthetic data (no external files,
no kernels required for the fast tier). `pytest.ini` puts the repo root on
`sys.path`, so bare `pytest` works from the repo root without any install step.

```bash
# Fast tier (~2.5 min on CPU) — the CI merge gate
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v -m "not slow"

# Full suite including integration (~13 min on CPU)
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v

# One module, or one test
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/test_recombination.py -v
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/test_recombination.py::test_mip_survival -v

# With coverage
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -m "not slow" --cov=tools --cov-report=term-missing
```

Always set `JAX_PLATFORM_NAME=cpu` — it forces CPU execution and keeps runs
deterministic and reproducible.

### Markers

| Marker | Meaning |
|---|---|
| `slow` | Kernel-dependent integration and the full-resolution pixel simulation. Excluded from the fast tier. |
| `requires_config` | Needs a detector config YAML. |
| `requires_kernels` | Needs the response-kernel NPZ files in `tools/responses/`. |

Run only the fast tier with `-m "not slow"`; run only the heavy tier with
`-m "slow"`.

### Fixtures

Shared fixtures live in `tests/conftest.py` — `jax_key`, a
`minimal_detector_config`, and small synthetic-deposit builders. `conftest.py`
also forces `JAX_PLATFORM_NAME=cpu` at collection time, so a plain `pytest`
invocation still runs on CPU.

For the full module-by-module breakdown (what each of the 264 tests checks), see
[`tests/TESTS.md`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/tests/TESTS.md).

## Notebooks: keep diffs clean

Notebooks under `notebooks/` are committed **stripped** — no cell outputs, no
execution counts — so diffs stay reviewable and the repo stays small. A
`nbstripout` clean filter is wired via `.gitattributes` (`*.ipynb filter=nbstripout`).
Install it once per clone:

```bash
nbstripout --install --attributes .gitattributes
```

After that, git strips outputs automatically on every commit. To strip a
notebook by hand, or to verify one is clean:

```bash
nbstripout notebooks/getting_started/00_quickstart.ipynb   # strip in place
nbstripout --verify notebooks/getting_started/00_quickstart.ipynb  # check
```

## CI gates

Two GitHub Actions workflows guard the repo:

- **`tests.yml`** — runs the fast tier (`pytest -m "not slow"`) on Python 3.10,
  3.11, and 3.12. This is the **merge gate**. A separate `pytest-slow` job runs
  the `slow` tier on one Python version and is *advisory* (`continue-on-error`):
  the full-resolution pixel sim is memory-heavy and shouldn't block merges if a
  hosted runner can't handle it.
- **`nbstripout-check.yml`** — a **hard gate** that fails if any committed
  notebook still contains outputs or execution counts. If it fails, run
  `nbstripout <notebook>` (or install the clean filter above) and recommit.

## The honesty principle

Two rules keep the docs from drifting away from the code:

- **Fix docstrings, not prose.** The API reference is generated from docstrings
  via `mkdocstrings`. When you change a public function, fix its docstring —
  don't duplicate the description in a hand-written page. Single-source the
  narrative tables too: the [units table](../physics/units.md) and the
  [capacity glossary](../concepts/capacities.md) are canonical; link to them
  rather than restating them.
- **Notebooks run against the real API.** Every notebook is CI-runnable on
  synthetic data (no external files), so an executed notebook that references a
  removed or renamed API fails loudly. Keep the synthetic-data path working when
  you change signatures.

!!! note "Notebook execution in CI is planned"
    The current CI runs pytest and the nbstripout gate; automated
    `jupyter nbconvert --execute` of the notebooks on the synthetic path is
    planned (see [`docs/PLAN.md`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/docs/PLAN.md) §12–§14), not yet wired. Until it
    lands, run the notebooks locally after changing a public API.
