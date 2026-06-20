# Installation

JAXTPC is installable as `jaxtpc`. The supported mode today is an **editable
install** from a clone (the `src/jaxtpc/` namespace migration is
[deferred](PLAN.md), so a non-editable wheel isn't fully supported yet).

```bash
git clone https://github.com/DeepLearnPhysics/JAXTPC.git
cd JAXTPC
pip install -e .
```

This installs the runtime dependencies (JAX, NumPy, SciPy, Matplotlib, h5py,
hdf5plugin, PyYAML, Pillow) and registers the CLIs (`jaxtpc-batch`,
`jaxtpc-setup-production`, `jaxtpc-make-labl`, `jaxtpc-viewer`,
`jaxtpc-export-gif`).

## GPU (JAX with CUDA)

By default JAX installs CPU-only. For GPU, install the CUDA build:

```bash
pip install -e ".[gpu]"          # jax[cuda12]
# or follow https://docs.jax.dev/en/latest/installation.html for your CUDA setup
```

## Optional extras

```bash
pip install -e ".[dev]"          # pytest, nbstripout
pip install -e ".[docs]"         # mkdocs-material, mkdocstrings, mkdocs-jupyter
```

## Notebook output stripping (once per clone)

Notebook outputs are kept out of git via an `nbstripout` filter. After cloning:

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

## Verify

```bash
python -c "import tools, jax; print('JAXTPC OK', jax.devices())"
JAX_PLATFORM_NAME=cpu pytest -m "not slow" -q      # ~2 min, CPU
```

!!! note "Use `python3` on systems where `python` is Python 2"
    All commands assume `python` is Python ≥ 3.10.
