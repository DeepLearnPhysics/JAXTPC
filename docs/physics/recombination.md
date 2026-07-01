# Recombination

Recombination converts an energy deposit into the two observable quanta of a
liquid-argon TPC: ionization electrons (charge, **Q**) and scintillation photons
(light, **L**). The split is anti-correlated — some electron–ion pairs recombine
before the electrons can drift away, and each recombination emits a photon — so
their sum is fixed by energy conservation.

Implemented in `tools/recombination.py`; the entry point is `compute_quanta`,
which both execution paths reach through the per-volume `recomb_fn` built in
`DetectorSimulator._setup_shared_factories`.

## The quanta

An energy deposit `ΔE` (MeV) over a step `dx` (cm) produces

- `N_i = ΔE / W_ion` electron–ion pairs, and
- `N_ex = α · N_i` excitons,

where `α` is the exciton/ion ratio (`excitation_ratio`, ≈ 0.21, a LAr constant)
and the two work functions are related by `W_ion = (1 + α) · W_ph`. In the code,
`W_ion` is `params.w_value` (the ionization work function, ≈ 23.6 eV) and `W_ph`
is derived as `w_value / (1 + excitation_ratio)` (≈ 19.5 eV).

After recombination with survival fraction `R`, the shared relations are:

```
Q = N_i · R                     electrons that escape
L = ΔE / W_ph − Q               photons (excitation + recombination), by energy conservation
```

All three models share the **survival fraction** formula, differing only in how
they compute the ionization-density parameter `ξ`:

```
R = ln(max(α + ξ, 1)) / ξ
```

The `max(·, 1)` clamp drives `R → 0` at very low `dE/dx`, and steps with `dx ≤ 0`
or `ΔE < 0` yield zero Q and zero L.

!!! note "Units are computed internally"
    `compute_quanta` takes `de` in MeV, `dx` in cm, and the E-field in V/cm
    (converted to kV/cm internally). Q is a number of electrons and L a number of
    photons. What happens to Q downstream (ENC vs ADC) is a separate concern —
    see [units](units.md).

## The three models

The model is chosen by the `model:` key in the YAML `charge_recombination`
block (or the `recombination_model=` constructor argument, which overrides it).
Valid names are in `RECOMB_MODELS = ('modified_box', 'emb', 'passthrough')`.

| Model | `ξ` formula | Angular dependence | Params bundle | Source |
|---|---|---|---|---|
| `modified_box` | `ξ = β/(ρ·E) · dE/dx` | none | `ModifiedBoxParams` | ArgoNeuT 2013 ([1306.1712](https://arxiv.org/abs/1306.1712)) |
| `emb` | `ξ(φ) = β_eff(φ)/(ρ·E) · dE/dx` | `β_eff(φ) = β_90 / √(sin²φ + cos²φ/R²)` | `EMBParams` | ICARUS 2024 ([2407.12969](https://arxiv.org/abs/2407.12969)) |
| `passthrough` | — | — | — | reconstruction / differentiable input |

Here `ρ` is the LAr `density`, `E` the local field (kV/cm), and `φ` (`phi_drift`)
the angle between the track direction and the local E-field.

### Modified Box (ArgoNeuT)

`_xi_modified_box` computes `ξ = (β / ρ) · dE/dx / E`, with **no** angular term.
Default ArgoNeuT parameters: `α = 0.93`, `β = 0.212` (kV/cm)(g/cm²)/MeV.

### EMB — Ellipsoid Modified Box (ICARUS 2024)

`_xi_emb` adds an angular correction. `β` becomes an effective
`β_eff(φ) = β_90 / √(sin²φ + cos²φ / R²)`, so tracks **parallel** to the drift
field (`φ → 0`) see a larger `β_eff`, hence larger `ξ`, hence **more**
recombination than perpendicular tracks (`φ → 90°`). Default ICARUS parameters:
`α = 0.904`, `β_90 = 0.204` (kV/cm)(g/cm²)/MeV, `R = 1.25` (anisotropy ratio).

`phi_drift` is computed in `physics.compute_phi_drift` from the track direction
`(theta, phi)` and the local (possibly SCE-distorted) E-field direction, so EMB
recombination automatically tracks field distortions.

### passthrough

Not a physics model — a bypass for reconstruction and the differentiable path.
The input `de` channel is interpreted as **Q directly** (electrons): no `ξ`, no
`R`, no `Q(dE)` inversion. It returns `Q = max(de, 0)` and `L = 0`, so the
downstream signal is exactly linear in this Q. Use it when you want to
reconstruct charge rather than deposited energy.

## Usage

`compute_quanta` takes an explicit `xi_fn`, selected from `XI_FN` by model name:

```python
import jax.numpy as jnp
from tools.recombination import compute_quanta, XI_FN
from tools.config import create_sim_params
from tools.geometry import generate_detector

cfg = generate_detector('config/cubic_wireplane_config.yaml')
params = create_sim_params(cfg, recombination_model='emb').recomb_params

de   = jnp.array([1.0, 2.5])          # MeV
dx   = jnp.array([0.3, 0.3])          # cm
phi  = jnp.array([0.0, 1.57])         # rad, track vs E-field
efld = jnp.array([500.0, 500.0])      # V/cm

Q, L = compute_quanta(de, dx, phi, efld, params, XI_FN['emb'])
```

In the simulator you do not call this directly — `_recomb_fn` wraps it (or
short-circuits to passthrough) and `compute_volume_physics` applies it, then
applies the single padding mask so padding deposits (`de = 0`) contribute
nothing.

!!! warning "α means different things per model"
    `alpha` is a field in both `ModifiedBoxParams` and `EMBParams` (the box
    `α`-parameter in the `ln(α + ξ)` numerator), and is **numerically different**
    from `excitation_ratio` (the exciton/ion ratio `N_ex/N_i`, also loosely
    called α in the LAr literature). They are separate fields; do not conflate
    them.
