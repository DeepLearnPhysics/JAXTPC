# Overcomplete MCMC Closure Optimization — Findings

## Goal

Reconstruct N_truth=5 point-charge segments from simulated wire signals using N_seg=10 optimizer segments (overcomplete). The extra 5 segments should be identified as unnecessary and either drain to minimal energy or get relocated. Based on the 3DGS-MCMC paper adapted for TPC simulation.

Init offset: [+30, -30, +30, +0.05] mm/MeV (~52mm Euclidean distance). Extra segments initialized randomly within detector bounds at median truth energy.

---

## Phase 1: 5-Seg Base Optimizer Tuning (no noise, no L1, no relocation)

Systematic sweep to find the optimal Adam + schedule configuration before adding MCMC components.

### Schedule Comparison (LR=0.3, lr_e_mult=0.03)

| Schedule | Pos (mm) | Loss | Step<3mm |
|----------|----------|------|----------|
| Cosine α=0.01 | 1.58 | 0.000165 | 450 |
| Exp d=0.999 | 1.22 | 0.000090 | 450 |
| Exp d=0.9995 | 1.21 | 0.000236 | 430 |
| Constant | 0.95 | 0.000374 | 380 |

### LR Sweep (exp d=0.9995, lr_e_mult=0.01)

| LR | Pos (mm) | Loss | Step<3mm |
|----|----------|------|----------|
| 0.5 | 1.09 | 0.000147 | 340 |
| **0.7** | **0.42** | **0.000058** | **260** |
| 1.0 | 0.64 | 0.000166 | 300 |

### Decay Rate Sweep (LR=0.5, lr_e_mult=0.01)

| Decay Rate | Pos (mm) | Loss |
|------------|----------|------|
| 0.999 | 1.28 | 0.000138 |
| 0.9995 | 1.09 | 0.000147 |
| 0.9998 | 0.75 | 0.000062 |

### lr_e_mult — Critical Phase Transition (LR=0.7, d=0.9995)

| lr_e_mult | Pos (mm) | Max (mm) | Loss | Notes |
|-----------|----------|----------|------|-------|
| 0.003 | 6.25 | 15.4 | 0.000651 | Energy too slow — stuck |
| 0.008 | 6.00 | 15.0 | 0.000560 | Still stuck |
| 0.01 | 6.38 | 16.4 | 0.000746 | Below threshold |
| **0.015** | **0.80** | **1.68** | **0.000054** | Above threshold |
| **0.02** | **0.52±0.24** | **0.98** | **0.000047** | **Robust (3 runs: 0.21, 0.78, 0.46)** |
| **0.025** | **1.14±0.80** | **1.90** | **0.000106** | Less stable (3 runs: 0.63, 0.72, 2.08) |
| 0.03 | 6.38 | 15.4 | 0.000712 | Above threshold — wraps around? |

**Key finding**: Sharp phase transition at lr_e_mult ≈ 0.015. Below this, energy learns too slowly to support position convergence. Above ~0.025, variance increases. **lr_e_mult=0.02 is the robust optimum.**

### Scaling to 8 Segments (8=8 equal, lr_e_mult=0.03/0.05)

| Config | Pos (mm) | Max (mm) | Loss |
|--------|----------|----------|------|
| 8=8, e=0.03 | 31.1–32.3 | 59–64 | 0.032 |
| 8=8, e=0.05 | 30.8–33.1 | 58–61 | 0.031 |

**8 segments completely stuck at ~31mm** — barely moved from 52mm init. The 5-seg hyperparams don't transfer. More segments = much harder loss landscape.

### Optimal 5-Seg Base Config

```
Schedule: Exponential decay, LR=0.7, decay_rate=0.9995
lr_e_mult=0.02, b1=0.95, b2=0.999
No noise, no L1, no relocation
Result: 0.52mm mean (0.21–0.78 range), 1000 steps
```

---

## Phase 1b: Noise on 5-Seg Base

Tested on the optimal base config (LR=0.7, d=0.9995, lr_e_mult=0.02).

### Quadratic Coupling (noise ∝ lr²) — TOO AGGRESSIVE

| NOISE_LR | Pos (mm) | Notes |
|----------|----------|-------|
| 0.5 | 2.00 | 4x worse than baseline |
| 1.0 | 2.58 | |
| 2.0 | 4.59 | |
| 5.0 | 122.7 | Catastrophic |

**Quadratic coupling decays too fast with exp schedule** — noise is too strong early, dies too quickly late. Not suitable.

### Constant Noise (fixed mm/step/dim)

| Noise (mm) | Pos (mm) | Max (mm) | Loss |
|------------|----------|----------|------|
| 0.02 | **0.39** | 0.90 | 0.000074 |
| 0.05 | 1.07 | 1.84 | 0.000094 |
| 0.1 | 1.07 | 1.88 | 0.000170 |

### Linear LR-Coupled (noise = lr × NOISE_LR)

| NOISE_LR | Init→Final (mm) | Pos (mm) | Max (mm) | Loss |
|----------|-----------------|----------|----------|------|
| 0.05 | 0.035→0.021 | **0.72** | 0.95 | 0.000056 |
| 0.1 | 0.070→0.043 | 1.17 | 1.94 | 0.000116 |
| 0.2 | 0.140→0.085 | 1.18 | 1.66 | 0.000100 |

**For 5=5 equal, noise doesn't help** — pure Adam already reaches the global minimum. Small noise (constant=0.02mm or linear=0.05) is tolerable but not beneficial. Larger noise monotonically degrades performance.

**For overcomplete, noise will be essential** to break symmetry and escape local minima. Safe starting values: constant=0.02mm or linear NOISE_LR=0.05.

---

## Phase 2: Overcomplete Runs (10 opt / 5 truth) — Previous Session

All runs used cosine or exponential schedule with LR=0.3–0.5, lr_e_mult=0.01–0.03.

| # | LR Schedule | LR | Noise | L1 | DT | Split | WU | Steps | Loss | Pos (mm) | dE (keV) | Extras Dead | Relocs |
|---|-------------|-----|-------|-----|------|-------|-----|-------|------|----------|----------|-------------|--------|
| 1 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.01 | 50/50 | 500 | 1500 | 0.001 | 6.6 | 82.3 | 0/5 | 11 |
| 2 | Cosine α=0.01 | 0.3 | 3 linear | 2e-4 fixed | 0.01 | 50/50 | 500 | 1500 | 0.011 | 28 | — | 0/5 | — |
| 3 | Cosine α=0.01 | 0.3 | 5 linear | 1e-3 coupled | 0.01 | 50/50 | 500 | 1500 | 0.004 | 18 | 83.8 | 0/5 | 16 |
| 4 | Cosine α=0.01 | 0.3 | 15 linear | 3e-3 coupled | 0.01 | 50/50 | 500 | 1500 | 0.025 | 44 | 73.1 | 0/5 | — |
| 5 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.03 | 50/50 | 500 | 1500 | 0.007 | 14 | 99.1 | 1/5 | 18 |
| 6 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.02 | 50/50 | 500 | 1500 | 0.004 | 11.4 | 67.4 | 0/5 | 14 |
| 7 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.001 | 50/50 | 500 | 1500 | 0.006 | 18.3 | **1.4** | **5/5** | 0 |
| 8 | Cosine α=0.01 | 0.3 | 3 linear | 5e-5 fixed | 0.001 | 50/50 | 500 | 1500 | 0.006 | 20.8 | 1.0 | 5/5 | 0 |
| 9 | Cosine α=0.01 | 0.3 | 8 linear | 5e-5 fixed | 0.001 | 50/50 | 500 | 1500 | 0.018 | 52 | 91 | 3/5 | 0 |
| 10 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.005 | 50/50 | 500 | 2500 | 0.0009 | 6.9 | 49.6 | 0/5 | 47 |
| 11 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.005 | 50/50 | 50 | 1000 | 0.005 | 15 | 68.7 | 0/5 | 17 |
| 12 | Cosine α=0.01 | 0.3 | 3 linear | 5e-4 coupled | 0.005 | 50/50 | 50 | 1000 | 0.004 | 16 | 88 | 0/5 | 15 |
| 13 | Cosine α=0.01 | 0.3 | 5 linear | 5e-4 coupled | 0.005 | 50/50 | 50 | 1000 | 0.007 | 19 | — | 0/5 | — |
| 14 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.005 | 90/10 | 50 | 1000 | 0.004 | 18 | 102.4 | 0/5 | 29 |
| 15 | Cosine α=0.01 | 0.3 | 5 linear | 1e-3 coupled | 0.005 | 80/20 | 50 | 1000 | 0.015 | 62 | 75 | 0/5 | 27 |
| 16 | Cosine α=0.01 | 0.3 | 5 linear | 5e-5 fixed | 0.005 | 50/50 | 500 | 4000 | **0.0003** | **2.96** | 82.6 | 0/5 | 82 |
| 17 | Constant | 0.05 | 5 const | 5e-5 fixed | 0.005 | 50/50 | 500 | 2000 | 0.005 | 20.8 | 48.5 | 0/5 | 7 |
| 18 | Constant | 0.1 | 5 const | 5e-5 fixed | 0.005 | 50/50 | 500 | 2000 | 0.006 | 11.9 | 48.3 | 0/5 | 13 |
| 19 | Constant | 0.2 | 5 const | 5e-5 fixed | 0.005 | 50/50 | 500 | 2000 | 0.007 | 13.5 | 33.3 | 0/5 | 29 |
| 20 | Constant | 0.2 | 5 const | 2e-4 fixed | 0.005 | 70/30 | 500 | 2000 | 0.010 | 17 | 53.7 | 0/5 | 29 |
| 21 | Exp d=0.998 | 0.3 | 2 linear | 5e-5 fixed | 0.005 | 50/50 | 500 | 2000 | 0.006 | 22 | 33.4 | 0/5 | 5 |
| 22 | Exp d=0.9995 | 0.5 | 2 linear | 5e-5 fixed | 0.005 | 50/50 | 500 | 2000 | 0.001 | 4.5 | 44.9 | 0/5 | 16 |
| 23 | Exp d=0.9995 | 0.5 | 5 lr² | 5e-5 fixed | 0.005 | 50/50 | 500 | 2000 | 0.001 | 3.5 | 55.1 | 0/5 | 28 |

### Key Findings from Overcomplete Runs

**Settled parameters** (consistent across all runs):
- **L1**: Fixed 5e-5 MeV/step always best. Coupled or stronger always worse.
- **Death threshold**: 0.005 best balance. Lower = better energy, worse positions.
- **Split ratio**: 50/50 best. 90/10, 80/20, 70/30 all worse.
- **Warmup**: 500 always better than 50.

**Fundamental limitations**:
- SW loss doesn't incentivize sparsity — extras never fully die
- Two-regime tradeoff: more relocations = better positions, worse energy
- Top-K gradient masking can leave weak segments with zero gradient

---

## Summary: What We Know

### Locked-In Base Config
```
Schedule: Exponential decay, LR=0.7, decay_rate=0.9995
b1=0.95, b2=0.999
lr_e_mult=0.02
```

### Locked-In MCMC Components (from Phase 2)
```
L1: Fixed rate (not coupled)
Death threshold: 0.005
Split ratio: 50/50
Warmup: 500
Relocation: Continuous (every step after warmup)
Donor selection: Energy-proportional multinomial
```

### To Be Tuned for Overcomplete with New Base Config
- **L1 rate**: Was 5e-5 with old LR=0.3 config. May need adjustment with LR=0.7 / lr_e_mult=0.02 since energy dynamics are different.
- **Noise**: Essential for overcomplete (breaks symmetry, escapes local minima). Constant or linear coupling — NOT quadratic. Start with constant=0.02mm or linear=0.05.
- **Warmup**: Was 500 with slow LR=0.3 cosine. With faster LR=0.7 exp, positions converge by ~260 steps. Could try 200–300.
