# 3D Gaussian Splatting as Markov Chain Monte Carlo — Paper Summary

**Paper**: Kheradmand et al., NeurIPS 2024 (Spotlight)
**arXiv**: [2404.09591](https://arxiv.org/abs/2404.09591)
**Code**: [github.com/ubc-vision/3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc)

---

## 1. Core Idea

Standard 3DGS treats Gaussians as deterministic parameters optimized via Adam. The paper reinterprets them as **samples from a probability distribution** over scene representations. Under this view, optimization becomes **MCMC sampling**, and adding noise to gradients converts SGD into **Stochastic Gradient Langevin Dynamics (SGLD)**.

The key observation: the standard 3DGS gradient update is already SGLD **minus the noise term**:

```
Standard:  g <- g - lr * grad L(g)
SGLD:      g <- g - lr * grad L(g) + noise * epsilon
```

Adding appropriately scaled noise turns exploitation-only optimization into exploration+exploitation sampling.

---

## 2. Why MCMC Instead of Pure Optimization

1. **Local minima trapping**: Pure Adam can only descend — no mechanism to escape poor basins
2. **Heuristic fragility**: Vanilla 3DGS uses ad-hoc clone/split/prune thresholds with no principled justification
3. **Initialization sensitivity**: Without exploration, bad initial placement is unrecoverable
4. **No count control**: Original 3DGS lets Gaussian count grow unboundedly

---

## 3. The SGLD Update Rule

Full update for Gaussian positions:

```
mu <- mu - lr * grad_mu L_total + noise_term
```

where the **noise term** (Eq. 8 in paper) is:

```
noise_term = lr * sigma(-k * (o_i - t)) * Sigma_i * eta,    eta ~ N(0, I)
```

### Noise design choices:

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Isotropic base** | `eta ~ N(0, I)` | Random exploration direction |
| **Covariance scaling** | `Sigma_i * eta` | Anisotropic noise aligned with Gaussian's shape — perturb along natural axes |
| **Sigmoid gating** | `sigma(-k * (o_i - t))` | Suppress noise for high-opacity (useful) Gaussians; allow noise for low-opacity (underutilized) ones |
| **LR coupling** | `lr * ...` | Noise decays naturally with learning rate schedule |

### Critical hyperparameters:
- `k = 100` (sigmoid steepness — effectively a sharp step function)
- `t = 0.995` (opacity threshold — noise only when opacity < 0.995)
- `lambda_noise = 5e-5` (noise magnitude)
- **Position-only**: Noise applied ONLY to positions `mu`, NOT to opacity, scale, rotation, or color

---

## 4. Loss Function

```
L_total = (1 - lambda_SSIM) * L1 + lambda_SSIM * L_D-SSIM + lambda_o * mean(|o_i|) + lambda_Sigma * mean(|sqrt(eig(Sigma_i))|)
```

| Term | Weight | Purpose |
|------|--------|---------|
| L1 pixel loss | 0.8 | Reconstruction accuracy |
| D-SSIM loss | 0.2 | Structural similarity |
| **L1 on opacity** | **0.01** | Push unused Gaussians toward zero opacity → creates "dead" pool for relocation |
| **L1 on scale** | **0.01** | Encourage compact Gaussians, prevent degenerate large ones |

### The opacity regularization is essential

It serves as a **Laplace (sparsity) prior** in the MCMC framework. Without it, there is no mechanism to create dead Gaussians for relocation. The ablation shows:
- **With regularization, without noise**: 27.41 PSNR (decent)
- **With noise, without regularization**: 17.60 PSNR (catastrophic failure)
- **With both**: 29.72 PSNR (best)

---

## 5. Relocation Strategy (Birth/Death)

### Philosophy
Rather than creating/destroying Gaussians (variable count), the paper keeps a **fixed budget** and **teleports** dead Gaussians to useful locations. This is grounded in MCMC: samples move through space, they are not created or destroyed.

### Death criterion
```
o_i < 0.005    (opacity below 0.5%)
```

The L1 regularization naturally pushes unused Gaussians toward zero opacity, creating dead candidates.

### Target selection
Dead Gaussians are relocated to alive Gaussians via **multinomial sampling proportional to opacity**. High-opacity (important) Gaussians are more likely to receive copies.

### Probability-preserving split formulas

When duplicating a Gaussian into N copies, the opacity is updated to preserve rendering:

```
(1 - o_new)^N = 1 - o_old
→ o_new = 1 - (1 - o_old)^{1/N}
```

The covariance is also updated via a complex formula involving binomial coefficients to match integrated color contribution.

### Schedule
- **Warmup**: 500 iterations (no relocation)
- **Frequency**: Every 100 iterations after warmup
- **Growth**: Add 5% of current count per step until cap reached
- **Moment reset**: Adam moments are **reset for the source** (donor) Gaussian but **retained for the relocated** (dead→new) Gaussian

---

## 6. Fixed Gaussian Budget

A `cap_max` parameter sets the maximum number of Gaussians. This is a **feature, not a limitation**:
- Eliminates unbounded growth
- Forces efficient resource allocation via relocation
- Makes memory usage predictable

---

## 7. Ablation Study Results

| Configuration | PSNR | Key takeaway |
|---|---|---|
| Vanilla 3DGS | 27.89 | Baseline |
| + L1 regularization only | 23.84 | Regularization alone **hurts** — kills useful Gaussians with no way to replace them |
| Full method, no noise | 27.41 | Relocation works somewhat without noise, but Gaussians still get trapped |
| Full method, no regularization | 17.60 | **Catastrophic** — without regularization, no dead Gaussians are created, relocation never triggers |
| Full method, noise on ALL params | 29.11 | Slightly worse than position-only noise |
| **Full method, position-only noise** | **29.72** | **Best** — spatial exploration is the key bottleneck |

### Key ablation insight
**Noise and regularization are co-dependent**. Noise provides exploration to escape local minima. Regularization creates the dead pool that enables relocation. Relocation redistributes capacity to where it's needed. Remove any one component and the system degrades.

---

## 8. Initialization Robustness

| Method | SfM Init | Random Init | Drop |
|---|---|---|---|
| Vanilla 3DGS | 27.89 | 22.72 | -5.17 |
| 3DGS-MCMC | 29.72 | 29.64 | **-0.08** |

The SGLD exploration makes the method nearly initialization-independent.

---

## 9. Theoretical Caveats

The MCMC connection is a **productive conceptual framework**, not a rigorous proof:

1. **Not true SGLD**: The sigmoid gating, covariance scaling, and separate `lambda_noise` break standard SGLD convergence conditions
2. **Adam breaks SGLD**: Adam's adaptive LR and momentum destroy theoretical guarantees (standard SGLD assumes vanilla SGD)
3. **No detailed balance**: Relocation moves preserve rendering approximately, not MCMC-rigorously
4. **No temperature annealing**: Effective temperature is implicitly `lambda_noise / lr`, which increases as lr decays — not formally analyzed

Despite these gaps, the framework **motivates the right algorithmic choices**: noise for exploration, regularization for sparsity, relocation for resource reallocation.

---

## 10. Complete Hyperparameter Table

| Parameter | Value | Notes |
|-----------|-------|-------|
| Position LR (start) | 1.6e-4 | Exponential decay |
| Position LR (end) | 1.6e-6 | Over 30k iterations |
| lambda_noise | 5e-5 | SGLD noise magnitude |
| Sigmoid k | 100 | Sharp gating |
| Sigmoid threshold t | 0.995 | Noise only for low-opacity |
| lambda_o (opacity reg) | 0.01 | L1 on opacity |
| lambda_Sigma (scale reg) | 0.01 | L1 on covariance eigenvalues |
| Dead threshold | 0.005 | Opacity below this = dead |
| Relocation interval | 100 steps | How often to relocate |
| Warmup | 500 steps | No relocation during warmup |
| Growth rate | 5% per step | Until cap_max reached |
| cap_max | Dataset-dependent | Fixed Gaussian budget |
| Adam beta1 | 0.9 | Standard |
| Adam beta2 | 0.999 | Standard |
| SSIM weight | 0.2 | Standard 3DGS value |
