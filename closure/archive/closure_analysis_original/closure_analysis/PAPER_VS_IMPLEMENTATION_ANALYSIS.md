# 3DGS-MCMC → JAXTPC Closure: Gap Analysis

This document compares the 3DGS-MCMC paper's approach with the current `sgld_closure.py` implementation, identifies fundamental mismatches, and proposes corrections.

---

## Analogy Mapping

| 3DGS Concept | JAXTPC Equivalent |
|---|---|
| 3D Gaussian | Point-charge segment (position + energy) |
| Opacity `o_i` | Segment energy `dE_i` |
| Gaussian covariance `Sigma_i` | No direct analog (segments are points, not extended objects) |
| Rendered image | Wire signals on U/V/Y planes |
| L1 + D-SSIM pixel loss | Sliced Wasserstein distance on (wire, time) pointclouds |
| L1 opacity regularization | L1 energy regularization (attempted) |
| Dead Gaussian (opacity < 0.005) | Dead segment (energy < 0.05 MeV) |

---

## Issue 1: Noise Gating Is Inverted

### Paper
Noise gate: `sigma(-k * (o_i - t))` with `t = 0.995`
- High opacity (useful, converged) → gate ≈ 0 → **no noise** (protect converged)
- Low opacity (useless, candidate for death) → gate ≈ 1 → **full noise** (explore freely)
- The idea: converged Gaussians should not be perturbed; dying Gaussians should explore aggressively before being relocated

### Current implementation
Noise gate: `sigmoid(-K_GATE * (energy - GATE_THRESHOLD))` with `GATE_THRESHOLD = 0.2 MeV`
- High energy (useful, alive) → gate ≈ 0 → no noise
- Low energy (dying) → gate ≈ 1 → full noise

### The problem
This is **directionally correct** (noise for weak segments, protection for strong ones), but the **threshold is wrong relative to the operating regime**. In the paper, t=0.995 means noise is suppressed only for the top 0.5% of opacity. Nearly all Gaussians receive some noise. In the implementation, GATE_THRESHOLD=0.2 MeV means noise only activates when a segment has lost 87% of its initial energy (1.6 MeV). As documented in SGLD_FINDINGS.md, segment energies **never fall below 0.5 MeV** in the 10-segment case, so **noise never activates**.

### Fix
The gate threshold should be set relative to the operating energy range, not at an absolute low value. If segments operate at 0.5–2.0 MeV, a threshold like 1.5 MeV would gate noise for most segments while protecting only the highest-energy (best-converged) ones. Alternatively, use a **relative threshold** (e.g., top 10% of current energies are protected).

---

## Issue 2: Regularization and Noise Are Decoupled

### Paper (the critical co-dependency)
The paper's central mechanism is a **three-part cycle**:
1. **L1 opacity regularization** continuously pushes unused Gaussians toward zero → creates dead candidates
2. **SGLD noise** lets those low-opacity Gaussians explore before dying
3. **Relocation** teleports dead Gaussians to high-opacity locations

All three are co-dependent. The ablation shows removing ANY one causes failure:
- No regularization → no dead Gaussians → relocation never triggers → **catastrophic (17.60 PSNR)**
- No noise → Gaussians trapped in local minima → regularization kills useful ones → **degraded (23.84 PSNR when added to vanilla 3DGS)**
- No relocation → dead Gaussians are wasted → capacity shrinks over time

### Current implementation
- L1 regularization on energy was tested (`test_l1_drain.py`) and found **completely ineffective** because:
  - L1 gradient is negligible vs. simulation gradients (0.001 vs 100)
  - Adam normalizes gradients, so absolute magnitude doesn't matter
  - SW loss distributes energy uniformly — no sparsity incentive
- Without effective regularization, segments never die
- Without dead segments, relocation never triggers
- Without relocation, capacity is never redistributed

### Root cause
The L1 regularization failure is **not a hyperparameter problem** — it's a fundamental difference between the two domains:

In 3DGS: opacity ∈ [0, 1] (sigmoid-bounded). An unused Gaussian can have its opacity pushed to ~0 by the regularizer while having minimal effect on the rendered image (opacity 0 = fully transparent = contributes nothing).

In JAXTPC: energy dE > 0, and every segment with nonzero energy produces a wire signal that contributes to the total. There is no "transparent" state — a segment at 0.1 MeV still generates a physical signal that the SW loss must account for. The sim gradients from matching this signal overwhelm any L1 pressure.

### Fix options
1. **Add an explicit opacity-like parameter**: Give each segment an opacity `o_i ∈ [0, 1]` that **multiplicatively scales its energy contribution**: `effective_energy = o_i * dE_i`. The L1 regularizer acts on `o_i`, not `dE_i`. When `o_i → 0`, the segment effectively vanishes regardless of its energy. This directly mirrors the paper's mechanism.

2. **Use energy fraction as opacity proxy**: Define `o_i = dE_i / sum(dE_j)`. Regularize this fraction. But this couples all segments and may create gradient issues.

3. **Stuck-segment detection instead of energy thresholds**: Monitor position/energy change rates over a window. If a segment hasn't moved in N steps, declare it dead and relocate. This sidesteps the regularization problem entirely (already suggested in SGLD_FINDINGS.md).

---

## Issue 3: No Covariance Scaling on Noise

### Paper
Noise is scaled by `Sigma_i * eta` — each Gaussian's covariance matrix. This makes noise **anisotropic**, aligned with the Gaussian's natural shape. A thin elongated Gaussian explores preferentially along its long axis.

### Current implementation
Noise is isotropic: `NOISE_LR * gate * random_normal` applied equally to x, y, z.

### Why this matters less for JAXTPC
Segments are point charges, not extended objects — there is no natural covariance/shape to align noise with. However, the **detector geometry** creates anisotropy: x is the drift direction (most sensitive), while y and z are along the wire planes. Noise could be scaled by detector-geometry-aware factors rather than per-segment covariance.

### Possible fix
Use anisotropic noise with detector-informed scales: e.g., less noise in x (drift direction, where signals are most sensitive) and more in y/z (along wire planes, where the loss landscape may be flatter).

---

## Issue 4: The Loss Function Difference

### Paper
- L1 + D-SSIM pixel-level loss: **dense, per-pixel gradients**
- Every Gaussian that contributes to any pixel gets a gradient
- The alpha-blending rendering equation provides smooth gradient flow through opacity

### Current implementation
- Sliced Wasserstein distance on pointclouds extracted via **top-K** selection
- **Top-K creates a hard gradient mask**: only the K brightest (wire, time) points contribute to the loss
- When 10 segments overlap, weaker segments fall below the K threshold and receive **zero gradient**
- This is the "gradient desert" problem documented in SGLD_FINDINGS.md

### Why this is the fundamental bottleneck
In 3DGS, even a nearly-transparent Gaussian (opacity 0.01) still contributes to the rendered pixel color and receives a gradient proportional to its contribution. The gradient is small but nonzero. This allows the L1 regularizer to gradually push it to zero opacity.

In JAXTPC with top-K, a weak segment that falls below the K threshold gets **exactly zero gradient** — not small, but zero. There is no gradient signal to push it toward death, and no gradient signal for noise to perturb. This creates the circular failure mode: no gradient → no energy change → thresholds never reached → no relocation → segment stays stuck.

### Fix options
1. **Soft top-K (differentiable relaxation)**: Replace hard top-K with a differentiable approximation (e.g., softmax weighting with temperature). All points contribute, but brighter ones contribute more.

2. **Use all points**: Remove top-K entirely and use the full (wire, time, weight) signal. This may be expensive but eliminates the gradient masking problem.

3. **Stratified sampling**: Instead of global top-K, divide the (wire, time) plane into spatial regions and sample K/R points from each region. This guarantees all spatial regions (and hence all segments) contribute to the loss.

4. **Dense loss instead of SW**: Use a direct L1/L2 loss on the wire signals (as 2D arrays, wires × time) instead of converting to pointclouds. This is more analogous to the paper's pixel-level loss and provides gradients to every segment that contributes to any wire/time bin.

---

## Issue 5: Energy Splitting During Relocation

### Paper
When relocating a dead Gaussian to a source Gaussian, the opacity is split via:
```
o_new = 1 - (1 - o_old)^{1/N}
```
This ensures N copies with `o_new` render identically to one copy with `o_old`. The covariance is also adjusted.

### Current implementation
Energy is split 50/50:
```python
params['dE'] = params['dE'].at[dead_idx].set(alive_energy * 0.5)
params['dE'] = params['dE'].at[alive_idx].set(alive_energy * 0.5)
```

### The problem
The 50/50 energy split is **physically correct for additive signals** (two segments each at half energy produce the same total wire signal as one at full energy, if at the same position). This is actually more appropriate than the paper's formula because:
- In 3DGS, opacity combines via alpha-blending (multiplicative): `(1-o)^N`
- In JAXTPC, energy combines additively: `E1 + E2 = E_total`

So the 50/50 split is the right analog. **This is one thing the implementation gets correct.**

### Minor improvement
Consider splitting **unequally** (e.g., 70/30) to give the relocated segment less energy initially, allowing it to explore without creating a large immediate signal disturbance. The paper's N-th root formula effectively does this for opacity.

---

## Issue 6: Adam Moment Reset Direction Is Inverted

### Paper
- Adam moments are **reset for the source** (donor) Gaussian: "The optimizer state is reset for the target sample to take large steps and encourage further optimization"
- Adam moments are **retained for the relocated** (dead → new) Gaussian: it inherits the new position but keeps its own optimizer momentum

### Current implementation (sgld_closure.py)
- Adam moments are **reset for the donor** segment (correct direction)
- Dead segment moments are **kept** (correct direction)

This actually matches the paper. No change needed.

---

## Issue 7: Temperature and Noise Schedule

### Paper
- `lambda_noise = 5e-5` is a fixed constant
- Noise naturally decays because it's multiplied by `lr`, which decays exponentially
- Effective temperature `~ lambda_noise / lr` **increases** over training (noise decays slower than lr)
- No explicit temperature annealing

### Current implementation
- Noise decays via its own exponential: `lr * exp(-step/noise_tau) * NOISE_LR`
- `noise_tau = total_steps / ln(100)` → noise drops by 100x over training
- Learning rate also decays via cosine schedule
- Combined effect: noise decays **much faster** than in the paper

### Fix
The paper couples noise to the learning rate only. The additional `exp(-step/noise_tau)` decay is overly aggressive and kills exploration too early. Remove the separate noise decay and let the noise inherit the learning rate schedule:
```python
noise = lr_current * NOISE_LR * gate * random_normal
```
where `lr_current` already decays via the cosine schedule.

---

## Issue 8: Warmup Period

### Paper
- 500 iterations warmup before relocation or densification
- Noise is applied from the start (no warmup for noise)

### Current implementation
- 500 steps warmup before both noise AND relocation

### Fix
Apply noise from step 0 (or very early). The warmup should only gate relocation, not exploration noise. Early noise helps segments escape bad initial positions before they get stuck.

---

## Summary: Priority-Ordered Fixes

| Priority | Issue | Impact | Difficulty |
|----------|-------|--------|------------|
| **1** | Top-K gradient masking (Issue 4) | Fundamental — blocks all other mechanisms | Medium (try dense loss or soft top-K) |
| **2** | Regularization mechanism (Issue 2) | Without it, no dead segments, no relocation | Medium (add opacity parameter or stuck-detection) |
| **3** | Noise gate threshold (Issue 1) | Noise never activates in practice | Easy (raise threshold to operating regime) |
| **4** | Noise schedule (Issue 7) | Exploration dies too fast | Easy (remove separate decay) |
| **5** | Warmup for noise (Issue 8) | Delays early exploration | Easy (remove noise warmup) |
| **6** | Anisotropic noise (Issue 3) | Minor improvement for detector geometry | Low priority |
| **7** | Energy split ratio (Issue 5) | 50/50 is already correct for additive physics | No change needed |
| **8** | Moment reset (Issue 6) | Already correct | No change needed |

---

## Recommended Implementation Order

### Phase 1: Fix the gradient flow
Replace top-K pointcloud extraction with either:
- (a) Dense L1/L2 loss directly on wire signal arrays, or
- (b) Soft top-K with differentiable weighting, or
- (c) Full pointcloud (no top-K) with SW loss

This alone may fix N=10 because all segments will receive gradients.

### Phase 2: Fix the noise mechanism
- Remove noise warmup (apply from step 0)
- Raise gate threshold to ~80th percentile of energy range
- Remove separate noise exponential decay (couple to LR schedule only)

### Phase 3: Fix the death/relocation cycle
- Add explicit opacity parameter `o_i ∈ [0,1]` multiplying energy, OR
- Implement stuck-segment detection (position/energy change rate monitoring)
- L1 regularize the opacity (not the energy directly)
- Tune death threshold relative to the new opacity parameter

### Phase 4: Tune and validate
- Run N=10 closure test with all fixes
- Compare to baseline Adam at N=5 to ensure no regression
- Ablation: remove each component individually to verify co-dependency
