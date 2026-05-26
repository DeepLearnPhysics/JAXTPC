# SGLD Closure Optimizer — Findings & Status

## Goal
Reconstruct N point-charge segments (position + energy) from simulated wire signals using differentiable simulation + optimal transport loss. Adam works well for N=1–5 but fails at N=10 due to local minima. The SGLD-inspired optimizer adds noise injection, segment relocation, and L1 regularization to escape these minima.

## What Works

### Baseline Adam (5 segments, N_seg = N_truth)
- **Settings**: lr=0.3, b1=0.95, cosine schedule (alpha=0.01), LR_ENERGY_MULT=0.1, K=10000, N_PROJ=200
- **Result**: Loss < 10^-3 by step ~320, mean pos error 1.1mm, max 2.7mm at 1200 steps
- **Key**: Cosine schedule is critical — exponential decay (100x) spends the learning budget too early (lr=0.03 at midpoint vs cosine's 0.15). Exponential only reached loss=0.004 at 1200 steps.
- **File**: `closure_analysis/sgld_closure.py --mode baseline`

### Noise injection (5 segments)
- Mode `--mode noise`, 1500 steps: loss=0.002, mean pos error 2.9mm
- Noise gating works correctly: gate opens smoothly at GATE_THRESHOLD=0.2 MeV
- Noise doesn't destabilize converged segments (0 dead throughout)

## What Fails

### 10-segment equal (N_seg = N_truth = 10)

**Baseline** (3000 steps): loss=0.126 (stuck from step ~500), mean pos error 30.5mm. Energy and positions freeze early. Seg 6 doesn't move at all from init (energy exactly 1.600 for all 3000 steps).

**Full SGLD** (noise + relocation, 3000 steps): loss=0.017 (7x better), 5/10 segments within 5mm. BUT Seg 6 still completely stuck, and Seg 1 absorbed energy (grew to 4.4 MeV).

**Why SGLD mechanisms didn't trigger**: All segment energies stayed above 0.5 MeV (DEAD_THRESHOLD=0.05, GATE_THRESHOLD=0.2). Zero dead segments throughout. The noise gate never opened and relocation never activated.

**Root cause of stuck segments**: Seg 6 gets zero sim gradients. Its signal is masked out by the top-K=10000 pointcloud extraction when 10 overlapping segments compete for the brightest points. With no gradient, energy can't change, so thresholds are never reached.

### Overcomplete (N_seg > N_truth) — Does NOT work

Tested with 3 segments, 2 truths. The hypothesis was that extra segments would be L1-drained to zero, triggering relocation. This fails for two reasons:

#### 1. L1 is negligible against sim gradients
- Sim gradients on energy: ~100 (at init)
- L1 gradient at λ=0.001: 0.001 (ratio: 0.00001x)
- L1 gradient at λ=50: 50 (still <1x)
- Even λ=50 produces identical energy distributions to λ=0

#### 2. Extra segments don't naturally drain
The SW loss distributes energy uniformly across all segments rather than concentrating it in the correct ones:

| | E_seg0 (truth=1.0) | E_seg1 (truth=1.0) | E_seg2 (EXTRA) |
|---|---|---|---|
| No L1, 800 steps | 0.589 | 0.407 | 0.391 |
| L1=10 | 0.551 | 0.417 | 0.409 |
| L1=50 | 0.470 | 0.443 | 0.452 |

All three segments converge to similar energies (~0.4–0.6 MeV). The extra segment is indistinguishable from truth segments. The SW loss on the combined signal can be minimized by distributing energy across many segments at wrong positions.

**Positions don't converge either**: After 1200 steps in the overcomplete test, Seg 0 is at [-70, +20, +132] vs truth [-100, +50, +100] — barely moved from init. The extra segment's signal confuses the loss landscape for truth segments.

#### 3. Energy learning rate sweep
- LR_ENERGY_MULT=0.1: Best separation (0.20 MeV gap between truth and extra), but slow
- LR_ENERGY_MULT=0.5: Less separation (0.12 MeV gap)
- LR_ENERGY_MULT=1.0: Separation REVERSES — extra ends up with MORE energy than truth seg 0. Chaotic oscillations.

## Key Insights

1. **Adam normalization**: Adam normalizes gradients, so the effective step is ~lr regardless of gradient magnitude. This means L1's absolute value doesn't matter for the update magnitude — only the direction matters. But since sim gradients and L1 push in the same direction (both positive = both push energy down), L1 doesn't change behavior.

2. **SW loss doesn't incentivize sparsity**: The fundamental issue is that the Sliced Wasserstein distance on the combined (W,T) signal treats all segment contributions additively. More segments at lower energy can match a target as well as fewer segments at correct energy. There's no mechanism to prefer concentrated solutions.

3. **Top-K masking kills gradients for weak segments**: With K=10000 points from a (W,T) signal containing 10 overlapping segments, weaker segments' contributions may fall below the K-th largest value. They get zero gradient and become permanently stuck.

4. **Cosine schedule >> exponential for Adam convergence**: Cosine maintains higher lr through the middle of training. Exponential drops too fast. For SGLD noise, a separate exponential decay (`noise_tau`) controls noise amplitude independently.

## Hyperparameters (current best)

```python
LR_POSITION = 0.3
B1 = 0.95
B2 = 0.999
LR_ENERGY_MULT = 0.1
K = 10000
N_PROJ = 200
DEAD_THRESHOLD = 0.05   # MeV
GATE_THRESHOLD = 0.2    # MeV
K_GATE = 100
NOISE_LR = 40.0
NOISE_DECAY_FACTOR = 100
RELOCATION_INTERVAL = 100
WARMUP = 500
```

Schedule: `optax.cosine_decay_schedule(init_value=0.3, decay_steps=total_steps, alpha=0.01)`

## Files

- `closure_analysis/sgld_closure.py` — Main optimizer with `--mode baseline|noise|full`
- `closure_analysis/test_l1_drain.py` — Diagnostic tests for L1/overcomplete/energy learning rate
- `closure_analysis/optimization_closure.py` — Original Adam baseline (reference)

## Open Questions / Next Directions

1. **Alternative loss functions**: There is a simpler projection-based loss in `ott_test/` (not yet investigated) that projects along a line rather than full SW. This might have different gradient properties that better separate segments.

2. **Per-segment gradient flow**: The top-K masking problem could potentially be addressed by:
   - Lowering K so weaker segments enter the pointcloud
   - Using a soft top-K (differentiable relaxation)
   - Stratified pointcloud extraction (guaranteed points per spatial region)

3. **Stuck-segment detection**: Instead of energy-based thresholds (which fail when sim gradients prevent energy drain), detect stuck segments by monitoring position/energy change rate over a window.

4. **Two-phase optimization**: Phase 1 — fix energies equal, optimize positions only. Phase 2 — unfreeze energies. This separates the permutation problem from the amplitude problem.

5. **N_seg = N_truth remains the viable path**: The overcomplete approach fails because SW loss doesn't incentivize sparsity. Focus should be on making N=N_truth=10 work by solving the stuck-segment problem directly.
