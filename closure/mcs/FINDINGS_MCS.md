# MCS wire-signal closure — systematic findings

Goal: recover a muon's **vertex, direction, energy, and per-segment multiple-
Coulomb-scattering (MCS) angles** from its simulated wire signals, by
differentiable closure through the JAXTPC forward. Data term is the **same
Sobolev geomean-log1p loss** as the segments closure (`make_sobolev_weight`,
s=1.5); the only MCS-specific addition is a **Highland scattering prior**:

```
loss = sobolev_loss_geomean_log1p(sigs, truth)  +  λ · highland_prior(dθ₁,dθ₂ | E) / N
```

Truth track: E=500 MeV, vertex (−200, 0, 100) mm, θ=45°, φ=30°, N=2000 segments
of 0.5 mm, in the dual-TPC `cubic_wireplane` detector. Init guess for the
globals is offset by **(+100,+100,−100) mm and +0.5 rad in θ,φ and +100 MeV**.

## 0. Status before this work: the closure did not run

`closure/mcs/forward.py` imported an external `MCS_muon.mcs_muon_generator`
module that was never migrated into the repo, so **every MCS entry point died at
import**. Reconstructed the missing physics (muon mass, LAr radiation length,
β·p kinematics, the PDG Highland formula, a branchless perpendicular basis, and
a scan-based reference truth generator) as the self-contained
`closure/mcs/mcs_physics.py`. The forward-model validation
(`validate_forward.py`, levels 0–2: position math, CSDA energy, AD-vs-FD
gradients) now passes **14/14**.

## 1. Globals-only (scattering fixed at truth): SOLVED

`python3 -m closure.mcs.run --mode globals-only --steps 300`

| Param | Truth | Init | Final | Error |
|---|---|---|---|---|
| x | −200.0 | −100 | −199.9 | **0.12 mm** |
| y | 0.0 | +100 | 0.2 | **0.16 mm** |
| z | 100.0 | 0 | 100.1 | **0.14 mm** |
| θ | 45.0° | 73.6° | 45.0° | **−0.01°** |
| φ | 30.0° | 58.6° | 30.0° | **−0.02°** |
| E | 500.0 | 600 | 505.5 | **+5.5 MeV (1.1 %)** |

Sub-mm vertex, sub-0.1° direction, ~1 % energy from a 170 mm-displaced start.
The energy is pinned mostly by the **CSDA range** (track length), not by
scattering. This part is robust and needs no further work.

## 2. The hard, under-developed part: the 2N scattering angles are ill-posed

The wire pitch (~3 mm) is **6× coarser** than the 0.5 mm step, so the wire
signals constrain only the **integrated trajectory**, not the fine angles.

**Stock setup fails** (λ=1e-3, no LR decay, run to fixed step count):
- scattering-only: loss falls to 0.021 by step 100 then **climbs back to ~0.10**;
  per-segment angle RMS grows monotonically 0.10→0.49 mrad (injecting spurious
  high-frequency scattering). Cumulative deflection overshoots to 136 mrad vs
  **32 truth (4×)**.
- full (staged): cumulative deflection **421 mrad vs 32 (13×)**, cumulative
  correlation collapses 0.88→0.37, and **global energy degrades 1.1 %→4 %**.

**Why raising λ alone doesn't fix it.** The over-bend uses per-segment angles
that are individually *smaller* than truth (fit RMS 1.68 < 1.98 mrad) but
**coherently aligned** — a low-frequency drift mode. The Highland prior
penalizes `Σ dθ²`, which is blind to coherence, so a stronger white prior just
shrinks *all* angles (including the real signal) rather than removing the
coherent bend. There is a genuine **degeneracy between the global direction and
a low-frequency scattering drift**, which is why freeing the globals (full mode)
makes the over-bend much worse.

## 3. What actually works: early-stop + LR decay + a moderate prior

Systematic sweep (`study.py --study scattering`, 250 steps, **early-stop at best
wire-loss**), metrics: `traj_mm` = RMS fit-vs-truth track position; `cum_ratio`
= |cumulative deflection| fit/truth (1.0 ideal); `cum_corr` = trajectory-shape
correlation; `win_corr` = 3 mm-windowed (fine) angle correlation.

Representative rows (G = segments per coarse angle):

| G | λ | decay | wire_loss | traj_mm | cum_ratio | cum_corr | win_corr |
|---|---|---|---|---|---|---|---|
| 1 | 1e-3 | 1.00 | 0.0145 | 0.28 | 1.62 | 0.961 | 0.131 |
| 1 | 3e-2 | 0.99 | 0.0147 | **0.25** | **0.77** | **0.968** | **0.249** |
| 1 | 3e-1 | 0.99 | 0.0165 | 0.27 | 0.54 | 0.961 | 0.251 |
| 10 | 1e-3 | 1.00 | 0.0407 | 0.71 | 4.58 | 0.853 | 0.019 |
| 20 | 3e-2 | 0.99 | 0.0148 | 0.25 | 0.78 | 0.968 | 0.249 |

Conclusions:
1. **The muon trajectory is recovered to ~0.25 mm** and its cumulative
   (low-frequency) deflection to **corr ≈ 0.97** in every well-behaved config.
2. **Fine per-3 mm scattering angles cap at corr ≈ 0.25** — the wire-pitch
   resolution floor. But this is only the *finest* scale: aggregated over longer
   windows the angle recovery climbs smoothly to corr ~0.95 by ~10 cm (see §6).
3. **Early-stopping at best wire-loss** is the single biggest stabilizer (it
   alone cut the stock cum_ratio from ~4 to 1.6). **decay=0.99** removes the
   residual blowups.
4. **λ ≈ 3e-2 is the magnitude sweet spot** (cum_ratio 0.77). Lower overshoots,
   higher over-shrinks the real scattering.
5. **Coarse-graining barely helps** — G=1 and G=20 are equivalent once λ+decay
   are set, confirming the ill-posedness is the resolution floor, not DOF count.
   So keep the natural per-segment parameterization; the prior is the right tool.

## 4. Recommended defaults

> **Superseded by §9** — the current recipe is s=1.0, λ=1e-3, converged joint
> fit (no early-stop). The early-stop / λ=3e-2 below was the best at s=1.5 and is
> kept for the historical record.

For the full closure: **early-stop at best wire-loss, lr=0.01–0.015 with
decay≈0.99, λ_prior≈3e-2, no coarse-graining**, optionally staged (globals-only
warm-up then open scattering) to protect the energy. See §5 for the full-mode
energy-protection numbers.

## 5. Full mode (free globals): staging is mandatory, and the tuned recipe ~halves the energy error

**Single-stage full fails completely.** `study.py --study full` (globals +
scattering both free from step 0, globals init 170 mm off) is catastrophic for
*every* (G, λ, decay): wire_loss 5.9–19, **track 165–178 mm off-truth**,
cumulative deflection 70–80× Highland. The 2N scattering DOF absorb the huge
initial global mismatch by bending the track wildly, and the globals never lock
on. Worse, **decay=0.99 hurts here** (E_err −180 MeV) because decaying the LR
freezes the globals before they converge from the far init. → **A globals-only
warm-up (staging) is not optional.**

**Staged, stock vs tuned** (`run.py --mode staged --steps 800 --stage1-steps 250`):

| Metric | stock (λ=1e-3, no decay, final params) | **tuned (λ=3e-2, --decay 0.99 --early-stop)** |
|---|---|---|
| x / y / z error | 0.08 / 0.41 / 0.62 mm | 0.20 / **0.05** / **0.21** mm |
| θ / φ error | 0.32° / 0.16° | **0.17° / 0.14°** |
| **E error** | **20.3 MeV (4.1 %)** | **12.1 MeV (2.4 %)** |
| cumulative deflection | **421 mrad (13×)** | **58.9 mrad (1.85×)** |
| cumulative corr (pl1/pl2) | 0.37 / 0.51 | **0.89 / 0.96** |
| windowed (3 mm) corr | ≈0 | ≈0 (resolution floor) |

The tuned recipe **halves the energy error** (4.1 %→2.4 %), **tames the
nonphysical over-bend 13×→1.85×**, and **restores the trajectory-shape
correlation 0.4→0.9**, while keeping the vertex sub-mm and direction <0.2°.

## 6. Aggregate scattering angles ARE recoverable — above the wire resolution

The per-3 mm angle sits at the resolution floor (corr ~0.25), but that is the
wrong *scale* to judge MCS, not a statement about the physics. The real
observable is the **net deflection aggregated over a longer window** — a net
angle over length L moves the track by ~L·θ, so the wire position signal that
constrains it grows with L. `window_scan.py` aggregates the per-segment fit/truth
angles (best scattering-only fit, λ=3e-2, decay=0.99, early-stop) at increasing
window lengths:

| window L | n_windows | aggregate-angle corr | RMS recovery (fit/truth) |
|---|---|---|---|
| 0.5 mm (per-step) | 2000 | 0.10 | 0.09 |
| 3 mm (wire pitch) | 333 | 0.25 | 0.23 |
| 12.5 mm | 80 | **0.49** | 0.43 |
| 25 mm | 40 | **0.69** | 0.58 |
| 50 mm | 20 | **0.84** | 0.64 |
| 100 mm | 10 | **0.95** | 0.77 |
| 200 mm | 5 | **0.99** | 0.89 |

(`mcs_aggregate_angle_recovery.png`)

- **Crossover to usable (corr > 0.5) is ~13 mm — about 4× the wire pitch** — but
  this row uses the **early-stop** fit (≈250 steps), which is under-converged for
  the fine angles. Converging the fit moves the crossover to ~6.7 mm; see §7.
- By the **standard MCS segment length (~10–14 cm)** the aggregate scattering
  angle is recovered at **corr ~0.95–0.99**, exactly the regime real MCS
  momentum measurements use.
- The RMS magnitude is **biased low** (the within-window high-frequency wiggle
  is filtered out by the wire signal), recovering 0.58 at 25 mm rising to 0.89 at
  200 mm. The *shape* is recovered far better than the *magnitude* — relevant for
  MCS momentum, which would need a per-window bias calibration.

## 7. CORRECTION: the fine-scale floor is **optimization-limited**, not an information limit

§3 and §6 measured the scattering fit at ~250–350 steps with **early-stop** —
which is correct for the trajectory/globals/magnitude but, it turns out,
**severely under-converged for the fine angles**. The scattering angles obey the
**spectral bias** of gradient descent: the coarse trajectory locks in the first
few hundred steps, but the fine, high-spatial-frequency angles fill in *slowly*
over thousands of steps — and the Sobolev loss makes this worse, because its
`1/(k²)^{s}` weight suppresses exactly the high-frequency gradients that drive
the fine angles (∝ k⁻³ at s=1.5).

Driving the scattering-only fit to convergence (`long_fit.py`, 9000 steps,
lr 0.02 × decay 0.99975, λ=3e-2) shows the aggregate-angle correlation **keeps
climbing as the loss falls**, far past the early-stop point:

| step | wire+prior loss | c@3 mm | c@6 mm | c@12.5 mm | c@25 mm | cum_ratio |
|---|---|---|---|---|---|---|
| 350 (early-stop) | 0.241 | 0.07 | 0.09 | 0.15 | 0.17 | 3.6 |
| 1000 | 0.179 | 0.15 | 0.19 | 0.28 | 0.36 | 2.7 |
| 2500 | 0.069 | 0.28 | 0.38 | 0.54 | 0.70 | 1.14 |
| 4000 | 0.040 | 0.32 | 0.44 | 0.60 | 0.78 | 0.66 |
| 8000 | 0.012 | 0.35 | 0.47 | 0.64 | 0.84 | 0.61 |
| 8999 | 0.0116 | **0.35** | **0.47** | **0.65** | **0.84** | 0.64 |

Converging properly moves the **corr = 0.5 crossover from ~13 mm down to ~6.7 mm**
(~2.2× the wire pitch) — roughly **a factor of two finer** — and raises c@25 mm
from 0.69 → 0.84. The plateau by step ~8000 (0.643 → 0.647 over the last 1000
steps) is the genuine wire information limit for this optimizer.

Two consequences:
1. **There is a real regime trade-off.** Early-stop (≈350 steps) gives the best
   trajectory/globals and a near-unbiased cumulative magnitude (cum_ratio ~0.77),
   but the worst fine angles. Long convergence (≈8000 steps) gives the finest
   angle resolution (crossover ~6.7 mm) but the cumulative magnitude under-shoots
   (cum_ratio ~0.61 at λ=3e-2). The two goals want different stopping points and
   different prior strengths.
2. **Sharpening the loss spectrum does NOT help — ~7 mm is the real floor.**
   The natural idea — lock the trajectory at s=1.5, then sharpen with a smaller s
   to feed the fine angles stronger high-frequency gradients (`coarse2fine.py`) —
   was **tested and refuted**. Starting from the *converged* s=1.5 solution
   (crossover 7.5 mm, c@25 mm = 0.81) and continuing 4000 steps at s=0.75 makes
   every scale **worse** (crossover 21 mm, c@25 mm = 0.53) and re-injects the
   over-bend (cum_ratio 0.66 → 1.22). The extra high-frequency weight gets fit by
   *spurious* angles, exactly as low-s does from zero. So the Sobolev s=1.5
   smoothing is the **correct** regularizer, and the high-frequency wire content
   is genuinely degenerate: weighting it more fits noise, not truth.

   **Conclusion: ~7 mm (~2× wire pitch) is the practical floor for this wire
   readout**, reached by *converging* under the smoothing loss (≈8000 steps), not
   by sharpening it. Going below ~7 mm would need more information (finer pitch /
   pixel readout / a non-white prior that distinguishes true from degenerate
   high-frequency structure), not a better wire-loss schedule.

## 8. Bottom line — how well can we do?

| Quantity | Result | Limited by |
|---|---|---|
| Vertex | **sub-mm** (≤0.6 mm staged, 0.15 mm globals-only) | — |
| Direction (θ, φ) | **<0.2°** | — |
| Energy | **1.1 %** (globals-only) → **2.4 %** (full) | scattering↔energy coupling |
| Muon trajectory | **~0.25 mm** RMS, cum-shape corr **~0.9–0.97** | — |
| Scattering angle @ 3 mm (wire pitch) | corr ~0.35 (converged) | optimizer / high-freq suppression |
| Scattering angle @ ~7 mm | corr **~0.5** (converged) | optimizer / high-freq suppression |
| Scattering angle @ ~2.5 cm | corr **~0.84** (converged) | smoothing / magnitude bias |
| Scattering angle @ ~10–14 cm (MCS scale) | corr **~0.95–0.99** | magnitude bias only |

Headline: **wire signals pin the muon's vertex, direction, range-energy and
trajectory very well; the MCS scattering angle is recoverable down to ~7 mm
(~2× the wire pitch) once the fit is converged — the ~13 mm "floor" was a
premature-stopping artifact, not physics.** ~7 mm is the genuine floor for this
wire readout: it is reached by converging under the Sobolev s=1.5 smoothing loss
(~8000 steps), and sharpening the loss to chase finer structure overfits and
does worse (§7). Going below ~7 mm needs more information — finer pitch / pixel
readout, or a non-white prior — not a better wire-loss schedule. The
aggregate-angle *magnitude* stays biased low at the strong prior, so an MCS
*momentum* estimate would still need a per-window bias calibration.

## 9. UPDATE — current recipe: s=1.0, λ=1e-3, converged joint fit (better + faster)

§1–8 used s=1.5 with early-stop (or very long runs). A systematic pass on the
loss landscape gives a strictly better, faster recipe.

**Sobolev order s (`sobolev_s_scan.py`).** The weight `1/(freq²+ε)^s` has ε
verified correct: `ε = 1/(π²·max_pad²)` ⇒ screening length `max_pad/2` samples
(= 512 for max_pad=1024). At s=1.5 the f→0 vs Nyquist weight ratio is ~4×10⁹ — a
huge dynamic range that makes the loss low-frequency-dominated and
ill-conditioned. Lowering to **s=1.0** shrinks it to ~3×10⁶: the joint fit
converges ~3× faster (reaches in ~800 steps what s=1.5 needs ~2300) with the
same recovery. s=0.5 is too far — the global energy loses its low-frequency
constraint and breaks (−21%). → default switched to **`SOBOLEV_S = 1.0`**.

**Prior weight λ (`decompose_sweep.py`).** The loss is two terms,
`wire + λ·prior`. Decomposed across the stage transition, the loss jump when
stage 2 turns on is **entirely the wire term** (scattering reset truth→0 → the
track goes straight → worse fit); the prior term is negligible (≤3×10⁻⁴) at
every λ. Sweeping λ at s=1.0: **λ=1e-3 is optimal** — best angle correlation at
every scale (0.50 / 0.87 / 0.97 at 12.5 / 50 / 200 mm) and net-deflection ratio
1.03. λ=0 overfits (1.83×, corr collapses); λ≥1e-2 over-regularizes. This is
run.py's default `LAMBDA_PRIOR=1e-3`; the earlier "λ=3e-2 sweet spot" was for
s=1.5 + early-stop.

**Convergence + timing (`convergence.py`, `multi_event.py`).** 0.15 s/step
(RTX 2080 Ti, N=2000, 6-plane differentiable sim). The wire loss plateaus in
~65 stage-2 steps (globals + coarse trajectory); the fine-angle correlation
keeps refining over ~800 steps at near-flat loss. JIT compile ~110 s, once,
amortized across events.

**Multi-event robustness** (5 scattering realizations, s=1.0, λ=1e-3):

| quantity | mean ± std |
|---|---|
| vertex error | 0.42 ± 0.17 mm |
| energy error | −6 ± 8 MeV (~1.6 %) |
| direction error | <0.5° |
| corr @ 5 cm | 0.83 ± 0.06 |
| 3D track residual | 0.93 ± 0.06 mm |
| net-deflection ratio | 0.97 ± 0.47 |
| loss-plateau | 65 ± 10 steps |
| time / event (warm) | 130 ± 1 s |

Globals + trajectory are robust event-to-event; the net-deflection *magnitude*
is the one high-variance quantity (low-scattering events under-recover under the
prior), so an MCS momentum estimate needs per-event/window calibration.

**Current recipe:** staged — globals-only warm-up (~250 steps), then a converged
**joint** fit at **s=1.0, λ=1e-3**, global lr 0.003 / angle lr 0.01, decay
0.9995, **no early-stop**. ~315 steps (~50 s) for globals + trajectory; ~1050
steps (~130 s) for full scattering refinement. (Do NOT raise the angle lr — the
LR scan showed 0.02 oscillates and 0.04 diverges; the fix for speed is the
better conditioning above, or preconditioning the double-integral forward.)

### Reproduce
```
python3 -m closure.mcs.validate_forward      # 14/14 forward-model physics checks
python3 -m closure.mcs.sobolev_s_scan        # eps check + s-scan -> s=1.0
python3 -m closure.mcs.decompose_sweep       # wire/prior decomposition + lambda sweep -> 1e-3
python3 -m closure.mcs.convergence           # convergence-by-scale + timing
python3 -m closure.mcs.multi_event           # convergence + quality across events
python3 -m closure.mcs.render_closure        # 2D wire event display (truth/reco/diff)
python3 -m closure.mcs.render_3d             # 3D track + transverse scattering
python3 -m closure.mcs.render_history        # loss + parameters vs iteration
# earlier exploratory scripts: study.py, window_scan.py, long_fit.py, coarse2fine.py,
#   resolve_scan.py, depth_scan.py, oracle_test.py, diagnose_fit.py, diagnose_rise.py
```
