# Full Closure Sweep Results

Event: `mpvmpr_20.h5`, event 0 (39,890 truth segments, 926.44 MeV total dE)
Optimizer: 10,000 segments, Sobolev geomean log1p loss (s=1.5)
Recombination: `modified_box` (consistent between truth sim, optimizer, relocation)
Optimizer dx: 0.5mm (build_forward default in sweep.py)
Truth dx: variable (~0.1mm from HDF5, NOT forced)

## 1. lr_e_mult Sweep — Baseline (no relocation, no L1, no noise)

Fixed: lr=0.5, decay_rate=0.9995, 1000 steps, baseline mode

| lr_e_mult | Final Loss | dE_ratio | Dead | Notes |
|-----------|-----------|----------|------|-------|
| 0.005 | 0.157 | 1.123 | 4551 | Best baseline |
| 0.018 | 0.308 | 1.206 | 5418 | |
| 0.05 | 0.475 | 1.350 | 6354 | Energy overshoots badly |

**Finding:** Smaller lr_e_mult consistently better in baseline. Energy step at 0.005 = 0.0025 MeV/step (2.7% of mean dE). All show energy drift upward (dE_ratio > 1) without L1.

## 2. lr_e_mult Sweep — Full Mode (relocation, no L1, no noise)

Fixed: lr=0.5, decay_rate=0.9995, reloc_every=25, max_reloc=100, 1500 steps

| lr_e_mult | Final Loss | dE_ratio | Dead | Steps to die (~) |
|-----------|-----------|----------|------|-----------------|
| 0.001 | 0.073 | 1.008 | 225 | ~160 |
| 0.002 | 0.048 | 1.021 | 941 | ~80 |
| 0.003 | **0.038** | 1.035 | 1382 | ~53 |
| 0.005 | 0.039 | 1.055 | 1944 | ~33 |
| 0.01 | 0.042 | 1.084 | 2448 | ~16 |
| 0.018 | 0.067 | 1.104 | 2930 | ~9 |

**Finding:** Minimum at lr_e_mult=0.003-0.005. Below 0.003, relocation starves (segments die too slowly). Above 0.005, energy convergence degrades. The optimal balances energy convergence speed vs relocation throughput.

**"Steps to die"** = mean_dE / (lr * lr_e_mult) = 0.093 / (0.5 * lr_e_mult). This is how many steps a useless segment takes to reach death_thresh=0.012 from mean dE, assuming the gradient consistently says "decrease energy."

## 3. Ad-hoc Runs (earlier in session, less controlled)

These used various settings before the systematic sweep. Truth dx was variable (~0.1mm), optimizer dx=0.5mm, some used EMB recombination (mismatch).

### Baseline runs (no reloc, no L1, no noise)
| lr | decay | lr_e_mult | Steps | Loss | dE_ratio | Notes |
|----|-------|-----------|-------|------|----------|-------|
| 0.3 | 0.9999 | 1.0 | 1000 | 1.92 | 3.492 | Energy blowup — lr_e_mult=1.0 catastrophic |
| 0.3 | 0.9999 | 1.0 | 2000 | 0.86 | 3.492 | Same, more steps |
| 0.3 | 0.9999 | 0.03 | 1000 | 0.34 | 1.215 | First fix — energy controlled |

### Full mode runs (reloc + L1 + noise, various settings)
| lr | decay | e_mult | noise | L1 | reloc | Steps | Loss | dE_ratio | Notes |
|----|-------|--------|-------|-----|-------|-------|------|----------|-------|
| 0.3 | 0.9999 | 0.03 | 0.05 | 0.001 | 50/10k | 1000 | 85.1 | 5.349 | Reloc spikes + energy chaos |
| 0.3 | 0.9999 | 0.03 | 0.1 | 0.0002 | 300/10k | 2000 | 0.027 | 0.953 | Large spikes every 300 steps |
| 0.5 | 0.9995 | 0.018 | 0.1 | 0.0002 | 300/10k | 2000 | 0.017 | 0.949 | Higher lr helps |
| 0.5 | 0.9995 | 0.018 | 0.1 | 0.0002 | 25/100 | 2000 | 0.024 | 0.993 | Smooth curve, slow recycling |
| 0.5 | 0.9995 | 0.018 | 0.0 | 0.0002 | 25/100 | 2000 | 0.023 | 1.005 | No noise = slightly better |
| 0.5 | 0.9995 | 0.018 | 1.0 | 0.0002 | 25/100 | 2000 | 0.040 | 0.991 | High noise hurts |
| 0.5 | 0.9995 | 0.018 | 0.3 | 0.0002 | 25/100 | 2000 | 0.024 | 0.993 | Mid noise ≈ no noise |
| 0.5 | 0.9995 | 0.018 | 0.3 | 0.0001 | 25/100 | 3000 | 0.015 | 0.958 | More steps helps, L1 over-drains |
| 0.5 | 0.9995 | 0.018 | 0.3 | 0.0001 | 25/100 | 3000 | 0.010 | 0.981 | + recomb=modified_box (consistency fix) |

### Energy-weighted donor experiments (reverted — random is better)
| Weighting | lr_e_mult | L1 | Steps | Loss | dE_ratio | Notes |
|-----------|-----------|-----|-------|------|----------|-------|
| linear | 0.018 | 0.0001 | 2000 | 0.016 | 0.895 | Over-drains, fights optimizer |
| sqrt | 0.018 | 0.00005 | 2000 | 0.044 | 0.897 | Still over-drains |

## 4. Key Findings

### Recombination model consistency
- Default recomb_model=None gives EMB for DetectorSimulator but modified_box for relocation split
- EMB is angular-dependent but SegmentData has no angles — systematic mismatch
- Fix: always pass `--recomb modified_box`
- Impact: 33% loss improvement (0.015 → 0.010) when fixed

### dx mismatch
- Truth HDF5 has variable dx (mean=0.098mm, median=0.1mm, 131 segments with dx=0)
- Optimizer uses fixed dx (0.5mm default in build_forward, changed to 0.1mm in full_closure.py)
- With matched dx, forward passes are identical (loss=0.000000)
- dE scaling does NOT preserve Q through nonlinear recombination: e_scale=3.99 gives signal ratio 0.69 (dx=0.1mm) or 1.48 (dx=0.5mm)
- sweep.py uses dx=0.5mm (build_forward default), full_closure.py uses dx=0.1mm

### Noise
- Tested noise_lr = 0, 0.1, 0.3, 1.0
- No benefit at any scale for closure test
- Sobolev loss provides smooth gradients — no local minima to escape

### Relocation spikes
- Spike magnitude scales linearly with max_reloc count
- max_reloc=100 gives ~1x spike (invisible), 500 gives 2.5x, 10000 gives 54x
- NOT correlated with which donors are selected
- Cause: energy split disrupts many segments simultaneously
- Adam moment zeroing contributes to recovery time but not spike magnitude

### L1 drain
- Purpose: push near-dead segments below death_thresh for relocation
- Global drain is harmful — over-drains at long runs
- With relocation + proper lr_e_mult, L1=0 works (sweep results above)
- If needed, L1 ≈ death_thresh / (reloc_every * 2) to target near-dead only

### Physical scales
- Wire pitch: 3mm
- Position step at lr=0.5: 0.5mm/step (17% of wire pitch)
- Mean dE: 0.093 MeV (with e_scale=3.99)
- Energy step at lr_e_mult=0.003: 0.0015 MeV/step (1.6% of mean dE)
- Death threshold: 0.012 MeV
- Init position jitter: 100mm
- Response kernel width: ~5-10 wires (~15-30mm)

## 5. Current Best Settings

From systematic sweep (Section 2):
```
lr=0.5, decay_rate=0.9995, lr_e_mult=0.003
noise_lr=0.0, l1=0.0
reloc_every=25, max_reloc=100
warmup=300, recomb=modified_box
```
Loss=0.038 at 1500 steps, dE_ratio=1.035, 1382 dead.

## 6. LR x Decay Sweep (systematic, sweep script)

Fixed: lr_e_mult=0.003, n_seg=10000, reloc 25/100, no noise/L1, warmup=300, 1500 steps, full mode

| Config | Loss | dE_ratio | Dead |
|--------|------|----------|------|
| lr=0.3, d=0.9995 | 0.058 | 1.039 | 1659 |
| lr=0.5, d=0.9995 | 0.038 | 1.035 | 1393 |
| lr=0.7, d=0.999 | 0.040 | 1.034 | 1207 |
| **lr=1.0, d=0.999** | **0.030** | 1.031 | 1138 |
| lr=0.7, d=0.9995 | 0.031 | 1.035 | 1277 |
| lr=1.0, d=0.9995 | 0.033 | 1.031 | 1422 |

**Finding:** Higher initial lr with faster decay wins. lr=1.0/d=0.999 best (0.030). The fast initial convergence more than compensates for aggressive cooling.

## 7. Warmup Sweep (systematic)

Fixed: lr=0.5, d=0.9995, lr_e_mult=0.003, n_seg=10000, 1500 steps, full mode

| Warmup | Loss | dE_ratio | Dead | Relocs |
|--------|------|----------|------|--------|
| **100** | **0.034** | 1.026 | 1020 | 5600 |
| 200 | 0.035 | 1.024 | 1178 | 5200 |
| 300 | 0.038 | 1.034 | 1351 | 4800 |
| 500 | 0.051 | 1.037 | 1796 | 4000 |

**Finding:** Shorter warmup is better. warmup=100 wins. Earlier relocation = more total redistribution time.

## 8. Step Count Sweep (best config)

Config: lr=1.0, d=0.999, lr_e_mult=0.003, warmup=100, reloc 25/100, no noise/L1, full mode

| Steps | Loss | dE_ratio | Dead | Time |
|-------|------|----------|------|------|
| 500 | 0.177 | 1.068 | 3038 | 121s |
| 1000 | 0.045 | 1.051 | 2085 | 234s |
| 1500 | 0.026 | 1.030 | 744 | 355s |
| 2000 | 0.015 | 1.020 | 25 | 479s |
| 3000 | **0.012** | 1.015 | 6 | 729s |

**Finding:** Loss scales roughly as 1/steps. No plateau at 3000 steps — still improving. All dead segments recycled by step ~1700.

## 9. Current Best Configuration

```
lr=1.0, decay_rate=0.999, lr_e_mult=0.003
warmup=100, noise_lr=0.0, l1=0.0
reloc_every=25, max_reloc=100
recomb=modified_box
```
At 3000 steps: loss=0.012, dE_ratio=1.015, dead=6.

## 10. Noise Sweep (best config: lr=1.0, d=0.999)

Fixed: lr_e_mult=0.003, warmup=100, reloc 25/100, no L1, 1500 steps, full mode

| noise_lr | Loss | dE_ratio | Dead |
|----------|------|----------|------|
| 0.0 | 0.027 | 1.032 | 744 |
| 0.1 | 0.024 | 1.024 | 657 |
| **0.3** | **0.022** | 1.013 | 386 |

**Finding:** Noise HELPS at lr=1.0 (unlike at lr=0.5 where it didn't). The fast lr decay (0.999) annihilates noise by step 1500 (noise=0.07mm), so it only affects early exploration. Noise helps segments explore neighboring wires during the high-lr phase.

## 11. Updated Best Configuration

```
lr=1.0, decay=0.999, lr_e_mult=0.003
warmup=100, noise_lr=0.3, l1=0.0
reloc_every=25, max_reloc=100
recomb=modified_box
```
At 1500 steps: loss=0.022, dE_ratio=1.013.
At 3000 steps (noise=0, previous best): loss=0.012, dE_ratio=1.015.
Expected at 3000 steps with noise=0.3: likely <0.012.

## 12. Not Yet Tested
- n_seg other than 10000
- Sobolev s parameter (always 1.5)
- Forced truth dx (to eliminate dx mismatch)
- Different events (always event 0)
- reloc_every / max_reloc at best lr/decay
