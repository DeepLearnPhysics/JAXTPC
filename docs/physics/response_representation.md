# Response representation: what the kernel means and why it is handled this way

The simulator must compute, for a deposit of charge `q` at transverse position `u0`
arriving at `t0` with diffusion `(sigma_T, sigma_L)`:

    S_n(tau_k) = q * INT INT G_sigmaT(y - u0) G_sigmaL(t - t0) R(y - n, tau_k - t) dy dt

`R` is the single-wire response (field response convolved with the electronics), stored in
`tools/responses/*_plane_kernel.npz` on a grid of 0.1 pitch x 0.5 us.

## The two axes are not the same kind of thing

| | pitch (Delta) | time (tau) |
|---|---|---|
| what it is | an **integral** over the charge distribution | a **sample** of a continuous waveform |
| why | wires are physical objects -- evaluation points, nothing is sampled transversally | the ADC point-samples; the 2.2 us shaping is the anti-alias filter |
| R's behaviour | **discontinuous** at \|Delta\| = 1/2 on the collection plane | smooth and band-limited (3e-4 of power above 40% of Nyquist) |
| what can go wrong | representing the discontinuity | the sub-grid **shift operator** |

That asymmetry is why each axis has exactly one failure mode, and why the fixes differ.

## Pitch: the tiling invariant  (`_enforce_wire_tiling`, kernels.py)

Wires tile the pitch, so every drift electron is collected by exactly one of them:

    S(x) = sum_n R(x - n)   must be CONSTANT in x

A collection response is `R = C * 1[|Delta| < 1/2] + induction`, and that boxcar's edge
falls **exactly on a grid sample**. A step landing on a sample must carry **half weight**,
so the two half-contributions from the wires either side sum to one. Then linear
interpolation in Delta tiles exactly:

    x in (0.4, 0.5):  home wire  C(1-a) + (C/2)a = C(1 - a/2)
                      neighbour  0(1-a) + (C/2)a = C a/2
                      sum        = C, for every a

The shipped default Y kernel stores the **non-collecting** branch there instead (column
integral 7e-6 against 0.060 one sample inside), so the boxcar is 0.9 pitch wide and the
tiling has a hole: residue-class spread 112%, mean tiling sum 89.4% of the on-wire column.

**Collection planes only, and that is physics, not convenience.** From the garfield
branches (the same physical position seen from the two adjacent regions):

| plane | branch L integral | branch R integral | \|L-R\| as % of on-wire peak |
|---|---|---|---|
| U | +0.00056 | +0.00053 | 1.0% |
| V | +0.00069 | +0.00027 | 3.2% |
| Y | -0.07869 (collected) | +0.00013 (not) | **104.7%** |

Only the collection plane bifurcates. Applying a half weight on U/V would corrupt a
legitimate sample of a smooth function, and measurement confirms it changes nothing there.
The detector (on-wire integral vs the largest integral a pitch away) is 101x on Y against
0.9x / 0.4x on U / V -- not marginal. The correction is **idempotent**, so kernels that
already tile (SBND stores the mean, which IS the half weight) are returned unchanged.

## Time: the sub-tick shift  (`interpolate_diffusion_kernel`, kernels.py)

The ADC point-samples a band-limited waveform, so the correct sub-tick operation is a
band-limited **fractional delay**. A 2-tap linear blend is not one: it is a shift
CONVOLVED WITH A TRIANGLE, injecting `a(1-a)` ticks^2 of width that depends only on where
the deposit lands inside its tick. Cubic Lagrange matches the first three moments, so it
places the pulse without inflating it, and is charge-neutral (weights sum to 1).

There is no half-weight analogue here: the response has no discontinuity in time, so there
is no representation error -- only the operator.

## Deliberately NOT done, with reasons

- **Do not "fix the blur".** Replacing the point-sampled gaussian with a band-limited or
  bin-integrated one measures neutral-to-worse on both axes. On pitch it cannot even affect
  charge: for any normalised g, `sum_n (g conv R)(x-n) = S * sum(g) = S`.
- **Do not raise the Delta interpolation order.** It rings on the discontinuity (Y is worse
  at short drift) and the gain elsewhere is below the reference's own model uncertainty.
- **Do not bin-integrate in time.** That is WireCell's quadrature, not the physics; it
  places the pulse half a tick early. JAXTPC's timing is already exact (centroid error
  +0.001 tick vs physics truth, where WireCell is -0.500).
- **Shift-aware Delta interpolation**: tried, much worse (aligning centroids of two nearly
  identical blurred columns is noise-driven).

## Known remaining limits

- **Exact-phase evaluation** (`sum_i R_i T_sigma(Delta - x_i)` at the deposit's actual
  Delta, instead of interpolating a phase-0 table) is the only variant that reaches the
  floor, matching WireCell to 4 significant figures. Its gain over the tiling fix is
  resolvable only at short drift: 4.7x the reference uncertainty at 2 cm, 2.3x at 8 cm,
  1.2x at 32 cm, 0.9x at 128 cm. Not implemented.
- **The floor itself** (2.9e-2 at 2 cm, ~5e-3 at long drift, shared exactly with WireCell)
  is set by garfield's 0.1-pitch sampling of a response whose pulse TRANSLATES in time with
  impact position (+2.61 bins over half a pitch against a 3.50-bin width). That is a data
  limitation; no algorithm change reaches below it.
- **`pre_window_us = 0`** truncates the response near the anode and currently masks the
  short-drift transverse behaviour. Coupled to the above -- assess together.
