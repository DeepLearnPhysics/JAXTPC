# Calibration

> **Status: no established method yet.** Unlike reconstruction (the
> [closure methods](../reco)), there is currently **no calibration code in the
> repository**. The notebooks previously sketched here were placeholders and
> have been removed pending a real method.

What calibration *would* mean here, concretely, and why it's a natural fit:

- The closure methods fix the **detector physics** (`SimParams`: velocity,
  lifetime, diffusion, recombination) and optimize the **event** (point charges /
  track params).
- **Calibration is the inverse**: fix a known event and optimize the **physics
  parameters** to match an observed readout — using the *same* differentiable
  forward (`forward_segments`) and a loss (the Sobolev loss, or a simpler
  charge-level loss). The capability already exists: gradients flow cleanly to
  `velocity_cm_us`, `lifetime_us`, and the recombination params (this is what
  `tests/test_pipeline_forward.py::TestFiniteDifferenceGradients` checks).

So a calibration notebook is feasible (fit e.g. electron lifetime or the
recombination α/β by gradient descent against a target event), but it would be a
**new method**, not a write-up of existing work. Decide the target quantity and
loss before building, so this folder reflects a real approach rather than a
guess.
