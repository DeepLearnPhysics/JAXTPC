# Calibration

Calibration workflows — using the simulator (and its gradients) to study or fit
detector calibration constants.

**Planned**
- `recombination_calibration.ipynb` — recover recombination parameters
  (modified-box α/β, EMB) from dE/dx vs charge, and compare models
- `gain_lifetime.ipynb` — electron lifetime and electrons-per-ADC gain:
  attenuation vs drift time, and fitting them back out
- `response_calibration.ipynb` — calibrate / validate the wire and pixel
  response kernels against a known input
- `field_response_validation.ipynb` — cross-check the field response and
  electronics shaping

This folder collects calibration-oriented notebooks; many naturally build on the
differentiable path (fit a calibration constant by gradient descent).
