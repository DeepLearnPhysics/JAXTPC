#!/usr/bin/env python3
"""
Position Gradient Analysis V2: Understanding Which Positions Are Differentiable

Key insight from V1:
- x position (drift direction): Differentiable through drift_time → attenuation, diffusion
- y position: Not used in simplified sim (would affect angles in full sim)
- z position (wire direction): Only affects discrete wire INDEX, not differentiable

This test explores:
1. Why x is differentiable (through values)
2. Why z is NOT differentiable (discrete indices)
3. How to make z-like positions differentiable (sub-bin offsets)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from physics_params import create_default_params, PhysicsParams
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import calculate_box_model_charge_with_physics_params


def test_simple_gradient(name, loss_fn, x0, eps=1e-3):
    """Test gradient for a simple scalar or array input."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)

    try:
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(x0)
        analytical_works = True
    except Exception as e:
        print(f"Analytical gradient FAILED: {e}")
        return False

    if isinstance(x0, jnp.ndarray) and x0.ndim > 0:
        print(f"Gradient shape: {grads.shape}")
        print(f"Gradient stats: min={float(jnp.min(grads)):.4e}, max={float(jnp.max(grads)):.4e}")
        print(f"               mean={float(jnp.mean(grads)):.4e}, std={float(jnp.std(grads)):.4e}")

        # Check if all zeros
        if jnp.allclose(grads, 0):
            print("WARNING: All gradients are ZERO!")
            return False

        # Numerical check for first element
        x_plus = x0.at[0].set(x0[0] + eps)
        x_minus = x0.at[0].set(x0[0] - eps)
        num_grad = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * eps)
        print(f"First element: analytical={float(grads[0]):.4e}, numerical={float(num_grad):.4e}")
    else:
        print(f"Analytical: {float(grads):.4e}")
        x_plus = x0 + eps
        x_minus = x0 - eps
        num_grad = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * eps)
        print(f"Numerical:  {float(num_grad):.4e}")

    return True


def main():
    print("="*70)
    print("Position Gradient Analysis V2")
    print("Understanding What Is and Isn't Differentiable")
    print("="*70)

    # Load data
    config_path = "../config/cubic_wireplane_config.yaml"
    data_path = "../mpvmpr_20.h5"

    detector_config = generate_detector(config_path)
    step_data = load_particle_step_data(data_path, 0)

    n_test = 50
    de = jnp.asarray(step_data['de'][:n_test], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'][:n_test], dtype=jnp.float32)
    positions_mm = jnp.asarray(step_data['position'][:n_test], dtype=jnp.float32)
    positions_cm = positions_mm / 10.0

    drift_velocity = detector_config['drift_velocity_cm_us']
    wire_spacing = float(detector_config['wire_spacings_cm'][0, 0])

    physics_params = create_default_params()

    # =========================================================================
    # TEST 1: X position through drift time (should work)
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 1: X Position (Drift Direction)")
    print("="*70)
    print("\nX position affects drift_time = |x| / velocity")
    print("Drift time affects: attenuation, diffusion sigma")
    print("These are continuous functions → DIFFERENTIABLE")

    x_positions = positions_cm[:, 0]
    base_drift_times = jnp.abs(x_positions) / drift_velocity

    def x_to_attenuation_loss(x_offsets):
        """Loss through drift time → attenuation."""
        x_new = x_positions + x_offsets
        drift_times = jnp.abs(x_new) / drift_velocity
        attenuation = jnp.exp(-drift_times / 10000.0)  # 10 ms lifetime
        return jnp.mean(attenuation)

    test_simple_gradient("X → Drift Time → Attenuation", x_to_attenuation_loss, jnp.zeros(n_test))

    def x_to_diffusion_loss(x_offsets):
        """Loss through drift time → diffusion sigma."""
        x_new = x_positions + x_offsets
        drift_times = jnp.abs(x_new) / drift_velocity
        sigma_sq = 2.0 * physics_params.diffusion_trans * drift_times
        return jnp.mean(sigma_sq)

    test_simple_gradient("X → Drift Time → Diffusion Sigma", x_to_diffusion_loss, jnp.zeros(n_test))

    # =========================================================================
    # TEST 2: Z position through wire index (should NOT work directly)
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 2: Z Position (Wire Direction) - Discrete Index")
    print("="*70)
    print("\nZ position affects wire_index = floor(z / wire_spacing)")
    print("floor() has zero gradient almost everywhere")
    print("This is NOT differentiable through indices!")

    z_positions = positions_cm[:, 2]
    wire_index_offset = 100

    def z_to_index_loss(z_offsets):
        """Loss through wire index (discrete)."""
        z_new = z_positions + z_offsets
        wire_indices = jnp.floor(z_new / wire_spacing + wire_index_offset).astype(jnp.int32)
        # Convert back to float to compute loss
        return jnp.mean(wire_indices.astype(jnp.float32))

    test_simple_gradient("Z → Wire Index (floor) - EXPECTED TO FAIL", z_to_index_loss, jnp.zeros(n_test))

    # =========================================================================
    # TEST 3: Z position through sub-bin offset (DIFFERENTIABLE!)
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 3: Z Position Through Sub-Bin Offset (Differentiable!)")
    print("="*70)
    print("\nThe KEY insight: wire_offset = z/spacing - floor(z/spacing)")
    print("This fractional part IS differentiable!")
    print("It determines the Gaussian weight at each wire.")

    def z_to_subbin_loss(z_offsets):
        """Loss through sub-bin wire offset (continuous)."""
        z_new = z_positions + z_offsets
        # Continuous wire position
        wire_pos = z_new / wire_spacing + wire_index_offset
        # Discrete index
        wire_idx = jnp.floor(wire_pos)
        # Sub-bin offset (0 to 1) - THIS IS DIFFERENTIABLE
        wire_offset = wire_pos - wire_idx
        return jnp.mean(wire_offset ** 2)

    test_simple_gradient("Z → Sub-Bin Offset (continuous)", z_to_subbin_loss, jnp.zeros(n_test))

    # =========================================================================
    # TEST 4: Z through Gaussian kernel weights (DIFFERENTIABLE!)
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 4: Z Position Through Gaussian Kernel Weights")
    print("="*70)
    print("\nWhen we compute Gaussian kernels at neighboring wires,")
    print("the z position determines the distance to each wire.")
    print("distance_to_wire[k] = z - (wire_idx + k) * wire_spacing")
    print("This flows into: exp(-d²/2σ²) which IS differentiable!")

    def z_through_gaussian_loss(z_offsets):
        """Loss through Gaussian kernel weights."""
        z_new = z_positions + z_offsets
        drift_times = jnp.abs(positions_cm[:, 0]) / drift_velocity

        # Wire position (continuous)
        wire_pos = z_new / wire_spacing + wire_index_offset
        wire_idx = jnp.floor(wire_pos)

        # Distances to neighboring wires
        wire_offsets_k = jnp.array([-1, 0, 1])  # 3 nearest wires
        distances = (wire_pos[:, None] - (wire_idx[:, None] + wire_offsets_k)) * wire_spacing

        # Diffusion sigma
        sigma_sq = 2.0 * physics_params.diffusion_trans * drift_times + 1e-10
        sigma = jnp.sqrt(sigma_sq)

        # Gaussian weights (DIFFERENTIABLE w.r.t. z through distances!)
        gauss_weights = jnp.exp(-distances**2 / (2 * sigma_sq[:, None]))

        return jnp.mean(gauss_weights)

    test_simple_gradient("Z → Gaussian Weights (distance-based)", z_through_gaussian_loss, jnp.zeros(n_test))

    # =========================================================================
    # TEST 5: Full Simulation with Position Offsets (Small Values)
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 5: Full Simulation - Position Offsets (N x 3)")
    print("="*70)
    print("\nTesting with SMALL offsets (0.1 cm scale) to ensure overlap.")
    print("Large offsets would cause non-overlapping signals → weak gradients.")
    print("\nExpected: gradients for x (through drift time)")
    print("          gradients for z (through Gaussian kernel weights)")

    electron_lifetime_us = detector_config['electron_lifetime_ms'] * 1000.0
    time_step_size = detector_config['time_step_size_us']
    num_wires = 200
    num_time_steps = 500
    K_wire = 3
    K_time = 5

    def compute_wireplane(pos):
        """Compute wireplane from positions (shared function)."""
        x_pos = pos[:, 0]
        z_pos = pos[:, 2]

        drift_times = jnp.abs(x_pos) / drift_velocity

        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        attenuation = jnp.exp(-drift_times / electron_lifetime_us)
        attenuated_charges = charges * attenuation

        sigma_wire_sq = 2.0 * physics_params.diffusion_trans * drift_times + 1e-12
        sigma_time_sq = 2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times + 1e-12
        sigma_wire = jnp.sqrt(sigma_wire_sq)
        sigma_time = jnp.sqrt(sigma_time_sq)

        wire_pos = z_pos / wire_spacing + num_wires // 2
        center_wire = jnp.floor(wire_pos).astype(jnp.int32)
        center_time = jnp.floor(drift_times / time_step_size).astype(jnp.int32)

        wire_offset = wire_pos - center_wire.astype(jnp.float32)
        time_offset = drift_times / time_step_size - center_time.astype(jnp.float32)

        wire_offsets_k = jnp.arange(-K_wire, K_wire + 1)
        time_offsets_k = jnp.arange(-K_time, K_time + 1)

        wire_distances = (wire_offsets_k[None, :] - wire_offset[:, None]) * wire_spacing
        time_distances = (time_offsets_k[None, :] - time_offset[:, None]) * time_step_size

        wire_gauss = jnp.exp(-wire_distances**2 / (2 * sigma_wire_sq[:, None])) / (jnp.sqrt(2 * jnp.pi) * sigma_wire[:, None] + 1e-12)
        time_gauss = jnp.exp(-time_distances**2 / (2 * sigma_time_sq[:, None])) / (jnp.sqrt(2 * jnp.pi) * sigma_time[:, None] + 1e-12)

        kernel_2d = wire_gauss[:, :, None] * time_gauss[:, None, :]
        weighted_kernel = kernel_2d * attenuated_charges[:, None, None]

        n_hits = de.shape[0]
        n_wire_k = 2 * K_wire + 1
        n_time_k = 2 * K_time + 1

        wire_indices_k = center_wire[:, None] + wire_offsets_k[None, :]
        time_indices_k = center_time[:, None] + time_offsets_k[None, :]

        wire_indices_k = jnp.clip(wire_indices_k, 0, num_wires - 1)
        time_indices_k = jnp.clip(time_indices_k, 0, num_time_steps - 1)

        wire_idx_3d = jnp.broadcast_to(wire_indices_k[:, :, None], (n_hits, n_wire_k, n_time_k))
        time_idx_3d = jnp.broadcast_to(time_indices_k[:, None, :], (n_hits, n_wire_k, n_time_k))

        wire_flat = wire_idx_3d.reshape(-1)
        time_flat = time_idx_3d.reshape(-1)
        values_flat = weighted_kernel.reshape(-1)

        wireplane = jnp.zeros((num_wires, num_time_steps))
        wireplane = wireplane.at[wire_flat, time_flat].add(values_flat)

        return wireplane

    # Compute reference wireplane at zero offset
    reference_wireplane = compute_wireplane(positions_cm)
    print(f"Reference wireplane: max={float(jnp.max(reference_wireplane)):.4e}")

    def full_sim_mse_loss(position_offsets):
        """MSE loss comparing offset simulation to reference."""
        pos = positions_cm + position_offsets
        wireplane = compute_wireplane(pos)
        # MSE against reference (gradient pushes toward zero offset)
        return jnp.mean((wireplane - reference_wireplane) ** 2)

    # Test with small offset to get non-zero gradient
    print("\nTesting at zero offset (should have zero gradient at minimum)...")
    position_offsets_zero = jnp.zeros((n_test, 3))
    loss_at_zero = full_sim_mse_loss(position_offsets_zero)
    print(f"Loss at zero offset: {float(loss_at_zero):.4e}")

    # Test with small non-zero offset
    print("\nTesting with small offset (0.1 cm in x and z)...")
    position_offsets_small = jnp.zeros((n_test, 3))
    position_offsets_small = position_offsets_small.at[:, 0].set(0.1)  # 0.1 cm x offset
    position_offsets_small = position_offsets_small.at[:, 2].set(0.05)  # 0.05 cm z offset

    loss_at_small = full_sim_mse_loss(position_offsets_small)
    print(f"Loss at small offset: {float(loss_at_small):.4e}")

    try:
        grad_fn = jax.grad(full_sim_mse_loss)
        grads = grad_fn(position_offsets_small)
        print(f"\nGradient shape: {grads.shape}")

        print("\nPer-Dimension Gradient Statistics:")
        dim_names = ['x (drift)', 'y', 'z (wire)']
        for d in range(3):
            g = grads[:, d]
            print(f"  {dim_names[d]:>12}: min={float(jnp.min(g)):>12.4e}, max={float(jnp.max(g)):>12.4e}")
            print(f"                mean={float(jnp.mean(g)):>12.4e}, std={float(jnp.std(g)):>12.4e}")

        # Numerical verification
        print("\nNumerical Verification (small eps=1e-4):")
        eps = 1e-4
        for d, dim_name in enumerate(dim_names):
            delta = jnp.zeros((n_test, 3))
            delta = delta.at[0, d].set(eps)
            num_grad = (full_sim_mse_loss(position_offsets_small + delta) - full_sim_mse_loss(position_offsets_small - delta)) / (2 * eps)
            ana_grad = grads[0, d]
            if abs(float(num_grad)) > 1e-10:
                rel_diff = abs(float(ana_grad) - float(num_grad)) / abs(float(num_grad))
                match = "MATCH" if rel_diff < 0.5 else "DIFF"
            else:
                match = "~0" if abs(float(ana_grad)) < 1e-8 else "NON-ZERO"
            print(f"  {dim_name}: analytical={float(ana_grad):>12.4e}, numerical={float(num_grad):>12.4e} [{match}]")

        # Check gradient direction - should point back toward zero
        print("\nGradient Direction Check:")
        print("  (Negative gradient means 'decrease offset to reduce loss')")
        for d, dim_name in enumerate(dim_names):
            mean_grad = float(jnp.mean(grads[:, d]))
            offset_val = float(position_offsets_small[0, d])
            if abs(mean_grad) > 1e-10:
                direction = "CORRECT" if (mean_grad * offset_val > 0) else "REVERSED"
                print(f"  {dim_name}: mean_grad={mean_grad:>12.4e}, offset={offset_val:.3f} → {direction}")
            else:
                print(f"  {dim_name}: mean_grad={mean_grad:>12.4e} → ZERO/WEAK")

    except Exception as e:
        print(f"FAILED: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: Position Gradient Differentiability")
    print("="*70)
    print("""
WHAT IS DIFFERENTIABLE:
-----------------------
✓ X position (drift direction):
  - Flows through drift_time = |x| / velocity
  - Affects: electron attenuation, diffusion sigmas
  - Gradient path: x → drift_time → exp(-t/τ), sqrt(2*D*t) → values

✓ Z position (wire direction) via sub-bin offset:
  - Flows through: wire_offset = z/spacing - floor(z/spacing)
  - Affects: distance_to_wire in Gaussian kernel
  - Gradient path: z → wire_offset → exp(-d²/2σ²) → kernel weight → values

✓ Y position (if used):
  - Would flow through angle calculations
  - Affects: angular response kernels

WHAT IS NOT DIFFERENTIABLE:
---------------------------
✗ Wire INDEX: floor(z/spacing) → integer → no gradient
✗ Time INDEX: floor(t/step) → integer → no gradient
✗ Index SELECTION: wireplane[wire_idx, time_idx] → discrete choice

KEY INSIGHT:
------------
Positions ARE differentiable, but only through the VALUES scattered to bins,
not through WHICH bins are selected. The gradient flows through:

1. Sub-bin offsets (continuous fractional position within bin)
2. Gaussian kernel weights (depend on distance to neighboring bins)
3. Attenuation and diffusion (depend on drift distance/time)

For practical inverse problems (learning particle positions):
- X position: LEARNABLE (strong gradient through drift physics)
- Z position: LEARNABLE with sub-bin precision (gradient through kernel weights)
- Y position: LEARNABLE if angular effects are modeled

The discrete binning creates a PIECEWISE smooth loss landscape:
- Within each bin: smooth gradients from sub-bin position
- At bin boundaries: discontinuity in gradient direction
""")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
