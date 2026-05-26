#!/usr/bin/env python3
"""
Test per-segment gradients with many segments and learnable dE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from physics_params import create_default_params
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import calculate_box_model_charge_with_physics_params


def main():
    print("="*60)
    print("Per-Segment Gradients: Positions + dE")
    print("="*60)

    # Load data
    config_path = "../config/cubic_wireplane_config.yaml"
    data_path = "../mpvmpr_20.h5"

    detector_config = generate_detector(config_path)
    step_data = load_particle_step_data(data_path, 0)

    n_segments = 500  # Many segments with overlap
    de_true = jnp.asarray(step_data['de'][:n_segments], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'][:n_segments], dtype=jnp.float32)
    positions_cm = jnp.asarray(step_data['position'][:n_segments], dtype=jnp.float32) / 10.0

    print(f"\nSegments: {n_segments}")
    print(f"Position range (cm):")
    print(f"  X: [{float(positions_cm[:,0].min()):.2f}, {float(positions_cm[:,0].max()):.2f}]")
    print(f"  Y: [{float(positions_cm[:,1].min()):.2f}, {float(positions_cm[:,1].max()):.2f}]")
    print(f"  Z: [{float(positions_cm[:,2].min()):.2f}, {float(positions_cm[:,2].max()):.2f}]")
    print(f"dE range: [{float(de_true.min()):.4f}, {float(de_true.max()):.4f}]")

    # Detector params
    drift_velocity = detector_config['drift_velocity_cm_us']
    electron_lifetime_us = detector_config['electron_lifetime_ms'] * 1000.0
    time_step_size = detector_config['time_step_size_us']
    num_time_steps = detector_config['num_time_steps']

    angles_rad = detector_config['angles_rad'][0]
    wire_spacings_cm = detector_config['wire_spacings_cm'][0]
    index_offsets = detector_config['index_offsets'][0]
    num_wires_actual = detector_config['num_wires_actual'][0]

    physics_params = create_default_params()
    K_wire = 3
    K_time = 5

    # Verify charges are non-zero
    charges = calculate_box_model_charge_with_physics_params(de_true, dx, physics_params)
    print(f"\nCharges range: [{float(charges.min()):.2e}, {float(charges.max()):.2e}]")
    print(f"Charges sum: {float(charges.sum()):.2e}")

    def compute_wireplane(positions, de_vals, plane_idx):
        """Compute wireplane for one plane."""
        n_seg = positions.shape[0]

        angle = angles_rad[plane_idx]
        wire_spacing = float(wire_spacings_cm[plane_idx])
        wire_offset = float(index_offsets[plane_idx])
        num_wires = int(num_wires_actual[plane_idx])

        x_pos = positions[:, 0]
        y_pos = positions[:, 1]
        z_pos = positions[:, 2]

        drift_times = jnp.abs(x_pos) / drift_velocity
        wire_coord = y_pos * jnp.sin(angle) + z_pos * jnp.cos(angle)

        # Use proper recombination function
        charges = calculate_box_model_charge_with_physics_params(de_vals, dx, physics_params)
        attenuation = jnp.exp(-drift_times / electron_lifetime_us)
        attenuated_charges = charges * attenuation

        sigma_wire_sq = 2.0 * physics_params.diffusion_trans * drift_times + 1e-12
        sigma_time_sq = 2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times + 1e-12
        sigma_wire = jnp.sqrt(sigma_wire_sq)
        sigma_time = jnp.sqrt(sigma_time_sq)

        wire_pos = wire_coord / wire_spacing + wire_offset
        center_wire = jnp.floor(wire_pos).astype(jnp.int32)
        center_time = jnp.floor(drift_times / time_step_size).astype(jnp.int32)

        wire_subbin = wire_pos - center_wire.astype(jnp.float32)
        time_subbin = drift_times / time_step_size - center_time.astype(jnp.float32)

        wire_offsets_k = jnp.arange(-K_wire, K_wire + 1)
        time_offsets_k = jnp.arange(-K_time, K_time + 1)

        wire_distances = (wire_offsets_k[None, :] - wire_subbin[:, None]) * wire_spacing
        time_distances = (time_offsets_k[None, :] - time_subbin[:, None]) * time_step_size

        wire_gauss = jnp.exp(-wire_distances**2 / (2 * sigma_wire_sq[:, None])) / (jnp.sqrt(2 * jnp.pi) * sigma_wire[:, None] + 1e-12)
        time_gauss = jnp.exp(-time_distances**2 / (2 * sigma_time_sq[:, None])) / (jnp.sqrt(2 * jnp.pi) * sigma_time[:, None] + 1e-12)

        kernel_2d = wire_gauss[:, :, None] * time_gauss[:, None, :]
        weighted_kernel = kernel_2d * attenuated_charges[:, None, None]

        n_wire_k = 2 * K_wire + 1
        n_time_k = 2 * K_time + 1

        wire_indices_k = center_wire[:, None] + wire_offsets_k[None, :]
        time_indices_k = center_time[:, None] + time_offsets_k[None, :]

        wire_indices_k = jnp.clip(wire_indices_k, 0, num_wires - 1)
        time_indices_k = jnp.clip(time_indices_k, 0, num_time_steps - 1)

        wire_idx_3d = jnp.broadcast_to(wire_indices_k[:, :, None], (n_seg, n_wire_k, n_time_k))
        time_idx_3d = jnp.broadcast_to(time_indices_k[:, None, :], (n_seg, n_wire_k, n_time_k))

        wire_flat = wire_idx_3d.reshape(-1)
        time_flat = time_idx_3d.reshape(-1)
        values_flat = weighted_kernel.reshape(-1)

        wireplane = jnp.zeros((num_wires, num_time_steps))
        wireplane = wireplane.at[wire_flat, time_flat].add(values_flat)

        return wireplane

    # Compute reference wireplanes with true values
    ref_wireplanes = [compute_wireplane(positions_cm, de_true, i) for i in range(3)]

    print(f"\nReference wireplane max signals:")
    for i, wp in enumerate(ref_wireplanes):
        print(f"  Plane {i}: max={float(jnp.max(wp)):.2e}, sum={float(jnp.sum(wp)):.2e}")

    # =========================================================================
    # Test 1: Position gradients
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 1: Position Gradients (500 segments)")
    print("="*60)

    def pos_loss(pos_offsets):
        pos = positions_cm + pos_offsets
        total = 0.0
        for i in range(3):
            wp = compute_wireplane(pos, de_true, i)
            total += jnp.mean((wp - ref_wireplanes[i]) ** 2)
        return total / 3.0

    # Check loss values first
    zero_offset = jnp.zeros((n_segments, 3))
    small_offset = jnp.ones((n_segments, 3)) * 0.05

    loss_at_zero = float(pos_loss(zero_offset))
    loss_at_offset = float(pos_loss(small_offset))

    print(f"\nLoss at zero offset: {loss_at_zero:.4e}")
    print(f"Loss at 0.05 offset: {loss_at_offset:.4e}")

    grad_pos_fn = jax.grad(pos_loss)
    grads_pos = grad_pos_fn(small_offset)

    print(f"\nGradient shape: {grads_pos.shape}")
    print(f"\nGradient statistics:")
    for d, dim in enumerate(['x', 'y', 'z']):
        g = grads_pos[:, d]
        print(f"  ∂L/∂{dim}: mean={float(jnp.mean(g)):.4e}, std={float(jnp.std(g)):.4e}")

    # Check a few segments numerically
    print("\nNumerical check (segments 0, 250, 499):")
    eps = 1e-4
    for seg_idx in [0, 250, 499]:
        print(f"\n  Segment {seg_idx}:")
        for d, dim in enumerate(['x', 'y', 'z']):
            delta = jnp.zeros((n_segments, 3))
            delta = delta.at[seg_idx, d].set(eps)
            loss_plus = float(pos_loss(small_offset + delta))
            loss_minus = float(pos_loss(small_offset - delta))
            num_grad = (loss_plus - loss_minus) / (2 * eps)
            ana_grad = float(grads_pos[seg_idx, d])

            if abs(num_grad) > 1e-12:
                rel_err = abs(ana_grad - num_grad) / abs(num_grad)
                status = "MATCH" if rel_err < 0.3 else "DIFF"
            else:
                status = "BOTH~0" if abs(ana_grad) < 1e-10 else "CHECK"

            print(f"    ∂L/∂{dim}: ana={ana_grad:.4e}, num={num_grad:.4e} [{status}]")

    nonzero_pos = jnp.sum(jnp.abs(grads_pos) > 1e-12)
    print(f"\nNon-zero gradients: {int(nonzero_pos)} / {n_segments * 3} ({100*int(nonzero_pos)/(n_segments*3):.1f}%)")

    # =========================================================================
    # Test 2: dE gradients
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 2: dE Gradients (500 segments)")
    print("="*60)

    def de_loss(de_scale):
        """Scale dE by per-segment factors."""
        de_scaled = de_true * de_scale
        total = 0.0
        for i in range(3):
            wp = compute_wireplane(positions_cm, de_scaled, i)
            total += jnp.mean((wp - ref_wireplanes[i]) ** 2)
        return total / 3.0

    # Check loss values
    ones_scale = jnp.ones(n_segments)
    perturbed_scale = jnp.ones(n_segments) * 1.1

    loss_at_ones = float(de_loss(ones_scale))
    loss_at_perturbed = float(de_loss(perturbed_scale))

    print(f"\nLoss at scale=1.0: {loss_at_ones:.4e}")
    print(f"Loss at scale=1.1: {loss_at_perturbed:.4e}")

    grad_de_fn = jax.grad(de_loss)
    grads_de = grad_de_fn(perturbed_scale)

    print(f"\nGradient shape: {grads_de.shape}")
    print(f"\nGradient statistics:")
    print(f"  ∂L/∂(dE_scale): mean={float(jnp.mean(grads_de)):.4e}, std={float(jnp.std(grads_de)):.4e}")

    # Numerical check
    print("\nNumerical check (segments 0, 250, 499):")
    for seg_idx in [0, 250, 499]:
        delta = jnp.zeros(n_segments)
        delta = delta.at[seg_idx].set(eps)
        loss_plus = float(de_loss(perturbed_scale + delta))
        loss_minus = float(de_loss(perturbed_scale - delta))
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        ana_grad = float(grads_de[seg_idx])

        if abs(num_grad) > 1e-12:
            rel_err = abs(ana_grad - num_grad) / abs(num_grad)
            status = "MATCH" if rel_err < 0.3 else "DIFF"
        else:
            status = "BOTH~0" if abs(ana_grad) < 1e-10 else "CHECK"
        print(f"  Segment {seg_idx}: ana={ana_grad:.4e}, num={num_grad:.4e} [{status}]")

    nonzero_de = jnp.sum(jnp.abs(grads_de) > 1e-12)
    print(f"\nNon-zero gradients: {int(nonzero_de)} / {n_segments} ({100*int(nonzero_de)/n_segments:.1f}%)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tested {n_segments} overlapping segments.")
    print(f"Position gradients shape: ({n_segments}, 3)")
    print(f"dE gradients shape: ({n_segments},)")

    pos_ok = int(nonzero_pos) == n_segments * 3
    de_ok = int(nonzero_de) == n_segments

    print(f"\nPosition gradients: {'PASS' if pos_ok else 'PARTIAL'} ({100*int(nonzero_pos)/(n_segments*3):.1f}% non-zero)")
    print(f"dE gradients: {'PASS' if de_ok else 'PARTIAL'} ({100*int(nonzero_de)/n_segments:.1f}% non-zero)")

    return True


if __name__ == "__main__":
    main()
