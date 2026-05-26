#!/usr/bin/env python3
"""
Per-Segment Position Gradient Test

Simple test: verify each segment has independent position gradients.
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
    print("Per-Segment Position Gradient Test")
    print("="*60)

    # Load data
    config_path = "../config/cubic_wireplane_config.yaml"
    data_path = "../mpvmpr_20.h5"

    detector_config = generate_detector(config_path)
    step_data = load_particle_step_data(data_path, 0)

    n_test = 10  # Small number for clarity
    de = jnp.asarray(step_data['de'][:n_test], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'][:n_test], dtype=jnp.float32)
    positions_cm = jnp.asarray(step_data['position'][:n_test], dtype=jnp.float32) / 10.0

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

    def compute_wireplane(positions_cm, plane_idx):
        """Compute wireplane for one plane."""
        angle = angles_rad[plane_idx]
        wire_spacing = float(wire_spacings_cm[plane_idx])
        wire_offset = float(index_offsets[plane_idx])
        num_wires = int(num_wires_actual[plane_idx])

        x_pos = positions_cm[:, 0]
        y_pos = positions_cm[:, 1]
        z_pos = positions_cm[:, 2]

        drift_times = jnp.abs(x_pos) / drift_velocity
        wire_coord = y_pos * jnp.sin(angle) + z_pos * jnp.cos(angle)

        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)
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

        wire_idx_3d = jnp.broadcast_to(wire_indices_k[:, :, None], (n_test, n_wire_k, n_time_k))
        time_idx_3d = jnp.broadcast_to(time_indices_k[:, None, :], (n_test, n_wire_k, n_time_k))

        wire_flat = wire_idx_3d.reshape(-1)
        time_flat = time_idx_3d.reshape(-1)
        values_flat = weighted_kernel.reshape(-1)

        wireplane = jnp.zeros((num_wires, num_time_steps))
        wireplane = wireplane.at[wire_flat, time_flat].add(values_flat)

        return wireplane

    # Compute reference wireplanes
    ref_wireplanes = [compute_wireplane(positions_cm, i) for i in range(3)]

    def loss_fn(pos_offsets):
        """MSE loss over all planes."""
        pos = positions_cm + pos_offsets
        total_loss = 0.0
        for i in range(3):
            wp = compute_wireplane(pos, i)
            total_loss += jnp.mean((wp - ref_wireplanes[i]) ** 2)
        return total_loss / 3.0

    # Test gradients
    print(f"\nTesting {n_test} segments")
    print("-"*60)

    offset = jnp.ones((n_test, 3)) * 0.05  # 0.5mm offset in each direction

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(offset)

    print("\nPer-segment gradients (analytical):")
    print(f"{'Seg':>4} {'∂L/∂x':>12} {'∂L/∂y':>12} {'∂L/∂z':>12}")
    print("-"*44)
    for i in range(n_test):
        print(f"{i:>4} {float(grads[i,0]):>12.4e} {float(grads[i,1]):>12.4e} {float(grads[i,2]):>12.4e}")

    # Numerical verification for first 3 segments
    print("\n" + "="*60)
    print("Numerical Verification (first 3 segments)")
    print("="*60)

    eps = 1e-4
    for seg_idx in range(3):
        print(f"\nSegment {seg_idx}:")
        for d, dim in enumerate(['x', 'y', 'z']):
            delta = jnp.zeros((n_test, 3))
            delta = delta.at[seg_idx, d].set(eps)

            loss_plus = loss_fn(offset + delta)
            loss_minus = loss_fn(offset - delta)
            num_grad = (loss_plus - loss_minus) / (2 * eps)
            ana_grad = float(grads[seg_idx, d])

            if abs(float(num_grad)) > 1e-12:
                rel_err = abs(ana_grad - float(num_grad)) / abs(float(num_grad))
                status = "MATCH" if rel_err < 0.3 else "DIFF"
            else:
                status = "BOTH~0" if abs(ana_grad) < 1e-10 else "CHECK"

            print(f"  ∂L/∂{dim}: ana={ana_grad:>10.4e}, num={float(num_grad):>10.4e} [{status}]")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total segments: {n_test}")
    print(f"Gradient shape: {grads.shape}")
    print(f"\nMean gradients:")
    print(f"  ∂L/∂x: {float(jnp.mean(grads[:,0])):.4e}")
    print(f"  ∂L/∂y: {float(jnp.mean(grads[:,1])):.4e}")
    print(f"  ∂L/∂z: {float(jnp.mean(grads[:,2])):.4e}")

    all_nonzero = jnp.all(jnp.abs(grads) > 1e-12)
    print(f"\nAll gradients non-zero: {bool(all_nonzero)}")

    return True


if __name__ == "__main__":
    main()
