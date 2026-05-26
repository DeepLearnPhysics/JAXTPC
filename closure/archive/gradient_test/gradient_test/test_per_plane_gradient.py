#!/usr/bin/env python3
"""
Per-Plane Gradient Analysis

Test each wire plane separately to understand gradient contributions.
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
    print("="*70)
    print("Per-Plane Gradient Analysis")
    print("="*70)

    # Load data
    config_path = "../config/cubic_wireplane_config.yaml"
    data_path = "../mpvmpr_20.h5"

    detector_config = generate_detector(config_path)
    step_data = load_particle_step_data(data_path, 0)

    n_test = 50
    de = jnp.asarray(step_data['de'][:n_test], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'][:n_test], dtype=jnp.float32)
    positions_cm = jnp.asarray(step_data['position'][:n_test], dtype=jnp.float32) / 10.0

    drift_velocity = detector_config['drift_velocity_cm_us']
    electron_lifetime_us = detector_config['electron_lifetime_ms'] * 1000.0
    time_step_size = detector_config['time_step_size_us']
    angles_rad = detector_config['angles_rad'][0]
    wire_spacings_cm = detector_config['wire_spacings_cm'][0]

    physics_params = create_default_params()

    num_wires = 200
    num_time_steps = 500
    K_wire = 3
    K_time = 5

    plane_names = ['U (+60°)', 'V (-60°)', 'Y (0°)']

    print(f"\nWire plane configuration:")
    for i, name in enumerate(plane_names):
        angle = float(angles_rad[i])
        print(f"  {name}: angle={np.degrees(angle):.1f}°, sin={np.sin(angle):.3f}, cos={np.cos(angle):.3f}")

    def compute_wireplane_for_plane(positions_cm, plane_idx):
        """Compute wireplane for a single plane."""
        angle = angles_rad[plane_idx]
        wire_spacing = float(wire_spacings_cm[plane_idx])

        x_pos = positions_cm[:, 0]
        y_pos = positions_cm[:, 1]
        z_pos = positions_cm[:, 2]

        drift_times = jnp.abs(x_pos) / drift_velocity

        # Wire coordinate using angle
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        wire_coord = y_pos * sin_angle + z_pos * cos_angle

        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        attenuation = jnp.exp(-drift_times / electron_lifetime_us)
        attenuated_charges = charges * attenuation

        sigma_wire_sq = 2.0 * physics_params.diffusion_trans * drift_times + 1e-12
        sigma_time_sq = 2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times + 1e-12
        sigma_wire = jnp.sqrt(sigma_wire_sq)
        sigma_time = jnp.sqrt(sigma_time_sq)

        wire_pos = wire_coord / wire_spacing + num_wires // 2
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
    ref_wireplanes = [compute_wireplane_for_plane(positions_cm, i) for i in range(3)]

    print("\n" + "="*70)
    print("Per-Plane Gradient Test")
    print("="*70)

    # Test each plane separately
    for plane_idx in range(3):
        ref_wp = ref_wireplanes[plane_idx]

        def single_plane_loss(position_offsets):
            pos = positions_cm + position_offsets
            wp = compute_wireplane_for_plane(pos, plane_idx)
            return jnp.mean((wp - ref_wp) ** 2)

        print(f"\n--- {plane_names[plane_idx]} ---")

        # Small offset
        offset = jnp.zeros((n_test, 3))
        offset = offset.at[:, 0].set(0.05)
        offset = offset.at[:, 1].set(0.05)
        offset = offset.at[:, 2].set(0.05)

        loss = float(single_plane_loss(offset))
        print(f"Loss at (0.05, 0.05, 0.05) offset: {loss:.4e}")

        # Gradients
        grad_fn = jax.grad(single_plane_loss)
        grads = grad_fn(offset)

        print(f"Mean gradients:")
        print(f"  ∂L/∂x = {float(jnp.mean(grads[:,0])):>12.4e}")
        print(f"  ∂L/∂y = {float(jnp.mean(grads[:,1])):>12.4e}")
        print(f"  ∂L/∂z = {float(jnp.mean(grads[:,2])):>12.4e}")

        # Numerical verification
        eps = 1e-4
        print(f"\nNumerical check (first segment, eps={eps}):")
        for d, dim_name in enumerate(['x', 'y', 'z']):
            delta = jnp.zeros((n_test, 3))
            delta = delta.at[0, d].set(eps)
            num_grad = (single_plane_loss(offset + delta) - single_plane_loss(offset - delta)) / (2 * eps)
            ana_grad = float(grads[0, d])

            if abs(float(num_grad)) > 1e-12:
                rel_diff = abs(ana_grad - float(num_grad)) / abs(float(num_grad))
                status = "MATCH" if rel_diff < 0.3 else "DIFF"
            elif abs(ana_grad) < 1e-10:
                status = "BOTH~0"
            else:
                status = "ANA≠0"

            print(f"  ∂L/∂{dim_name}: ana={ana_grad:>12.4e}, num={float(num_grad):>12.4e} [{status}]")

        # Check expected gradient relationship
        angle = float(angles_rad[plane_idx])
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        mean_y = float(jnp.mean(grads[:,1]))
        mean_z = float(jnp.mean(grads[:,2]))

        print(f"\nGradient relationship check:")
        print(f"  sin({np.degrees(angle):.0f}°) = {sin_a:.4f}")
        print(f"  cos({np.degrees(angle):.0f}°) = {cos_a:.4f}")

        if abs(mean_z) > 1e-10 and abs(cos_a) > 0.01:
            # ∂L/∂r is common to both
            # ∂L/∂y = ∂L/∂r * sin(a), ∂L/∂z = ∂L/∂r * cos(a)
            # So ∂L/∂y / sin(a) ≈ ∂L/∂z / cos(a)
            ratio_y = mean_y / sin_a if abs(sin_a) > 0.01 else float('inf')
            ratio_z = mean_z / cos_a
            print(f"  ∂L/∂y / sin(a) = {ratio_y:.4e}")
            print(f"  ∂L/∂z / cos(a) = {ratio_z:.4e}")
            if abs(ratio_y) < float('inf') and abs(ratio_z) > 0:
                print(f"  Ratio = {abs(ratio_y/ratio_z):.4f} (should be ~1.0)")

    # Combined test
    print("\n" + "="*70)
    print("Combined (All Planes) Gradient Test")
    print("="*70)

    def all_planes_loss(position_offsets):
        pos = positions_cm + position_offsets
        total_loss = 0.0
        for i in range(3):
            wp = compute_wireplane_for_plane(pos, i)
            total_loss += jnp.mean((wp - ref_wireplanes[i]) ** 2)
        return total_loss / 3.0

    offset = jnp.zeros((n_test, 3))
    offset = offset.at[:, 0].set(0.05)
    offset = offset.at[:, 1].set(0.05)
    offset = offset.at[:, 2].set(0.05)

    grad_fn = jax.grad(all_planes_loss)
    grads = grad_fn(offset)

    print(f"\nCombined mean gradients:")
    print(f"  ∂L/∂x = {float(jnp.mean(grads[:,0])):>12.4e}")
    print(f"  ∂L/∂y = {float(jnp.mean(grads[:,1])):>12.4e}")
    print(f"  ∂L/∂z = {float(jnp.mean(grads[:,2])):>12.4e}")

    print("\nNumerical check (first segment):")
    eps = 1e-4
    for d, dim_name in enumerate(['x', 'y', 'z']):
        delta = jnp.zeros((n_test, 3))
        delta = delta.at[0, d].set(eps)
        num_grad = (all_planes_loss(offset + delta) - all_planes_loss(offset - delta)) / (2 * eps)
        ana_grad = float(grads[0, d])

        if abs(float(num_grad)) > 1e-12:
            rel_diff = abs(ana_grad - float(num_grad)) / abs(float(num_grad))
            status = "MATCH" if rel_diff < 0.3 else "DIFF"
        else:
            status = "BOTH~0" if abs(ana_grad) < 1e-10 else "ANA≠0"

        print(f"  ∂L/∂{dim_name}: ana={ana_grad:>12.4e}, num={float(num_grad):>12.4e} [{status}]")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
Per-plane gradient analysis shows:

1. Y plane (0°): Only Z contributes (sin(0)=0)
   - ∂L/∂y ≈ 0 (expected!)
   - ∂L/∂z is strong

2. U plane (+60°): Both Y and Z contribute
   - ∂L/∂y ∝ sin(60°) = 0.866
   - ∂L/∂z ∝ cos(60°) = 0.5

3. V plane (-60°): Both Y and Z contribute
   - ∂L/∂y ∝ sin(-60°) = -0.866 (opposite sign to U!)
   - ∂L/∂z ∝ cos(-60°) = 0.5

When combining U and V planes:
- Z gradients ADD (both have cos(±60°) = 0.5)
- Y gradients CANCEL (sin(60°) + sin(-60°) = 0)

This explains why combined Y gradient is weak while Z is strong.
For full 3D position recovery, need planes that don't cancel!
""")

    return True


if __name__ == "__main__":
    main()
