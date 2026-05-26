#!/usr/bin/env python3
"""
Position Gradient Analysis with Properly Centered Wire Planes

Key fix: Compute wire plane offsets based on actual data positions so
all planes have valid wire indices.
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
    print("Position Gradient Analysis - Properly Centered")
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

    # Compute proper center offsets for each plane based on data
    y_mean = float(jnp.mean(positions_cm[:, 1]))
    z_mean = float(jnp.mean(positions_cm[:, 2]))

    print(f"\nData center: y={y_mean:.2f} cm, z={z_mean:.2f} cm")

    # For each plane, compute the wire coordinate at data center
    # and use that to set the offset so data lands at num_wires//2
    plane_offsets = []
    for i, (angle, spacing) in enumerate(zip(angles_rad, wire_spacings_cm)):
        cos_a = float(jnp.cos(angle))
        sin_a = float(jnp.sin(angle))
        wire_coord_at_center = y_mean * sin_a + z_mean * cos_a
        # We want wire_coord_at_center / spacing + offset = num_wires // 2
        offset = num_wires // 2 - wire_coord_at_center / spacing
        plane_offsets.append(offset)
        print(f"  {plane_names[i]}: wire_coord_center={wire_coord_at_center:.2f}, offset={offset:.2f}")

    def compute_wireplane_for_plane(positions_cm, plane_idx):
        """Compute wireplane for a single plane with proper offset."""
        angle = angles_rad[plane_idx]
        wire_spacing = float(wire_spacings_cm[plane_idx])
        wire_offset = plane_offsets[plane_idx]

        x_pos = positions_cm[:, 0]
        y_pos = positions_cm[:, 1]
        z_pos = positions_cm[:, 2]

        drift_times = jnp.abs(x_pos) / drift_velocity

        # Wire coordinate using angle - THIS IS WHERE Y AND Z BOTH CONTRIBUTE
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

        # Wire position with proper offset
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

    # Verify wire positions are now valid
    print("\nVerifying wire positions after centering:")
    y = positions_cm[:, 1]
    z = positions_cm[:, 2]
    for i, (angle, spacing, offset) in enumerate(zip(angles_rad, wire_spacings_cm, plane_offsets)):
        cos_a = np.cos(float(angle))
        sin_a = np.sin(float(angle))
        wire_coord = y * sin_a + z * cos_a
        wire_pos = wire_coord / spacing + offset
        print(f"  {plane_names[i]}: wire_pos = [{float(wire_pos.min()):.1f}, {float(wire_pos.max()):.1f}]")

    # Compute reference wireplanes
    ref_wireplanes = [compute_wireplane_for_plane(positions_cm, i) for i in range(3)]

    print("\n" + "="*70)
    print("Per-Plane Gradient Test (Centered)")
    print("="*70)

    # Test each plane separately
    for plane_idx in range(3):
        ref_wp = ref_wireplanes[plane_idx]

        def single_plane_loss(position_offsets):
            pos = positions_cm + position_offsets
            wp = compute_wireplane_for_plane(pos, plane_idx)
            return jnp.mean((wp - ref_wp) ** 2)

        print(f"\n--- {plane_names[plane_idx]} ---")

        offset = jnp.zeros((n_test, 3))
        offset = offset.at[:, 0].set(0.05)
        offset = offset.at[:, 1].set(0.05)
        offset = offset.at[:, 2].set(0.05)

        loss = float(single_plane_loss(offset))
        print(f"Loss at (0.05, 0.05, 0.05): {loss:.4e}")

        grad_fn = jax.grad(single_plane_loss)
        grads = grad_fn(offset)

        print(f"Mean gradients:")
        print(f"  ∂L/∂x = {float(jnp.mean(grads[:,0])):>12.4e}")
        print(f"  ∂L/∂y = {float(jnp.mean(grads[:,1])):>12.4e}")
        print(f"  ∂L/∂z = {float(jnp.mean(grads[:,2])):>12.4e}")

        # Numerical verification
        eps = 1e-4
        print(f"\nNumerical check (segment 0, eps={eps}):")
        for d, dim_name in enumerate(['x', 'y', 'z']):
            delta = jnp.zeros((n_test, 3))
            delta = delta.at[0, d].set(eps)
            num_grad = (single_plane_loss(offset + delta) - single_plane_loss(offset - delta)) / (2 * eps)
            ana_grad = float(grads[0, d])

            if abs(float(num_grad)) > 1e-12:
                rel_diff = abs(ana_grad - float(num_grad)) / abs(float(num_grad))
                status = "MATCH" if rel_diff < 0.5 else "DIFF"
            elif abs(ana_grad) < 1e-10:
                status = "BOTH~0"
            else:
                status = "ANA≠0"

            print(f"  ∂L/∂{dim_name}: ana={ana_grad:>12.4e}, num={float(num_grad):>12.4e} [{status}]")

        # Gradient relationship
        angle = float(angles_rad[plane_idx])
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)
        mean_y = float(jnp.mean(grads[:,1]))
        mean_z = float(jnp.mean(grads[:,2]))

        print(f"\nTheoretical relationship:")
        print(f"  sin({np.degrees(angle):.0f}°) = {sin_a:.4f}")
        print(f"  cos({np.degrees(angle):.0f}°) = {cos_a:.4f}")
        if abs(sin_a) > 0.01 and abs(cos_a) > 0.01:
            ratio = (mean_y / sin_a) / (mean_z / cos_a) if abs(mean_z) > 1e-10 else float('inf')
            print(f"  (∂L/∂y / sin) / (∂L/∂z / cos) = {ratio:.4f} (should be ~1.0)")

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

    loss_combined = float(all_planes_loss(offset))
    print(f"\nCombined loss at offset: {loss_combined:.4e}")

    grad_fn = jax.grad(all_planes_loss)
    grads = grad_fn(offset)

    print(f"\nCombined mean gradients:")
    print(f"  ∂L/∂x = {float(jnp.mean(grads[:,0])):>12.4e}")
    print(f"  ∂L/∂y = {float(jnp.mean(grads[:,1])):>12.4e}")
    print(f"  ∂L/∂z = {float(jnp.mean(grads[:,2])):>12.4e}")

    print("\nNumerical check (segment 0):")
    eps = 1e-4
    for d, dim_name in enumerate(['x', 'y', 'z']):
        delta = jnp.zeros((n_test, 3))
        delta = delta.at[0, d].set(eps)
        num_grad = (all_planes_loss(offset + delta) - all_planes_loss(offset - delta)) / (2 * eps)
        ana_grad = float(grads[0, d])

        if abs(float(num_grad)) > 1e-12:
            rel_diff = abs(ana_grad - float(num_grad)) / abs(float(num_grad))
            status = "MATCH" if rel_diff < 0.5 else "DIFF"
        else:
            status = "BOTH~0" if abs(ana_grad) < 1e-10 else "ANA≠0"

        print(f"  ∂L/∂{dim_name}: ana={ana_grad:>12.4e}, num={float(num_grad):>12.4e} [{status}]")

    # Loss landscape
    print("\n" + "="*70)
    print("Loss Landscape (single dimension variations)")
    print("="*70)

    offsets_1d = jnp.linspace(-0.1, 0.1, 11)

    for dim, dim_name in enumerate(['X', 'Y', 'Z']):
        print(f"\nVarying {dim_name}:")
        print(f"{'Offset':>8} {'Loss':>12}")
        print("-"*22)
        for off in offsets_1d:
            pos_offset = jnp.zeros((n_test, 3))
            pos_offset = pos_offset.at[:, dim].set(float(off))
            loss = float(all_planes_loss(pos_offset))
            print(f"{float(off):>8.3f} {loss:>12.4e}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
With proper centering so all planes have valid wire indices:

1. Each plane should show gradients for its contributing dimensions:
   - U (+60°): ∂L/∂y ∝ sin(60°), ∂L/∂z ∝ cos(60°)
   - V (-60°): ∂L/∂y ∝ sin(-60°), ∂L/∂z ∝ cos(-60°)
   - Y (0°):   ∂L/∂y = 0,         ∂L/∂z ∝ cos(0°) = 1

2. Combined gradients:
   - ∂L/∂z should be strong (all planes contribute)
   - ∂L/∂y should be NON-ZERO from U and V planes
     (Note: U and V have opposite-sign y contributions, but the LOSS
     increases for either direction of y shift, so gradients add!)

3. All three coordinates (x, y, z) should be learnable.
""")

    return True


if __name__ == "__main__":
    main()
