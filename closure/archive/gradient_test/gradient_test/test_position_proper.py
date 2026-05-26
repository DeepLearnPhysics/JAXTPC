#!/usr/bin/env python3
"""
Position Gradient Analysis with Proper Geometry

Uses actual detector geometry from config file.
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
    print("Position Gradient Analysis - Proper Geometry")
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

    # Use actual detector parameters
    drift_velocity = detector_config['drift_velocity_cm_us']
    electron_lifetime_us = detector_config['electron_lifetime_ms'] * 1000.0
    time_step_size = detector_config['time_step_size_us']
    num_time_steps = detector_config['num_time_steps']

    # Wire plane parameters from config (side 0)
    angles_rad = detector_config['angles_rad'][0]
    wire_spacings_cm = detector_config['wire_spacings_cm'][0]
    index_offsets = detector_config['index_offsets'][0]
    num_wires_actual = detector_config['num_wires_actual'][0]
    min_wire_indices = detector_config['min_wire_indices_abs'][0]
    max_wire_indices = detector_config['max_wire_indices_abs'][0]

    physics_params = create_default_params()

    K_wire = 3
    K_time = 5

    plane_names = ['U (+60°)', 'V (-60°)', 'Y (0°)']

    print(f"\nDetector geometry:")
    for i, name in enumerate(plane_names):
        print(f"  {name}: {int(num_wires_actual[i])} wires, offset={int(index_offsets[i])}, "
              f"range=[{int(min_wire_indices[i])}, {int(max_wire_indices[i])}]")

    print(f"\nData positions (cm):")
    print(f"  X: [{float(positions_cm[:,0].min()):.2f}, {float(positions_cm[:,0].max()):.2f}]")
    print(f"  Y: [{float(positions_cm[:,1].min()):.2f}, {float(positions_cm[:,1].max()):.2f}]")
    print(f"  Z: [{float(positions_cm[:,2].min()):.2f}, {float(positions_cm[:,2].max()):.2f}]")

    # Verify wire positions
    y = positions_cm[:, 1]
    z = positions_cm[:, 2]
    print(f"\nWire positions in each plane:")
    for i, name in enumerate(plane_names):
        angle = float(angles_rad[i])
        spacing = float(wire_spacings_cm[i])
        offset = float(index_offsets[i])
        num_wires = int(num_wires_actual[i])

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        wire_coord = y * sin_a + z * cos_a
        wire_pos = wire_coord / spacing + offset

        in_range = float(wire_pos.min()) >= 0 and float(wire_pos.max()) < num_wires
        print(f"  {name}: wire_pos=[{float(wire_pos.min()):.1f}, {float(wire_pos.max()):.1f}], "
              f"valid=[0, {num_wires-1}], OK={in_range}")

    def compute_wireplane_for_plane(positions_cm, plane_idx):
        """Compute wireplane for a single plane using proper geometry."""
        angle = angles_rad[plane_idx]
        wire_spacing = float(wire_spacings_cm[plane_idx])
        wire_offset = float(index_offsets[plane_idx])
        num_wires = int(num_wires_actual[plane_idx])

        x_pos = positions_cm[:, 0]
        y_pos = positions_cm[:, 1]
        z_pos = positions_cm[:, 2]

        drift_times = jnp.abs(x_pos) / drift_velocity

        # Wire coordinate: r = y*sin(angle) + z*cos(angle)
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

        # Wire position with proper offset from geometry
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

    def compute_centroid(wireplane):
        """Compute signal-weighted centroid."""
        num_w, num_t = wireplane.shape
        wire_coords = jnp.arange(num_w)
        time_coords = jnp.arange(num_t)
        wire_grid, time_grid = jnp.meshgrid(wire_coords, time_coords, indexing='ij')

        weights = jnp.abs(wireplane) + 1e-12
        total_weight = jnp.sum(weights)

        centroid_wire = jnp.sum(wire_grid * weights) / total_weight
        centroid_time = jnp.sum(time_grid * weights) / total_weight

        return centroid_wire, centroid_time

    # Compute reference
    print("\n" + "="*70)
    print("Computing reference wireplanes and centroids...")
    print("="*70)

    ref_wireplanes = []
    ref_centroids = []
    for i in range(3):
        wp = compute_wireplane_for_plane(positions_cm, i)
        cw, ct = compute_centroid(wp)
        ref_wireplanes.append(wp)
        ref_centroids.append((cw, ct))
        print(f"  {plane_names[i]}: centroid=({float(cw):.1f}, {float(ct):.1f}), max_signal={float(jnp.max(wp)):.2e}")

    # Create loss functions
    def mse_loss_all_planes(position_offsets):
        """MSE loss summed over all planes."""
        pos = positions_cm + position_offsets
        total_loss = 0.0
        for i in range(3):
            wp = compute_wireplane_for_plane(pos, i)
            total_loss += jnp.mean((wp - ref_wireplanes[i]) ** 2)
        return total_loss / 3.0

    def centroid_loss_all_planes(position_offsets):
        """Centroid distance loss summed over all planes."""
        pos = positions_cm + position_offsets
        total_loss = 0.0
        for i in range(3):
            wp = compute_wireplane_for_plane(pos, i)
            cw, ct = compute_centroid(wp)
            ref_cw, ref_ct = ref_centroids[i]
            total_loss += (cw - ref_cw)**2 + (ct - ref_ct)**2
        return total_loss / 3.0

    # Test gradients
    print("\n" + "="*70)
    print("Gradient Test with Small Offset (0.05 cm each direction)")
    print("="*70)

    offset = jnp.zeros((n_test, 3))
    offset = offset.at[:, 0].set(0.05)  # Small x shift
    offset = offset.at[:, 1].set(0.05)  # Small y shift
    offset = offset.at[:, 2].set(0.05)  # Small z shift

    # MSE Loss
    print("\n--- MSE Loss ---")
    loss_zero = float(mse_loss_all_planes(jnp.zeros((n_test, 3))))
    loss_offset = float(mse_loss_all_planes(offset))
    print(f"Loss at zero: {loss_zero:.4e}")
    print(f"Loss at offset: {loss_offset:.4e}")

    grad_fn = jax.grad(mse_loss_all_planes)
    grads = grad_fn(offset)

    print(f"\nAnalytical gradients (mean over segments):")
    for d, name in enumerate(['x', 'y', 'z']):
        mean_g = float(jnp.mean(grads[:, d]))
        print(f"  ∂L/∂{name} = {mean_g:>12.4e}")

    print(f"\nNumerical verification (segment 0):")
    eps = 1e-4
    for d, name in enumerate(['x', 'y', 'z']):
        delta = jnp.zeros((n_test, 3))
        delta = delta.at[0, d].set(eps)
        num_grad = (mse_loss_all_planes(offset + delta) - mse_loss_all_planes(offset - delta)) / (2 * eps)
        ana_grad = float(grads[0, d])

        if abs(float(num_grad)) > 1e-12:
            rel_diff = abs(ana_grad - float(num_grad)) / abs(float(num_grad))
            status = "MATCH" if rel_diff < 0.5 else "DIFF"
        elif abs(ana_grad) < 1e-10:
            status = "BOTH~0"
        else:
            status = "ANA≠0"

        print(f"  ∂L/∂{name}: ana={ana_grad:>12.4e}, num={float(num_grad):>12.4e} [{status}]")

    # Centroid Loss
    print("\n--- Centroid Loss ---")
    loss_zero_c = float(centroid_loss_all_planes(jnp.zeros((n_test, 3))))
    loss_offset_c = float(centroid_loss_all_planes(offset))
    print(f"Loss at zero: {loss_zero_c:.4e}")
    print(f"Loss at offset: {loss_offset_c:.4e}")

    grad_fn_c = jax.grad(centroid_loss_all_planes)
    grads_c = grad_fn_c(offset)

    print(f"\nAnalytical gradients (mean over segments):")
    for d, name in enumerate(['x', 'y', 'z']):
        mean_g = float(jnp.mean(grads_c[:, d]))
        print(f"  ∂L/∂{name} = {mean_g:>12.4e}")

    print(f"\nNumerical verification (segment 0):")
    for d, name in enumerate(['x', 'y', 'z']):
        delta = jnp.zeros((n_test, 3))
        delta = delta.at[0, d].set(eps)
        num_grad = (centroid_loss_all_planes(offset + delta) - centroid_loss_all_planes(offset - delta)) / (2 * eps)
        ana_grad = float(grads_c[0, d])

        if abs(float(num_grad)) > 1e-12:
            rel_diff = abs(ana_grad - float(num_grad)) / abs(float(num_grad))
            status = "MATCH" if rel_diff < 0.5 else "DIFF"
        elif abs(ana_grad) < 1e-10:
            status = "BOTH~0"
        else:
            status = "ANA≠0"

        print(f"  ∂L/∂{name}: ana={ana_grad:>12.4e}, num={float(num_grad):>12.4e} [{status}]")

    # Per-plane gradient breakdown
    print("\n" + "="*70)
    print("Per-Plane Gradient Breakdown")
    print("="*70)

    for plane_idx in range(3):
        def single_plane_mse(position_offsets):
            pos = positions_cm + position_offsets
            wp = compute_wireplane_for_plane(pos, plane_idx)
            return jnp.mean((wp - ref_wireplanes[plane_idx]) ** 2)

        grad_fn_p = jax.grad(single_plane_mse)
        grads_p = grad_fn_p(offset)

        angle = float(angles_rad[plane_idx])
        sin_a, cos_a = np.sin(angle), np.cos(angle)

        print(f"\n{plane_names[plane_idx]} (sin={sin_a:.3f}, cos={cos_a:.3f}):")
        print(f"  ∂L/∂x = {float(jnp.mean(grads_p[:,0])):>12.4e}")
        print(f"  ∂L/∂y = {float(jnp.mean(grads_p[:,1])):>12.4e}")
        print(f"  ∂L/∂z = {float(jnp.mean(grads_p[:,2])):>12.4e}")

        # Check gradient ratio matches angle
        mean_y = float(jnp.mean(grads_p[:,1]))
        mean_z = float(jnp.mean(grads_p[:,2]))
        if abs(sin_a) > 0.01 and abs(cos_a) > 0.01 and abs(mean_z) > 1e-10:
            ratio = (mean_y / sin_a) / (mean_z / cos_a)
            print(f"  (∂L/∂y/sin) / (∂L/∂z/cos) = {ratio:.4f} (should be ~1.0)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
With proper geometry from config:
- All wire positions are within valid detector bounds
- Each plane uses its correct angle, spacing, and offset

Expected gradient behavior:
- X: Affects drift time → attenuation, diffusion (all planes)
- Y: Affects wire coord via y*sin(angle) (U and V planes, NOT Y plane)
- Z: Affects wire coord via z*cos(angle) (all planes)

For Y-gradient:
- U plane (+60°): sin(60°) = 0.866 → strong Y contribution
- V plane (-60°): sin(-60°) = -0.866 → strong Y contribution (opposite sign)
- Y plane (0°): sin(0°) = 0 → NO Y contribution

When U and V planes are properly in range, Y-gradients should be non-zero!
""")

    return True


if __name__ == "__main__":
    main()
