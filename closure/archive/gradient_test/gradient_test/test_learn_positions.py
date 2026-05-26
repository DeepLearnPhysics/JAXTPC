#!/usr/bin/env python3
"""
Learn positions via gradient descent - start 5cm away.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from physics_params import create_default_params
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import calculate_box_model_charge_with_physics_params


def main():
    print("="*60)
    print("Learning Positions - Starting 1cm Away")
    print("="*60)

    config_path = "../config/cubic_wireplane_config.yaml"
    data_path = "../mpvmpr_20.h5"

    detector_config = generate_detector(config_path)
    step_data = load_particle_step_data(data_path, 0)

    n_segments = 50
    de = jnp.asarray(step_data['de'][:n_segments], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'][:n_segments], dtype=jnp.float32)
    true_positions = jnp.asarray(step_data['position'][:n_segments], dtype=jnp.float32) / 10.0

    drift_velocity = detector_config['drift_velocity_cm_us']
    electron_lifetime_us = detector_config['electron_lifetime_ms'] * 1000.0
    time_step_size = detector_config['time_step_size_us']
    num_time_steps = detector_config['num_time_steps']

    angles_rad = detector_config['angles_rad'][0]
    wire_spacings_cm = detector_config['wire_spacings_cm'][0]
    index_offsets = detector_config['index_offsets'][0]
    num_wires_actual = detector_config['num_wires_actual'][0]

    physics_params = create_default_params()
    K_wire, K_time = 3, 5

    def compute_wireplane(positions, plane_idx):
        n_seg = positions.shape[0]
        angle = angles_rad[plane_idx]
        wire_spacing = float(wire_spacings_cm[plane_idx])
        wire_offset = float(index_offsets[plane_idx])
        num_wires = int(num_wires_actual[plane_idx])

        drift_times = jnp.abs(positions[:, 0]) / drift_velocity
        wire_coord = positions[:, 1] * jnp.sin(angle) + positions[:, 2] * jnp.cos(angle)

        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        attenuated_charges = charges * jnp.exp(-drift_times / electron_lifetime_us)

        sigma_wire = jnp.sqrt(2.0 * physics_params.diffusion_trans * drift_times + 1e-12)
        sigma_time = jnp.sqrt(2.0 * (physics_params.diffusion_long / drift_velocity**2) * drift_times + 1e-12)

        wire_pos = wire_coord / wire_spacing + wire_offset
        center_wire = jnp.floor(wire_pos).astype(jnp.int32)
        center_time = jnp.floor(drift_times / time_step_size).astype(jnp.int32)

        wire_subbin = wire_pos - center_wire.astype(jnp.float32)
        time_subbin = drift_times / time_step_size - center_time.astype(jnp.float32)

        wire_k = jnp.arange(-K_wire, K_wire + 1)
        time_k = jnp.arange(-K_time, K_time + 1)

        wire_dist = (wire_k[None, :] - wire_subbin[:, None]) * wire_spacing
        time_dist = (time_k[None, :] - time_subbin[:, None]) * time_step_size

        wire_gauss = jnp.exp(-wire_dist**2 / (2 * sigma_wire[:, None]**2 + 1e-12))
        time_gauss = jnp.exp(-time_dist**2 / (2 * sigma_time[:, None]**2 + 1e-12))

        kernel = wire_gauss[:, :, None] * time_gauss[:, None, :] * attenuated_charges[:, None, None]

        n_w, n_t = 2*K_wire+1, 2*K_time+1
        w_idx = jnp.clip(center_wire[:, None] + wire_k, 0, num_wires-1)
        t_idx = jnp.clip(center_time[:, None] + time_k, 0, num_time_steps-1)

        w_3d = jnp.broadcast_to(w_idx[:, :, None], (n_seg, n_w, n_t))
        t_3d = jnp.broadcast_to(t_idx[:, None, :], (n_seg, n_w, n_t))

        wp = jnp.zeros((num_wires, num_time_steps))
        wp = wp.at[w_3d.reshape(-1), t_3d.reshape(-1)].add(kernel.reshape(-1))
        return wp

    target_wps = [compute_wireplane(true_positions, i) for i in range(3)]

    def loss_fn(pos):
        return sum(jnp.mean((compute_wireplane(pos, i) - target_wps[i])**2) for i in range(3)) / 3.0

    # Start 1cm away
    np.random.seed(42)
    init_offset = jnp.array(np.random.randn(n_segments, 3) * 1.0, dtype=jnp.float32)
    positions = true_positions + init_offset

    init_err = float(jnp.mean(jnp.abs(init_offset)))
    print(f"\nSegments: {n_segments}")
    print(f"Initial mean error: {init_err:.2f} cm ({init_err*10:.1f} mm)")

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(positions)

    print(f"\n{'Iter':>5} {'Loss':>12} {'Mean Err':>10} {'Max Err':>10}")
    print("-"*42)

    for i in range(1000):
        loss, grads = jax.value_and_grad(loss_fn)(positions)
        updates, opt_state = optimizer.update(grads, opt_state)
        positions = optax.apply_updates(positions, updates)

        if i % 100 == 0 or i == 999:
            err = positions - true_positions
            print(f"{i:>5} {float(loss):>12.2e} {float(jnp.mean(jnp.abs(err))):>10.3f} {float(jnp.max(jnp.abs(err))):>10.3f}")

    final_err = positions - true_positions
    final_mean = float(jnp.mean(jnp.abs(final_err)))
    improvement = (init_err - final_mean) / init_err * 100

    print(f"\nResult: {init_err:.2f} cm → {final_mean:.3f} cm ({improvement:.1f}% improvement)")


if __name__ == "__main__":
    main()
