#!/usr/bin/env python3
"""
Comprehensive gradient test with on-the-fly kernel computation.

This script tests gradients through:
1. Recombination parameters (recomb_A, recomb_B, field_strength, density, w_value)
2. Diffusion parameters (diffusion_long, diffusion_trans) with on-the-fly computation

The key insight is that when kernels are computed on-the-fly (hit path), the
diffusion parameters flow through the Gaussian computations and are differentiable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from physics_params import create_default_params, PhysicsParams, PARAM_INFO
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import calculate_box_model_charge_with_physics_params
from tools.wires import (
    compute_transverse_diffusion,
    compute_longitudinal_diffusion,
    compute_wire_distances,
    accumulate_signals
)
# Attenuation computed inline - no need for import


def test_gradient(name, loss_fn, params, param_names, eps=1e-3):
    """Test analytical vs numerical gradients with detailed reporting."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)

    # Try analytical gradient
    try:
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        analytical_works = True
    except Exception as e:
        print(f"Analytical gradient FAILED: {e}")
        analytical_works = False
        grads = None

    # Numerical comparison
    print(f"\n{'Parameter':<28} {'Analytical':>14} {'Numerical':>14} {'Rel Diff':>12} {'Match':>8}")
    print('-'*80)

    all_match = True
    results = {}

    for param_name in param_names:
        base_val = getattr(params, param_name)
        step = eps * max(abs(base_val), 1e-8)

        params_plus = params._replace(**{param_name: base_val + step})
        params_minus = params._replace(**{param_name: base_val - step})

        loss_plus = float(loss_fn(params_plus))
        loss_minus = float(loss_fn(params_minus))
        num_grad = (loss_plus - loss_minus) / (2 * step)

        if analytical_works:
            ana_grad = float(getattr(grads, param_name))
            if abs(num_grad) > 1e-10:
                rel_diff = abs(ana_grad - num_grad) / abs(num_grad)
                match = "YES" if rel_diff < 0.1 else "NO"
                if rel_diff >= 0.1:
                    all_match = False
            else:
                rel_diff = 0.0 if abs(ana_grad) < 1e-6 else float('inf')
                match = "~0" if abs(ana_grad) < 1e-6 else "CHECK"
            print(f"{param_name:<28} {ana_grad:>14.4e} {num_grad:>14.4e} {rel_diff:>12.2e} {match:>8}")
            results[param_name] = {'analytical': ana_grad, 'numerical': num_grad, 'rel_diff': rel_diff, 'match': match == "YES"}
        else:
            print(f"{param_name:<28} {'FAILED':>14} {num_grad:>14.4e} {'N/A':>12} {'N/A':>8}")
            results[param_name] = {'analytical': None, 'numerical': num_grad, 'rel_diff': None, 'match': False}

    status = "PASS" if analytical_works and all_match else "PARTIAL" if analytical_works else "FAIL"
    print(f"\nStatus: {status}")
    return {'success': analytical_works and all_match, 'results': results, 'grads': grads}


def create_full_simulation_loss(de, dx, positions_cm, detector_config):
    """
    Create a loss function that runs the full simulation with on-the-fly diffusion.

    This implements the "hit path" where diffusion is computed directly.
    """
    drift_velocity = detector_config['drift_velocity_cm_us']
    electron_lifetime_ms = detector_config['electron_lifetime_ms']
    electron_lifetime_us = electron_lifetime_ms * 1000.0  # convert to us
    # Wire spacing from the first plane of first side
    wire_spacing = float(detector_config['wire_spacings_cm'][0, 0])
    time_step_size = detector_config['time_step_size_us']

    # Calculate drift times from x positions
    drift_times = jnp.abs(positions_cm[:, 0]) / drift_velocity

    # Wire plane parameters (simplified - using Y plane at z=0)
    num_wires = 200
    num_time_steps = 500
    K_wire = 5  # Half-width for wire kernel
    K_time = 10  # Half-width for time kernel

    # Wire indices based on z positions
    z_positions = positions_cm[:, 2]
    wire_index_offset = num_wires // 2

    def simulation_loss(physics_params):
        """Full simulation loss with on-the-fly diffusion."""
        # 1. Recombination: energy -> charge
        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)

        # 2. Attenuation
        attenuation = jnp.exp(-drift_times / electron_lifetime_us)
        attenuated_charges = charges * attenuation

        # 3. Compute diffusion sigmas from physics params
        sigma_wire_sq = 2.0 * physics_params.diffusion_trans * drift_times
        sigma_time_sq = 2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times

        # Ensure minimum sigma
        min_sigma_sq = 1e-8
        sigma_wire_sq = jnp.maximum(sigma_wire_sq, min_sigma_sq)
        sigma_time_sq = jnp.maximum(sigma_time_sq, min_sigma_sq)

        sigma_wire = jnp.sqrt(sigma_wire_sq)
        sigma_time = jnp.sqrt(sigma_time_sq)

        # 4. Wire and time indices (discrete - not differentiable)
        center_wire = jnp.floor(z_positions / wire_spacing + wire_index_offset).astype(jnp.int32)
        center_time = jnp.floor(drift_times / time_step_size).astype(jnp.int32)

        # Wire and time offsets from bin centers
        wire_offset = z_positions / wire_spacing + wire_index_offset - center_wire.astype(jnp.float32)
        time_offset = drift_times / time_step_size - center_time.astype(jnp.float32)

        # 5. Build kernel contributions with on-the-fly Gaussian diffusion
        # Create kernel grids
        wire_offsets_k = jnp.arange(-K_wire, K_wire + 1)  # [-K_wire, ..., K_wire]
        time_offsets_k = jnp.arange(-K_time, K_time + 1)  # [-K_time, ..., K_time]

        # Expand for broadcasting: (n_hits, 1), (2K+1,) -> (n_hits, 2K+1)
        wire_distances = (wire_offsets_k[None, :] - wire_offset[:, None]) * wire_spacing
        time_distances = (time_offsets_k[None, :] - time_offset[:, None]) * time_step_size

        # Gaussian diffusion (on-the-fly computation - differentiable!)
        # wire_gauss: shape (n_hits, 2*K_wire+1)
        wire_gauss = jnp.exp(-wire_distances**2 / (2 * sigma_wire_sq[:, None])) / (jnp.sqrt(2 * jnp.pi) * sigma_wire[:, None])
        # time_gauss: shape (n_hits, 2*K_time+1)
        time_gauss = jnp.exp(-time_distances**2 / (2 * sigma_time_sq[:, None])) / (jnp.sqrt(2 * jnp.pi) * sigma_time[:, None])

        # Outer product: (n_hits, 2*K_wire+1, 2*K_time+1)
        kernel_2d = wire_gauss[:, :, None] * time_gauss[:, None, :]

        # Scale by charge: (n_hits, 2*K_wire+1, 2*K_time+1)
        weighted_kernel = kernel_2d * attenuated_charges[:, None, None]

        # 6. Compute wire and time indices for each kernel element
        wire_indices_k = center_wire[:, None] + wire_offsets_k[None, :]  # (n_hits, 2*K_wire+1)
        time_indices_k = center_time[:, None] + time_offsets_k[None, :]  # (n_hits, 2*K_time+1)

        # Clip to valid range
        wire_indices_k = jnp.clip(wire_indices_k, 0, num_wires - 1)
        time_indices_k = jnp.clip(time_indices_k, 0, num_time_steps - 1)

        # 7. Scatter-add to wireplane
        # Flatten for scatter
        n_hits = de.shape[0]
        n_wire_k = 2 * K_wire + 1
        n_time_k = 2 * K_time + 1

        # Create full index arrays for broadcasting
        # wire_indices: (n_hits, n_wire_k) -> (n_hits, n_wire_k, n_time_k)
        # time_indices: (n_hits, n_time_k) -> (n_hits, n_wire_k, n_time_k)
        wire_idx_3d = jnp.broadcast_to(wire_indices_k[:, :, None], (n_hits, n_wire_k, n_time_k))
        time_idx_3d = jnp.broadcast_to(time_indices_k[:, None, :], (n_hits, n_wire_k, n_time_k))

        # Flatten everything
        wire_flat = wire_idx_3d.reshape(-1)
        time_flat = time_idx_3d.reshape(-1)
        values_flat = weighted_kernel.reshape(-1)

        # Initialize and scatter
        wireplane = jnp.zeros((num_wires, num_time_steps))
        wireplane = wireplane.at[wire_flat, time_flat].add(values_flat)

        # 8. Loss: sum of squares (simple loss that exercises all operations)
        return jnp.sum(wireplane ** 2)

    return simulation_loss


def main():
    print("="*70)
    print("JAXTPC Gradient Analysis: On-The-Fly Kernel Computation")
    print("="*70)
    print(f"\nJAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # Load data
    print("\nLoading data...")
    config_path = "../config/cubic_wireplane_config.yaml"
    data_path = "../mpvmpr_20.h5"

    detector_config = generate_detector(config_path)
    step_data = load_particle_step_data(data_path, 0)

    # Use subset for faster testing
    n_test = 500
    de = jnp.asarray(step_data['de'][:n_test], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'][:n_test], dtype=jnp.float32)
    positions_mm = jnp.asarray(step_data['position'][:n_test], dtype=jnp.float32)
    positions_cm = positions_mm / 10.0

    print(f"Using {n_test} particle steps")
    print(f"Total energy deposited: {float(jnp.sum(de)):.4f} MeV")

    params = create_default_params()
    drift_velocity = detector_config['drift_velocity_cm_us']
    drift_times = jnp.abs(positions_cm[:, 0]) / drift_velocity

    results = {}

    # =========================================================================
    # TEST 1: Recombination Only
    # =========================================================================
    def recomb_loss(physics_params):
        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        return jnp.sum(charges ** 2)

    results['recombination'] = test_gradient(
        "Recombination Parameters",
        recomb_loss, params,
        ['recomb_A', 'recomb_B', 'field_strength', 'density', 'w_value']
    )

    # =========================================================================
    # TEST 2: Diffusion Functions Only (On-The-Fly)
    # =========================================================================
    def diffusion_only_loss(physics_params):
        # Compute diffusion sigmas
        sigma_wire_sq = 2.0 * physics_params.diffusion_trans * drift_times
        sigma_time_sq = 2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times

        sigma_wire = jnp.sqrt(jnp.maximum(sigma_wire_sq, 1e-10))
        sigma_time = jnp.sqrt(jnp.maximum(sigma_time_sq, 1e-10))

        return jnp.sum(sigma_wire + sigma_time)

    results['diffusion_only'] = test_gradient(
        "Diffusion Parameters (Direct Computation)",
        diffusion_only_loss, params,
        ['diffusion_long', 'diffusion_trans']
    )

    # =========================================================================
    # TEST 3: Combined Recombination + Diffusion
    # =========================================================================
    def combined_loss(physics_params):
        # Recombination
        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)

        # Diffusion sigmas
        sigma_wire_sq = 2.0 * physics_params.diffusion_trans * drift_times
        sigma_time_sq = 2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times
        sigma_wire = jnp.sqrt(jnp.maximum(sigma_wire_sq, 1e-10))
        sigma_time = jnp.sqrt(jnp.maximum(sigma_time_sq, 1e-10))

        # Combine: charge weighted by inverse spread
        spread = 1.0 / (sigma_wire * sigma_time + 1e-10)
        return jnp.sum(charges * spread)

    results['combined'] = test_gradient(
        "Combined Recombination + Diffusion",
        combined_loss, params,
        ['recomb_A', 'recomb_B', 'diffusion_long', 'diffusion_trans']
    )

    # =========================================================================
    # TEST 4: Full Simulation with On-The-Fly Kernels
    # =========================================================================
    simulation_loss = create_full_simulation_loss(de, dx, positions_cm, detector_config)

    results['full_simulation'] = test_gradient(
        "Full Simulation (On-The-Fly Kernels)",
        simulation_loss, params,
        ['recomb_A', 'recomb_B', 'diffusion_long', 'diffusion_trans']
    )

    # =========================================================================
    # TEST 5: All Recombination Parameters in Full Simulation
    # =========================================================================
    results['full_recomb'] = test_gradient(
        "Full Simulation - All Recombination Params",
        simulation_loss, params,
        ['recomb_A', 'recomb_B', 'field_strength', 'density', 'w_value']
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: Gradient Analysis Results")
    print("="*70)

    print("\nTest Results:")
    print("-"*50)
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"  {test_name:<35}: {status}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    print("""
Differentiable Operations (Gradients Flow Through):
  ✓ Recombination (Box Model): recomb_A, recomb_B, field_strength, density, w_value
  ✓ Diffusion sigma computation: diffusion_long, diffusion_trans
  ✓ On-the-fly Gaussian kernels: exp(-d²/2σ²) where σ depends on params
  ✓ Scatter-add VALUES: wireplane.at[idx].add(values)
  ✓ Attenuation: exp(-t/τ)

Non-Differentiable (Constant w.r.t. params):
  ✗ Wire/time INDEX computation: floor() -> astype(int32)
  ✗ K_wire, K_time: static kernel sizes

Key Insight:
  When computing kernels ON-THE-FLY, diffusion parameters ARE differentiable.
  The Gaussian kernel values depend on σ which depends on diffusion coefficients.
  Gradients flow through the kernel VALUES, not the kernel indices.
""")

    print("="*70)

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/onthefly_gradient_report.txt', 'w') as f:
        f.write("JAXTPC Gradient Analysis: On-The-Fly Kernel Computation\n")
        f.write("="*60 + "\n\n")

        for test_name, result in results.items():
            f.write(f"\n{test_name.upper()}\n")
            f.write("-"*40 + "\n")
            f.write(f"Status: {'PASS' if result['success'] else 'FAIL'}\n\n")

            for param_name, data in result['results'].items():
                f.write(f"  {param_name}:\n")
                f.write(f"    Analytical: {data['analytical']}\n")
                f.write(f"    Numerical:  {data['numerical']}\n")
                if data['rel_diff'] is not None:
                    f.write(f"    Rel Diff:   {data['rel_diff']:.2e}\n")
                f.write(f"    Match:      {data['match']}\n\n")

    print(f"\nReport saved to: results/onthefly_gradient_report.txt")

    return all(r['success'] for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
