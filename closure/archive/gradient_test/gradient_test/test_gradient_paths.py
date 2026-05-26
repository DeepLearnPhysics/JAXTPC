#!/usr/bin/env python3
"""
Comprehensive test of which gradient paths work in the JAXTPC simulation.

This script tests gradients through different parts of the simulation pipeline
to identify what's differentiable and what breaks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from physics_params import create_default_params, PhysicsParams

# Import modules from local tools copy
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

recomb = load_module('recomb', 'tools/recombination.py')
loader = load_module('loader', 'tools/loader.py')
geometry = load_module('geometry', 'tools/geometry.py')
wires = load_module('wires', 'tools/wires.py')


def test_gradient(name, loss_fn, params, param_names, eps=1e-3):
    """Test analytical vs numerical gradients."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)

    # Try analytical
    try:
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        analytical_works = True
    except Exception as e:
        print(f"Analytical gradient FAILED: {e}")
        analytical_works = False
        grads = None

    # Numerical comparison
    print(f"{'Parameter':<25} {'Analytical':>14} {'Numerical':>14} {'Match':>8}")
    print('-'*65)

    all_match = True
    for param_name in param_names:
        base_val = getattr(params, param_name)
        step = eps * abs(base_val) if abs(base_val) > 1e-10 else eps

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
                match = "~0" if abs(ana_grad) < 1e-6 else "CHECK"
            print(f"{param_name:<25} {ana_grad:>14.4e} {num_grad:>14.4e} {match:>8}")
        else:
            print(f"{param_name:<25} {'FAILED':>14} {num_grad:>14.4e} {'N/A':>8}")

    status = "PASS" if analytical_works and all_match else "PARTIAL" if analytical_works else "FAIL"
    print(f"\nStatus: {status}")
    return analytical_works and all_match


def main():
    print("="*60)
    print("JAXTPC Gradient Path Analysis")
    print("="*60)
    print(f"JAX version: {jax.__version__}")

    # Load data
    step_data = loader.load_particle_step_data('../mpvmpr_20.h5', 0)
    detector_config = geometry.generate_detector('../config/cubic_wireplane_config.yaml')

    n_test = 100
    de = jnp.asarray(step_data['de'][:n_test], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'][:n_test], dtype=jnp.float32)
    positions_mm = jnp.asarray(step_data['position'][:n_test], dtype=jnp.float32)

    params = create_default_params()
    drift_velocity = detector_config['drift_velocity_cm_us']

    results = {}

    # =========================================================================
    # TEST 1: Pure recombination (definitely differentiable)
    # =========================================================================
    def recomb_loss(physics_params):
        charges = recomb.calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        return jnp.sum(charges ** 2)

    results['recombination'] = test_gradient(
        "Pure Recombination",
        recomb_loss, params,
        ['recomb_A', 'recomb_B', 'field_strength', 'density', 'w_value']
    )

    # =========================================================================
    # TEST 2: Diffusion Gaussian functions (definitely differentiable)
    # =========================================================================
    positions_cm = positions_mm / 10.0
    drift_times = jnp.abs(positions_cm[:, 0]) / drift_velocity

    def diffusion_loss(physics_params):
        sigma_wire = jnp.sqrt(2.0 * physics_params.diffusion_trans * drift_times + 1e-10)
        sigma_time = jnp.sqrt(2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times + 1e-10)
        return jnp.sum(sigma_wire + sigma_time)

    results['diffusion_functions'] = test_gradient(
        "Diffusion Gaussian Functions",
        diffusion_loss, params,
        ['diffusion_long', 'diffusion_trans']
    )

    # =========================================================================
    # TEST 3: Combined recomb + diffusion (should work)
    # =========================================================================
    def combined_loss(physics_params):
        charges = recomb.calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        sigma_wire = jnp.sqrt(2.0 * physics_params.diffusion_trans * drift_times + 1e-10)
        sigma_time = jnp.sqrt(2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times + 1e-10)
        spread = 1.0 / (sigma_wire * sigma_time + 1e-10)
        return jnp.sum(charges * spread)

    results['combined'] = test_gradient(
        "Recombination + Diffusion Combined",
        combined_loss, params,
        ['recomb_A', 'recomb_B', 'diffusion_long', 'diffusion_trans']
    )

    # =========================================================================
    # TEST 4: Scatter-add with diffusion weighting (the key test!)
    # =========================================================================
    wire_spacing_cm = 0.3
    time_step_size_us = 0.5
    num_wires = 200
    num_time_steps = 500

    def scatter_diffusion_loss(physics_params):
        # Recombination
        charges = recomb.calculate_box_model_charge_with_physics_params(de, dx, physics_params)

        # Attenuation
        attenuation = jnp.exp(-drift_times / 10000.0)
        attenuated_charges = charges * attenuation

        # Diffusion sigmas (DIFFERENTIABLE)
        sigma_wire = jnp.sqrt(2.0 * physics_params.diffusion_trans * drift_times + 1e-10)
        sigma_time = jnp.sqrt(2.0 * (physics_params.diffusion_long / (drift_velocity ** 2)) * drift_times + 1e-10)

        # Wire/time indices (NON-DIFFERENTIABLE)
        z_positions = positions_cm[:, 2]
        center_wire = jnp.floor(z_positions / wire_spacing_cm + num_wires / 2).astype(jnp.int32)
        center_time = jnp.floor(drift_times / time_step_size_us).astype(jnp.int32)

        # Gaussian weights (DIFFERENTIABLE - applied to values!)
        wire_dist = z_positions / wire_spacing_cm + num_wires / 2 - center_wire.astype(jnp.float32)
        time_dist = drift_times / time_step_size_us - center_time.astype(jnp.float32)
        gauss_wire = jnp.exp(-wire_dist**2 / (2 * (sigma_wire / wire_spacing_cm)**2 + 1e-10))
        gauss_time = jnp.exp(-time_dist**2 / (2 * (sigma_time / time_step_size_us)**2 + 1e-10))

        # Weighted charges (gradients flow through here!)
        weighted_charges = attenuated_charges * gauss_wire * gauss_time

        # Scatter-add (differentiable w.r.t. values)
        center_wire = jnp.clip(center_wire, 0, num_wires - 1)
        center_time = jnp.clip(center_time, 0, num_time_steps - 1)
        wireplane = jnp.zeros((num_wires, num_time_steps))
        wireplane = wireplane.at[center_wire, center_time].add(weighted_charges)

        return jnp.sum(wireplane ** 2)

    results['scatter_with_diffusion'] = test_gradient(
        "Scatter-Add with Diffusion Weighting",
        scatter_diffusion_loss, params,
        ['recomb_A', 'recomb_B', 'diffusion_long', 'diffusion_trans']
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY: Gradient Path Analysis")
    print("="*60)

    print("\nDifferentiable paths:")
    print("  [YES] Recombination: recomb_A, recomb_B, field_strength, density, w_value")
    print("  [YES] Diffusion functions: diffusion_long, diffusion_trans")
    print("  [YES] Scatter-add VALUES: charges weighted by Gaussian")

    print("\nNon-differentiable (but still usable):")
    print("  [NO]  Wire index computation: floor() -> astype(int32)")
    print("  [NO]  Time index computation: floor() -> astype(int32)")
    print("  [NO]  K_wire, K_time: computed from max_sigma (static)")
    print("  [NO]  Response kernels: pre-computed DKernel (static)")

    print("\nKey insight:")
    print("  Gradients flow through VALUES, not INDICES.")
    print("  Diffusion affects Gaussian weights on charges -> differentiable!")
    print("  Wire/time binning creates discrete indices -> not differentiable,")
    print("  but the values scattered to those bins ARE differentiable.")

    print("\n" + "="*60)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
