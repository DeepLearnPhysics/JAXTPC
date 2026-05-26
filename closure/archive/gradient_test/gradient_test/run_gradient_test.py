#!/usr/bin/env python3
"""
Main gradient testing script for JAXTPC simulation.

This script tests gradients of the simulation with respect to physics
parameters (diffusion and recombination) to assess differentiability.

Usage:
    python run_gradient_test.py [--test recomb|diffusion|all] [--event N]
"""

import os
import sys
import argparse
import time

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add this directory first so we import from gradient_test/tools, not main tools/
sys.path.insert(0, SCRIPT_DIR)

# Add parent directory for main package imports if needed
sys.path.insert(1, os.path.dirname(SCRIPT_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import from gradient_test package
from physics_params import PhysicsParams, create_default_params, PARAM_INFO
from loss_functions import charge_sum_loss, mse_loss_dense, total_charge_loss
from gradient_utils import (
    gradient_check, loss_landscape_1d,
    compute_numerical_gradient, compute_all_numerical_gradients
)
from visualization import (
    plot_loss_landscape, plot_all_loss_landscapes,
    plot_gradient_comparison, create_gradient_report
)

# Import from local tools copy (gradient_test/tools/)
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import (
    recombine_steps_with_physics_params,
    calculate_box_model_charge_with_physics_params
)


# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH = "../config/cubic_wireplane_config.yaml"
DATA_PATH = "../mpvmpr_20.h5"
EVENT_IDX = 0
RESULTS_DIR = "results"

# Parameters to test
TEST_PARAMS = ['diffusion_long', 'diffusion_trans', 'recomb_A', 'recomb_B']
RECOMB_PARAMS = ['recomb_A', 'recomb_B', 'field_strength', 'density', 'w_value']


# =============================================================================
# Test Functions
# =============================================================================

def test_recombination_gradients(step_data, verbose=True):
    """
    Test gradients through the recombination model only.

    This is the simplest test as recombination is fully differentiable.

    Parameters
    ----------
    step_data : dict
        Particle step data with 'de' and 'dx' arrays.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    dict
        Test results including gradients and loss landscapes.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Recombination Gradients")
    print("=" * 60)

    # Create default parameters
    params = create_default_params()

    # Get reference charge with default parameters
    de = jnp.asarray(step_data['de'], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'], dtype=jnp.float32)

    reference_charges = calculate_box_model_charge_with_physics_params(de, dx, params)
    reference_sum = float(jnp.sum(reference_charges))

    if verbose:
        print(f"\nReference configuration:")
        print(f"  Number of steps: {len(de):,}")
        print(f"  Total energy deposited: {float(jnp.sum(de)):.2f} MeV")
        print(f"  Total collected charge: {reference_sum:.2e} electrons")

    # Define loss function for recombination only
    def recomb_loss(physics_params):
        """Loss = (sum(charges) - reference)^2"""
        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        return (jnp.sum(charges) - reference_sum * 1.01) ** 2  # Target 1% higher

    # Test gradient computation
    if verbose:
        print("\n--- Gradient Check ---")

    # Use eps=1e-3 for numerical gradients (1e-6 is too small and causes precision issues)
    results = gradient_check(
        recomb_loss,
        params,
        param_names=RECOMB_PARAMS,
        eps=1e-3,
        verbose=verbose
    )

    # Compute loss landscapes
    if verbose:
        print("\n--- Computing Loss Landscapes ---")

    loss_landscapes = {}
    for param_name in RECOMB_PARAMS:
        if verbose:
            print(f"  Sweeping {param_name}...")
        param_vals, loss_vals = loss_landscape_1d(
            recomb_loss, params, param_name,
            relative_range=0.3, num_points=21
        )
        loss_landscapes[param_name] = (param_vals, loss_vals)

    results['loss_landscapes'] = loss_landscapes
    results['reference_sum'] = reference_sum

    return results


def test_full_simulation_gradients(step_data, detector_config, verbose=True):
    """
    Test gradients through the full simulation pipeline.

    This is more complex and may encounter non-differentiable operations.

    Parameters
    ----------
    step_data : dict
        Particle step data.
    detector_config : dict
        Detector configuration.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    dict
        Test results.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Full Simulation Gradients (Recombination Stage)")
    print("=" * 60)

    # For now, we test the recombination step which feeds into the simulation
    # Full simulation gradient testing requires more modifications

    params = create_default_params()

    de = jnp.asarray(step_data['de'], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'], dtype=jnp.float32)

    # Reference charges
    reference_charges = calculate_box_model_charge_with_physics_params(de, dx, params)

    # Loss function: MSE from reference
    def simulation_loss(physics_params):
        """MSE loss on charge output."""
        charges = calculate_box_model_charge_with_physics_params(de, dx, physics_params)
        return jnp.mean((charges - reference_charges * 0.95) ** 2)

    if verbose:
        print("\n--- Testing gradients through charge calculation ---")

    results = gradient_check(
        simulation_loss,
        params,
        param_names=['recomb_A', 'recomb_B'],
        eps=1e-3,
        verbose=verbose
    )

    return results


def run_parameter_sensitivity_analysis(step_data, verbose=True):
    """
    Analyze sensitivity of simulation output to parameter changes.

    Parameters
    ----------
    step_data : dict
        Particle step data.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    dict
        Sensitivity analysis results.
    """
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis")
    print("=" * 60)

    params = create_default_params()
    de = jnp.asarray(step_data['de'], dtype=jnp.float32)
    dx = jnp.asarray(step_data['dx'], dtype=jnp.float32)

    # Reference
    ref_charges = calculate_box_model_charge_with_physics_params(de, dx, params)
    ref_sum = float(jnp.sum(ref_charges))

    results = {}
    perturbation = 0.01  # 1% change

    print(f"\nReference total charge: {ref_sum:.4e}")
    print(f"\nSensitivity to 1% parameter change:")
    print("-" * 50)
    print(f"{'Parameter':<25} {'d(charge)/d(param)':<20} {'Relative':<15}")
    print("-" * 50)

    for param_name in RECOMB_PARAMS:
        base_val = getattr(params, param_name)
        delta = base_val * perturbation

        # Perturbed calculation
        perturbed_params = params._replace(**{param_name: base_val + delta})
        perturbed_charges = calculate_box_model_charge_with_physics_params(
            de, dx, perturbed_params
        )
        perturbed_sum = float(jnp.sum(perturbed_charges))

        # Sensitivity
        sensitivity = (perturbed_sum - ref_sum) / delta
        relative_change = (perturbed_sum - ref_sum) / ref_sum

        results[param_name] = {
            'sensitivity': sensitivity,
            'relative_change': relative_change,
            'base_value': base_val,
        }

        info = PARAM_INFO.get(param_name, {'name': param_name})
        print(f"{info['name']:<25} {sensitivity:<20.4e} {relative_change*100:<15.4f}%")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='JAXTPC Gradient Testing')
    parser.add_argument('--test', choices=['recomb', 'full', 'sensitivity', 'all'],
                        default='all', help='Which test to run')
    parser.add_argument('--event', type=int, default=EVENT_IDX,
                        help='Event index to use')
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='Save plots to results directory')
    args = parser.parse_args()

    print("=" * 60)
    print("JAXTPC Gradient Testing Framework")
    print("=" * 60)
    print(f"\nJAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load detector configuration
    print(f"\nLoading detector config from: {CONFIG_PATH}")
    detector_config = generate_detector(CONFIG_PATH)
    if detector_config is None:
        print("ERROR: Failed to load detector configuration")
        return 1

    # Load event data
    print(f"Loading event {args.event} from: {DATA_PATH}")
    try:
        step_data = load_particle_step_data(DATA_PATH, args.event)
        n_steps = len(step_data.get('de', []))
        print(f"Loaded {n_steps:,} particle steps")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return 1

    # Run tests
    all_results = {}

    if args.test in ['recomb', 'all']:
        results = test_recombination_gradients(step_data, verbose=True)
        all_results['recombination'] = results

        # Save plots
        if args.save_plots and 'loss_landscapes' in results:
            print("\n--- Saving Plots ---")

            # Individual loss landscape plots
            for param_name, (param_vals, loss_vals) in results['loss_landscapes'].items():
                numerical_grad = results.get('numerical_grads', {}).get(param_name)
                analytical_grad = None
                if results.get('analytical_grads'):
                    analytical_grad = getattr(results['analytical_grads'], param_name, None)

                save_path = os.path.join(RESULTS_DIR, f'loss_landscape_{param_name}.png')
                plot_loss_landscape(
                    param_vals, loss_vals, param_name,
                    numerical_gradient=numerical_grad,
                    analytical_gradient=analytical_grad,
                    save_path=save_path,
                    title_suffix='(Recombination Test)'
                )
                plt.close()

            # Combined plot
            fig = plot_all_loss_landscapes(
                results['loss_landscapes'],
                gradients=results.get('comparison'),
                save_dir=RESULTS_DIR
            )
            plt.close(fig)

            # Gradient comparison
            if results.get('comparison'):
                save_path = os.path.join(RESULTS_DIR, 'gradient_comparison.png')
                fig = plot_gradient_comparison(results['comparison'], save_path=save_path)
                plt.close(fig)

    if args.test in ['full', 'all']:
        results = test_full_simulation_gradients(step_data, detector_config, verbose=True)
        all_results['full_simulation'] = results

    if args.test in ['sensitivity', 'all']:
        results = run_parameter_sensitivity_analysis(step_data, verbose=True)
        all_results['sensitivity'] = results

    # Save report
    if args.save_plots and all_results:
        report_path = os.path.join(RESULTS_DIR, 'gradient_report.txt')
        with open(report_path, 'w') as f:
            f.write("JAXTPC Gradient Testing Report\n")
            f.write("=" * 60 + "\n\n")

            for test_name, results in all_results.items():
                f.write(f"\n{test_name.upper()} TEST\n")
                f.write("-" * 40 + "\n")

                if 'analytical_success' in results:
                    status = "SUCCESS" if results['analytical_success'] else "FAILED"
                    f.write(f"Analytical gradients: {status}\n")

                if 'numerical_grads' in results:
                    f.write("\nNumerical gradients:\n")
                    for name, grad in results['numerical_grads'].items():
                        f.write(f"  {name}: {grad:.6e}\n")

                if 'comparison' in results and results['comparison']:
                    f.write("\nComparison:\n")
                    for name, comp in results['comparison'].items():
                        match = "MATCH" if comp['is_close'] else "MISMATCH"
                        f.write(f"  {name}: {match} (rel_diff={comp['rel_diff']:.2e})\n")

        print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print("Gradient testing complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
