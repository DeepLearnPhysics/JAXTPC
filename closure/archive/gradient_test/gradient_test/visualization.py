"""
Visualization utilities for gradient testing.

This module provides plotting functions for loss landscapes,
gradient comparisons, and diagnostic visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional
import os

try:
    from .physics_params import PARAM_INFO
except ImportError:
    from physics_params import PARAM_INFO


def plot_loss_landscape(
    param_values: jnp.ndarray,
    loss_values: jnp.ndarray,
    param_name: str,
    numerical_gradient: Optional[float] = None,
    analytical_gradient: Optional[float] = None,
    save_path: Optional[str] = None,
    title_suffix: str = ""
) -> plt.Figure:
    """
    Plot 1D loss landscape with optional gradient tangent lines.

    Parameters
    ----------
    param_values : jnp.ndarray
        Parameter values (x-axis).
    loss_values : jnp.ndarray
        Loss values (y-axis).
    param_name : str
        Name of the parameter.
    numerical_gradient : float, optional
        Numerical gradient at center point.
    analytical_gradient : float, optional
        Analytical gradient at center point.
    save_path : str, optional
        Path to save the figure.
    title_suffix : str
        Additional text for title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    info = PARAM_INFO.get(param_name, {'name': param_name, 'units': '', 'latex': param_name})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot loss curve
    ax.plot(param_values, loss_values, 'b-', linewidth=2, label='Loss')
    ax.scatter(param_values, loss_values, c='blue', s=30, zorder=5)

    # Mark center point
    center_idx = len(param_values) // 2
    center_val = param_values[center_idx]
    center_loss = loss_values[center_idx]
    ax.scatter([center_val], [center_loss], c='red', s=100, marker='*',
               zorder=10, label='Default value')

    # Plot gradient tangent lines
    x_range = param_values[-1] - param_values[0]
    tangent_x = np.array([center_val - 0.1 * x_range, center_val + 0.1 * x_range])

    if numerical_gradient is not None:
        tangent_y = center_loss + numerical_gradient * (tangent_x - center_val)
        ax.plot(tangent_x, tangent_y, 'g--', linewidth=2,
                label=f'Numerical grad: {numerical_gradient:.2e}')

    if analytical_gradient is not None:
        tangent_y = center_loss + analytical_gradient * (tangent_x - center_val)
        ax.plot(tangent_x, tangent_y, 'm:', linewidth=2,
                label=f'Analytical grad: {analytical_gradient:.2e}')

    ax.set_xlabel(f'{info["name"]} ({info["units"]})', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Loss Landscape: {info["name"]} {title_suffix}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_gradient_comparison(
    comparison_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing analytical and numerical gradients.

    Parameters
    ----------
    comparison_results : dict
        Results from gradient_check().
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    param_names = list(comparison_results.keys())
    n_params = len(param_names)

    analytical = [comparison_results[p]['analytical'] for p in param_names]
    numerical = [comparison_results[p]['numerical'] for p in param_names]

    x = np.arange(n_params)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, analytical, width, label='Analytical', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, numerical, width, label='Numerical', color='green', alpha=0.7)

    # Color bars based on whether they match
    for i, name in enumerate(param_names):
        if not comparison_results[name]['is_close']:
            bars1[i].set_color('red')
            bars2[i].set_color('orange')

    # Labels
    labels = [PARAM_INFO.get(p, {'name': p})['name'] for p in param_names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Gradient Value')
    ax.set_title('Analytical vs Numerical Gradients')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add relative difference text
    for i, name in enumerate(param_names):
        rel_diff = comparison_results[name]['rel_diff']
        y_pos = max(abs(analytical[i]), abs(numerical[i]))
        ax.annotate(f'{rel_diff:.1e}', (x[i], y_pos),
                    textcoords="offset points", xytext=(0, 5),
                    ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_all_loss_landscapes(
    loss_landscapes: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    gradients: Optional[Dict[str, Dict]] = None,
    save_dir: Optional[str] = None
) -> plt.Figure:
    """
    Plot all loss landscapes in a grid.

    Parameters
    ----------
    loss_landscapes : dict
        Dictionary of param_name -> (param_values, loss_values).
    gradients : dict, optional
        Dictionary of param_name -> {'numerical': ..., 'analytical': ...}.
    save_dir : str, optional
        Directory to save individual plots.

    Returns
    -------
    plt.Figure
        Matplotlib figure with grid of subplots.
    """
    n_params = len(loss_landscapes)
    n_cols = 2
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]

    for i, (param_name, (param_values, loss_values)) in enumerate(loss_landscapes.items()):
        ax = axes[i]
        info = PARAM_INFO.get(param_name, {'name': param_name, 'units': ''})

        ax.plot(param_values, loss_values, 'b-', linewidth=2)
        ax.scatter(param_values, loss_values, c='blue', s=20)

        # Mark center
        center_idx = len(param_values) // 2
        ax.scatter([param_values[center_idx]], [loss_values[center_idx]],
                   c='red', s=80, marker='*', zorder=10)

        ax.set_xlabel(f'{info["name"]} ({info["units"]})')
        ax.set_ylabel('Loss')
        ax.set_title(f'{info["name"]}')
        ax.grid(True, alpha=0.3)

        # Add gradient info if available
        if gradients and param_name in gradients:
            grad_info = gradients[param_name]
            text = f"Num: {grad_info.get('numerical', 'N/A'):.2e}"
            if 'analytical' in grad_info:
                text += f"\nAna: {grad_info['analytical']:.2e}"
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Loss Landscapes for Physics Parameters', fontsize=14)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'all_loss_landscapes.png'),
                    dpi=150, bbox_inches='tight')

    return fig


def create_gradient_report(
    gradient_results: Dict,
    loss_landscapes: Dict,
    save_path: str
):
    """
    Create a text report of gradient testing results.

    Parameters
    ----------
    gradient_results : dict
        Results from gradient_check().
    loss_landscapes : dict
        Loss landscape data.
    save_path : str
        Path to save the report.
    """
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("JAXTPC Gradient Testing Report\n")
        f.write("=" * 60 + "\n\n")

        # Analytical gradient status
        f.write("Analytical Gradient Computation:\n")
        f.write("-" * 40 + "\n")
        if gradient_results.get('analytical_success', False):
            f.write("Status: SUCCESS\n")
        else:
            f.write("Status: FAILED\n")
            if gradient_results.get('analytical_error'):
                f.write(f"Error: {gradient_results['analytical_error']}\n")
        f.write("\n")

        # Numerical gradients
        f.write("Numerical Gradients:\n")
        f.write("-" * 40 + "\n")
        for name, grad in gradient_results.get('numerical_grads', {}).items():
            info = PARAM_INFO.get(name, {'name': name})
            f.write(f"  {info['name']:<25}: {grad:>12.6e}\n")
        f.write("\n")

        # Comparison
        if gradient_results.get('comparison'):
            f.write("Gradient Comparison (Analytical vs Numerical):\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Parameter':<20} {'Analytical':>12} {'Numerical':>12} {'Rel Diff':>10} {'Match':>6}\n")
            f.write("-" * 60 + "\n")
            for name, comp in gradient_results['comparison'].items():
                match = "YES" if comp['is_close'] else "NO"
                f.write(f"{name:<20} {comp['analytical']:>12.4e} {comp['numerical']:>12.4e} "
                        f"{comp['rel_diff']:>10.2e} {match:>6}\n")
            f.write("\n")

        # Loss landscape summary
        f.write("Loss Landscape Summary:\n")
        f.write("-" * 40 + "\n")
        for name, (params, losses) in loss_landscapes.items():
            info = PARAM_INFO.get(name, {'name': name})
            f.write(f"\n  {info['name']}:\n")
            f.write(f"    Parameter range: [{float(params[0]):.4e}, {float(params[-1]):.4e}]\n")
            f.write(f"    Loss range: [{float(min(losses)):.4e}, {float(max(losses)):.4e}]\n")
            f.write(f"    Loss at default: {float(losses[len(losses)//2]):.4e}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Report saved to: {save_path}")
