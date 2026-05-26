"""
Gradient utilities for testing and comparing gradients.

This module provides functions for computing analytical and numerical
gradients, and comparing them to assess gradient accuracy.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple, Optional, Any
from functools import partial

try:
    from .physics_params import PhysicsParams, PARAM_INFO
except ImportError:
    from physics_params import PhysicsParams, PARAM_INFO


def compute_numerical_gradient(
    loss_fn: Callable,
    params: PhysicsParams,
    param_name: str,
    eps: float = 1e-4,
    relative: bool = True,
    *args,
    **kwargs
) -> float:
    """
    Compute numerical gradient using central finite differences.

    Parameters
    ----------
    loss_fn : Callable
        Loss function that takes PhysicsParams as first argument.
    params : PhysicsParams
        Current parameter values.
    param_name : str
        Name of the parameter to differentiate with respect to.
    eps : float
        Finite difference step size. If relative=True, this is relative
        to the parameter value.
    relative : bool
        If True, use relative step size (eps * |param_value|).
    *args, **kwargs
        Additional arguments to pass to loss_fn.

    Returns
    -------
    float
        Numerical gradient estimate.
    """
    current_value = getattr(params, param_name)

    # Compute step size
    if relative:
        step = eps * max(abs(current_value), 1e-8)
    else:
        step = eps

    # Forward step
    params_plus = params._replace(**{param_name: current_value + step})
    loss_plus = loss_fn(params_plus, *args, **kwargs)

    # Backward step
    params_minus = params._replace(**{param_name: current_value - step})
    loss_minus = loss_fn(params_minus, *args, **kwargs)

    # Central difference
    gradient = (loss_plus - loss_minus) / (2 * step)

    return float(gradient)


def compute_all_numerical_gradients(
    loss_fn: Callable,
    params: PhysicsParams,
    eps: float = 1e-4,
    param_names: Optional[list] = None,
    *args,
    **kwargs
) -> Dict[str, float]:
    """
    Compute numerical gradients for all (or specified) parameters.

    Parameters
    ----------
    loss_fn : Callable
        Loss function.
    params : PhysicsParams
        Current parameters.
    eps : float
        Finite difference step size.
    param_names : list, optional
        List of parameter names to compute. If None, computes all.
    *args, **kwargs
        Additional arguments to loss_fn.

    Returns
    -------
    dict
        Dictionary of parameter name -> gradient value.
    """
    if param_names is None:
        param_names = list(params._fields)

    gradients = {}
    for name in param_names:
        gradients[name] = compute_numerical_gradient(
            loss_fn, params, name, eps, *args, **kwargs
        )

    return gradients


def try_analytical_gradient(
    loss_fn: Callable,
    params: PhysicsParams,
    *args,
    **kwargs
) -> Tuple[Optional[PhysicsParams], Optional[Exception]]:
    """
    Try to compute analytical gradient using JAX.

    Parameters
    ----------
    loss_fn : Callable
        Loss function that takes PhysicsParams as first argument.
    params : PhysicsParams
        Current parameter values.
    *args, **kwargs
        Additional arguments to pass to loss_fn.

    Returns
    -------
    tuple
        (gradients, None) if successful, (None, exception) if failed.
    """
    try:
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(params, *args, **kwargs)
        return gradients, None
    except Exception as e:
        return None, e


def compare_gradients(
    analytical_grads: PhysicsParams,
    numerical_grads: Dict[str, float],
    param_names: Optional[list] = None,
    rtol: float = 1e-2,
    atol: float = 1e-6
) -> Dict[str, Dict]:
    """
    Compare analytical and numerical gradients.

    Parameters
    ----------
    analytical_grads : PhysicsParams
        Analytical gradients from jax.grad.
    numerical_grads : dict
        Numerical gradients from finite differences.
    param_names : list, optional
        Parameters to compare. If None, compares all.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    Returns
    -------
    dict
        Comparison results for each parameter.
    """
    if param_names is None:
        param_names = list(numerical_grads.keys())

    results = {}
    for name in param_names:
        analytical = getattr(analytical_grads, name)
        numerical = numerical_grads[name]

        abs_diff = abs(analytical - numerical)
        rel_diff = abs_diff / max(abs(numerical), 1e-10)

        is_close = jnp.allclose(
            jnp.array([analytical]),
            jnp.array([numerical]),
            rtol=rtol,
            atol=atol
        )

        results[name] = {
            'analytical': float(analytical),
            'numerical': float(numerical),
            'abs_diff': float(abs_diff),
            'rel_diff': float(rel_diff),
            'is_close': bool(is_close),
        }

    return results


def gradient_check(
    loss_fn: Callable,
    params: PhysicsParams,
    param_names: Optional[list] = None,
    eps: float = 1e-4,
    rtol: float = 1e-2,
    atol: float = 1e-6,
    verbose: bool = True,
    *args,
    **kwargs
) -> Dict:
    """
    Comprehensive gradient check comparing analytical and numerical.

    Parameters
    ----------
    loss_fn : Callable
        Loss function.
    params : PhysicsParams
        Current parameters.
    param_names : list, optional
        Parameters to check.
    eps : float
        Finite difference step size.
    rtol, atol : float
        Tolerances for comparison.
    verbose : bool
        Whether to print results.
    *args, **kwargs
        Additional arguments to loss_fn.

    Returns
    -------
    dict
        Results including:
        - 'analytical_success': bool
        - 'analytical_error': Exception or None
        - 'numerical_grads': dict
        - 'analytical_grads': PhysicsParams or None
        - 'comparison': dict or None
    """
    if param_names is None:
        param_names = ['diffusion_long', 'diffusion_trans', 'recomb_A', 'recomb_B']

    results = {
        'analytical_success': False,
        'analytical_error': None,
        'numerical_grads': {},
        'analytical_grads': None,
        'comparison': None,
    }

    # Compute numerical gradients first (always works)
    if verbose:
        print("Computing numerical gradients...")

    results['numerical_grads'] = compute_all_numerical_gradients(
        loss_fn, params, eps, param_names, *args, **kwargs
    )

    if verbose:
        print("Numerical gradients:")
        for name, grad in results['numerical_grads'].items():
            info = PARAM_INFO.get(name, {'name': name, 'units': ''})
            print(f"  {info['name']}: {grad:.6e}")

    # Try analytical gradients
    if verbose:
        print("\nAttempting analytical gradients...")

    analytical_grads, error = try_analytical_gradient(
        loss_fn, params, *args, **kwargs
    )

    if error is None:
        results['analytical_success'] = True
        results['analytical_grads'] = analytical_grads

        if verbose:
            print("Analytical gradients computed successfully!")

        # Compare
        results['comparison'] = compare_gradients(
            analytical_grads, results['numerical_grads'],
            param_names, rtol, atol
        )

        if verbose:
            print("\nGradient comparison:")
            print(f"{'Parameter':<20} {'Analytical':>12} {'Numerical':>12} {'Rel Diff':>10} {'OK?':>5}")
            print("-" * 65)
            for name, comp in results['comparison'].items():
                ok_str = "YES" if comp['is_close'] else "NO"
                print(f"{name:<20} {comp['analytical']:>12.4e} {comp['numerical']:>12.4e} "
                      f"{comp['rel_diff']:>10.2e} {ok_str:>5}")

    else:
        results['analytical_error'] = error
        if verbose:
            print(f"Analytical gradient failed: {error}")

    return results


def loss_landscape_1d(
    loss_fn: Callable,
    params: PhysicsParams,
    param_name: str,
    relative_range: float = 0.5,
    num_points: int = 21,
    *args,
    **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute loss values over a 1D parameter sweep.

    Parameters
    ----------
    loss_fn : Callable
        Loss function.
    params : PhysicsParams
        Base parameters.
    param_name : str
        Parameter to sweep.
    relative_range : float
        Range as fraction of default value.
    num_points : int
        Number of points in sweep.
    *args, **kwargs
        Additional arguments to loss_fn.

    Returns
    -------
    tuple
        (param_values, loss_values) arrays.
    """
    base_value = getattr(params, param_name)
    scales = jnp.linspace(1.0 - relative_range, 1.0 + relative_range, num_points)
    param_values = base_value * scales

    loss_values = []
    for val in param_values:
        test_params = params._replace(**{param_name: float(val)})
        loss = loss_fn(test_params, *args, **kwargs)
        loss_values.append(float(loss))

    return jnp.array(param_values), jnp.array(loss_values)
