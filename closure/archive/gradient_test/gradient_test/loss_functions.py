"""
Loss functions for gradient testing.

This module provides MSE and other loss functions for comparing
simulated wireplane signals to reference signals.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Union


def mse_loss_dense(
    predicted: jnp.ndarray,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None
) -> float:
    """
    Compute Mean Squared Error between predicted and target arrays.

    Parameters
    ----------
    predicted : jnp.ndarray
        Predicted signal array (num_wires, num_time_steps).
    target : jnp.ndarray
        Target signal array (num_wires, num_time_steps).
    mask : jnp.ndarray, optional
        Boolean mask for valid elements.

    Returns
    -------
    float
        Mean squared error.
    """
    diff = predicted - target
    if mask is not None:
        diff = jnp.where(mask, diff, 0.0)
        n_valid = jnp.sum(mask)
        return jnp.sum(diff ** 2) / jnp.maximum(n_valid, 1.0)
    else:
        return jnp.mean(diff ** 2)


def mse_loss_sparse(
    predicted_indices: jnp.ndarray,
    predicted_values: jnp.ndarray,
    target_indices: jnp.ndarray,
    target_values: jnp.ndarray,
    num_wires: int,
    num_time_steps: int
) -> float:
    """
    Compute MSE from sparse representations by converting to dense.

    This is less efficient but simpler for gradient testing.

    Parameters
    ----------
    predicted_indices : jnp.ndarray
        Indices of predicted non-zero values (N, 2).
    predicted_values : jnp.ndarray
        Values at predicted indices (N,).
    target_indices : jnp.ndarray
        Indices of target non-zero values (M, 2).
    target_values : jnp.ndarray
        Values at target indices (M,).
    num_wires : int
        Number of wires.
    num_time_steps : int
        Number of time steps.

    Returns
    -------
    float
        Mean squared error.
    """
    # Convert to dense
    predicted_dense = jnp.zeros((num_wires, num_time_steps))
    if len(predicted_values) > 0:
        predicted_dense = predicted_dense.at[
            predicted_indices[:, 0], predicted_indices[:, 1]
        ].add(predicted_values)

    target_dense = jnp.zeros((num_wires, num_time_steps))
    if len(target_values) > 0:
        target_dense = target_dense.at[
            target_indices[:, 0], target_indices[:, 1]
        ].add(target_values)

    return mse_loss_dense(predicted_dense, target_dense)


def total_charge_loss(
    predicted_charges: jnp.ndarray,
    target_charges: jnp.ndarray
) -> float:
    """
    Compute loss based on total charge difference.

    This is useful for testing recombination gradients in isolation.

    Parameters
    ----------
    predicted_charges : jnp.ndarray
        Predicted charge array.
    target_charges : jnp.ndarray
        Target charge array.

    Returns
    -------
    float
        Mean squared error of total charge.
    """
    pred_total = jnp.sum(predicted_charges)
    target_total = jnp.sum(target_charges)
    return (pred_total - target_total) ** 2


def normalized_mse_loss(
    predicted: jnp.ndarray,
    target: jnp.ndarray
) -> float:
    """
    Compute normalized MSE (divided by target variance).

    Parameters
    ----------
    predicted : jnp.ndarray
        Predicted values.
    target : jnp.ndarray
        Target values.

    Returns
    -------
    float
        Normalized MSE.
    """
    mse = jnp.mean((predicted - target) ** 2)
    variance = jnp.var(target)
    return mse / jnp.maximum(variance, 1e-10)


def per_plane_mse_loss(
    predicted_signals: Dict,
    target_signals: Dict,
    weights: Optional[Dict] = None
) -> float:
    """
    Compute weighted MSE loss across all wire planes.

    Parameters
    ----------
    predicted_signals : dict
        Dictionary of predicted signals keyed by (side_idx, plane_idx).
    target_signals : dict
        Dictionary of target signals with same keys.
    weights : dict, optional
        Per-plane weights. If None, equal weights are used.

    Returns
    -------
    float
        Total weighted MSE loss.
    """
    total_loss = 0.0
    n_planes = 0

    for key in predicted_signals:
        if key in target_signals:
            pred = predicted_signals[key]
            target = target_signals[key]

            # Handle sparse format (tuple of indices, values)
            if isinstance(pred, tuple) and len(pred) == 2:
                # For now, just sum the values for a simple loss
                pred_sum = jnp.sum(pred[1])
                target_sum = jnp.sum(target[1])
                plane_loss = (pred_sum - target_sum) ** 2
            else:
                plane_loss = mse_loss_dense(pred, target)

            weight = 1.0
            if weights is not None and key in weights:
                weight = weights[key]

            total_loss += weight * plane_loss
            n_planes += 1

    if n_planes > 0:
        total_loss = total_loss / n_planes

    return total_loss


def charge_sum_loss(charges: jnp.ndarray, target_sum: float) -> float:
    """
    Simple loss based on sum of charges.

    Useful for isolated recombination testing.

    Parameters
    ----------
    charges : jnp.ndarray
        Charge array.
    target_sum : float
        Target sum of charges.

    Returns
    -------
    float
        Squared difference from target.
    """
    return (jnp.sum(charges) - target_sum) ** 2
