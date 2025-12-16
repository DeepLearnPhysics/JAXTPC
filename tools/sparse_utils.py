"""
Sparse utilities for converting between dense, bucket, and truly sparse formats.

Truly sparse format:
    - indices: (N, 2) int32 array with [wire_idx, time_idx] per row
    - values: (N,) float32 array with signal values
"""

import jax.numpy as jnp
import numpy as np


def dense_to_sparse(dense_array, threshold=0.0):
    """
    Convert dense (num_wires, num_time_steps) array to sparse format.

    Args:
        dense_array: (num_wires, num_time_steps) array
        threshold: Only include values with |value| > threshold (keeps sign)

    Returns:
        indices: (N, 2) int32 array with [wire_idx, time_idx] per row
        values: (N,) float32 array with signal values (preserves sign)
    """
    # Use absolute value for threshold check, but keep original signed values
    mask = jnp.abs(dense_array) > threshold
    wire_indices, time_indices = jnp.where(mask)
    indices = jnp.stack([wire_indices, time_indices], axis=1)
    values = dense_array[mask]
    return indices.astype(jnp.int32), values.astype(jnp.float32)


def sparse_buckets_to_sparse(buckets, compact_to_key, num_active,
                              B1, B2, num_wires, num_time_steps,
                              threshold=0.0):
    """
    Convert sparse buckets format directly to truly sparse (indices, values).

    Args:
        buckets: (max_buckets, B1, B2) bucket data
        compact_to_key: (max_buckets,) mapping to bucket keys
        num_active: number of active buckets
        B1, B2: bucket dimensions
        num_wires, num_time_steps: detector dimensions
        threshold: Only include values with |value| > threshold (keeps sign)

    Returns:
        indices: (N, 2) int32 array with [wire_idx, time_idx] per row
        values: (N,) float32 array with signal values (preserves sign)
    """
    NUM_BUCKETS_T = (num_time_steps + B2 - 1) // B2

    all_indices = []
    all_values = []

    for i in range(int(num_active)):
        key = int(compact_to_key[i])
        bucket_w = key // NUM_BUCKETS_T
        bucket_t = key % NUM_BUCKETS_T

        w_start = bucket_w * B1
        t_start = bucket_t * B2

        bucket_data = np.array(buckets[i])
        # Use absolute value for threshold check, but keep original signed values
        mask = np.abs(bucket_data) > threshold
        local_w, local_t = np.where(mask)

        global_w = w_start + local_w
        global_t = t_start + local_t

        # Filter to valid detector range
        valid = (global_w < num_wires) & (global_t < num_time_steps)

        if np.any(valid):
            all_indices.append(np.stack([global_w[valid], global_t[valid]], axis=1))
            all_values.append(bucket_data[mask][valid])

    if all_indices:
        indices = np.concatenate(all_indices, axis=0)
        values = np.concatenate(all_values, axis=0)
    else:
        indices = np.empty((0, 2), dtype=np.int32)
        values = np.empty((0,), dtype=np.float32)

    return jnp.array(indices, dtype=jnp.int32), jnp.array(values, dtype=jnp.float32)


def sparse_to_dense(indices, values, num_wires, num_time_steps):
    """
    Convert sparse (indices, values) format to dense array using .at.add.

    Args:
        indices: (N, 2) int32 array with [wire_idx, time_idx] per row
        values: (N,) float32 array with signal values
        num_wires, num_time_steps: output array dimensions

    Returns:
        dense: (num_wires, num_time_steps) float32 array
    """
    dense = jnp.zeros((num_wires, num_time_steps), dtype=jnp.float32)

    if len(indices) == 0:
        return dense

    wire_idx = indices[:, 0]
    time_idx = indices[:, 1]

    # Use .at.add to handle duplicate indices (sum contributions)
    dense = dense.at[wire_idx, time_idx].add(values)

    return dense
