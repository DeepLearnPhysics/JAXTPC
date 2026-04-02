"""
Output format conversion for JAXTPC simulation results.

Converts between the three output formats produced by DetectorSimulator:
    - dense: (num_wires, num_time) arrays
    - bucketed: 5-tuple (buckets, num_active, compact_to_key, B1, B2)
    - wire_sparse: 3-tuple (active_signals, wire_indices, n_active)

Two target formats for downstream use:
    - dense: {(vol, plane): (W, T) ndarray}
    - sparse: {(vol, plane): (wire, time, values) ndarrays}
"""

import numpy as np
from tools.wires import sparse_buckets_to_dense


def to_dense(response_signals, config):
    """Convert any output format to dense (W, T) arrays.

    Parameters
    ----------
    response_signals : dict
        From process_event(). Keyed by (vol_idx, plane_idx).
    config : SimConfig

    Returns
    -------
    dict mapping (vol, plane) to (num_wires, num_time_steps) ndarray.
    """
    output = {}
    for (vol_idx, plane_idx), signal in response_signals.items():
        num_wires = config.volumes[vol_idx].num_wires[plane_idx]
        num_time = config.num_time_steps

        if isinstance(signal, tuple) and len(signal) == 5:
            buckets, num_active, compact_to_key, B1, B2 = signal
            dense = sparse_buckets_to_dense(
                buckets, compact_to_key, num_active,
                int(B1), int(B2), num_wires, num_time,
                buckets.shape[0])
            output[(vol_idx, plane_idx)] = np.asarray(dense)

        elif isinstance(signal, tuple) and len(signal) == 3:
            active_signals, wire_indices, n_active = signal
            n = int(n_active)
            dense = np.zeros((num_wires, num_time), dtype=np.float32)
            wire_idx = np.asarray(wire_indices[:n])
            active = np.asarray(active_signals[:n])
            for i in range(n):
                w = int(wire_idx[i])
                if 0 <= w < num_wires:
                    dense[w] = active[i, :num_time]
            output[(vol_idx, plane_idx)] = dense

        else:
            output[(vol_idx, plane_idx)] = np.asarray(signal)

    return output


def to_sparse(response_signals, config, threshold_adc=0.0):
    """Convert any output format to sparse (wire, time, values) arrays.

    Parameters
    ----------
    response_signals : dict
        From process_event(). Any format.
    config : SimConfig
    threshold_adc : float
        Minimum absolute value to keep. 0 keeps all nonzero pixels.

    Returns
    -------
    dict mapping (vol, plane) to dict with 'wire', 'time', 'values'.
    """
    dense = to_dense(response_signals, config)

    output = {}
    for key, arr in dense.items():
        if threshold_adc > 0:
            mask = np.abs(arr) >= threshold_adc
        else:
            mask = arr != 0

        wire_idx, time_idx = np.where(mask)
        values = arr[mask].astype(np.float32)

        output[key] = {
            'wire': wire_idx.astype(np.int32),
            'time': time_idx.astype(np.int32),
            'values': values,
        }

    return output
