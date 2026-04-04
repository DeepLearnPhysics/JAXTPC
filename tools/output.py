"""
Output format conversion for JAXTPC simulation results.

Three user-facing output formats:
    - dense: full arrays — (W, T) for wire, (py, pz, T) for pixel
    - sparse: thresholded (coords, values) as flat numpy arrays
    - bucketed: raw tile tuples from accumulation (no conversion)

Handles all internal formats:
    - dense array (2D wire or 3D pixel)
    - wire bucketed 5-tuple (buckets, num_active, ctk, B1, B2)
    - pixel bucketed 6-tuple (buckets, num_active, ctk, B1, B2, B3)
    - wire sparse 3-tuple (active_signals, wire_indices, n_active)
"""

import warnings
import numpy as np
from tools.wires import sparse_buckets_to_dense, sparse_pixel_buckets_to_dense


def _detect_format(signal):
    """Detect the output format of a signal."""
    if isinstance(signal, tuple):
        if len(signal) == 6:
            return 'pixel_bucketed'
        elif len(signal) == 5:
            return 'wire_bucketed'
        elif len(signal) == 3:
            return 'wire_sparse'
    arr = np.asarray(signal) if not isinstance(signal, np.ndarray) else signal
    if arr.ndim == 3:
        return 'pixel_dense'
    return 'wire_dense'


def to_dense(response_signals, config):
    """Convert any output format to dense arrays.

    Parameters
    ----------
    response_signals : dict
        From process_event(). Keyed by (vol_idx, plane_idx).
    config : SimConfig

    Returns
    -------
    dict mapping (vol, plane) to numpy arrays.
        Wire: (num_wires, num_time_steps)
        Pixel: (num_py, num_pz, num_time_steps)
    """
    output = {}
    for (vol_idx, plane_idx), signal in response_signals.items():
        vol = config.volumes[vol_idx]
        num_time = config.num_time_steps
        fmt = _detect_format(signal)

        if fmt == 'wire_bucketed':
            buckets, num_active, ctk, B1, B2 = signal
            num_wires = vol.num_wires[plane_idx]
            dense = sparse_buckets_to_dense(
                buckets, ctk, num_active,
                int(B1), int(B2), num_wires, num_time,
                buckets.shape[0])
            output[(vol_idx, plane_idx)] = np.asarray(dense)

        elif fmt == 'pixel_bucketed':
            buckets, num_active, ctk, B1, B2, B3 = signal
            num_py, num_pz = vol.pixel_shape
            if num_py * num_pz * num_time > 100_000_000:
                warnings.warn(
                    f"Pixel to_dense for volume {vol_idx}: "
                    f"({num_py}x{num_pz}x{num_time}) = "
                    f"{num_py*num_pz*num_time*4/1e9:.1f} GB. "
                    f"Consider using to_sparse instead.")
            dense = sparse_pixel_buckets_to_dense(
                buckets, ctk, num_active,
                int(B1), int(B2), int(B3),
                num_py, num_pz, num_time,
                buckets.shape[0])
            output[(vol_idx, plane_idx)] = np.asarray(dense)

        elif fmt == 'wire_sparse':
            active_signals, wire_indices, n_active = signal
            num_wires = vol.num_wires[plane_idx]
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
            # Dense (wire 2D or pixel 3D) — just convert
            output[(vol_idx, plane_idx)] = np.asarray(signal)

    return output


def _sparse_from_wire_bucketed(signal, vol, plane_idx, num_time, threshold):
    """Extract sparse entries directly from wire bucketed 5-tuple."""
    buckets, num_active, ctk, B1, B2 = signal
    B1, B2 = int(B1), int(B2)
    na = int(num_active)
    num_wires = vol.num_wires[plane_idx]

    NUM_BT = (num_time + B2 - 1) // B2

    # Slice active tiles and move to numpy
    bk = np.asarray(buckets[:na])      # (na, B1, B2)
    keys = np.asarray(ctk[:na])        # (na,)

    # Decode tile origins
    bw = keys // NUM_BT               # (na,)
    bt = keys % NUM_BT                # (na,)
    w_start = bw * B1                  # (na,)
    t_start = bt * B2                  # (na,)

    # Global coordinates via broadcast: (na, B1, B2)
    full_shape = (na, B1, B2)
    global_w = np.broadcast_to(
        w_start[:, None, None] + np.arange(B1)[None, :, None], full_shape)
    global_t = np.broadcast_to(
        t_start[:, None, None] + np.arange(B2)[None, None, :], full_shape)

    flat_w = global_w.ravel()
    flat_t = global_t.ravel()
    flat_v = bk.ravel()

    # Mask: in bounds and above threshold
    valid = ((flat_w >= 0) & (flat_w < num_wires) &
             (flat_t >= 0) & (flat_t < num_time) &
             (np.abs(flat_v) >= threshold))

    return {
        'wire': flat_w[valid].astype(np.int32),
        'time': flat_t[valid].astype(np.int32),
        'values': flat_v[valid].astype(np.float32),
    }


def _sparse_from_pixel_bucketed(signal, vol, num_time, threshold):
    """Extract sparse entries directly from pixel bucketed 6-tuple."""
    buckets, num_active, ctk, B1, B2, B3 = signal
    B1, B2, B3 = int(B1), int(B2), int(B3)
    na = int(num_active)
    num_py, num_pz = vol.pixel_shape

    NUM_BPZ = (num_pz + B2 - 1) // B2
    NUM_BT = (num_time + B3 - 1) // B3

    bk = np.asarray(buckets[:na])      # (na, B1, B2, B3)
    keys = np.asarray(ctk[:na])        # (na,)

    # Decode tile origins
    bpy = keys // (NUM_BPZ * NUM_BT)
    remainder = keys % (NUM_BPZ * NUM_BT)
    bpz = remainder // NUM_BT
    bt = remainder % NUM_BT
    py_start = bpy * B1
    pz_start = bpz * B2
    t_start = bt * B3

    # Global coordinates via broadcast: (na, B1, B2, B3)
    full_shape = (na, B1, B2, B3)
    global_py = np.broadcast_to(
        py_start[:, None, None, None] + np.arange(B1)[None, :, None, None], full_shape)
    global_pz = np.broadcast_to(
        pz_start[:, None, None, None] + np.arange(B2)[None, None, :, None], full_shape)
    global_t = np.broadcast_to(
        t_start[:, None, None, None] + np.arange(B3)[None, None, None, :], full_shape)

    flat_py = global_py.ravel()
    flat_pz = global_pz.ravel()
    flat_t = global_t.ravel()
    flat_v = bk.ravel()

    valid = ((flat_py >= 0) & (flat_py < num_py) &
             (flat_pz >= 0) & (flat_pz < num_pz) &
             (flat_t >= 0) & (flat_t < num_time) &
             (np.abs(flat_v) >= threshold))

    return {
        'pixel_y': flat_py[valid].astype(np.int32),
        'pixel_z': flat_pz[valid].astype(np.int32),
        'time': flat_t[valid].astype(np.int32),
        'values': flat_v[valid].astype(np.float32),
    }


def to_sparse(response_signals, config, threshold_adc=0.0):
    """Convert any output format to sparse coordinate arrays.

    Extracts directly from bucketed format without densifying.

    Parameters
    ----------
    response_signals : dict
        From process_event(). Any format.
    config : SimConfig
    threshold_adc : float
        Minimum absolute value to keep. 0 keeps all nonzero.

    Returns
    -------
    dict mapping (vol, plane) to dict with coordinate arrays.
        Wire: {'wire': int32, 'time': int32, 'values': float32}
        Pixel: {'pixel_y': int32, 'pixel_z': int32, 'time': int32, 'values': float32}
    """
    thresh = threshold_adc if threshold_adc > 0 else 1e-30

    output = {}
    for (vol_idx, plane_idx), signal in response_signals.items():
        vol = config.volumes[vol_idx]
        num_time = config.num_time_steps
        fmt = _detect_format(signal)

        if fmt == 'wire_bucketed':
            output[(vol_idx, plane_idx)] = _sparse_from_wire_bucketed(
                signal, vol, plane_idx, num_time, thresh)

        elif fmt == 'pixel_bucketed':
            output[(vol_idx, plane_idx)] = _sparse_from_pixel_bucketed(
                signal, vol, num_time, thresh)

        elif fmt == 'wire_sparse':
            active_signals, wire_indices, n_active = signal
            n = int(n_active)
            active = np.asarray(active_signals[:n])
            wires = np.asarray(wire_indices[:n])
            # Expand per-wire time series to flat sparse
            w_list, t_list, v_list = [], [], []
            for i in range(n):
                row = active[i, :num_time]
                mask = np.abs(row) >= thresh
                t_idx = np.where(mask)[0]
                if len(t_idx) > 0:
                    w_list.append(np.full(len(t_idx), int(wires[i]), dtype=np.int32))
                    t_list.append(t_idx.astype(np.int32))
                    v_list.append(row[t_idx].astype(np.float32))
            if w_list:
                output[(vol_idx, plane_idx)] = {
                    'wire': np.concatenate(w_list),
                    'time': np.concatenate(t_list),
                    'values': np.concatenate(v_list),
                }
            else:
                output[(vol_idx, plane_idx)] = {
                    'wire': np.array([], dtype=np.int32),
                    'time': np.array([], dtype=np.int32),
                    'values': np.array([], dtype=np.float32),
                }

        elif fmt == 'pixel_dense':
            arr = np.asarray(signal)
            mask = np.abs(arr) >= thresh
            py, pz, t = np.where(mask)
            output[(vol_idx, plane_idx)] = {
                'pixel_y': py.astype(np.int32),
                'pixel_z': pz.astype(np.int32),
                'time': t.astype(np.int32),
                'values': arr[mask].astype(np.float32),
            }

        else:
            # wire_dense
            arr = np.asarray(signal)
            mask = np.abs(arr) >= thresh
            wire_idx, time_idx = np.where(mask)
            output[(vol_idx, plane_idx)] = {
                'wire': wire_idx.astype(np.int32),
                'time': time_idx.astype(np.int32),
                'values': arr[mask].astype(np.float32),
            }

    return output
