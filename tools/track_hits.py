"""
Track hit labeling module for LArTPC simulation.

This module provides functions to track which particles (tracks) contribute
to each wire signal in the detector simulation. It uses a K_wire x K_time
neighbor system to aggregate charge deposits and determine the dominant
contributor at each (wire, time) location.

The track labeling runs in parallel with the main simulation to provide
particle attribution data for physics analysis.
"""

import jax
import jax.numpy as jnp
from functools import partial

from tools.wires import (
    compute_angular_scaling_vmap,
    compute_deposit_wire_angles_vmap,
    prepare_deposit_with_diffusion,
    prepare_pixel_deposit_with_diffusion,
)


@partial(jax.jit, static_argnames=['max_tracks', 'max_wires', 'max_time', 'max_keys'])
def group_hits_by_track(wire_time_indices, track_ids, charge_deposits,
                        min_charge_threshold=0.0,
                        max_tracks=10000, max_wires=2000, max_time=2000,
                        max_keys=1000000):
    """
    Aggregate charge deposits by track and location for LArTPC simulation data.

    Standalone function for direct use outside the simulation pipeline.
    NOT used by create_track_hits_fn (which uses the chunk-merge path instead).

    IMPORTANT: All static sizes must be set correctly by the caller:
        - max_tracks: must be >= number of unique track IDs in the input
        - max_wires: must be >= max wire index in wire_time_indices
        - max_time: must be >= max time index in wire_time_indices
        - max_keys: must be >= number of unique (track, wire, time) combinations
    If any are too small, results will be silently truncated. Check num_hits
    and num_tracks in the output against these limits.

    Parameters
    ----------
    wire_time_indices : jnp.ndarray, shape (N, 2)
        Wire and time indices for each charge deposit.
    track_ids : jnp.ndarray, shape (N,)
        Track ID for each charge deposit.
    charge_deposits : jnp.ndarray, shape (N,)
        Charge value for each deposit.
    min_charge_threshold : float
        Minimum charge to keep a hit (default: 0.0).
    max_tracks : int
        Maximum number of tracks (static). Must be >= unique track count.
    max_wires : int
        Maximum number of wires (static). Must be >= max wire index.
    max_time : int
        Maximum time bins (static). Must be >= max time index.
    max_keys : int
        Maximum number of unique (track, wire, time) combinations (static).

    Returns
    -------
    hits_by_track : jnp.ndarray, shape (max_keys, 3)
        Aggregated hits [wire, time, charge].
    num_hits : int
        Number of unique (track, wire, time) combinations found.
    track_boundaries : jnp.ndarray, shape (max_tracks,)
        Indices where track boundaries occur in sorted data.
    num_tracks : int
        Number of track boundaries found.
    track_ids_at_boundaries : jnp.ndarray, shape (max_tracks,)
        Track IDs at each boundary position.
    """
    N = wire_time_indices.shape[0]

    # Step 1: Sort by (track, wire, time) via two-pass stable sort (int32)
    # Single composite key overflows int32 when max_wires*max_time*max_tracks > 2^31
    wire_time_key = (wire_time_indices[:, 0].astype(jnp.int32) * max_time +
                     wire_time_indices[:, 1].astype(jnp.int32))
    _, idx = jax.lax.sort_key_val(wire_time_key, jnp.arange(N, dtype=jnp.int32))
    _, sort_indices = jax.lax.sort_key_val(track_ids[idx], idx)

    sorted_wires = wire_time_indices[sort_indices, 0]
    sorted_times = wire_time_indices[sort_indices, 1]
    sorted_tracks = track_ids[sort_indices]
    sorted_charges = charge_deposits[sort_indices]

    # Step 2: Find boundaries and aggregate
    # Boundary when track or (wire,time) changes — avoids int32 overflow in composite key
    sorted_wt = sorted_wires.astype(jnp.int32) * max_time + sorted_times.astype(jnp.int32)
    key_boundaries = jnp.ones(N, dtype=bool)
    key_boundaries = key_boundaries.at[1:].set(
        (sorted_tracks[1:] != sorted_tracks[:-1]) |
        (sorted_wt[1:] != sorted_wt[:-1])
    )

    segment_ids = jnp.cumsum(key_boundaries) - 1
    summed_charges = jax.ops.segment_sum(sorted_charges, segment_ids, num_segments=N)
    aggregated_charges = summed_charges[segment_ids]

    # Step 3: Apply threshold and extract unique entries
    segment_ends = jnp.roll(key_boundaries, -1).at[-1].set(True)
    threshold_mask = segment_ends & (aggregated_charges > min_charge_threshold)

    unique_indices = jnp.where(threshold_mask, size=max_keys, fill_value=0)[0]
    num_unique = jnp.sum(threshold_mask)

    num_stored = jnp.minimum(num_unique, max_keys)
    valid_mask = jnp.arange(max_keys) < num_stored

    # Step 4: Build hits_by_track array (without track column)
    hits_by_track = jnp.stack([
        jnp.where(valid_mask, sorted_wires[unique_indices], 0),
        jnp.where(valid_mask, sorted_times[unique_indices], 0),
        jnp.where(valid_mask, aggregated_charges[unique_indices], 0)
    ], axis=1)

    # Step 5: Key boundaries count
    num_hits = num_unique

    # Step 6: Track boundaries and track IDs
    # Build temporary track array for boundary detection
    temp_track_array = jnp.where(valid_mask, sorted_tracks[unique_indices], 0)
    track_changes = jnp.ones(max_keys, dtype=bool)
    track_changes = track_changes.at[1:].set(
        (temp_track_array[1:] != temp_track_array[:-1]) &
        (jnp.arange(1, max_keys) < num_stored)
    )
    track_boundaries = jnp.where(track_changes, size=max_tracks, fill_value=max_keys)[0]
    num_tracks = jnp.sum(track_changes & (jnp.arange(max_keys) < num_stored))

    # Extract track IDs at boundary positions
    track_ids_at_boundaries = jnp.where(
        jnp.arange(max_tracks) < num_tracks,
        temp_track_array[track_boundaries],
        0
    )

    return (hits_by_track, num_hits,
            track_boundaries, num_tracks,
            track_ids_at_boundaries)


@partial(jax.jit, static_argnames=['max_keys', 'max_time'])
def label_hits(hits_by_track, num_stored, track_ids_at_boundaries,
               track_boundaries, num_tracks,
               max_keys=1000000, max_time=2000):
    """
    Determine which track owns each pixel based on maximum charge.

    Standalone function — companion to group_hits_by_track.
    NOT used by the simulation pipeline's create_track_hits_fn.

    IMPORTANT: max_keys and max_time must match the values used in
    group_hits_by_track. If num_stored >= max_keys, the input was
    already truncated and results may be incomplete.

    Parameters
    ----------
    hits_by_track : jnp.ndarray, shape (max_keys, 3)
        Array of aggregated hits [wire, time, total_charge].
    num_stored : int
        Number of valid entries in hits_by_track.
    track_ids_at_boundaries : jnp.ndarray, shape (max_tracks,)
        Track IDs at each boundary position.
    track_boundaries : jnp.ndarray, shape (max_tracks,)
        Indices where track boundaries occur.
    num_tracks : int
        Number of valid track boundaries.
    max_keys : int
        Maximum size of arrays (must match hits_by_track shape).
    max_time : int
        Maximum time index for creating composite keys.

    Returns
    -------
    labeled_hits : jnp.ndarray, shape (max_keys, 4)
        Array of labeled hits with highest charge per (wire,time) [track_id, wire, time, charge].
    num_labeled : int
        Number of unique (wire,time) locations.
    """
    # Rebroadcast track IDs from boundaries using searchsorted
    # Each entry belongs to the track segment it falls into
    # Use masking to avoid dynamic slicing in JIT
    max_tracks_arr = track_boundaries.shape[0]
    valid_boundaries_mask = jnp.arange(max_tracks_arr) < num_tracks
    masked_boundaries = jnp.where(valid_boundaries_mask, track_boundaries, 1e9)

    boundary_idx = jnp.searchsorted(masked_boundaries,
                                    jnp.arange(max_keys),
                                    side='right') - 1
    boundary_idx = jnp.clip(boundary_idx, 0, jnp.maximum(num_tracks - 1, 0))

    # Only assign track IDs to valid entries
    rebroadcast_track_ids = jnp.where(
        jnp.arange(max_keys) < num_stored,
        track_ids_at_boundaries[boundary_idx],
        0
    )

    # Create composite key for (wire, time, -charge) sorting
    wire_time_key = hits_by_track[:, 0].astype(jnp.int32) * max_time + hits_by_track[:, 1].astype(jnp.int32)
    sort_keys = jnp.stack([wire_time_key, -hits_by_track[:, 2]], axis=1)
    sorted_indices = jnp.lexsort((sort_keys[:, 1], sort_keys[:, 0]))

    sorted_hits_by_track = hits_by_track[sorted_indices]
    sorted_track_ids = rebroadcast_track_ids[sorted_indices]

    # Mark first occurrence of each (wire, time)
    wire_time_first = jnp.ones(max_keys, dtype=bool)
    wire_time_first = wire_time_first.at[1:].set(
        (sorted_hits_by_track[1:, 0] != sorted_hits_by_track[:-1, 0]) |
        (sorted_hits_by_track[1:, 1] != sorted_hits_by_track[:-1, 1])
    )

    # Only consider valid entries
    is_valid_entry = (jnp.arange(max_keys) < num_stored)[sorted_indices]
    wire_time_first = wire_time_first & is_valid_entry

    # Extract labeled hits
    labeled_indices = jnp.where(wire_time_first, size=max_keys, fill_value=0)[0]
    num_labeled = jnp.sum(wire_time_first)

    valid_labeled_mask = jnp.arange(max_keys) < num_labeled

    labeled_hits = jnp.stack([
        jnp.where(valid_labeled_mask, sorted_track_ids[labeled_indices], 0),
        jnp.where(valid_labeled_mask, sorted_hits_by_track[labeled_indices, 0], 0),
        jnp.where(valid_labeled_mask, sorted_hits_by_track[labeled_indices, 1], 0),
        jnp.where(valid_labeled_mask, sorted_hits_by_track[labeled_indices, 2], 0)
    ], axis=1)

    return labeled_hits, num_labeled


def merge_chunk_sensor_hits(state_sk, state_tk, state_gk, state_ch,
                            chunk_sk, chunk_tk, chunk_gk, chunk_ch,
                            inter_thresh):
    """
    Merge a chunk of expanded hits into running state via sort-aggregate-compact.

    Uses three-pass stable sort (all int32) to achieve (spatial_key, time_key,
    group_id) lexicographic ordering. Works for both wire and pixel readout:
      wire:  spatial_key = wire_idx
      pixel: spatial_key = py * max_pz + pz

    Called from within JIT context (not separately decorated).

    Parameters
    ----------
    state_sk : jnp.ndarray, shape (max_keys,), int32
        Running spatial keys. Sentinels = 2^30.
    state_tk : jnp.ndarray, shape (max_keys,), int32
        Running time indices.
    state_gk : jnp.ndarray, shape (max_keys,), int32
        Running group IDs.
    state_ch : jnp.ndarray, shape (max_keys,), float32
        Running charges.
    chunk_sk : jnp.ndarray, shape (exp_size,), int32
        New chunk spatial keys.
    chunk_tk : jnp.ndarray, shape (exp_size,), int32
        New chunk time indices.
    chunk_gk : jnp.ndarray, shape (exp_size,), int32
        New chunk group IDs.
    chunk_ch : jnp.ndarray, shape (exp_size,), float32
        New chunk charges.
    inter_thresh : float
        Intermediate pruning threshold.

    Returns
    -------
    new_sk, new_tk, new_gk, new_ch : jnp.ndarray, shape (max_keys,)
        Compacted state arrays.
    count : jnp.int32
        Number of valid entries in compacted state.
    """
    SENTINEL = jnp.int32(2**30)
    max_keys = state_sk.shape[0]
    merge_size = max_keys + chunk_sk.shape[0]

    all_sk = jnp.concatenate([state_sk, chunk_sk])
    all_tk = jnp.concatenate([state_tk, chunk_tk])
    all_gk = jnp.concatenate([state_gk, chunk_gk])
    all_ch = jnp.concatenate([state_ch, chunk_ch])

    # Three-pass stable sort: group (tertiary) → time (secondary) → spatial (primary)
    _, idx1 = jax.lax.sort_key_val(all_gk, jnp.arange(merge_size, dtype=jnp.int32))
    _, idx2 = jax.lax.sort_key_val(all_tk[idx1], idx1)
    _, idx3 = jax.lax.sort_key_val(all_sk[idx2], idx2)

    sorted_sk = all_sk[idx3]
    sorted_tk = all_tk[idx3]
    sorted_gk = all_gk[idx3]
    sorted_ch = all_ch[idx3]

    # Boundary: new segment where any key changes
    boundaries = jnp.ones(merge_size, dtype=bool).at[1:].set(
        (sorted_sk[1:] != sorted_sk[:-1]) |
        (sorted_tk[1:] != sorted_tk[:-1]) |
        (sorted_gk[1:] != sorted_gk[:-1])
    )
    seg_ids = jnp.cumsum(boundaries) - 1
    summed = jax.ops.segment_sum(sorted_ch, seg_ids, num_segments=merge_size)
    agg = summed[seg_ids]

    # Filter: segment ends, exclude sentinels, apply intermediate threshold
    seg_ends = jnp.roll(boundaries, -1).at[-1].set(True)
    valid_entry = seg_ends & (sorted_sk < SENTINEL) & (agg > inter_thresh)

    # Compact into max_keys
    compact_idx = jnp.where(valid_entry, size=max_keys, fill_value=0)[0]
    count = jnp.sum(valid_entry).astype(jnp.int32)
    vmask = jnp.arange(max_keys) < count

    new_sk = jnp.where(vmask, sorted_sk[compact_idx], SENTINEL)
    new_tk = jnp.where(vmask, sorted_tk[compact_idx], jnp.int32(0))
    new_gk = jnp.where(vmask, sorted_gk[compact_idx], jnp.int32(0))
    new_ch = jnp.where(vmask, agg[compact_idx], 0.0).astype(jnp.float32)

    return new_sk, new_tk, new_gk, new_ch, count


def label_merged_hits(state_pk, state_tr, state_ch, state_count,
                      threshold, max_time):
    """
    Find dominant track per pixel from merged state.

    State is already sorted by (pixel_key, track) from the merge loop.
    Applies final threshold, then uses segment_max to find the track
    with highest charge at each (wire, time) pixel.

    Called from within JIT context (not separately decorated).

    Parameters
    ----------
    state_pk : jnp.ndarray, shape (max_keys,), int32
        Final pixel keys (wire * max_time + time).
    state_tr : jnp.ndarray, shape (max_keys,), int32
        Final track IDs.
    state_ch : jnp.ndarray, shape (max_keys,), float32
        Final charges per (pixel, track).
    state_count : jnp.int32
        Number of valid entries.
    threshold : float
        Final charge threshold.
    max_time : int
        Time dimension for decoding pixel_key.

    Returns
    -------
    dict with:
        labeled_hits : jnp.ndarray, shape (max_keys, 4)
            [track_id, wire, time, charge] per unique pixel.
        num_labeled : int
            Number of labeled pixels.
        hits_by_track : jnp.ndarray, shape (max_keys, 3)
            [wire, time, charge] per valid (pixel, track) entry.
        num_hits : int
            Number of valid entries after threshold.
    """
    max_keys = state_pk.shape[0]

    # Decode pixel_key
    state_wires = state_pk // max_time
    state_times = state_pk % max_time

    # Apply final threshold
    is_valid = (jnp.arange(max_keys) < state_count) & (state_ch > threshold)
    num_valid = jnp.sum(is_valid)
    valid_idx = jnp.where(is_valid, size=max_keys, fill_value=0)[0]
    vmask = jnp.arange(max_keys) < num_valid

    c_wires = jnp.where(vmask, state_wires[valid_idx], 0)
    c_times = jnp.where(vmask, state_times[valid_idx], 0)
    c_tracks = jnp.where(vmask, state_tr[valid_idx], 0)
    c_charges = jnp.where(vmask, state_ch[valid_idx], 0.0)

    hits_by_track = jnp.stack([c_wires, c_times, c_charges], axis=1).astype(jnp.float32)

    # Find pixel boundaries (different wire or time)
    pixel_boundary = jnp.ones(max_keys, dtype=bool).at[1:].set(
        ((c_wires[1:] != c_wires[:-1]) | (c_times[1:] != c_times[:-1]))
        & (jnp.arange(1, max_keys) < num_valid)
    )
    pixel_ids = jnp.cumsum(pixel_boundary) - 1

    # Find max charge per pixel
    charges_for_max = jnp.where(vmask, c_charges, -1e30)
    max_per_pixel = jax.ops.segment_max(
        charges_for_max, pixel_ids, num_segments=max_keys)
    max_for_entry = max_per_pixel[pixel_ids]

    # First entry matching max (tie-break by position)
    is_winner = vmask & (c_charges >= max_for_entry) & (max_for_entry > -1e29)
    winner_indices = jnp.where(is_winner, jnp.arange(max_keys), max_keys)
    first_winner = jax.ops.segment_min(
        winner_indices, pixel_ids, num_segments=max_keys)

    num_pixels = jnp.sum(pixel_boundary & vmask)
    pixel_range = jnp.arange(max_keys)
    valid_pixel = pixel_range < num_pixels
    widx = jnp.where(valid_pixel, first_winner[pixel_range], 0)

    labeled_hits = jnp.stack([
        jnp.where(valid_pixel, c_tracks[widx], 0),
        jnp.where(valid_pixel, c_wires[widx], 0),
        jnp.where(valid_pixel, c_times[widx], 0),
        jnp.where(valid_pixel, c_charges[widx], 0),
    ], axis=1).astype(jnp.float32)

    return {
        'labeled_hits': labeled_hits,
        'num_labeled': num_pixels,
        'hits_by_track': hits_by_track,
        'num_hits': num_valid,
    }


def sparse_hits_to_dense(track_hit_result, num_wires, num_time_steps):
    """
    Convert sparse track hits to dense 2D array.

    Uses hits_by_track which contains [wire, time, charge] per hit.
    This function should be called OUTSIDE the simulation when dense
    hit signals are needed for visualization.

    Parameters
    ----------
    track_hit_result : dict
        Output from track labeling containing:
        - 'hits_by_track': jnp.ndarray, shape (max_keys, 3) with [wire, time, charge]
        - 'num_hits': int, number of valid entries
    num_wires : int
        Number of wires in the output array.
    num_time_steps : int
        Number of time steps in the output array.

    Returns
    -------
    dense : jnp.ndarray, shape (num_wires, num_time_steps)
        Dense 2D array of accumulated charges.
    """
    hits_by_track = track_hit_result['hits_by_track']
    num_hits = track_hit_result['num_hits']

    # Initialize dense array
    dense = jnp.zeros((num_wires, num_time_steps), dtype=jnp.float32)

    # Get valid hits
    num_valid = int(num_hits)
    if num_valid > 0:
        valid_hits = hits_by_track[:num_valid]
        wire_indices = valid_hits[:, 0].astype(jnp.int32)
        time_indices = valid_hits[:, 1].astype(jnp.int32)
        charges = valid_hits[:, 2]

        # Clip indices to valid range
        wire_indices = jnp.clip(wire_indices, 0, num_wires - 1)
        time_indices = jnp.clip(time_indices, 0, num_time_steps - 1)

        # Accumulate charges
        dense = dense.at[wire_indices, time_indices].add(charges)

    return dense


def label_from_groups(state_sk, state_tk, state_gk, state_ch, state_count,
                      group_to_track, decode_spatial_fn=None):
    """Derive track hits from group merge state (postprocessing, outside JIT).

    Maps groups → tracks, aggregates per (sensor_pos, track), finds dominant
    track per sensor position. Works for both wire and pixel readout.

    Parameters
    ----------
    state_sk : jnp.ndarray, shape (max_keys,), int32
        Final spatial keys (wire_idx for wire, py*max_pz+pz for pixel).
    state_tk : jnp.ndarray, shape (max_keys,), int32
        Final time indices.
    state_gk : jnp.ndarray, shape (max_keys,), int32
        Final group IDs.
    state_ch : jnp.ndarray, shape (max_keys,), float32
        Final charges per (sensor_pos, time, group).
    state_count : jnp.int32
        Number of valid entries.
    group_to_track : np.ndarray, shape (n_groups,), int32
        Lookup: group_to_track[group_id] = track_id.
    decode_spatial_fn : callable, optional
        Decodes spatial_key → coordinate columns for output.
        Wire (default): sk → [wire] (1 column).
        Pixel: sk → [py, pz] (2 columns).
        Returns np.ndarray of shape (N, n_spatial_dims).

    Returns
    -------
    dict with labeled_hits, labeled_track_ids, num_labeled,
    hits_by_track, num_hits, group_correspondence.
    """
    import numpy as np_host

    count = int(state_count)
    sks = np_host.asarray(state_sk[:count])
    tks = np_host.asarray(state_tk[:count])
    gids = np_host.asarray(state_gk[:count])
    chs = np_host.asarray(state_ch[:count])
    g2t = np_host.asarray(group_to_track)

    # Default decode: wire (spatial_key = wire_idx, 1 column)
    if decode_spatial_fn is None:
        decode_spatial_fn = lambda sk: sk[:, None]

    if count == 0:
        n_cols = 3  # minimum: [spatial..., time, charge]
        empty = np_host.zeros((0, n_cols), dtype=np_host.float32)
        return {
            'labeled_hits': empty,
            'labeled_track_ids': np_host.zeros((0,), dtype=np_host.int32),
            'num_labeled': 0,
            'hits_by_track': empty,
            'num_hits': 0,
            'group_correspondence': (state_sk[:count], state_tk[:count],
                                     state_gk[:count], state_ch[:count], count),
        }

    # Map group → track
    tids = g2t[gids]

    # Composite pixel key for boundary detection: spatial * large_prime + time
    # Only used for sorting/grouping, not stored
    composite_pk = sks.astype(np_host.int64) * 100000 + tks.astype(np_host.int64)

    # Aggregate by (sensor_pos, time, track): sum group charges per track
    # Two-pass stable sort: by track (secondary), then composite pixel (primary)
    order1 = np_host.argsort(tids, kind='stable')
    order2 = np_host.argsort(composite_pk[order1], kind='stable')
    order = order1[order2]
    s_sks = sks[order]
    s_tks = tks[order]
    s_tids = tids[order]
    s_chs = chs[order]
    s_cpk = composite_pk[order]

    # (pixel, track) boundaries
    n_valid = len(sks)
    pt_boundary = np_host.ones(n_valid, dtype=bool)
    pt_boundary[1:] = (s_cpk[1:] != s_cpk[:-1]) | (s_tids[1:] != s_tids[:-1])
    pt_starts = np_host.where(pt_boundary)[0]
    n_pt = len(pt_starts)

    # Sum charges within each (sensor_pos, time, track) group
    pt_charges = np_host.add.reduceat(s_chs, pt_starts)
    pt_sks = s_sks[pt_starts]
    pt_tks = s_tks[pt_starts]
    pt_tracks = s_tids[pt_starts]
    pt_cpks = s_cpk[pt_starts]

    # Decode spatial keys to coordinate columns
    spatial_cols = decode_spatial_fn(pt_sks)  # (n_pt, n_spatial_dims)
    time_col = pt_tks.astype(np_host.float32)
    charge_col = pt_charges.astype(np_host.float32)

    hits_by_track = np_host.column_stack([
        spatial_cols.astype(np_host.float32),
        time_col[:, None],
        charge_col[:, None],
    ])

    # Find dominant track per sensor position (unique composite pixel key)
    px_boundary = np_host.ones(n_pt, dtype=bool)
    px_boundary[1:] = pt_cpks[1:] != pt_cpks[:-1]
    px_starts = np_host.where(px_boundary)[0]
    n_pixels = len(px_starts)

    max_charges = np_host.maximum.reduceat(pt_charges, px_starts)
    px_ids = np_host.zeros(n_pt, dtype=np_host.int64)
    px_ids[px_starts] = 1
    px_ids = np_host.cumsum(px_ids) - 1
    is_max = pt_charges >= max_charges[px_ids]

    # First max per pixel
    max_positions = np_host.where(is_max)[0]
    winner_idx = np_host.full(n_pixels, max_positions[0], dtype=np_host.int64)
    seen = np_host.zeros(n_pixels, dtype=bool)
    for pos in max_positions:
        pid = px_ids[pos]
        if not seen[pid]:
            winner_idx[pid] = pos
            seen[pid] = True

    labeled_hits = hits_by_track[winner_idx]
    labeled_track_ids = pt_tracks[winner_idx].astype(np_host.int32)

    return {
        'labeled_hits': labeled_hits,
        'labeled_track_ids': labeled_track_ids,
        'num_labeled': n_pixels,
        'hits_by_track': hits_by_track,
        'num_hits': n_pt,
        'group_correspondence': (state_sk[:count], state_tk[:count],
                                 state_gk[:count], state_ch[:count], count),
    }


def finalize_track_hits(track_hits, decode_fns=None):
    """Derive track labels from raw group merge state.

    Applies label_from_groups to each plane's raw
    (sk, tk, gk, ch, count, row_sums) 6-tuple. The group_to_track
    lookup is stored in track_hits['group_to_track'].

    Call after moving response_signals off GPU to avoid memory pressure.

    Parameters
    ----------
    track_hits : dict
        From process_event(). Plane keys map to raw 6-tuples.
        Contains 'group_to_track' metadata key.
    decode_fns : dict, optional
        Maps plane_key → decode_spatial_fn for label_from_groups.
        If None, uses default wire decode (spatial_key = wire_idx).

    Returns
    -------
    track_hits : dict
        Keyed by (vol, plane) with labeled_hits, hits_by_track,
        group_correspondence, row_sums per plane.
    """
    group_to_track = track_hits.pop('group_to_track')
    max_keys = None

    for plane_key, raw in list(track_hits.items()):
        state_sk, state_tk, state_gk, state_ch, state_count, row_sums = raw
        decode_fn = decode_fns.get(plane_key) if decode_fns else None
        result = label_from_groups(
            state_sk, state_tk, state_gk, state_ch, state_count,
            group_to_track, decode_spatial_fn=decode_fn)
        result['row_sums'] = row_sums
        track_hits[plane_key] = result

        # Validate
        gp = result.get('group_correspondence')
        if gp is not None:
            count_val = int(gp[-1])
            if max_keys is None:
                max_keys = state_sk.shape[0]
            if count_val >= max_keys:
                print(f"ERROR: Plane {plane_key}: group merge count ({count_val:,}) >= "
                      f"max_keys ({max_keys:,}). Correspondence data TRUNCATED!")

    return track_hits


# =============================================================================
# Q_s FRACTIONS (inside JIT, per-side)
# =============================================================================

def compute_qs_fractions(charges, group_ids, num_segments):
    """Compute Q_s fractions: each deposit's share of its group's charge.

    Called inside JIT after compute_volume_physics. Uses the already-computed
    recombined charges (before attenuation). Groups must not span sides
    (enforced by compute_group_ids with side-change splitting).

    Parameters
    ----------
    charges : jnp.ndarray (total_pad,)
        Recombined charge per deposit (valid_mask already applied, padded zeros).
    group_ids : jnp.ndarray (total_pad,)
        Group assignment per deposit.
    num_segments : int
        Static upper bound for segment_sum (use cfg.total_pad).

    Returns
    -------
    qs : jnp.ndarray (total_pad,)
        Fraction of group charge per deposit. Padded entries = 0.
    """
    group_sums = jax.ops.segment_sum(charges, group_ids, num_segments=num_segments)
    denom = jnp.maximum(group_sums[group_ids], 1e-10)
    return charges / denom


# =============================================================================
# FACTORY FUNCTION (create closure for use inside JIT)
# =============================================================================

def _make_noop_track_hits(max_keys, total_pad):
    """Build a no-op track hits function that returns matching zero 6-tuple."""
    SENTINEL_PK = jnp.int32(2**30)
    zero_hits = (
        jnp.full(max_keys, SENTINEL_PK, dtype=jnp.int32),
        jnp.zeros(max_keys, dtype=jnp.int32),
        jnp.zeros(max_keys, dtype=jnp.int32),
        jnp.zeros(max_keys, dtype=jnp.float32),
        jnp.int32(0),
        jnp.zeros(total_pad, dtype=jnp.float32),
    )
    def noop(intermediates, deposits, vol_geom, plane_idx, n_actual):
        return zero_hits
    return noop, zero_hits


def create_pixel_track_hits_fn(cfg, vol_geom):
    """Create pixel track hits labeling closure for one volume.

    Uses 3D CDF-integrated diffusion kernel (py, pz, time) and
    the unified 3-pass merge.

    Parameters
    ----------
    cfg : SimConfig
    vol_geom : VolumeGeometry (readout_type='pixel')

    Returns
    -------
    track_hits_fn : callable
        (pixel_int, deposits, vol_geom, plane_idx, n_actual) -> 6-tuple
    """
    diffusion = vol_geom.diffusion
    K_py = diffusion.K_wire
    K_pz = diffusion.K_wire
    K_time = diffusion.K_time
    K_total = (2 * K_py + 1) * (2 * K_pz + 1) * (2 * K_time + 1)
    exp_size = cfg.track_hits.hits_chunk_size * K_total
    SENTINEL_PK = jnp.int32(2**30)

    num_py, num_pz = vol_geom.pixel_shape
    pixel_pitch = vol_geom.pixel_pitch_cm

    prepare_pixel_vmap = jax.vmap(
        prepare_pixel_deposit_with_diffusion,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0,
                 None, None, None, None, None, None, None, None, None, None, None),
    )

    def track_hits_fn(pixel_int, deposits, vol_geom, plane_idx, n_actual):
        charges = pixel_int.charges
        drift_time_us = pixel_int.drift_time_us
        tick_us = pixel_int.tick_us
        attenuation_factors = pixel_int.attenuation
        py_idx = pixel_int.pixel_y_idx
        pz_idx = pixel_int.pixel_z_idx
        py_offset = pixel_int.pixel_y_offset
        pz_offset = pixel_int.pixel_z_offset
        valid_mask = jnp.arange(charges.shape[0]) < deposits.n_actual
        group_ids = deposits.group_ids

        max_safe_chunks = charges.shape[0] // cfg.track_hits.hits_chunk_size
        num_chunks = jnp.minimum(
            (n_actual + cfg.track_hits.hits_chunk_size - 1) // cfg.track_hits.hits_chunk_size,
            max_safe_chunks
        )

        def body(i, state):
            s_sk, s_tk, s_gk, s_ch, s_count, s_rowsums = state
            start = i * cfg.track_hits.hits_chunk_size
            cs = cfg.track_hits.hits_chunk_size

            c_charges = jax.lax.dynamic_slice(charges, (start,), (cs,))
            c_drift_time = jax.lax.dynamic_slice(drift_time_us, (start,), (cs,))
            c_tick = jax.lax.dynamic_slice(tick_us, (start,), (cs,))
            c_py = jax.lax.dynamic_slice(py_idx, (start,), (cs,))
            c_pz = jax.lax.dynamic_slice(pz_idx, (start,), (cs,))
            c_py_off = jax.lax.dynamic_slice(py_offset, (start,), (cs,))
            c_pz_off = jax.lax.dynamic_slice(pz_offset, (start,), (cs,))
            c_atten = jax.lax.dynamic_slice(attenuation_factors, (start,), (cs,))
            c_valid = jax.lax.dynamic_slice(valid_mask, (start,), (cs,))
            c_gids = jax.lax.dynamic_slice(group_ids, (start,), (cs,))

            spatial_keys, time_idx, sig_val = prepare_pixel_vmap(
                c_charges, c_drift_time, c_tick, c_py, c_pz,
                c_py_off, c_pz_off, c_atten, c_valid,
                K_py, K_pz, K_time, pixel_pitch, cfg.time_step_us,
                diffusion.trans_cm2_us, diffusion.long_cm2_us,
                diffusion.velocity_cm_us, num_py, num_pz,
                cfg.num_time_steps
            )

            chunk_rowsums = jnp.sum(
                jnp.where(sig_val > cfg.track_hits.inter_thresh, sig_val, 0.0),
                axis=1)
            s_rowsums = jax.lax.dynamic_update_slice(
                s_rowsums, chunk_rowsums, (start,))

            gid_exp = jnp.repeat(c_gids[:, jnp.newaxis], K_total, axis=1)

            sk_flat = spatial_keys.reshape(exp_size).astype(jnp.int32)
            t_flat = time_idx.reshape(exp_size).astype(jnp.int32)
            gid_flat = gid_exp.reshape(exp_size).astype(jnp.int32)
            ch_flat = sig_val.reshape(exp_size)

            chunk_valid = ch_flat > 0.0
            chunk_sk = jnp.where(chunk_valid, sk_flat, SENTINEL_PK)
            chunk_tk = jnp.where(chunk_valid, t_flat, jnp.int32(0))
            chunk_gk = jnp.where(chunk_valid, gid_flat, jnp.int32(0))
            chunk_ch = jnp.where(chunk_valid, ch_flat, 0.0).astype(jnp.float32)

            new_sk, new_tk, new_gk, new_ch, new_count = merge_chunk_sensor_hits(
                s_sk, s_tk, s_gk, s_ch,
                chunk_sk, chunk_tk, chunk_gk, chunk_ch,
                cfg.track_hits.inter_thresh
            )
            return (new_sk, new_tk, new_gk, new_ch, new_count, s_rowsums)

        init_state = (
            jnp.full(cfg.track_hits.max_keys, SENTINEL_PK, dtype=jnp.int32),
            jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.int32),
            jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.int32),
            jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.float32),
            jnp.int32(0),
            jnp.zeros(charges.shape[0], dtype=jnp.float32),
        )

        final_sk, final_tk, final_gk, final_ch, final_count, final_rowsums = \
            jax.lax.fori_loop(0, num_chunks, body, init_state)

        return (final_sk, final_tk, final_gk, final_ch, final_count, final_rowsums)

    return track_hits_fn


def create_track_hits_fn_for_volume(cfg, vol_geom):
    """Create track hits labeling closure for one volume.

    Dispatches to wire or pixel factory based on readout_type.

    Returns
    -------
    track_hits_fn : callable
        (intermediates, deposits, vol_geom, plane_idx, n_actual) -> 6-tuple
    zero_hits : tuple
        Pre-allocated zero 6-tuple matching output shape.
    decode_fn : callable or None
        spatial_key -> coordinate columns for finalize_track_hits.
    """
    import numpy as np_host

    if not cfg.include_track_hits:
        noop, zero_hits = _make_noop_track_hits(
            cfg.track_hits.max_keys if cfg.track_hits is not None else 1,
            cfg.total_pad)
        return noop, zero_hits, None

    SENTINEL_PK = jnp.int32(2**30)
    zero_hits = (
        jnp.full(cfg.track_hits.max_keys, SENTINEL_PK, dtype=jnp.int32),
        jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.int32),
        jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.int32),
        jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.float32),
        jnp.int32(0),
        jnp.zeros(cfg.total_pad, dtype=jnp.float32),
    )

    if vol_geom.readout_type == 'pixel':
        num_pz = vol_geom.pixel_shape[1]
        decode_fn = lambda sk, _npz=num_pz: np_host.column_stack([sk // _npz, sk % _npz])
        return create_pixel_track_hits_fn(cfg, vol_geom), zero_hits, decode_fn

    # Wire factory below
    diffusion = vol_geom.diffusion
    K_total = (2 * diffusion.K_wire + 1) * (2 * diffusion.K_time + 1)
    exp_size = cfg.track_hits.hits_chunk_size * K_total
    SENTINEL_PK = jnp.int32(2**30)

    prepare_deposit_vmap_hit = jax.vmap(
        prepare_deposit_with_diffusion,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None,
                 None, None, None, None, None),
    )

    def track_hits_fn(plane_int, deposits, vol_geom, plane_idx, n_actual):
        charges = plane_int.charges
        drift_time_us = plane_int.drift_time_us
        tick_us = plane_int.tick_us
        drift_distance_cm = plane_int.drift_distance_cm
        closest_wire_idx = plane_int.closest_wire_idx
        closest_wire_distances = plane_int.closest_wire_dist
        attenuation_factors = plane_int.attenuation
        valid_mask = jnp.arange(charges.shape[0]) < deposits.n_actual
        group_ids = deposits.group_ids
        spacing_cm = vol_geom.wire_spacings_cm[plane_idx]
        num_wires_plane = vol_geom.num_wires[plane_idx]
        angle_rad = vol_geom.angles_rad[plane_idx]

        theta_xz, theta_y = compute_deposit_wire_angles_vmap(
            deposits.theta, deposits.phi, angle_rad
        )
        angular_scaling_factor = compute_angular_scaling_vmap(theta_xz, theta_y)

        max_safe_chunks = charges.shape[0] // cfg.track_hits.hits_chunk_size
        num_chunks = jnp.minimum(
            (n_actual + cfg.track_hits.hits_chunk_size - 1) // cfg.track_hits.hits_chunk_size,
            max_safe_chunks
        )

        def body(i, state):
            s_sk, s_tk, s_gk, s_ch, s_count, s_rowsums = state
            start = i * cfg.track_hits.hits_chunk_size

            c_charges = jax.lax.dynamic_slice(charges, (start,), (cfg.track_hits.hits_chunk_size,))
            c_drift_time = jax.lax.dynamic_slice(drift_time_us, (start,), (cfg.track_hits.hits_chunk_size,))
            c_tick = jax.lax.dynamic_slice(tick_us, (start,), (cfg.track_hits.hits_chunk_size,))
            c_drift_dist = jax.lax.dynamic_slice(drift_distance_cm, (start,), (cfg.track_hits.hits_chunk_size,))
            c_wire_idx = jax.lax.dynamic_slice(closest_wire_idx, (start,), (cfg.track_hits.hits_chunk_size,))
            c_wire_dist = jax.lax.dynamic_slice(closest_wire_distances, (start,), (cfg.track_hits.hits_chunk_size,))
            c_atten = jax.lax.dynamic_slice(attenuation_factors, (start,), (cfg.track_hits.hits_chunk_size,))
            c_theta_xz = jax.lax.dynamic_slice(theta_xz, (start,), (cfg.track_hits.hits_chunk_size,))
            c_theta_y = jax.lax.dynamic_slice(theta_y, (start,), (cfg.track_hits.hits_chunk_size,))
            c_ang_scale = jax.lax.dynamic_slice(angular_scaling_factor, (start,), (cfg.track_hits.hits_chunk_size,))
            c_valid = jax.lax.dynamic_slice(valid_mask, (start,), (cfg.track_hits.hits_chunk_size,))
            c_gids = jax.lax.dynamic_slice(group_ids, (start,), (cfg.track_hits.hits_chunk_size,))

            wire_idx, time_idx, sig_val = prepare_deposit_vmap_hit(
                c_charges, c_drift_time, c_tick, c_drift_dist,
                c_wire_idx, c_wire_dist, c_atten,
                c_theta_xz, c_theta_y, c_ang_scale, c_valid,
                diffusion.K_wire, diffusion.K_time, spacing_cm, cfg.time_step_us,
                diffusion.long_cm2_us, diffusion.trans_cm2_us,
                diffusion.velocity_cm_us, num_wires_plane,
                cfg.num_time_steps
            )

            chunk_rowsums = jnp.sum(
                jnp.where(sig_val > cfg.track_hits.inter_thresh, sig_val, 0.0),
                axis=1,
            )
            s_rowsums = jax.lax.dynamic_update_slice(
                s_rowsums, chunk_rowsums, (start,))

            gid_exp = jnp.repeat(c_gids[:, jnp.newaxis], K_total, axis=1)

            w_flat = wire_idx.reshape(exp_size).astype(jnp.int32)
            t_flat = time_idx.reshape(exp_size).astype(jnp.int32)
            gid_flat = gid_exp.reshape(exp_size).astype(jnp.int32)
            ch_flat = sig_val.reshape(exp_size)

            # Separate spatial and time keys (no composite pk)
            chunk_valid = ch_flat > 0.0
            chunk_sk = jnp.where(chunk_valid, w_flat, SENTINEL_PK)
            chunk_tk = jnp.where(chunk_valid, t_flat, jnp.int32(0))
            chunk_gk = jnp.where(chunk_valid, gid_flat, jnp.int32(0))
            chunk_ch = jnp.where(chunk_valid, ch_flat, 0.0).astype(jnp.float32)

            new_sk, new_tk, new_gk, new_ch, new_count = merge_chunk_sensor_hits(
                s_sk, s_tk, s_gk, s_ch,
                chunk_sk, chunk_tk, chunk_gk, chunk_ch,
                cfg.track_hits.inter_thresh
            )

            return (new_sk, new_tk, new_gk, new_ch, new_count, s_rowsums)

        init_state = (
            jnp.full(cfg.track_hits.max_keys, SENTINEL_PK, dtype=jnp.int32),
            jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.int32),
            jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.int32),
            jnp.zeros(cfg.track_hits.max_keys, dtype=jnp.float32),
            jnp.int32(0),
            jnp.zeros(charges.shape[0], dtype=jnp.float32),
        )

        final_sk, final_tk, final_gk, final_ch, final_count, final_rowsums = \
            jax.lax.fori_loop(0, num_chunks, body, init_state)

        return (final_sk, final_tk, final_gk, final_ch, final_count, final_rowsums)

    # Wire: spatial_key = wire_idx, default decode (None → identity in label_from_groups)
    return track_hits_fn, zero_hits, None
