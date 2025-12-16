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


@partial(jax.jit, static_argnames=['max_tracks', 'max_wires', 'max_time', 'max_keys'])
def group_hits_by_track(wire_time_indices, track_ids, charge_deposits,
                        min_charge_threshold=0.0,
                        max_tracks=10000, max_wires=2000, max_time=2000,
                        max_keys=1000000):
    """
    Aggregate charge deposits by track and location for LArTPC simulation data.

    This function groups charge deposits by (track_id, wire, time) and sums
    the charges at each location. It's used in the hit path to
    track particle contributions.

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
        Maximum number of tracks (static).
    max_wires : int
        Maximum number of wires (static).
    max_time : int
        Maximum time bins (static).
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

    # Step 1: Sort by (track, wire, time)
    composite_key = (track_ids.astype(jnp.int32) * (max_wires * max_time) +
                     wire_time_indices[:, 0].astype(jnp.int32) * max_time +
                     wire_time_indices[:, 1].astype(jnp.int32))

    sorted_keys, sort_indices = jax.lax.sort_key_val(composite_key, jnp.arange(N))

    sorted_wires = wire_time_indices[sort_indices, 0]
    sorted_times = wire_time_indices[sort_indices, 1]
    sorted_tracks = track_ids[sort_indices]
    sorted_charges = charge_deposits[sort_indices]

    # Step 2: Find boundaries and aggregate
    key_boundaries = jnp.ones(N, dtype=bool)
    key_boundaries = key_boundaries.at[1:].set(sorted_keys[1:] != sorted_keys[:-1])

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

    For each unique (wire, time) location, this function determines which
    track contributed the most charge and returns that as the "labeled" hit.

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


def sparse_hits_to_dense(track_hit_result, num_wires, num_time_steps, min_wire_idx=0):
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
    min_wire_idx : int, optional
        Minimum wire index to subtract (for relative indexing). Default: 0.

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
        wire_indices = valid_hits[:, 0].astype(jnp.int32) - min_wire_idx
        time_indices = valid_hits[:, 1].astype(jnp.int32)
        charges = valid_hits[:, 2]

        # Clip indices to valid range
        wire_indices = jnp.clip(wire_indices, 0, num_wires - 1)
        time_indices = jnp.clip(time_indices, 0, num_time_steps - 1)

        # Accumulate charges
        dense = dense.at[wire_indices, time_indices].add(charges)

    return dense


# Backward compatibility aliases (deprecated)
aggregate_hits_by_track = group_hits_by_track
get_matched_hits = label_hits
