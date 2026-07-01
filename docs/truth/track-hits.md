# Track hits: group → track correspondence

Track hits are JAXTPC's **truth layer**: alongside the readout signal, the
simulation records which particle contributed to each wire/pixel cell, so
downstream reconstruction has per-particle ground truth. The mechanism is a
two-level correspondence — deposits are collapsed into **groups**, groups map
back to Geant4 **tracks** — plus a per-deposit charge share (`qs_fractions`).

!!! tip "Runnable notebook"
    [`notebooks/reco/segments_closure.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/reco/segments_closure.ipynb)
    walks the full group→track closure on a synthetic event.

## Groups: short per-track runs

`loader.compute_group_ids` (`tools/loader.py:539`) assigns every deposit a
**group id**. A group is a run of up to `group_size` (default 5) *consecutive*
deposits of a single track, and a new group is forced whenever:

- the **track id changes**, or
- the **spatial gap** between consecutive deposits exceeds `gap_threshold_mm`
  (default 5.0 mm) — this splits physically distant deposits of the same track
  (e.g. a neutron/gamma that deposits far away).

Deposits are stable-sorted by `track_id` (preserving trajectory order within a
track) before the run-length grouping, so a group is always a compact,
same-track, spatially-contiguous cluster. This compactness is what lets the
production path bound each group's footprint analytically (see
[box mode](#box-mode-default)).

!!! warning "Group ids are 1-based"
    `group_ids == 0` is reserved for **invalid / padding** deposits. Real groups
    start at 1 (`tools/loader.py:606`, "Consecutive group IDs (1-based; 0
    reserved for invalid)"). Every downstream stage encodes this: masks use
    `group_ids > 0`, and `group_to_track[0]` is an unused sentinel entry. In
    production `hits/` files the same 1-based convention holds — see
    [data formats](../production/data-formats.md).

Grouping is done **per volume** — deposits are already split by volume before
`compute_group_ids` is called, so groups never span volumes. The function
returns:

- `group_ids` — `(N,)` int32, one group per deposit (0 = padding).
- `group_to_track` — `(n_groups,)` lookup, `group_to_track[gid] = track_id`
  (NumPy, host-side, outside JIT).
- `n_groups` — total group count *including* the unused group 0.

## `qs_fractions`: each deposit's charge share

`compute_qs_fractions` (`tools/track_hits.py:629`) runs **inside JIT** after
`compute_volume_physics` and gives each deposit its fractional share of its
group's total recombined charge:

```python
group_sums = jax.ops.segment_sum(charges, group_ids, num_segments=num_segments)
denom      = jnp.maximum(group_sums[group_ids], 1e-10)
return charges / denom          # each deposit's share of its group's charge
```

It uses the recombined charge **before attenuation**; groups must not span
readout sides (guaranteed by the grouping rule). These fractions let truth be
attributed at deposit granularity even though the signal is accumulated at group
granularity — a group's signal can be redistributed to its constituent deposits
by their `qs_fractions`.

## Two production modes: box vs merge

The JIT emits, per `(volume, plane)`, a **raw 6-tuple** of group-level sensor
state:

```
(state_sk, state_tk, state_gk, state_ch, state_count, row_sums)
#  spatial  time      group     charge   n_valid      per-deposit rowsums
```

`create_track_hits_fn_for_volume` (`tools/track_hits.py:972`) builds the closure
that produces it, and selects one of two accumulation strategies via
`cfg.track_hits.box_enabled`.

### Box mode (default)

`box_enabled=True` — the **group-as-bucket** path. Each group's diffusion-spread
contributions are scatter-added into a small fixed-size box
`(MAXG, BW, BT)` located at the group's minimum wire/tick corner
(`tools/track_hits.py:1055`). Because a group's footprint is bounded by the
grouping rule (`span = (group_size - 1) * gap_threshold_mm`), the box dimensions
(`box_bw`/`box_btw` for wire, `box_bpy`/`box_bpz`/`box_bt` for pixel) are
computed **analytically** by `compute_box_dims` — no per-dataset scan needed.
After the chunk loop the box is compacted: cells clearing `inter_thresh` become
the raw 6-tuple entries. This is the production default (see PLAN §8b).

### Merge mode

`box_enabled=False` — the non-default path (`tools/track_hits.py:1121`). Instead
of local boxes, each chunk's `(spatial, time, group)` keys are merged into a
running sorted state by `merge_chunk_sensor_hits` (`tools/track_hits.py:297`),
pruning entries below `inter_thresh` per chunk. It produces the same raw 6-tuple
shape, so the host-side finalize step is identical. Merge is a live but
non-default mode; box is preferred because its per-group boxes avoid the global
sort-merge.

Both modes are real, supported options — box is the default, merge is the
alternate (PLAN §8b).

## Two-stage finalize

Labeling is deliberately split: the **JIT emits raw group state**, then the
**host maps groups → tracks**. The production entry point is the
**`DetectorSimulator.finalize_track_hits` method** (`tools/simulation.py:881`):

```python
response_signals, track_hits_raw, deposits = sim.process_event(deposits, key=key)
track_hits = sim.finalize_track_hits(track_hits_raw)   # host-side labeling
```

The method pops the per-volume `group_to_track` lookups and calls
`label_from_groups` (`tools/track_hits.py:435`) for each plane, passing the
plane's spatial decode function. `label_from_groups` runs on the host (NumPy):
it maps each group to its track, aggregates charge per `(sensor_position, time,
track)`, and finds the dominant track per sensor cell — returning
`labeled_hits`, `labeled_track_ids`, `hits_by_track`, and `group_correspondence`.
Doing the label step off-GPU keeps peak device memory down (call it after moving
`response_signals` off the GPU).

!!! note "Name collision: method vs module function"
    There is **also** a module-level `finalize_track_hits`
    (`tools/track_hits.py:572`). It is a **name-collision duplicate** — only
    stale-imported by one test, *not* the production entry point. Always use the
    `DetectorSimulator.finalize_track_hits` **method**. (PLAN §8b flags this for
    later de-duplication.)

## Legacy standalone trio (reference / test-only)

`track_hits.py` also carries an older, self-contained labeling path:
`group_hits_by_track` (`:85`) / `label_hits` (`:206`) / `sparse_hits_to_dense`
(`:384`). This trio uses a `K_wire × K_time` neighbor system and is **not used
by the simulator** — the module docstring marks it kept "for out-of-pipeline use
and tests." Do not mistake it for the production API; the box/merge +
`label_from_groups` path above is what `process_event` actually runs.

## End to end

```python
# 1. Load: deposits get group_ids (1-based) + group_to_track per volume
deposits = load_event('data.h5', sim.config, event_idx=0)

# 2. Simulate: JIT emits raw group state; deposits get qs_fractions filled
response_signals, track_hits_raw, deposits = sim.process_event(deposits, key=key)

# 3. Finalize on host: groups → tracks (use the METHOD)
track_hits = sim.finalize_track_hits(track_hits_raw)
track_hits[(0, 2)]['labeled_hits']        # (n, [wire, time, charge]) for vol 0, plane Y
track_hits[(0, 2)]['labeled_track_ids']   # dominant track id per sensor cell
```

## See also

- [Data formats](../production/data-formats.md) — how groups, `qs_fractions`,
  and `group_to_track` land in production `hits/` files (1-based groups, CSR
  encoding).
- [`segments_closure.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/reco/segments_closure.ipynb)
  — group→track closure on a synthetic event.
- [Execution paths](../architecture/execution-paths.md) — where track-hits sits
  in `process_event` (and why the differentiable path has none).
