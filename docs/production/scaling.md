# Running at scale

Once a [batch run](batch.md) works on a handful of events, scaling to a full
dataset is a matter of **splitting the work across processes** and **resuming**
cleanly after interruptions. This page covers the mechanisms `run_batch.py`
provides for both, independent of any particular cluster or scheduler.

The unit of parallelism is the **source file**: each input HDF5 file is
processed independently and writes its own output files, so files distribute
trivially across processes with no coordination.

## Always smoke-test first

A full run can be hours of GPU time; a mis-sized capacity or a bad path wastes
all of it. Run a few events on one or two source files first:

```bash
jaxtpc-batch \
    --data one_source.h5 \
    --config config/detector.yaml \
    --production-config config/production_detector.yaml \
    --outdir /tmp/smoke \
    --events 5
```

Then verify:

- the three output files exist under `/tmp/smoke/{sensor,step,hits}/`,
- `logs/summary_shard000.txt` reports `events_processed=5 events_skipped=0`,
- `logs/overflow_events_shard000.csv` has only its header row.

If events were skipped, the CSV names the overflow type — re-tune the
[capacities](../concepts/capacities.md) before committing to the long run.

## Sharding

`--num-shards K` splits the resolved file list into K disjoint subsets, and
`--shard-id N` (0-based) selects subset N. The split is **round-robin** by file
index (`files[i]` goes to shard `i % K`), which balances load even when files
vary in size. Launch one process per shard, each pinned to its own GPU:

```bash
K=4
for S in $(seq 0 $((K-1))); do
    # Pin this shard to GPU $S however your environment exposes it
    jaxtpc-batch \
        --data /path/to/run_dir/ \
        --config config/detector.yaml \
        --production-config config/production_detector.yaml \
        --outdir /path/to/output \
        --shard-id $S --num-shards $K &
done
wait
```

`--data` accepts files, directories (globbed for `*.h5`), and glob patterns,
and multiple paths may be given; they are resolved into one sorted, de-duped
list before sharding. The shards are independent — they read disjoint files and
write disjoint output — so there is no locking or communication between them.

!!! note "Adapt the loop to your scheduler"
    The `for … &` / `wait` loop above is just a generic "one process per GPU"
    pattern. Under a batch scheduler, submit K array tasks instead and pass
    each task's index as `--shard-id` (with `--num-shards K`). Nothing in
    `run_batch.py` is scheduler-specific.

### Explicit ranges

`--file-range "start:stop[:step]"` is a Python slice into the file list,
applied **after** shard selection. It is handy for reprocessing a known
subset:

```bash
# The first 10 files of shard 0
jaxtpc-batch --data run_dir/ --shard-id 0 --num-shards 4 --file-range 0:10 ...
```

## Resume

!!! note "Resume exists"
    Older notes said there was "no resume-skip yet" and that re-running
    overwrites finished files. That is out of date. `--skip-existing` provides
    crash-safe resume via per-file `.done` markers.

Pass `--skip-existing` to make a run resumable:

- When an output file's sensor/step/hits HDF5 are fully written **and closed**,
  the driver drops a marker under `{outdir}/.done/run_<id>/{dataset}_<suffix>.done`.
- On a re-run, any output whose `.done` marker already exists is **skipped**,
  not recomputed or overwritten.

Because the marker is written only *after* the HDF5 files are closed, a file
that crashed mid-write has no marker and is redone on the next run. Re-running
the same command — same shard, same range — after an interruption therefore
picks up exactly where it left off. It is safe to re-run any shard or range:
finished work is never touched.

```bash
# First attempt (interrupted partway)
jaxtpc-batch --data run_dir/ --shard-id 0 --num-shards 4 \
    --outdir /path/to/output --skip-existing ...

# Re-run the identical command — finished files are skipped, the rest resume
jaxtpc-batch --data run_dir/ --shard-id 0 --num-shards 4 \
    --outdir /path/to/output --skip-existing ...
```

## Save workers and the write ceiling

Within one process, saving is offloaded to `--workers` threads (see the
[threaded save architecture](batch.md#threaded-save-architecture)). More
workers help only up to a point:

- Save workers hold GPU-resident results while pulling them to host, so the
  worker count is effectively **GPU-memory bound** — too many workers in flight
  at once can exhaust device memory and cause failures. Raise `--workers`
  gradually and watch memory.
- When the workers are contending on the per-file locks rather than on memory,
  `--per-worker-files` lets each worker write its own file set (suffixed
  `_wNN`) so the HDF5 writes parallelize across cores. Downstream readers
  consume the `_wNN` files as one set.

Use `JAXTPC_PROFILE_SAVE=1` to see whether the pipeline is sim-bound,
load-bound, or save-bound before adding workers — throwing more workers at a
sim-bound run does nothing.

!!! tip "Reader threads too"
    On dense events the per-event load (HDF5 read + grouping) can stall the
    main loop. `--read-workers N` runs N prefetch threads so loading overlaps
    the GPU sim; grouping releases the GIL, so the readers genuinely
    parallelize.

## Overflow logging

Events larger than the config was tuned for do not crash the run — the driver
catches the `RuntimeError`, classifies it (`total_pad_overflow`,
`maxg_overflow`, `max_keys_overflow`, `max_buckets_overflow`, or
`runtime_error`), logs one row to
`{outdir}/logs/overflow_events_shard<SSS>.csv`, and continues. The output files
contain no entries for skipped events. Each shard also writes
`{outdir}/logs/summary_shard<SSS>.txt` with the per-shard tally.

To audit a completed run:

```bash
# Count skipped events by type across all shards
awk -F, 'NR>1 {print $7}' <outdir>/logs/overflow_events_*.csv | sort | uniq -c
```

If overflows are common, re-profile with a larger probe and re-tune the
[capacities](../concepts/capacities.md), or (for `maxg`) enable
[tiered routing](batch.md#tiered-routing-optional).

## See also

- [Batch pipeline](batch.md) — the driver, its flags, and the save architecture.
- [Capacities & overflow](../concepts/capacities.md) — sizing and failure modes.
- [Data formats](data-formats.md) — reading the output files as a set.
