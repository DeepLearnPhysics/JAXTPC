#!/bin/bash
# =============================================================================
# run_doraemon_pixel.sh — SLURM array driver for PIXEL production (run_batch.py)
#
# Splits a file-range into BATCH_SIZE-file chunks and runs each chunk as one
# task of a SLURM job array on `ampere` (1 A100 / task). A task asks for 2h but
# RELEASES THE NODE the instant run_batch.py returns, so 2h is just a ceiling.
#
# SELF-SUBMITTING — run it directly on a login node (do NOT `sbatch` it).
#
#   CHOOSING WHICH FILES (all env overrides of the CONFIG defaults; the run-set
#   knobs are forwarded to the array tasks via sbatch --export). Selection is
#   layered: VOXEL/RUN_GLOB/DROP_LAST_RUNS pick the run folders, then START:STOP
#   is a window into the resulting sorted file list (global file index).
#     ./slurm/run_doraemon_pixel.sh                     # submit full set
#     START=300 STOP=500 ./slurm/run_doraemon_pixel.sh  # only files 300:500
#     START=300 STOP=500 SKIP_EXISTING=0 ./...          # FORCE redo 300:500
#     DROP_LAST_RUNS=0 ./...                            # include the last folder too
#     DROP_LAST_RUNS=2 ./...                            # drop the last 2 folders
#     RUN_GLOB='run_002757*' ./...                      # only matching run folders
#     VOXEL=test_00_00_03 ./...                         # a different voxelization
#     ./slurm/run_doraemon_pixel.sh status              # report done/missing
#     MAX_CONCURRENT=5 ./...                            # GPUs in flight (default 5)
#   Dropping the LAST folder(s) is index-stable (earlier files keep their index
#   + .done marker), so raising/lowering DROP_LAST_RUNS later just adds/removes
#   tail files and the rest resume cleanly.
#
# RESUME / RERUN MODEL (nothing is erased, finished work is never redone):
#   * --skip-existing  : run_batch skips any input file already marked done (a
#                        per-file marker under $OUTDIR/.done/, written only AFTER
#                        its sensor/step/hits are fully closed — crash-safe). So a
#                        plain re-submit only re-runs files that FAILED / never
#                        finished (those have no marker). Set SKIP_EXISTING=0 to
#                        force a range to be recomputed even if marked done.
#   * `status` mode    : lists which files in the range are done vs missing and
#                        prints the missing ones as ready-to-paste START/STOP
#                        ranges, so you can re-run exactly the failures.
#   * shard-id = the task's global start file index -> per-shard logs
#                (summary_shardNNN.txt / overflow_events_shardNNN.csv) are unique
#                per slice and never collide across re-runs or new subsets.
#   * SLURM logs use %A_%a (job id + task) -> unique per submission, never erased.
#   * output paths are addressed by (run_id, file_idx) -> idempotent; different
#     subsets never touch each other's files.
# =============================================================================

#SBATCH --job-name=dora_pixel
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=230016M
#SBATCH --time=02:00:00
# NOTE: --output/--error/--array/--export are supplied on the sbatch command line
#       by the submit section (so logs follow $OUTDIR). Do not set them here.

set -euo pipefail
MODE="${1:-run}"            # "run" (default) | "status"

# ============================= CONFIG (edit me) ==============================
WORKDIR=${WORKDIR:-/sdf/group/neutrino/omara/JAXTPC}  # env-overridable: point at a branch checkout/worktree
IMAGE=/sdf/group/neutrino/images/develop.sif
export TMPDIR=/sdf/data/neutrino/omara/tmp

# --- which input subset (all env-overridable; see "choosing which files") ----
VOXEL=${VOXEL:-test_00_00_02}                        # 300um voxelization
DATA_PARENT=/sdf/data/neutrino/doraemon/${VOXEL}
RUN_GLOB=${RUN_GLOB:-run_*}                          # which run dirs (pattern)
# Trailing run folders still being generated are excluded until complete. Only
# the LAST folder is ever dropped (dropping the tail keeps every earlier file's
# global index — and its .done marker — stable). test_00_00_02's last folder
# (run_0027670361) is not done yet -> drop 1. Set DROP_LAST_RUNS=0 once complete.
DROP_LAST_RUNS=${DROP_LAST_RUNS:-1}
DATA_DIRS=( "${DATA_PARENT}"/${RUN_GLOB}/ )
NRUNS_ALL=${#DATA_DIRS[@]}
(( DROP_LAST_RUNS > 0 && DROP_LAST_RUNS < NRUNS_ALL )) && \
    DATA_DIRS=( "${DATA_DIRS[@]:0:NRUNS_ALL-DROP_LAST_RUNS}" )
NRUNS=${#DATA_DIRS[@]}
# run_batch resolves each --data dir to a sorted *.h5 list, concatenated in the
# order above; this wrapper enumerates identically so file indices line up.

# --- readout-specific (the ONLY block that differs from the wire script) ----
CONFIG=config/cubic_pixel_config.yaml
# Capacity config — override to redo a folder that overflowed (e.g. max_keys):
#   RUN_GLOB='run_0027658640' SKIP_EXISTING=0 \
#     PROD_CONFIG=config/production_cubic_pixel_doraemon_300micro_bigkeys.yaml ./...
PROD_CONFIG=${PROD_CONFIG:-config/production_cubic_pixel_doraemon_300micro.yaml}
OUTDIR=/sdf/data/neutrino/omara/JAXTPC_Pixel/${VOXEL}
DATASET=sim_pixel
WORKERS=12          # save workers. Pixel uses bucketed (sparse) accumulation so
                    # queued results are small; 12 is safe + plenty.
PER_WORKER=0        # 1 = each worker writes its own _wNN files (parallel writes,
                    # N files/source). 0 = one clean sensor/step/hits per source.

# --- common run_batch knobs -------------------------------------------------
EVENTS=200          # events per input file (doraemon = 200 = all)
READ_WORKERS=4
CODEC=blosc-lz4

# --- file-range / batching / concurrency / resume (env-overridable) ---------
START=${START:-0}                    # first global file index (inclusive)
STOP=${STOP:-0}                      # exclusive; 0 => auto = total files found
BATCH_SIZE=${BATCH_SIZE:-20}         # input files per array task (~1.3 h pixel, ~1.5 h worst-case)
MAX_CONCURRENT=${MAX_CONCURRENT:-3}  # max array tasks (= GPUs) running at once (env-overridable)
SKIP_EXISTING=${SKIP_EXISTING:-1}    # 1=skip files already done; 0=force redo
# SLURM account selector ("mode"): cider -> mli:cider-ml (our allocation);
# default -> mli:default (shared queue); any other value is passed verbatim.
MODE=${MODE:-cider}
case "$MODE" in
  cider)   ACCOUNT=mli:cider-ml ;;
  default) ACCOUNT=mli:default ;;
  *)       ACCOUNT=$MODE ;;
esac
# =============================================================================

# Enumerate the combined file list exactly as run_batch will (per-dir sorted
# *.h5, in DATA_DIRS order) to get the total count for STOP-auto and NBATCH.
mapfile -t ALL_FILES < <(for d in "${DATA_DIRS[@]}"; do
                           ls -1 "$d"*.h5 2>/dev/null | sort
                         done)
NFILES=${#ALL_FILES[@]}
(( STOP == 0 || STOP > NFILES )) && STOP=$NFILES
NBATCH=$(( (STOP - START + BATCH_SIZE - 1) / BATCH_SIZE ))

# helper: marker path for a given global file index (mirrors run_batch naming)
done_marker_for() {
    local f="${ALL_FILES[$1]}" rundir base
    rundir=$(basename "$(dirname "$f")")          # run_00XXXXXXXX
    base=$(basename "$f"); base="${base#edepsim_}"; base="${base%.h5}"
    printf '%s/.done/%s/%s_%04d.done' "$OUTDIR" "$rundir" "$DATASET" "$((10#$base))"
}

# ---------------------------------------------------------------------------
# SUBMIT / STATUS MODE: run from a login node (no SLURM_ARRAY_TASK_ID yet).
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    (( NFILES == 0 )) && { echo "ERROR: no .h5 under ${DATA_PARENT}/${RUN_GLOB}"; exit 1; }

    # ---- STATUS: report done vs missing in [START,STOP), missing as ranges ----
    if [[ "$MODE" == "status" ]]; then
        echo "Status  $OUTDIR"
        echo "  run folders $NRUNS of $NRUNS_ALL (drop_last=$DROP_LAST_RUNS)   range $START:$STOP   ($((STOP-START)) files)"
        ndone=0; missing=()
        for ((i=START; i<STOP; i++)); do
            if [[ -f "$(done_marker_for "$i")" ]]; then ndone=$((ndone+1))
            else missing+=("$i"); fi
        done
        echo "  done=$ndone   missing=${#missing[@]}"
        if (( ${#missing[@]} )); then
            echo "  re-run the missing files with:"
            s=${missing[0]}; p=${missing[0]}
            for idx in "${missing[@]:1}"; do
                if (( idx == p+1 )); then p=$idx
                else echo "    START=$s STOP=$((p+1)) ./$(basename "$0")"; s=$idx; p=$idx; fi
            done
            echo "    START=$s STOP=$((p+1)) ./$(basename "$0")"
        fi
        exit 0
    fi

    # ---- SUBMIT ----
    SCRIPT="$(readlink -f "$0")"
    mkdir -p "$OUTDIR/logs" "$TMPDIR"
    (( NBATCH <= 0 )) && { echo "ERROR: empty range ${START}:${STOP}"; exit 1; }

    echo "Readout     : pixel   outdir $OUTDIR"
    echo "Run folders : $NRUNS of $NRUNS_ALL used (drop_last=$DROP_LAST_RUNS, glob=$RUN_GLOB)"
    echo "Files found : $NFILES   range $START:$STOP  ->  $NBATCH task(s) of <=$BATCH_SIZE files"
    echo "Workers     : $WORKERS (per_worker=$PER_WORKER)   skip_existing=$SKIP_EXISTING"
    echo "Account     : $ACCOUNT (mode=$MODE)"
    echo "Concurrency : <=$MAX_CONCURRENT GPU(s) at once   2h ceiling/task"
    for ((b=0; b<NBATCH; b++)); do
        s=$(( START + b*BATCH_SIZE ))
        e=$(( s + BATCH_SIZE )); (( e > STOP )) && e=$STOP
        printf "  task %3d : files %5d:%-5d (%3d files)  shard-id %d\n" \
               "$b" "$s" "$e" "$((e-s))" "$s"
    done

    # Forward every selection knob so each array task rebuilds the identical
    # file list (RUN_GLOB holds a '*', so quote the whole --export value).
    EXPORT="ALL,VOXEL=$VOXEL,RUN_GLOB=$RUN_GLOB,DROP_LAST_RUNS=$DROP_LAST_RUNS,PROD_CONFIG=$PROD_CONFIG"
    EXPORT="$EXPORT,START=$START,STOP=$STOP,BATCH_SIZE=$BATCH_SIZE,SKIP_EXISTING=$SKIP_EXISTING"
    exec sbatch \
        --account="$ACCOUNT" \
        --export="$EXPORT" \
        --output="$OUTDIR/logs/slurm_%x_%A_%a.out" \
        --error="$OUTDIR/logs/slurm_%x_%A_%a.err" \
        --array="0-$((NBATCH-1))%${MAX_CONCURRENT}" \
        "$SCRIPT"
fi

# ---------------------------------------------------------------------------
# WORKER MODE: inside one array task. Compute this task's slice + shard-id.
# ---------------------------------------------------------------------------
b=$SLURM_ARRAY_TASK_ID
RANGE_START=$(( START + b*BATCH_SIZE ))
RANGE_STOP=$(( RANGE_START + BATCH_SIZE )); (( RANGE_STOP > STOP )) && RANGE_STOP=$STOP
SHARD=$RANGE_START   # deterministic + unique per slice -> per-shard logs never collide

echo "============================================================"
echo " PIXEL array task $b : files ${RANGE_START}:${RANGE_STOP}  shard-id ${SHARD}"
echo " host $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}  $(date)"
echo "============================================================"

export PYTHONPATH=''
mkdir -p "$TMPDIR"

PW_FLAG=""; (( PER_WORKER )) && PW_FLAG="--per-worker-files"
SK_FLAG=""; (( SKIP_EXISTING )) && SK_FLAG="--skip-existing"

rc=0
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch "$IMAGE" \
  bash -lc "cd '$WORKDIR' && TMPDIR='$TMPDIR' python3 production/run_batch.py \
      --data ${DATA_DIRS[*]} \
      --file-range '${RANGE_START}:${RANGE_STOP}' --events ${EVENTS} \
      --config '${CONFIG}' --production-config '${PROD_CONFIG}' \
      --outdir '${OUTDIR}' --dataset '${DATASET}' \
      --shard-id ${SHARD} --workers ${WORKERS} ${PW_FLAG} \
      --read-workers ${READ_WORKERS} --codec ${CODEC} \
      ${SK_FLAG}" || rc=$?

echo "array task $b finished at $(date) with exit code $rc"
# Returning here ends the step -> SLURM frees the node now, not at the 2h mark.
exit $rc
