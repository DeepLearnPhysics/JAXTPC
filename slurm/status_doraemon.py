#!/usr/bin/env python3
"""Fast status for the doraemon wire/pixel SLURM production.

Standalone, read-only replacement for `run_doraemon_{wire,pixel}.sh status`.
The shell version forks basename/dirname (~3 procs) and stats one marker per
file, serially -> thousands of forks + network stats. This does ONE os.scandir
walk of the .done tree, then diffs against the input file list in memory.

It mirrors the wrapper's file enumeration exactly (run dirs in sorted order,
each dir's *.h5 sorted, concatenated; only the LAST folders dropped via
--drop-last) so global indices line up with run_batch / the wrapper. Missing
files are compressed into ready-to-paste START/STOP rerun lines.

Examples:
  ./slurm/status_doraemon.py wire
  ./slurm/status_doraemon.py pixel --start 2886
  ./slurm/status_doraemon.py wire --drop-last 0 --summary
"""
import argparse
import os
import sys

READOUTS = {
    'wire':  ('wire',  'sim_wire'),
    'pixel': ('pixel', 'sim_pixel'),
}


def marker_name(h5_basename, dataset):
    """edepsim_000006.h5 -> sim_wire_0006.done  (matches run_batch naming)."""
    base = h5_basename
    if base.startswith('edepsim_'):
        base = base[len('edepsim_'):]
    if base.endswith('.h5'):
        base = base[:-len('.h5')]
    return f'{dataset}_{int(base):04d}.done'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('readout', choices=READOUTS.keys())
    ap.add_argument('--voxel', default='test_00_00_02')
    ap.add_argument('--run-glob', default='run_')
    ap.add_argument('--drop-last', type=int, default=1,
                    help='drop the last N run folders (still being generated). '
                         'NOTE: the wrapper default is 1; raise it if more than '
                         'one trailing folder is mid-generation.')
    ap.add_argument('--start', type=int, default=0, help='first global index (incl)')
    ap.add_argument('--stop', type=int, default=0, help='exclusive; 0 => all')
    ap.add_argument('--data-parent', default=None,
                    help='override input parent (default /sdf/data/neutrino/doraemon/<voxel>)')
    ap.add_argument('--outdir', default=None,
                    help='override output base (default /sdf/data/neutrino/doraemon/<wire|pixel>_<voxel>)')
    ap.add_argument('--summary', action='store_true',
                    help='just done/total counts, skip the missing-range report')
    args = ap.parse_args()

    out_root, dataset = READOUTS[args.readout]
    data_parent = args.data_parent or f'/sdf/data/neutrino/doraemon/{args.voxel}'
    outdir = args.outdir or f'/sdf/data/neutrino/doraemon/{out_root}_{args.voxel}'
    done_dir = os.path.join(outdir, '.done')

    # --- enumerate run folders, drop the trailing ones (index-stable) ---------
    if not os.path.isdir(data_parent):
        sys.exit(f'ERROR: input parent not found: {data_parent}')
    run_dirs = sorted(e.path for e in os.scandir(data_parent)
                      if e.is_dir() and e.name.startswith(args.run_glob))
    n_runs_all = len(run_dirs)
    if 0 < args.drop_last < n_runs_all:
        run_dirs = run_dirs[:n_runs_all - args.drop_last]
    n_runs = len(run_dirs)
    if not run_dirs:
        sys.exit(f'ERROR: no run folders under {data_parent}/{args.run_glob}*')

    # --- combined input file list (same order run_batch sees) -----------------
    # keep (run_folder_name, expected_marker_name) per global index
    files = []
    for d in run_dirs:
        run_name = os.path.basename(d.rstrip('/'))
        for name in sorted(n for n in os.listdir(d) if n.endswith('.h5')):
            files.append((run_name, marker_name(name, dataset)))
    n_files = len(files)
    if n_files == 0:
        sys.exit(f'ERROR: no .h5 under {data_parent}/{args.run_glob}*')

    start = args.start
    stop = args.stop if (0 < args.stop <= n_files) else n_files

    # --- ONE walk of the .done tree -> {run_name: set(marker_names)} -----------
    present = {}
    if os.path.isdir(done_dir):
        for run_e in os.scandir(done_dir):
            if run_e.is_dir():
                present[run_e.name] = {m.name for m in os.scandir(run_e.path)
                                       if m.name.endswith('.done')}
    total_markers = sum(len(v) for v in present.values())

    print(f'Status  {outdir}')
    print(f'  run folders {n_runs} of {n_runs_all} (drop_last={args.drop_last})   '
          f'input files {n_files}   markers on disk {total_markers}')
    print(f'  range {start}:{stop}  ({stop - start} files)')

    if args.summary:
        print(f'  done(total)={total_markers}/{n_files}')
        return

    # --- diff in memory -------------------------------------------------------
    missing = []
    for i in range(start, stop):
        run_name, mname = files[i]
        if mname not in present.get(run_name, ()):
            missing.append(i)

    print(f'  done={stop - start - len(missing)}   missing={len(missing)}')
    if not missing:
        return

    # compress consecutive indices into START/STOP rerun lines
    print('  re-run the missing files with:')
    script = f'./slurm/run_doraemon_{args.readout}.sh'
    s = p = missing[0]
    for idx in missing[1:]:
        if idx == p + 1:
            p = idx
        else:
            print(f'    START={s} STOP={p + 1} {script}')
            s = p = idx
    print(f'    START={s} STOP={p + 1} {script}')


if __name__ == '__main__':
    main()
