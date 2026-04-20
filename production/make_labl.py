"""
Temporary labl generator.

Produces the ``labl/`` directory from the inst/ output + the edepsim
source file. Seg is **not** consulted because after the rename seg is
pure physics and no longer carries track-related arrays.

Per-deposit → track_id comes from inst::

    deposit i  -> inst.volume_N.segment_to_group[i] = g
               -> inst.volume_N.group_to_track[g]   = track_id

Per-track metadata (pdg, interaction, ancestor) comes from the edepsim
source via :class:`tools.loader.ParticleStepExtractor`.

Output layout (matches pimm-data's ``JAXTPCLablReader``)::

    {outdir}/labl/{dataset}_labl_{NNNN}.h5
        /config/ attrs
        /event_NNN/volume_N/
            # Per-deposit (N,) foreign key row-aligned with seg rows
            segment_to_track    (N,) int32

            # Per-unique-track (T,) dimension table
            track_ids           (T,) int32 — primary key
            track_pdg           (T,) int32 — raw PDG code
            track_interaction   (T,) int32 — raw interaction_id
            track_cluster       (T,) int32 — dummy (= track_id)
            track_ancestor      (T,) int32 — raw ancestor track id

All columns are raw / unmapped. Conversion from PDG (or any column) to
task-specific class indices happens downstream (see pimm-data's
``RemapSegment`` transform), not here.

Not JIT-compiled, not part of the batch pipeline. Intended to be
replaced by a proper edepsim-side labl writer integrated into
production.

Usage::

    python3 production/make_labl.py --outdir dataset_20 --source out.h5
"""

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

# Ensure tools/ is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.loader import ParticleStepExtractor, compute_interaction_ids


def _build_track_lookup(extractor, source_event_idx):
    """Extract per-step arrays from edepsim, collapse to per-unique-track
    metadata.

    Returns ``dict`` keyed by track_id (``int``) with fields:
    ``pdg``, ``interaction``, ``ancestor`` (all int).
    """
    step_data = extractor.extract_step_arrays(source_event_idx)
    if not step_data or 'track_id' not in step_data:
        return {}

    track_ids = np.asarray(step_data['track_id'], dtype=np.int64)
    if track_ids.size == 0:
        return {}

    pdata = getattr(extractor, '_last_particle_data', None) or {}
    interaction_ids = compute_interaction_ids(
        extractor.file, source_event_idx,
        ancestor_track_ids=step_data.get('ancestor_track_id'),
        particle_track_ids=pdata.get('track_id'),
        particle_parent_ids=pdata.get('parent_track_id'))
    interaction_ids = np.asarray(interaction_ids, dtype=np.int32)

    pdg = np.asarray(step_data.get('pdg', np.zeros_like(track_ids)),
                     dtype=np.int32)
    ancestor = np.asarray(step_data.get('ancestor_track_id',
                                        np.zeros_like(track_ids)),
                          dtype=np.int32)

    # First occurrence per unique track wins (all steps of a track share
    # the same per-track metadata).
    uniq, first_idx = np.unique(track_ids, return_index=True)
    return {
        int(tid): {
            'pdg': int(pdg[first_idx[i]]),
            'interaction': int(interaction_ids[first_idx[i]]),
            'ancestor': int(ancestor[first_idx[i]]),
        }
        for i, tid in enumerate(uniq)
    }


def _volume_labels(inst_vol_group, track_lookup):
    """Build per-deposit FK + per-unique-track dimension table for one
    volume, sourcing deposit → track from inst.

    Parameters
    ----------
    inst_vol_group : h5py.Group
        Volume group from the inst file. Must contain
        ``segment_to_group`` (N,) and ``group_to_track`` (G,).
    track_lookup : dict[int, dict]
        Output of :func:`_build_track_lookup` for the matching edepsim
        event.
    """
    if ('segment_to_group' not in inst_vol_group
            or 'group_to_track' not in inst_vol_group):
        empty = np.array([], dtype=np.int32)
        return dict(segment_to_track=empty, track_ids=empty,
                    track_pdg=empty, track_interaction=empty,
                    track_cluster=empty, track_ancestor=empty)

    seg_to_grp = inst_vol_group['segment_to_group'][:].astype(np.int32)
    g2t = inst_vol_group['group_to_track'][:].astype(np.int32)

    # Per-deposit track via seg_to_grp ∘ g2t (bound-safe).
    valid = (seg_to_grp >= 0) & (seg_to_grp < len(g2t))
    segment_to_track = np.full(len(seg_to_grp), -1, dtype=np.int32)
    segment_to_track[valid] = g2t[seg_to_grp[valid]]

    # Per-unique-track table (T rows): every track_id that appears at
    # least once in this volume's deposits (drops -1 sentinel).
    uniq = np.unique(segment_to_track[segment_to_track >= 0])
    T = len(uniq)

    track_pdg = np.full(T, -1, dtype=np.int32)
    track_interaction = np.full(T, -1, dtype=np.int32)
    track_ancestor = np.full(T, -1, dtype=np.int32)
    for i, tid in enumerate(uniq):
        meta = track_lookup.get(int(tid))
        if meta is None:
            continue
        track_pdg[i] = meta['pdg']
        track_interaction[i] = meta['interaction']
        track_ancestor[i] = meta['ancestor']

    return dict(
        segment_to_track=segment_to_track,
        track_ids=uniq.astype(np.int32),
        track_pdg=track_pdg,
        track_interaction=track_interaction,
        track_cluster=uniq.astype(np.int32),  # dummy: 1 track = 1 cluster
        track_ancestor=track_ancestor,
    )


def make_labl(inst_path, labl_path, source_file, dataset_name, file_index):
    """Generate a labl file from inst + edepsim source."""
    with h5py.File(inst_path, 'r') as f_inst, \
            h5py.File(labl_path, 'w') as f_labl, \
            ParticleStepExtractor(source_file) as extractor:

        inst_cfg = f_inst['config']
        n_events = int(inst_cfg.attrs['n_events'])
        n_volumes = int(inst_cfg.attrs.get('n_volumes', 1))

        g_cfg = f_labl.create_group('config')
        g_cfg.attrs['dataset_name'] = dataset_name
        g_cfg.attrs['file_index'] = file_index
        g_cfg.attrs['n_events'] = n_events
        g_cfg.attrs['n_volumes'] = n_volumes
        g_cfg.attrs['label_names'] = np.array(
            ['track_pdg', 'track_cluster', 'track_interaction',
             'track_ancestor'], dtype=object)
        g_cfg.attrs['source'] = f'dummy-from-edepsim:{os.path.basename(source_file)}'
        g_cfg.attrs['generator'] = 'production/make_labl.py'
        for key in ('source_file', 'global_event_offset', 'group_size',
                    'gap_threshold_mm', 'run_id', 'git_commit', 'git_dirty',
                    'git_repo'):
            if key in inst_cfg.attrs:
                g_cfg.attrs[key] = inst_cfg.attrs[key]

        for i in range(n_events):
            evt_key = f'event_{i:03d}'
            if evt_key not in f_inst:
                continue
            inst_evt = f_inst[evt_key]
            source_event_idx = int(inst_evt.attrs.get('source_event_idx', i))

            track_lookup = _build_track_lookup(extractor, source_event_idx)

            labl_evt = f_labl.create_group(evt_key)
            for v in range(n_volumes):
                vol_key = f'volume_{v}'
                if vol_key not in inst_evt:
                    continue
                inst_vol = inst_evt[vol_key]
                labl_vol = labl_evt.create_group(vol_key)

                labels = _volume_labels(inst_vol, track_lookup)
                for name, arr in labels.items():
                    labl_vol.create_dataset(name, data=arr, compression='gzip')


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dummy labl/ directory from inst + edepsim.")
    parser.add_argument('--outdir', required=True,
                        help='Base dataset directory (must contain inst/). '
                             'labl/ is created here.')
    parser.add_argument('--source', default=None,
                        help='Path to the edepsim source HDF5 file. '
                             'Defaults to the per-file source_file attr '
                             'stored in inst, resolved relative to cwd.')
    parser.add_argument('--dataset', default='sim',
                        help="File prefix (default: 'sim')")
    parser.add_argument('--inst-subdir', default='inst')
    parser.add_argument('--labl-subdir', default='labl')
    args = parser.parse_args()

    inst_dir = os.path.join(args.outdir, args.inst_subdir)
    labl_dir = os.path.join(args.outdir, args.labl_subdir)

    if not os.path.isdir(inst_dir):
        sys.exit(f"Inst directory not found: {inst_dir}")
    os.makedirs(labl_dir, exist_ok=True)

    pattern = os.path.join(inst_dir, f'{args.dataset}_inst_*.h5')
    inst_files = sorted(glob.glob(pattern))
    if not inst_files:
        sys.exit(f"No inst files matching {pattern}")

    for inst_path in inst_files:
        basename = os.path.basename(inst_path)
        stem = basename.rsplit('.', 1)[0]
        file_index = int(stem.rsplit('_', 1)[-1])
        labl_path = os.path.join(
            labl_dir, f'{args.dataset}_labl_{file_index:04d}.h5')

        if args.source:
            source = args.source
        else:
            with h5py.File(inst_path, 'r') as f:
                source = str(f['config'].attrs.get('source_file', 'out.h5'))
        if not os.path.exists(source):
            sys.exit(f"edepsim source not found: {source} (use --source to override)")

        t0 = time.time()
        make_labl(inst_path, labl_path, source, args.dataset, file_index)
        print(f'{basename} -> {os.path.basename(labl_path)} '
              f'[{time.time() - t0:.2f}s] (source={os.path.basename(source)})')

    print(f'\nDone. {len(inst_files)} labl file(s) written to {labl_dir}/')


if __name__ == '__main__':
    main()
