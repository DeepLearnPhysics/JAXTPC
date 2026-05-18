"""
Temporary labl generator.

Produces the ``labl/`` directory from the hits/ output + the edepsim
source file. Edep is **not** consulted because edep is pure physics
and does not carry track-related arrays.

Per-deposit → track_id comes from hits::

    deposit i  -> hits.volume_N.deposit_to_group[i] = g
               -> hits.volume_N.group_to_track[g]   = track_id

Per-track metadata (pdg, interaction, ancestor) comes from the edepsim
source via :class:`tools.loader.ParticleStepExtractor`.

Output layout (matches pimm-data's ``JAXTPCLablReader``)::

    {outdir}/labl/{dataset}_labl_{NNNN}.h5
        /config/ attrs
        /event_NNN/volume_N/
            # Per-deposit (N,) foreign key row-aligned with edep rows
            deposit_to_track    (N,) int32

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
        root_track_ids=step_data.get('root_track_id'),
        particle_track_ids=pdata.get('track_id'),
        particle_parent_ids=pdata.get('parent_track_id'))
    interaction_ids = np.asarray(interaction_ids, dtype=np.int32)

    pdg = np.asarray(step_data.get('pdg', np.zeros_like(track_ids)),
                     dtype=np.int32)
    ancestor = np.asarray(step_data.get('root_track_id',
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


def _volume_labels(hits_vol_group, track_lookup):
    """Build per-deposit FK + per-unique-track dimension table for one
    volume, sourcing deposit → track from hits.

    Parameters
    ----------
    hits_vol_group : h5py.Group
        Volume group from the hits file. Must contain
        ``deposit_to_group`` (N,) and ``group_to_track`` (G,).
    track_lookup : dict[int, dict]
        Output of :func:`_build_track_lookup` for the matching edepsim
        event.
    """
    d2g_key = 'deposit_to_group' if 'deposit_to_group' in hits_vol_group else 'segment_to_group'
    if (d2g_key not in hits_vol_group
            or 'group_to_track' not in hits_vol_group):
        empty = np.array([], dtype=np.int32)
        return dict(deposit_to_track=empty, track_ids=empty,
                    track_pdg=empty, track_interaction=empty,
                    track_cluster=empty, track_ancestor=empty)

    dep_to_grp = hits_vol_group[d2g_key][:].astype(np.int32)
    g2t = hits_vol_group['group_to_track'][:].astype(np.int32)

    valid = (dep_to_grp >= 0) & (dep_to_grp < len(g2t))
    deposit_to_track = np.full(len(dep_to_grp), -1, dtype=np.int32)
    deposit_to_track[valid] = g2t[dep_to_grp[valid]]

    uniq = np.unique(deposit_to_track[deposit_to_track >= 0])
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
        deposit_to_track=deposit_to_track,
        track_ids=uniq.astype(np.int32),
        track_pdg=track_pdg,
        track_interaction=track_interaction,
        track_cluster=uniq.astype(np.int32),
        track_ancestor=track_ancestor,
    )


def make_labl(hits_path, labl_path, source_file, dataset_name, file_index):
    """Generate a labl file from hits + edepsim source."""
    with h5py.File(hits_path, 'r') as f_hits, \
            h5py.File(labl_path, 'w') as f_labl, \
            ParticleStepExtractor(source_file) as extractor:

        hits_cfg = f_hits['config']
        n_events = int(hits_cfg.attrs['n_events'])
        n_volumes = int(hits_cfg.attrs.get('n_volumes', 1))

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
                    'gap_threshold_mm', 'production_version', 'run_id',
                    'batch_timestamp', 'git_commit', 'git_dirty', 'git_repo'):
            if key in hits_cfg.attrs:
                g_cfg.attrs[key] = hits_cfg.attrs[key]

        for i in range(n_events):
            evt_key = f'event_{i:03d}'
            if evt_key not in f_hits:
                continue
            hits_evt = f_hits[evt_key]
            source_event_idx = int(hits_evt.attrs.get('source_event_idx', i))

            track_lookup = _build_track_lookup(extractor, source_event_idx)

            labl_evt = f_labl.create_group(evt_key)
            if 'event_id' in hits_evt.attrs:
                labl_evt.attrs['event_id'] = hits_evt.attrs['event_id']
            for v in range(n_volumes):
                vol_key = f'volume_{v}'
                if vol_key not in hits_evt:
                    continue
                hits_vol = hits_evt[vol_key]
                labl_vol = labl_evt.create_group(vol_key)

                labels = _volume_labels(hits_vol, track_lookup)
                for name, arr in labels.items():
                    labl_vol.create_dataset(name, data=arr, compression='gzip')


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dummy labl/ directory from hits + edepsim.")
    parser.add_argument('--outdir', required=True,
                        help='Base dataset directory (must contain hits/). '
                             'labl/ is created here.')
    parser.add_argument('--source', default=None,
                        help='Path to the edepsim source HDF5 file. '
                             'Defaults to the per-file source_file attr '
                             'stored in hits, resolved relative to cwd.')
    parser.add_argument('--dataset', default='sim',
                        help="File prefix (default: 'sim')")
    parser.add_argument('--hits-subdir', default='hits')
    parser.add_argument('--labl-subdir', default='labl')
    args = parser.parse_args()

    hits_dir = os.path.join(args.outdir, args.hits_subdir)
    labl_dir = os.path.join(args.outdir, args.labl_subdir)

    if not os.path.isdir(hits_dir):
        sys.exit(f"Hits directory not found: {hits_dir}")
    os.makedirs(labl_dir, exist_ok=True)

    pattern = os.path.join(hits_dir, f'{args.dataset}_hits_*.h5')
    hits_files = sorted(glob.glob(pattern))
    if not hits_files:
        sys.exit(f"No hits files matching {pattern}")

    for hits_path in hits_files:
        basename = os.path.basename(hits_path)
        stem = basename.rsplit('.', 1)[0]
        file_index = int(stem.rsplit('_', 1)[-1])
        labl_path = os.path.join(
            labl_dir, f'{args.dataset}_labl_{file_index:04d}.h5')

        if args.source:
            source = args.source
        else:
            with h5py.File(hits_path, 'r') as f:
                source = str(f['config'].attrs.get('source_file', 'out.h5'))
        if not os.path.exists(source):
            sys.exit(f"edepsim source not found: {source} (use --source to override)")

        t0 = time.time()
        make_labl(hits_path, labl_path, source, args.dataset, file_index)
        print(f'{basename} -> {os.path.basename(labl_path)} '
              f'[{time.time() - t0:.2f}s] (source={os.path.basename(source)})')

    print(f'\nDone. {len(hits_files)} labl file(s) written to {labl_dir}/')


if __name__ == '__main__':
    main()
