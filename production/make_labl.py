"""
Temporary labl generator.

Produces the ``labl/`` directory from the hits/ output + the edepsim
source file. Step is **not** consulted because step is pure physics
and does not carry track-related arrays.

Per-deposit → track_id comes from hits::

    deposit i  -> hits.volume_N.deposit_to_group[i] = g
               -> hits.volume_N.group_to_track[g]   = track_id

Per-track metadata (pdg, interaction, ancestor) comes from the edepsim
source via :class:`tools.loader.ParticleStepExtractor`.

Output layout (matches pimm-data's ``JAXTPCLablReader``)::

    {outdir}/labl/{dataset}_labl_{NNNN}.h5
        /config/ attrs
        /entry_NNNNNN/volume_N/   (same entry names and identity attrs as hits)
            # Per-deposit (N,) foreign key row-aligned with step rows
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

# Register blosc/zstd/lz4 HDF5 filters so hits output written with the
# default run_batch codec (blosc-zstd) is readable. No-op if absent.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# Ensure tools/ is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.loader import ParticleStepExtractor, compute_interaction_ids
from production.entries import DPROD_ID_ATTRS, entry_keys, source_files


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
    if ('deposit_to_group' not in hits_vol_group
            or 'group_to_track' not in hits_vol_group):
        empty = np.array([], dtype=np.int32)
        return dict(deposit_to_track=empty, track_ids=empty,
                    track_pdg=empty, track_interaction=empty,
                    track_cluster=empty, track_ancestor=empty)

    dep_to_grp = hits_vol_group['deposit_to_group'][:].astype(np.int32)
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
    """Generate a labl file from hits + edepsim source(s).

    *source_file* overrides the edep-sim input of a single-input hits file;
    None takes each entry's input from the hits file (config/source_files and
    the entry's source_index), which is how a merged file is read.
    """
    extractors = {}

    def _extractor(path):
        if path not in extractors:
            extractors[path] = ParticleStepExtractor(path).__enter__()
        return extractors[path]

    try:
        _make_labl(hits_path, labl_path, source_file, dataset_name, file_index,
                   _extractor)
    finally:
        for ex in extractors.values():
            ex.__exit__(None, None, None)


def _make_labl(hits_path, labl_path, source_file, dataset_name, file_index,
               _extractor):
    with h5py.File(hits_path, 'r') as f_hits, \
            h5py.File(labl_path, 'w') as f_labl:

        hits_cfg = f_hits['config']
        sources = source_files(f_hits)
        if source_file is not None:
            if len(sources) > 1:
                raise ValueError(f'{hits_path} merges {len(sources)} inputs; '
                                 f'take them from the file (no --source)')
            sources = [source_file]
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
        g_cfg.attrs['source'] = 'dummy-from-edepsim:' + ','.join(
            os.path.basename(p) for p in sources)
        g_cfg.attrs['generator'] = 'production/make_labl.py'
        for key in ('format_version', 'n_sources', 'source_file', 'global_event_offset',
                    'group_size', 'gap_threshold_mm', 'production_version', 'run_id',
                    'batch_timestamp', 'git_commit', 'git_dirty', 'git_repo'):
            if key in hits_cfg.attrs:
                g_cfg.attrs[key] = hits_cfg.attrs[key]
        if 'source_files' in hits_cfg:
            g_cfg.create_dataset('source_files', data=hits_cfg['source_files'][()])

        for i, evt_key in enumerate(entry_keys(f_hits)):
            hits_evt = f_hits[evt_key]
            source_event_idx = int(hits_evt.attrs.get('source_event_idx', i))
            src = sources[int(hits_evt.attrs.get('source_index', 0))]

            track_lookup = _build_track_lookup(_extractor(src), source_event_idx)

            labl_evt = f_labl.create_group(evt_key)
            for key in ('source_index', 'source_event_idx') + DPROD_ID_ATTRS:
                if key in hits_evt.attrs:
                    labl_evt.attrs[key] = hits_evt.attrs[key]
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
        # {dataset}_hits_{suffix}.h5 -> {dataset}_labl_{suffix}.h5 (suffix: NNNN,
        # or the --merge-output name of a merged file)
        suffix = basename[len(f'{args.dataset}_hits_'):-len('.h5')]
        labl_path = os.path.join(labl_dir, f'{args.dataset}_labl_{suffix}.h5')
        with h5py.File(hits_path, 'r') as f:
            file_index = int(f['config'].attrs.get('file_index', 0))
            sources = [args.source] if args.source else source_files(f)
        missing = [s for s in sources if not os.path.exists(s)]
        if not sources or missing:
            sys.exit(f"edepsim source not found: {missing or '(none recorded)'} "
                     f"(use --source to override)")

        t0 = time.time()
        make_labl(hits_path, labl_path, args.source, args.dataset, file_index)
        print(f'{basename} -> {os.path.basename(labl_path)} '
              f'[{time.time() - t0:.2f}s] '
              f'(source={",".join(os.path.basename(s) for s in sources)})')

    print(f'\nDone. {len(hits_files)} labl file(s) written to {labl_dir}/')


if __name__ == '__main__':
    main()
