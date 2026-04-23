"""
Production HDF5 load functions.

Reads simulation output from the three file types produced by
``run_batch.py`` (canonical names):
    sensor — sparse thresholded raw readout
    seg    — 3D truth deposits (per-volume)
    inst   — per-instance sensor decomposition
"""

import os
import numpy as np
import h5py
from collections import namedtuple


_PLANE_LABELS = {0: 'U', 1: 'V', 2: 'Y'}


def _plane_label(plane_idx):
    """Plane index → short label."""
    return _PLANE_LABELS.get(plane_idx, str(plane_idx))


# =============================================================================
# File paths
# =============================================================================

def get_file_paths(production_dir, dataset, file_index):
    """Return (sensor_path, seg_path, inst_path) for a given batch file."""
    return tuple(
        os.path.join(production_dir, stem,
                     f'{dataset}_{stem}_{file_index:04d}.h5')
        for stem in ('sensor', 'seg', 'inst')
    )


# =============================================================================
# Config / metadata
# =============================================================================

# Minimal config objects that satisfy visualization functions
_VolGeomMin = namedtuple('_VolGeomMin', ['num_wires', 'n_planes'])
_ConfigMin = namedtuple('_ConfigMin', [
    'volumes', 'n_volumes', 'num_time_steps', 'time_step_us', 'electrons_per_adc',
    'plane_names'])


def load_config(sensor_path):
    """Load production metadata from a response file.

    Returns a dict with all config attributes plus 'num_wires_arr'.
    """
    with h5py.File(sensor_path, 'r') as f:
        cfg = f['config']
        meta = dict(cfg.attrs)
        meta['num_wires_arr'] = cfg['num_wires'][:]
    return meta


def build_viz_config(sensor_path):
    """Build a minimal config object for visualization functions.

    Only requires the response file — no YAML or generate_detector needed.
    """
    meta = load_config(sensor_path)
    readout_type = meta.get('readout_type', 'wire')
    nw = meta['num_wires_arr']
    n_vol = nw.shape[0]
    n_planes = nw.shape[1]
    volumes = tuple(
        _VolGeomMin(
            num_wires=tuple(int(nw[v, p]) for p in range(n_planes)),
            n_planes=int(np.sum(nw[v] > 0)),
        )
        for v in range(n_vol)
    )
    _PLANE_LABELS = ('U', 'V', 'Y')
    plane_names = tuple(
        ('Pixel',) if readout_type == 'pixel'
        else tuple(_PLANE_LABELS[:vol.n_planes])
        for vol in volumes)
    return _ConfigMin(
        volumes=volumes,
        n_volumes=n_vol,
        num_time_steps=int(meta['num_time_steps']),
        time_step_us=float(meta['time_step_us']),
        electrons_per_adc=float(meta['electrons_per_adc']),
        plane_names=plane_names,
    )


# =============================================================================
# Response loading
# =============================================================================

def load_event_sensor(sensor_path, event_idx):
    """Load one event's response signals as dense arrays.

    Automatically detects wire (2D) vs pixel (3D) format, and
    uint16 (digitized) vs float32.

    Returns
    -------
    dense_signals : dict
        Wire: {(vol, plane): (num_wires, num_time_steps) ndarray}
        Pixel: {(vol, plane): (num_py, num_pz, num_time_steps) ndarray}
    event_attrs : dict
    pedestals : dict or None
    """
    event_key = f'event_{event_idx:03d}'

    with h5py.File(sensor_path, 'r') as f:
        cfg_grp = f['config']
        num_time_steps = int(cfg_grp.attrs['num_time_steps'])
        num_wires_arr = cfg_grp['num_wires'][:]
        n_vol = num_wires_arr.shape[0]
        n_planes = num_wires_arr.shape[1]

        evt = f[event_key]
        event_attrs = dict(evt.attrs)

        dense_signals = {}
        pedestals = {}
        digitized = False

        for v in range(n_vol):
            vol_key = f'volume_{v}'
            if vol_key not in evt:
                continue

            # Discover plane labels from HDF5 groups
            vol_grp = evt[vol_key]
            for plabel in vol_grp:
                g = vol_grp[plabel]
                if not isinstance(g, h5py.Group):
                    continue

                # Detect pixel vs wire by presence of delta_py
                is_pixel = 'delta_py' in g

                if is_pixel:
                    # Pixel 3D format
                    if 'n_pixels' not in g.attrs:
                        continue
                    py_start = int(g.attrs['py_start'])
                    pz_start = int(g.attrs['pz_start'])
                    time_start = int(g.attrs['time_start'])

                    pys = py_start + np.cumsum(g['delta_py'][:]).astype(np.int32)
                    pzs = pz_start + np.cumsum(g['delta_pz'][:]).astype(np.int32)
                    times = time_start + np.cumsum(g['delta_time'][:]).astype(np.int32)

                    # Infer pixel grid size from volume_ranges if available
                    if 'volume_ranges' in cfg_grp:
                        vr = cfg_grp['volume_ranges'][v]  # (3, 2) in mm
                        # Assume pixel pitch from range / max pixel index
                        num_py = int(np.max(pys)) + 1 if len(pys) > 0 else 1
                        num_pz = int(np.max(pzs)) + 1 if len(pzs) > 0 else 1
                    else:
                        num_py = int(np.max(pys)) + 1 if len(pys) > 0 else 1
                        num_pz = int(np.max(pzs)) + 1 if len(pzs) > 0 else 1

                    values = g['values'][:].astype(np.float32)
                    valid = ((pys >= 0) & (pys < num_py) &
                             (pzs >= 0) & (pzs < num_pz) &
                             (times >= 0) & (times < num_time_steps))

                    dense = np.zeros((num_py, num_pz, num_time_steps), dtype=np.float32)
                    dense[pys[valid], pzs[valid], times[valid]] = values[valid]

                    # Find plane index from label
                    p = 0  # pixel volumes have plane_idx=0
                    dense_signals[(v, p)] = dense

                else:
                    # Wire 2D format
                    if 'delta_wire' not in g:
                        continue
                    # Find plane index from label
                    p = {'U': 0, 'V': 1, 'Y': 2}.get(plabel, 0)
                    nw = int(num_wires_arr[v, p])
                    if nw == 0:
                        continue

                    wire_start = int(g.attrs['wire_start'])
                    time_start = int(g.attrs['time_start'])

                    wires = wire_start + np.cumsum(g['delta_wire'][:]).astype(np.int32)
                    times = time_start + np.cumsum(g['delta_time'][:]).astype(np.int32)
                    valid = ((wires >= 0) & (wires < nw) &
                             (times >= 0) & (times < num_time_steps))

                    raw_values = g['values'][:]
                    if raw_values.dtype == np.uint16:
                        digitized = True
                        ped = int(g.attrs['pedestal'])
                        pedestals[(v, p)] = ped
                        dense = np.zeros((nw, num_time_steps), dtype=np.uint16)
                        dense[wires[valid], times[valid]] = raw_values[valid]
                    else:
                        dense = np.zeros((nw, num_time_steps), dtype=np.float32)
                        dense[wires[valid], times[valid]] = raw_values[valid]

                    dense_signals[(v, p)] = dense

    return dense_signals, event_attrs, pedestals if digitized else None


# =============================================================================
# Segment loading
# =============================================================================

def load_event_seg(seg_path, event_idx):
    """Load one event's 3D truth deposits (per-volume).

    Per the design (see particle-imaging-models/docs/DATASET_DESIGN.md),
    seg holds pure physics only. Group/instance fields live in inst,
    per-track labels in labl.

    Returns
    -------
    volumes : list of dict, one per volume. Each has:
        positions_mm (N, 3), de (N,), dx (N,), theta (N,), phi (N,),
        t0_us (N,), charge (N,), photons (N,), n_actual (int).
    """
    event_key = f'event_{event_idx:03d}'

    with h5py.File(seg_path, 'r') as f:
        evt = f[event_key]
        n_volumes = int(evt.attrs.get('n_volumes', 2))

        volumes = []
        for v in range(n_volumes):
            vg_key = f'volume_{v}'
            if vg_key not in evt:
                volumes.append({'n_actual': 0})
                continue

            vg = evt[vg_key]
            n = int(vg.attrs['n_actual'])

            if n == 0:
                volumes.append({'n_actual': 0})
                continue

            pos_step = float(vg.attrs['pos_step_mm'])
            origin = np.array([vg.attrs['pos_origin_x'],
                               vg.attrs['pos_origin_y'],
                               vg.attrs['pos_origin_z']])
            positions_mm = vg['positions'][:].astype(np.float32) * pos_step + origin

            vol = {
                'positions_mm': positions_mm,
                'de': vg['de'][:].astype(np.float32),
                'dx': vg['dx'][:].astype(np.float32),
                'theta': vg['theta'][:].astype(np.float32),
                'phi': vg['phi'][:].astype(np.float32),
                't0_us': vg['t0_us'][:].astype(np.float32),
                'charge': vg['charge'][:].astype(np.float32),
                'photons': vg['photons'][:].astype(np.float32),
                'n_actual': n,
            }
            volumes.append(vol)

    return volumes


# =============================================================================
# Correspondence loading
# =============================================================================

def _decode_plane_inst(g, num_time_steps):
    """Decode one plane's CSR correspondence into flat arrays."""
    grp_ids = g['group_ids'][:]
    grp_sizes = g['group_sizes'][:]
    center_wires = g['center_wires'][:]
    center_times = g['center_times'][:]
    peak_charges = g['peak_charges'][:]
    delta_wires = g['delta_wires'][:]
    delta_times = g['delta_times'][:]
    charges_u16 = g['charges_u16'][:]

    group_starts = np.cumsum(grp_sizes) - grp_sizes
    n_entries = int(grp_sizes.sum())

    pk_flat = np.empty(n_entries, dtype=np.int32)
    gid_flat = np.empty(n_entries, dtype=np.int32)
    ch_flat = np.empty(n_entries, dtype=np.float32)

    for i in range(len(grp_ids)):
        s = int(group_starts[i])
        sz = int(grp_sizes[i])
        w = int(center_wires[i]) + delta_wires[s:s + sz].astype(np.int32)
        t = int(center_times[i]) + delta_times[s:s + sz].astype(np.int32)
        ch = float(peak_charges[i]) * charges_u16[s:s + sz].astype(np.float32) / 65535.0

        pk_flat[s:s + sz] = w * num_time_steps + t
        gid_flat[s:s + sz] = grp_ids[i]
        ch_flat[s:s + sz] = ch

    return pk_flat, gid_flat, ch_flat, n_entries


def _decode_plane_inst_pixel(g):
    """Decode one pixel plane's CSR correspondence into flat arrays."""
    grp_ids = g['group_ids'][:]
    grp_sizes = g['group_sizes'][:]
    center_py = g['center_py'][:]
    center_pz = g['center_pz'][:]
    center_times = g['center_times'][:]
    peak_charges = g['peak_charges'][:]
    delta_py = g['delta_py'][:]
    delta_pz = g['delta_pz'][:]
    delta_times = g['delta_times'][:]
    charges_u16 = g['charges_u16'][:]

    group_starts = np.cumsum(grp_sizes) - grp_sizes
    n_entries = int(grp_sizes.sum())

    py_flat = np.empty(n_entries, dtype=np.int32)
    pz_flat = np.empty(n_entries, dtype=np.int32)
    t_flat = np.empty(n_entries, dtype=np.int32)
    gid_flat = np.empty(n_entries, dtype=np.int32)
    ch_flat = np.empty(n_entries, dtype=np.float32)

    for i in range(len(grp_ids)):
        s = int(group_starts[i])
        sz = int(grp_sizes[i])
        py = int(center_py[i]) + delta_py[s:s + sz].astype(np.int32)
        pz = int(center_pz[i]) + delta_pz[s:s + sz].astype(np.int32)
        t = int(center_times[i]) + delta_times[s:s + sz].astype(np.int32)
        ch = float(peak_charges[i]) * charges_u16[s:s + sz].astype(np.float32) / 65535.0

        py_flat[s:s + sz] = py
        pz_flat[s:s + sz] = pz
        t_flat[s:s + sz] = t
        gid_flat[s:s + sz] = grp_ids[i]
        ch_flat[s:s + sz] = ch

    return py_flat, pz_flat, t_flat, gid_flat, ch_flat, n_entries


def load_event_inst(inst_path, event_idx, num_time_steps=None,
                    n_volumes=None, max_planes=3):
    """Load correspondence and derive track labels + diffused charge.

    Parameters
    ----------
    inst_path : str
    event_idx : int
    num_time_steps : int, optional
        If ``None``, read from inst ``/config`` attrs.
    n_volumes : int, optional
        If ``None``, read from inst ``/config`` attrs.
    max_planes : int

    Returns
    -------
    track_hits : dict
        {(vol, plane): result from label_from_groups}
    truth_dense : dict
        {(vol, plane): (num_wires, num_time) ndarray}
    group_to_track : list of arrays, one per volume
    segment_to_group : list of arrays, one per volume (per-deposit group id;
        row-aligned with seg deposits in the same volume; None where missing)
    qs_fractions : list of arrays, one per volume (per-deposit fraction of
        its group's recombined charge; None where missing)
    """
    from tools.track_hits import label_from_groups

    event_key = f'event_{event_idx:03d}'
    track_hits = {}
    truth_dense = {}

    with h5py.File(inst_path, 'r') as f:
        if num_time_steps is None:
            num_time_steps = int(f['config'].attrs['num_time_steps'])
        if n_volumes is None:
            n_volumes = int(f['config'].attrs['n_volumes'])
        evt = f[event_key]
        nw_arr = f['config']['num_wires'][:]

        # Load per-volume group_to_track + per-deposit group/qs assignments
        g2t_per_vol = []
        s2g_per_vol = []
        qs_per_vol = []
        for v in range(n_volumes):
            vol_key = f'volume_{v}'
            if vol_key in evt:
                vol_grp = evt[vol_key]
                g2t_per_vol.append(vol_grp['group_to_track'][:]
                                   if 'group_to_track' in vol_grp
                                   else np.array([0], dtype=np.int32))
                s2g_per_vol.append(vol_grp['segment_to_group'][:]
                                   if 'segment_to_group' in vol_grp else None)
                qs_per_vol.append(vol_grp['qs_fractions'][:].astype(np.float32)
                                  if 'qs_fractions' in vol_grp else None)
            else:
                g2t_per_vol.append(np.array([0], dtype=np.int32))
                s2g_per_vol.append(None)
                qs_per_vol.append(None)

        for v in range(n_volumes):
            vol_key = f'volume_{v}'
            if vol_key not in evt:
                continue

            vol_grp = evt[vol_key]
            for plabel in vol_grp:
                g = vol_grp[plabel]
                if not isinstance(g, h5py.Group) or 'group_ids' not in g:
                    continue

                # Detect pixel vs wire
                is_pixel = 'delta_py' in g

                # Find plane index from label
                p = {'U': 0, 'V': 1, 'Y': 2, 'Pixel': 0}.get(plabel, 0)

                if is_pixel:
                    py, pz, t, gid, ch, n_entries = _decode_plane_inst_pixel(g)

                    # Diffused charge as sparse dict (no dense 3D array)
                    truth_dense[(v, p)] = {
                        'pixel_y': py, 'pixel_z': pz,
                        'time': t, 'values': ch,
                    }

                    # Track labels with pixel decode
                    num_pz = int(np.max(pz)) + 1 if len(pz) > 0 else 1
                    def decode_pixel(sk, _npz=num_pz):
                        return np.column_stack([sk // _npz, sk % _npz])

                    # Pack spatial key for label_from_groups
                    sk = py * num_pz + pz
                    result = label_from_groups(
                        sk, t, gid, ch, n_entries,
                        g2t_per_vol[v], decode_spatial_fn=decode_pixel)
                    track_hits[(v, p)] = result

                else:
                    nw = int(nw_arr[v, p])
                    if nw == 0:
                        continue

                    pk, gid, ch, n_entries = _decode_plane_inst(g, num_time_steps)

                    dense = np.zeros((nw, num_time_steps), dtype=np.float32)
                    all_w = pk // num_time_steps
                    all_t = pk % num_time_steps
                    valid = (all_w >= 0) & (all_w < nw) & (all_t >= 0) & (all_t < num_time_steps)
                    np.add.at(dense, (all_w[valid], all_t[valid]), ch[valid])
                    truth_dense[(v, p)] = dense

                    result = label_from_groups(
                        all_w[valid], all_t[valid], gid[valid], ch[valid],
                        np.sum(valid),
                        g2t_per_vol[v])
                    track_hits[(v, p)] = result

    return track_hits, truth_dense, g2t_per_vol, s2g_per_vol, qs_per_vol


# =============================================================================
# Segment <-> pixel correspondence
# =============================================================================

def load_correspondence(inst_path, event_idx, v, num_time_steps=None):
    """Per-volume deposit<->pixel correspondence tables from inst.

    Unlike ``load_event_inst`` (which sums per-pixel charge and erases
    group identity), this returns each plane's raw group-keyed entries
    so forward/backward correspondence is a direct lookup.

    Parameters
    ----------
    inst_path : str
    event_idx : int
    v : int
        Volume index.
    num_time_steps : int, optional
        If ``None``, read from the file's ``/config`` attrs.

    Returns
    -------
    dict with keys:
        's2g'    (N_dep,) int32    — segment_to_group (row-aligned with seg[v])
        'qs'     (N_dep,) float32  — qs_fractions
        'g2t'    (n_groups,) int32 — group_to_track (optional label)
        'planes' {label: {'wire', 'time', 'charge', 'gid'}} — per-plane flat
                 arrays; wire/pixel-y + pixel-z for pixel readouts.
    """
    with h5py.File(inst_path, 'r') as f:
        if num_time_steps is None:
            num_time_steps = int(f['config'].attrs['num_time_steps'])
        evt = f[f'event_{event_idx:03d}']
        vg_key = f'volume_{v}'
        if vg_key not in evt:
            return None
        vg = evt[vg_key]
        s2g = vg['segment_to_group'][:]
        qs  = vg['qs_fractions'][:].astype(np.float32)
        g2t = vg['group_to_track'][:]

        planes = {}
        for plabel in vg:
            g = vg[plabel]
            if not isinstance(g, h5py.Group) or 'group_ids' not in g:
                continue
            if 'delta_py' in g:
                py, pz, t, gid, ch, _ = _decode_plane_inst_pixel(g)
                planes[plabel] = dict(pixel_y=py, pixel_z=pz,
                                      time=t, charge=ch, gid=gid)
            else:
                pk, gid, ch, _ = _decode_plane_inst(g, num_time_steps)
                planes[plabel] = dict(wire=pk // num_time_steps,
                                      time=pk % num_time_steps,
                                      charge=ch, gid=gid)

    return {'s2g': s2g, 'qs': qs, 'g2t': g2t, 'planes': planes}


def segment_charge_per_plane(corr, plane_label):
    """Total electrons from each deposit landing on pixels of one plane.

    Parameters
    ----------
    corr : dict
        Output of :func:`load_correspondence`.
    plane_label : str
        'U', 'V', 'Y', or 'Pixel'.

    Returns
    -------
    (N_dep,) float32. Entries are zero for deposits whose group fell
    below the inst write threshold on this plane (no pixel entries).
    """
    d = corr['planes'][plane_label]
    grp_tot = np.bincount(d['gid'], weights=d['charge'],
                          minlength=corr['g2t'].size).astype(np.float32)
    return corr['qs'] * grp_tot[corr['s2g']]
