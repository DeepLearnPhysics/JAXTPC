"""Export per-plane wire-readout geometry from a JAXTPC detector config to a
portable JSON that pimm-data (and other downstream consumers) can read WITHOUT
importing JAXTPC. This is the source-of-truth geometry the dense sensor path
needs (n_wires, per-wire lengths, pedestal, num_time_steps), instead of
hardcoding it.

Usage:
    python3 scripts/export_plane_geometry.py config/cubic_wireplane_config.yaml out.json
"""
import json
import sys

import numpy as np
import yaml

from tools.geometry import generate_detector
from tools.config import create_sim_config


def export(config_path, out_path):
    raw = yaml.safe_load(open(config_path))
    det = generate_detector(config_path)
    cfg = create_sim_config(det)

    # pedestals: prefer the readout/digitization section, else JAXTPC defaults
    dig = (raw.get('readout', {}) or {}).get('digitization', {}) or {}
    ped_coll = int(dig.get('pedestal_collection', 410))
    ped_ind = int(dig.get('pedestal_induction', 1843))
    sr_hz = float((raw.get('readout', {}) or {}).get('sampling_rate', 2.0)) * 1e6

    out = {
        'detector': config_path.split('/')[-1].replace('.yaml', ''),
        'num_time_steps': int(cfg.num_time_steps),
        'sampling_rate_hz': sr_hz,
        'coherent': (raw.get('simulation', {}) or {}).get('coherent_noise', {}),
        'planes': {},
    }
    for v in range(cfg.n_volumes):
        vol = cfg.volumes[v]
        for p in range(vol.n_planes):
            ptype = cfg.plane_names[v][p]
            label = f'volume_{v}_{ptype}'
            wl = np.asarray(vol.wire_lengths_m[p], dtype=np.float64)
            out['planes'][label] = {
                'n_wires': int(vol.num_wires[p]),
                'pedestal': ped_coll if ptype == 'Y' else ped_ind,
                # constant-length planes (collection) store a scalar; varying
                # (induction) store the full per-wire array.
                'wire_lengths_m': (float(wl[0]) if wl.min() == wl.max()
                                   else wl.tolist()),
            }
    json.dump(out, open(out_path, 'w'))
    print(f"wrote {out_path}: detector={out['detector']} "
          f"num_time_steps={out['num_time_steps']} planes={list(out['planes'])}")
    for label, e in out['planes'].items():
        wl = e['wire_lengths_m']
        wlrep = f"const {wl:.3f}m" if isinstance(wl, float) else \
            f"vary {min(wl):.3f}-{max(wl):.3f}m ({len(wl)})"
        print(f"  {label}: n_wires={e['n_wires']} pedestal={e['pedestal']} wire_len={wlrep}")


if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2])
