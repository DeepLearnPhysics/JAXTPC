"""CPU-only value scan: charge-aware max_keys + maxg + box dims + plots.

Runs find_optimal_maxg with the charge-aware estimator (threshold c*xinter_thresh)
over all files, then patches an existing production config's max_keys/maxg/box
(keeping its GPU-derived chunks + maxg_medium). No GPU needed -- only the chunk
optimization does. Saves distribution plots.
"""
import sys, glob, yaml, math, os, argparse
sys.path.insert(0, '/sdf/group/neutrino/omara/JAXTPC')
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--existing', required=True, help='config to patch (chunks/maxg_medium kept)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--cstar', type=float, default=2.5,
                    help='threshold multiplier (pixel: 2.5 -> 1.0x). Wire plateaus, '
                         'so wire uses --cstar 1 with --divisor instead.')
    ap.add_argument('--divisor', type=float, default=1.0,
                    help='divide the estimate by this calibrated overlap factor '
                         '(wire ~3.79; pixel 1.0). Applied after cstar.')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--headroom', type=float, default=1.1)
    a = ap.parse_args()

    from tools.geometry import generate_detector
    from tools.config import create_sim_config
    from profiler.find_optimal_maxg import find_optimal_maxg
    from profiler.estimate_max_keys import (build_pixel_value_table,
        build_wire_value_table, build_charge_model)

    dc = generate_detector(a.config)
    sc = create_sim_config(dc, total_pad=2_000_000)
    cm = build_charge_model(yaml.safe_load(open(a.config)))
    vt = {v: (build_pixel_value_table(sc, vg) if vg.readout_type == 'pixel'
              else build_wire_value_table(vg.diffusion))
          for v, vg in enumerate(sc.volumes)}
    files = []
    for d in a.data:
        files += sorted(glob.glob(os.path.join(d, '*.h5'))) if os.path.isdir(d) else [d]
    print(f"[{a.tag}] scanning {len(files)} files, cstar={a.cstar}, workers={a.workers}", flush=True)

    maxg, info = find_optimal_maxg(
        files, a.config, n_workers=a.workers, dim_files=20,
        value_tables=vt, charge_model=cm, key_thresh=a.cstar * 1.0, pctile=99.95)

    dep = np.asarray(info['deposits']); keys = np.asarray(info['keys']) / a.divisor
    ng = np.asarray(info['n_groups']); box = info.get('box_dims', {})
    max_keys = int(math.ceil(keys.max() * a.headroom / 1e5) * 1e5)

    cfg = yaml.safe_load(open(a.existing))
    old_pad = cfg['total_pad']; pad_needed = int(dep.max())
    note = (f'  *** deposit max {pad_needed:,} > total_pad {old_pad:,}: BUMP+REALIGN CHUNKS ***'
            if pad_needed > old_pad else '')
    cfg['maxg'] = maxg; cfg['max_keys'] = max_keys
    if box:
        if 'BPY' in box:
            cfg['box_bpy'] = box['BPY']; cfg['box_bpz'] = box['BPZ']; cfg['box_bt'] = box['BT']
        else:
            cfg['box_bw'] = box['BW']; cfg['box_btw'] = box['BT']

    print(f"[{a.tag}] maxg(p99.95)={maxg:,}  max_keys={max_keys:,} "
          f"(est max {int(keys.max()):,} x{a.headroom})  box={box}  "
          f"deposit_max={pad_needed:,}/total_pad={old_pad:,}{note}", flush=True)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figd = 'profiler/figures'; os.makedirs(figd, exist_ok=True)
    for arr, name, xl, line in [(dep, 'deposit_distribution', 'deposits/vol', old_pad),
                                (ng, 'maxg_distribution', 'n_groups/event-vol', maxg),
                                (keys, 'maxkeys_distribution', 'est keys', max_keys / a.headroom)]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(arr, bins=80, color='steelblue', alpha=0.8)
        ax.axvline(line, color='r', ls='--', lw=1.5)
        ax.set_xlabel(xl); ax.set_ylabel('count')
        ax.set_title(f'{name} ({a.tag}, {len(arr):,} event-vols)')
        fig.tight_layout()
        p = f'{figd}/{name}_{a.tag}.png'; fig.savefig(p, dpi=120); plt.close(fig)
        print(f'[{a.tag}] saved {p}', flush=True)

    with open(a.out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f'[{a.tag}] wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
