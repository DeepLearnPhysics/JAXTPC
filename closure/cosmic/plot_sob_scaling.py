#!/usr/bin/env python3
"""SCE recovery scaling under the REAL (correct) intrinsic noise, Sobolev loss.

Left:  |E| MAE vs number of cosmic muons M (coverage scaling).
Right: |E| MAE vs optimization step, one curve per M (convergence).

Reads closure/cosmic/scan_sob/m*.json (run_accum, --real-noise, Sobolev loss).
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sob_scaling.png')
NOISELESS = 1.63   # M=256/4k-step reference (Sobolev, no noise)


def main():
    files = sorted(glob.glob(os.path.join(HERE, 'scan_sob', 'm*.json')),
                   key=lambda f: json.load(open(f))['n_muons'])
    res = [json.load(open(f)) for f in files]
    if not res:
        print('no scan_sob results yet'); return

    Ms = np.array([r['n_muons'] for r in res])
    val = np.array([r['val_mae'] for r in res])
    last = np.array([r['emae'][-1] for r in res])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # -- coverage scaling --
    ax[0].loglog(Ms, val, 'o-', color='C0', lw=2, ms=8, label='val-selected')
    ax[0].loglog(Ms, last, 's--', color='C0', alpha=0.4, label='last step')
    ax[0].loglog(Ms, val[0] * np.sqrt(Ms[0] / Ms), 'k:', alpha=0.5, label=r'$\propto M^{-1/2}$')
    ax[0].axhline(NOISELESS, color='g', ls='-', lw=1.2, alpha=0.7,
                  label=f'noiseless ref ≈ {NOISELESS}')
    for m, v in zip(Ms, val):
        ax[0].annotate(f'{v:.2f}', (m, v), textcoords='offset points', xytext=(5, 6), fontsize=8)
    sl = np.polyfit(np.log(Ms), np.log(val), 1)[0]
    ax[0].set(xlabel='number of cosmic muons M', ylabel='|E| MAE vs truth (V/cm)',
              title=f'Coverage scaling (Sobolev + real noise)\noverall slope M^{sl:.2f}')
    ax[0].legend(fontsize=8); ax[0].grid(True, which='both', alpha=0.3)

    # -- convergence (step scaling) --
    cmap = plt.get_cmap('viridis')
    for i, r in enumerate(res):
        e = np.array(r['emae'])
        steps = np.linspace(0, r['steps'], len(e))
        ax[1].loglog(np.maximum(steps, 1), e, color=cmap(i / max(1, len(res) - 1)),
                     lw=1.6, label=f"M={r['n_muons']}")
    ax[1].axhline(NOISELESS, color='g', ls='-', lw=1.2, alpha=0.7)
    ax[1].set(xlabel='optimization step', ylabel='|E| MAE vs truth (V/cm)',
              title='Convergence vs steps (per M)')
    ax[1].legend(fontsize=8); ax[1].grid(True, which='both', alpha=0.3)

    fig.suptitle('SCE field recovery under REAL intrinsic noise (~1.25 ADC, SNR~90-990): '
                 'steps × muons', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'saved {OUT}')
    print(f"{'M':>7}{'val':>9}{'last':>9}")
    for m, v, l in zip(Ms, val, last):
        print(f'{m:>7}{v:>9.2f}{l:>9.2f}')


if __name__ == '__main__':
    main()
