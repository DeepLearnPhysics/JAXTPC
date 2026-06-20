#!/usr/bin/env python3
"""Aggregate per-case JSON results into a track-count scaling figure.

Overlays the unregularised scan (scan/) and the smoothness-regularised scan
(scan_reg/) so the effect of the prior on coverage scaling is visible.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'closure_scaling.png')


def load(d):
    files = [f for f in glob.glob(os.path.join(HERE, d, 'n*.json'))
             if 'smoke' not in f]
    res = sorted((json.load(open(f)) for f in files), key=lambda r: r['n_tracks'])
    return res


def series(res):
    nt = np.array([r['n_tracks'] for r in res])
    best = np.array([min(r['emae']) for r in res])
    last = np.array([r['emae'][-1] for r in res])
    return nt, best, last


def main():
    unreg = load('scan')
    reg = load('scan_reg')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for res, lab, c in [(unreg, 'no reg', 'C0'), (reg, 'smoothness λ=1e2', 'C1')]:
        if not res:
            continue
        nt, best, last = series(res)
        lam = res[0].get('reg', 0.0)
        print(f"--- {lab} ---")
        print(f"{'tracks':>7}{'best MAE':>10}{'best step':>10}{'last MAE':>10}")
        for r in res:
            e = np.array(r['emae'])
            print(f"{r['n_tracks']:>7}{e.min():>10.2f}{int(e.argmin()):>10}{e[-1]:>10.2f}")
        ax[0].loglog(nt, best, '-o', color=c, label=f'best (early-stop), {lab}')
        ax[0].loglog(nt, last, '--s', color=c, alpha=0.5, label=f'last step, {lab}')

    if unreg:
        nt, best, _ = series(unreg)
        ax[0].loglog(nt, best[0] * np.sqrt(nt[0] / nt), 'k:', alpha=0.5,
                     label=r'$\propto N^{-1/2}$')
    ax[0].set(xlabel='number of cosmic tracks', ylabel='|E| MAE vs truth (V/cm)',
              title='full-field recovery error vs track coverage')
    ax[0].legend(fontsize=8); ax[0].grid(True, which='both', alpha=0.3)

    for res, lab, cm in [(unreg, 'no reg', 'Blues'), (reg, 'λ=1e2', 'Oranges')]:
        for i, r in enumerate(res):
            e = r['emae']
            ax[1].plot(np.arange(len(e)), e, color=plt.get_cmap(cm)(0.4 + 0.5 * i / max(1, len(res) - 1)),
                       label=f"{r['n_tracks']}t {lab}", lw=1)
    ax[1].set(xlabel='optimization step', ylabel='|E| MAE (V/cm)', yscale='log',
              title='convergence (overfitting after the early minimum)')
    ax[1].legend(fontsize=6, ncol=2); ax[1].grid(True, alpha=0.3)

    fig.suptitle('SCE full-field recovery: track coverage × smoothness prior '
                 '(parallel GPUs)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
