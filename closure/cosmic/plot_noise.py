#!/usr/bin/env python3
"""Consolidate the noise analysis into one figure.

Left  : the SCE field's signal SIGNATURE per readout plane vs the real
        intrinsic-noise floor (the observability diagnostic) — why single-event
        recovery is noise-limited.
Right : |E| MAE for each condition against the two anchors (noiseless recovery
        and the do-nothing flat-field baseline), showing the free recovery
        diverges far past "do nothing" under real noise and which lever (if any)
        pulls it back.

Numbers come from scan_batch/scan_real JSONs plus the measured diagnostics
(printed by the analysis; hard-coded here as annotations).
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'noise_analysis.png')

# measured diagnostics (ENC) for the 40 cm geometry, fine cosmics
SIG = {'U (induction)': 15.6, 'V (induction)': 16.8, 'Y (collection)': 120.1}
NOISE = 227.0
NOISELESS = 0.98  # SIREN representation floor (noiseless, 12k steps)
BASELINE = 13.4   # do-nothing: predict flat E0


def val_or_last(path):
    d = json.load(open(path)); e = np.array(d['emae'])
    return d.get('val_mae', e[-1])


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # -- observability diagnostic --
    names = list(SIG); vals = [SIG[n] for n in names]
    ax[0].bar(names, vals, color=['C0', 'C0', 'C2'])
    ax[0].axhline(NOISE, color='r', ls='--', label=f'real noise floor = {NOISE:.0f} ENC')
    ax[0].set(ylabel='per-event signal signature (ENC, RMS on-track)',
              title='Field signature vs noise: induction is buried,\nonly collection peeks through')
    for i, v in enumerate(vals):
        ax[0].text(i, v + 5, f'{v:.0f}', ha='center', fontsize=9)
    ax[0].legend(); ax[0].grid(True, axis='y', alpha=0.3)

    # -- recovery vs number of muons: noise-limited √M, then floor saturation --
    Ms, white = [], []
    for M, f in [(384, 'm384'), (768, 'm768'), (1536, 'm1536'),
                 (3072, 'm3072'), (12288, 'm12288')]:
        p = os.path.join(HERE, 'scan_big', f + '.json')
        if os.path.exists(p):
            Ms.append(M); white.append(val_or_last(p))
    Ms = np.array(Ms); white = np.array(white)
    ax[1].loglog(Ms, white, 'o-', color='C2', lw=2, ms=8,
                 label='whitened χ², real noise (12k steps)')
    if len(Ms):
        ax[1].loglog(Ms, white[0] * np.sqrt(Ms[0] / Ms), 'k:', alpha=0.6,
                     label=r'$\propto M^{-1/2}$ (coherent averaging)')
    sob = os.path.join(HERE, 'scan_white', 'm384_sobolev.json')
    if os.path.exists(sob):
        ax[1].loglog([384], [val_or_last(sob)], 'X', color='C3', ms=14,
                     label='Sobolev loss (noise-blind) → diverges')
    ax[1].axhline(NOISELESS, color='g', ls='-', lw=1.5,
                  label=f'noiseless SIREN floor = {NOISELESS}')
    ax[1].axhline(BASELINE, color='k', ls='--', lw=1.5, label=f'do-nothing baseline = {BASELINE}')
    for m, v in zip(Ms, white):
        ax[1].annotate(f'{v:.2f}', (m, v), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax[1].set(xlabel='number of cosmic muons M', ylabel='|E| MAE vs truth (V/cm)',
              title='Recovery under REAL noise: √M while noise-limited,\nthen saturates at the noiseless floor (~1 V/cm)')
    ax[1].legend(fontsize=8); ax[1].grid(True, which='both', alpha=0.3)

    fig.suptitle('SCE recovery under realistic intrinsic noise (227 ENC): an observability limit',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
