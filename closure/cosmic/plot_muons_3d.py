#!/usr/bin/env python3
"""3D plot of all cosmic muon chords used in the SCE-recovery study.

Reproduces the exact surface-to-surface sampling (RandomState(0), x clipped to
the drift volume) used by run_accum, and draws every entrance→exit chord through
the 40 cm closure detector box, coloured by zenith angle.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.particle_generator import sample_box_endpoints

HERE = os.path.dirname(os.path.abspath(__file__))
# 40 cm closure detector: drift x in [-200,0] mm (anode at 0), transverse ±200
LO, HI = (-200.0, -200.0, -200.0), (0.0, 200.0, 200.0)
# display ranges (cm)
XR, YR, ZR = (-20, 0), (-20, 20), (-20, 20)


def box_edges(xr, yr, zr):
    import itertools
    corners = np.array(list(itertools.product(xr, yr, zr)))
    edges = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if np.sum(corners[i] != corners[j]) == 1:   # share 2 coords => edge
                edges.append([corners[i], corners[j]])
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=768, help='number of muons to draw')
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    segs, thetas = [], []
    for _ in range(args.n):
        a, b = sample_box_endpoints(rng, LO, HI)
        a, b = a / 10.0, b / 10.0                        # mm -> cm
        ch = b - a
        thetas.append(np.degrees(np.arccos(abs(ch[2]) / (np.linalg.norm(ch) + 1e-9))))
        segs.append([a, b])
    thetas = np.array(thetas)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(thetas.min(), thetas.max())
    lc = Line3DCollection(segs, colors=cmap(norm(thetas)), linewidths=0.5, alpha=0.45)
    ax.add_collection3d(lc)

    for e in box_edges(XR, YR, ZR):
        e = np.array(e)
        ax.plot(e[:, 0], e[:, 1], e[:, 2], color='k', lw=1.0, alpha=0.7)
    # anode plane (x=0) shaded
    yy, zz = np.meshgrid([YR[0], YR[1]], [ZR[0], ZR[1]])
    ax.plot_surface(np.zeros_like(yy), yy, zz, alpha=0.12, color='red')

    ax.set(xlabel='x  (drift, cm)', ylabel='y (cm)', zlabel='z (cm)',
           xlim=XR, ylim=YR, zlim=ZR)
    ax.set_title(f'All {args.n} cosmic muon chords through the 40 cm detector\n'
                 f'(anode plane x=0 shaded red; coloured by zenith angle)', fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label='angle to z-axis (deg)')
    ax.view_init(elev=18, azim=-60)
    out = os.path.join(HERE, 'muons_3d.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'saved {out}  ({args.n} muons, chord lengths span surface-to-surface)')


if __name__ == '__main__':
    main()
