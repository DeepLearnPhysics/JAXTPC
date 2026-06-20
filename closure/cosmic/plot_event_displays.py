#!/usr/bin/env python3
"""Example event displays of the cosmic muons used in the SCE-recovery study,
rendered with the repo's wire-plane viewer (tools.visualization).

For a handful of example cosmics: forward-simulate the chord through the truth
SCE field and show the U/V/Y wire-plane images — both clean and with the real
MicroBooNE intrinsic noise (what the recovery actually sees). One PNG per muon.
"""
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from closure.cosmic.recover_field import build
from tools.particle_generator import (load_dedx_table_jax, generate_cosmic_chord,
                                       sample_box_endpoints)
from tools.visualization import visualize_wire_signals
from tools.noise import (load_noise_params, _get_noise_spectrum_shape,
                         _generate_noise_for_plane)

HERE = os.path.dirname(os.path.abspath(__file__))
HALF = (200.0, 200.0, 200.0)
# actual drift detector box (mm): x in [-200,0] (anode at 0), y,z in [-200,200]
LO, HI = (-200.0, -200.0, -200.0), (0.0, 200.0, 200.0)


def add_real_noise(obs, cfg, seed):
    epa = float(cfg.electrons_per_adc); nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path)
    spectrum = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    k = jax.random.PRNGKey(seed); out = {}
    for (s, p), o in obs.items():
        nw = o.shape[0]
        L = jnp.asarray(cfg.volumes[s].wire_lengths_m[p], jnp.float32)
        series = (ny + nz * L) * epa
        k, sk = jax.random.split(k)
        noise = _generate_noise_for_plane(sk, nw, nt, spectrum, series, float(nx) * epa)
        out[(s, p)] = np.asarray(o) + np.asarray(noise)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4, help='number of example muons')
    ap.add_argument('--noisy', action='store_true', help='also render with real intrinsic noise')
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    args = ap.parse_args()

    sim, _, _, _ = build(160, truth_scale=1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)

    made = 0; tried = 0
    while made < args.n and tried < 200:
        tried += 1
        a, b = sample_box_endpoints(rng, LO, HI)
        chord_cm = np.linalg.norm(b - a) / 10
        if chord_cm < 25:                       # skip short grazers for clear displays
            continue
        p, d, theta, phi, _ = generate_cosmic_chord(
            jnp.array(a), jnp.array(b), 4000., 160, logT, dedx,
            half_extents_mm=HALF, step_mm=4.0)
        sg = sim.forward_segments(base, p, d, dx=4.0)
        obs = {(0, pl): np.asarray(sg[pl]) for pl in range(len(sg))}

        fig = visualize_wire_signals(obs, sim.config, threshold_enc=0, cmap='obsidian')
        fig.suptitle(f'Cosmic muon #{made}: chord {chord_cm:.0f} cm, '
                     f'θ={np.degrees(float(theta)):.0f}°, φ={np.degrees(float(phi)):.0f}°  '
                     f'(clean signal, ENC)', y=1.02, fontsize=12)
        out = os.path.join(HERE, f'event_muon_{made}.png')
        fig.savefig(out, dpi=120, bbox_inches='tight'); plt.close(fig)
        print(f'saved {out}')

        if args.noisy:
            nobs = add_real_noise(obs, sim.config, seed=100 + made)
            fig = visualize_wire_signals(nobs, sim.config, threshold_enc=0, cmap='obsidian')
            fig.suptitle(f'Cosmic muon #{made}: + real MicroBooNE intrinsic noise '
                         f'(~227 ENC/wire) — what the recovery sees', y=1.02, fontsize=12)
            out = os.path.join(HERE, f'event_muon_{made}_noisy.png')
            fig.savefig(out, dpi=120, bbox_inches='tight'); plt.close(fig)
            print(f'saved {out}')
        made += 1


if __name__ == '__main__':
    main()
