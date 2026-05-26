#!/usr/bin/env python3
"""Sweep n_points for the consistent CSDA table to find the sweet spot."""

import jax
import jax.numpy as jnp
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    build_consistent_csda_table,
    generate_muon_segments_scan,
    generate_muon_segments_csda,
)

STEP_SIZE_MM = 0.5
N_SEGMENTS = 4000
MIN_ENERGY = 10.0
SMOOTH_T = 0.2

log_T, dedx = load_dedx_table_jax()
start = jnp.zeros(3)

# Reference: scan results (compute once)
cases = [
    ("500 MeV",  500.0, 0.8, 1.2),
    ("5000 MeV", 5000.0, 0.3, 0.7),
    ("50 MeV",   50.0, 0.5, 0.3),
    ("20 MeV",   20.0, 1.0, 2.0),
]

print(f"{'n_pts':>8} | ", end="")
for name, *_ in cases:
    print(f" {name:>10} fwd  {name:>10} grad |", end="")
print(f" {'time ms':>8}")
print("-" * 140)

scan_refs = {}
for name, energy, theta, phi in cases:
    E, th, ph = jnp.float32(energy), jnp.float32(theta), jnp.float32(phi)
    _, de_s = generate_muon_segments_scan(
        E, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
    jax.block_until_ready(de_s)

    g_s = float(jax.grad(lambda e: jnp.sum(generate_muon_segments_scan(
        e, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, min_energy_mev=MIN_ENERGY,
        smooth_temperature=SMOOTH_T)[1]))(E))

    scan_refs[name] = (float(jnp.sum(de_s)), g_s)

for n_pts in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
    R_cm, T_MeV = build_consistent_csda_table(log_T, dedx, n_points=n_pts)

    row = f"{n_pts:>8} | "
    for name, energy, theta, phi in cases:
        E, th, ph = jnp.float32(energy), jnp.float32(theta), jnp.float32(phi)
        ref_total, ref_grad = scan_refs[name]

        _, de_c = generate_muon_segments_csda(
            E, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, R_cm, T_MeV)
        jax.block_until_ready(de_c)
        total_c = float(jnp.sum(de_c))

        g_c = float(jax.grad(lambda e: jnp.sum(generate_muon_segments_csda(
            e, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, R_cm, T_MeV)[1]))(E))

        fwd_rel = abs(total_c - ref_total) / max(abs(ref_total), 1e-12) * 100
        grad_rel = abs(g_c - ref_grad) / max(abs(ref_grad), abs(g_c), 1e-12) * 100

        row += f" {fwd_rel:>9.4f}%  {grad_rel:>9.2f}% |"

    # Time one forward call (warm up already done from above)
    fwd_jit = jax.jit(lambda e, t, p: generate_muon_segments_csda(
        e, start, t, p, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, R_cm, T_MeV))
    E0, th0, ph0 = jnp.float32(500.0), jnp.float32(0.8), jnp.float32(1.2)
    jax.block_until_ready(fwd_jit(E0, th0, ph0))
    t0 = time.perf_counter()
    for _ in range(20):
        jax.block_until_ready(fwd_jit(E0, th0, ph0))
    ms = (time.perf_counter() - t0) / 20 * 1000

    row += f" {ms:>7.2f}"
    print(row)
