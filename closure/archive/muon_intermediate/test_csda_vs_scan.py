#!/usr/bin/env python3
"""Compare CSDA parallel vs scan sequential muon segment generation.

Tests:
1. Table consistency: verify dE/dx table integrates to CSDA range
2. Forward values (hard mode) — raw 82-pt table and densified 2000-pt table
3. Gradients (smooth mode) — both table densities
4. Speed: forward and value_and_grad timing
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    build_consistent_csda_table,
    generate_muon_segments_scan,
    generate_muon_segments_csda,
    LAR_DENSITY,
    _DEDX_FILE,
)

# ---- Configuration ----
STEP_SIZE_MM = 0.5
N_SEGMENTS = 4000
MIN_ENERGY = 10.0
SMOOTH_T = 0.2

TEST_CASES = [
    ("Standard (500 MeV)",       500.0, 0.8, 1.2),
    ("Low energy (50 MeV)",       50.0, 0.5, 0.3),
    ("Near threshold (20 MeV)",   20.0, 1.0, 2.0),
    ("High energy (5000 MeV)",  5000.0, 0.3, 0.7),
]


def _load_csda_table(n_dense=0):
    """Load CSDA range table, optionally densified (no global cache)."""
    data = np.loadtxt(_DEDX_FILE, delimiter=",", comments="#")
    T_MeV = data[:, 0]
    R_cm = data[:, 3] / LAR_DENSITY
    if n_dense > 0:
        log_T_raw = np.log(T_MeV)
        log_T_d = np.linspace(log_T_raw[0], log_T_raw[-1], n_dense)
        R_cm = np.interp(log_T_d, log_T_raw, R_cm)
        T_MeV = np.exp(log_T_d)
    return jnp.array(R_cm), jnp.array(T_MeV)


# ---- Table consistency check ----
def check_table_consistency():
    data = np.loadtxt(_DEDX_FILE, delimiter=",", comments="#")
    T = data[:, 0]
    dedx = data[:, 2] * LAR_DENSITY
    R_tab = data[:, 3] / LAR_DENSITY

    inv_dedx = 1.0 / dedx
    R_int = np.zeros_like(T)
    for i in range(1, len(T)):
        R_int[i] = R_int[i-1] + 0.5 * (inv_dedx[i-1] + inv_dedx[i]) * (T[i] - T[i-1])

    mask = T >= 10.0
    rel_err = np.abs(R_int[mask] - R_tab[mask]) / R_tab[mask]
    print("Table consistency (trapezoidal integral of 1/dE/dx vs tabulated R):")
    print(f"  E >= 10 MeV: max rel err = {rel_err.max()*100:.2f}%, "
          f"mean = {rel_err.mean()*100:.2f}%")


def _rel(a, b):
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom * 100


# ---- Forward comparison ----
def compare_forward(name, energy, theta, phi, log_T, dedx, R_cm, T_MeV,
                    de_scan=None):
    """Run CSDA forward and compare to scan result. Returns (de_scan, de_csda)."""
    start = jnp.zeros(3)
    E, th, ph = jnp.float32(energy), jnp.float32(theta), jnp.float32(phi)

    if de_scan is None:
        _, de_scan = generate_muon_segments_scan(
            E, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
        jax.block_until_ready(de_scan)

    _, de_csda = generate_muon_segments_csda(
        E, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, R_cm, T_MeV)
    jax.block_until_ready(de_csda)

    total_s = float(jnp.sum(de_scan))
    total_c = float(jnp.sum(de_csda))
    de_diff = jnp.abs(de_scan - de_csda)

    active_mask = de_scan > 0.01  # meaningful segments only
    if jnp.any(active_mask):
        max_rel = float(jnp.max(jnp.where(
            active_mask, de_diff / jnp.maximum(de_scan, 1e-12), 0.0))) * 100
    else:
        max_rel = 0.0

    print(f"    Total dE: scan={total_s:.4f}, csda={total_c:.4f}  "
          f"(rel {_rel(total_s, total_c):.4f}%)")
    print(f"    Max |dE_i| diff: {float(jnp.max(de_diff)):.6f} MeV, "
          f"mean: {float(jnp.mean(de_diff)):.6f} MeV")
    print(f"    Max per-seg rel (dE>0.01): {max_rel:.2f}%")

    return de_scan, de_csda


# ---- Gradient comparison ----
def compare_gradients(name, energy, theta, phi, log_T, dedx, R_cm, T_MeV):
    start = jnp.zeros(3)
    E, th, ph = jnp.float32(energy), jnp.float32(theta), jnp.float32(phi)

    def loss_scan_E(e):
        _, de = generate_muon_segments_scan(
            e, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, min_energy_mev=MIN_ENERGY,
            smooth_temperature=SMOOTH_T)
        return jnp.sum(de)

    def loss_csda_E(e):
        _, de = generate_muon_segments_csda(
            e, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, R_cm, T_MeV)
        return jnp.sum(de)

    gse = float(jax.grad(loss_scan_E)(E))
    gce = float(jax.grad(loss_csda_E)(E))

    def loss_scan_a(tv, pv):
        pos, de = generate_muon_segments_scan(
            E, start, tv, pv, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, min_energy_mev=MIN_ENERGY,
            smooth_temperature=SMOOTH_T)
        return jnp.sum(de * pos[:, 2])

    def loss_csda_a(tv, pv):
        pos, de = generate_muon_segments_csda(
            E, start, tv, pv, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, R_cm, T_MeV)
        return jnp.sum(de * pos[:, 2])

    gst, gsp = [float(x) for x in jax.grad(loss_scan_a, argnums=(0, 1))(th, ph)]
    gct, gcp = [float(x) for x in jax.grad(loss_csda_a, argnums=(0, 1))(th, ph)]

    print(f"    d(sum(de))/dE:   scan={gse: .6f}  csda={gce: .6f}  "
          f"(rel {_rel(gse, gce):.2f}%)")
    print(f"    d(de*z)/dtheta:  scan={gst: .2f}  csda={gct: .2f}  "
          f"(rel {_rel(gst, gct):.2f}%)")
    print(f"    d(de*z)/dphi:    scan={gsp: .2f}  csda={gcp: .2f}  "
          f"(rel {_rel(gsp, gcp):.2f}%)")

    return dict(gse=gse, gce=gce, gst=gst, gct=gct)


# ---- Speed comparison ----
def compare_speed(log_T, dedx, R_cm, T_MeV, n_runs=50):
    start = jnp.zeros(3)
    E, th, ph = jnp.float32(500.0), jnp.float32(0.8), jnp.float32(1.2)

    fwd_s = jax.jit(lambda e, t, p: generate_muon_segments_scan(
        e, start, t, p, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, min_energy_mev=MIN_ENERGY, smooth_temperature=0.0))
    fwd_c = jax.jit(lambda e, t, p: generate_muon_segments_csda(
        e, start, t, p, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, R_cm, T_MeV))

    # Warm up
    jax.block_until_ready(fwd_s(E, th, ph))
    jax.block_until_ready(fwd_c(E, th, ph))

    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(fwd_s(E, th, ph))
    ts = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(fwd_c(E, th, ph))
    tc = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\n  Forward (500 MeV, hard, {n_runs} runs):")
    print(f"    Scan: {ts:.2f} ms   CSDA: {tc:.2f} ms   Speedup: {ts/max(tc,1e-6):.0f}x")

    # Gradient speed
    gs = jax.jit(jax.value_and_grad(lambda e: jnp.sum(generate_muon_segments_scan(
        e, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, min_energy_mev=MIN_ENERGY, smooth_temperature=SMOOTH_T)[1])))
    gc = jax.jit(jax.value_and_grad(lambda e: jnp.sum(generate_muon_segments_csda(
        e, start, th, ph, STEP_SIZE_MM, N_SEGMENTS,
        log_T, dedx, R_cm, T_MeV)[1])))

    jax.block_until_ready(gs(E))
    jax.block_until_ready(gc(E))

    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(gs(E))
    tgs = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(gc(E))
    tgc = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\n  value_and_grad(sum(de)) w.r.t. energy (smooth T={SMOOTH_T}):")
    print(f"    Scan: {tgs:.2f} ms   CSDA: {tgc:.2f} ms   Speedup: {tgs/max(tgc,1e-6):.0f}x")

    # Angle gradient speed
    def _loss_s(t, p):
        pos, de = generate_muon_segments_scan(
            E, start, t, p, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, min_energy_mev=MIN_ENERGY, smooth_temperature=SMOOTH_T)
        return jnp.sum(de * pos[:, 2])
    def _loss_c(t, p):
        pos, de = generate_muon_segments_csda(
            E, start, t, p, STEP_SIZE_MM, N_SEGMENTS,
            log_T, dedx, R_cm, T_MeV)
        return jnp.sum(de * pos[:, 2])

    gas = jax.jit(jax.value_and_grad(_loss_s, argnums=(0,)))
    gac = jax.jit(jax.value_and_grad(_loss_c, argnums=(0,)))
    jax.block_until_ready(gas(th, ph))
    jax.block_until_ready(gac(th, ph))

    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(gas(th, ph))
    tas = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(gac(th, ph))
    tac = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\n  value_and_grad(sum(de*z)) w.r.t. theta:")
    print(f"    Scan: {tas:.2f} ms   CSDA: {tac:.2f} ms   Speedup: {tas/max(tac,1e-6):.0f}x")


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Config: step={STEP_SIZE_MM}mm, n={N_SEGMENTS}, min_E={MIN_ENERGY}MeV\n")

    check_table_consistency()

    log_T, dedx = load_dedx_table_jax()
    R_pdg, T_pdg = _load_csda_table(n_dense=0)
    R_con, T_con = build_consistent_csda_table(log_T, dedx, n_points=10000)

    tables = [
        ("82-pt PDG R column", R_pdg, T_pdg),
        ("10k-pt integrated from dE/dx", R_con, T_con),
    ]

    # ---- Forward ----
    for label, R_cm, T_MeV in tables:
        print(f"\n{'='*60}")
        print(f"  FORWARD — {label}")
        print(f"{'='*60}")
        for name, energy, theta, phi in TEST_CASES:
            print(f"\n  {name}:")
            compare_forward(name, energy, theta, phi, log_T, dedx, R_cm, T_MeV)

    # ---- Gradients ----
    for label, R_cm, T_MeV in tables:
        print(f"\n{'='*60}")
        print(f"  GRADIENTS (smooth T={SMOOTH_T}) — {label}")
        print(f"{'='*60}")
        for name, energy, theta, phi in TEST_CASES:
            print(f"\n  {name}:")
            compare_gradients(name, energy, theta, phi, log_T, dedx, R_cm, T_MeV)

    # ---- Speed (use consistent table) ----
    print(f"\n{'='*60}")
    print(f"  SPEED (10k-pt consistent table)")
    print(f"{'='*60}")
    compare_speed(log_T, dedx, R_con, T_con)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
