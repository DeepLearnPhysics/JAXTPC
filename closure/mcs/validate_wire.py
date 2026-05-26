"""
Validation level 4: wire signal sensitivity to MCS scattering.

4a: L2 difference between straight and MCS wire signals.
4b: dL/d(dtheta1[k]) is nonzero at multiple segment positions.

Run from project root:
    python3 -m closure.mcs.validate_wire

Migrated from closure_analysis_MCS/validate_wire_sensitivity.py.
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import time

from closure.mcs.forward import (
    mcs_cumsum_forward,
    generate_mcs_truth,
    build_mcs_forward,
)
from tools.particle_generator import (
    load_dedx_table_jax,
    generate_muon_segments_trig,
    mask_outside_volume,
    get_half_extents_mm,
)
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEGMENTS = 2000
STEP_SIZE_MM = 0.5
ENERGY = 500.0
START = np.array([-200.0, 0.0, 100.0])
THETA = np.pi / 4
PHI = np.pi / 6

PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

results = []

def test(name, passed, message=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, message))
    print(f"  [{status}] {name}: {message}")


# ---------------------------------------------------------------------------
# 4a: Signal difference between straight and MCS tracks
# ---------------------------------------------------------------------------

def test_4a(forward, log_T, dedx, half_ext):
    print("\n=== 4a: Signal difference: straight vs MCS ===")

    sin_th = jnp.sin(jnp.float32(THETA))
    cos_th = jnp.cos(jnp.float32(THETA))
    sin_ph = jnp.sin(jnp.float32(PHI))
    cos_ph = jnp.cos(jnp.float32(PHI))
    start_jax = jnp.array(START, dtype=jnp.float32)
    energy_jax = jnp.float32(ENERGY)

    # Straight-line track
    pos_str, de_str = generate_muon_segments_trig(
        energy_jax, start_jax,
        sin_th, cos_th, sin_ph, cos_ph,
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    )
    de_str = mask_outside_volume(pos_str, de_str, half_ext)

    # MCS track
    rng_key = jax.random.PRNGKey(42)
    pos_mcs, de_mcs, dt1, dt2 = generate_mcs_truth(
        energy_jax, start_jax,
        jnp.float32(THETA), jnp.float32(PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, rng_key,
    )
    de_mcs = mask_outside_volume(pos_mcs, de_mcs, half_ext)

    # Wire signals
    print("  Computing straight-line wire signals...")
    t0 = time.time()
    sigs_str = jax.jit(forward)(pos_str, de_str)
    for s in sigs_str:
        jax.block_until_ready(s)
    print(f"    Compiled + ran ({time.time()-t0:.1f}s)")

    print("  Computing MCS wire signals...")
    t0 = time.time()
    sigs_mcs = jax.jit(forward)(pos_mcs, de_mcs)
    for s in sigs_mcs:
        jax.block_until_ready(s)
    print(f"    Ran ({time.time()-t0:.1f}s)")

    # L2 differences per plane
    all_nonzero = True
    details = []
    for i, name in enumerate(PLANE_NAMES):
        diff = sigs_mcs[i] - sigs_str[i]
        l2 = float(jnp.sqrt(jnp.sum(diff ** 2)))
        sig_norm = float(jnp.sqrt(jnp.sum(sigs_str[i] ** 2)))
        rel = l2 / (sig_norm + 1e-12)
        details.append(f"{name}: L2={l2:.4f} (rel={rel:.4f})")
        if l2 < 1e-8:
            all_nonzero = False

    test("4a: Straight vs MCS signals differ",
         all_nonzero,
         "; ".join(details))

    return sigs_str, sigs_mcs, dt1, dt2


# ---------------------------------------------------------------------------
# 4b: Gradient of wire loss w.r.t. scattering angles
# ---------------------------------------------------------------------------

def test_4b(forward, log_T, dedx, truth_signals, half_ext):
    print("\n=== 4b: dL/d(dtheta1[k]) nonzero ===")

    sin_th = jnp.sin(jnp.float32(THETA))
    cos_th = jnp.cos(jnp.float32(THETA))
    sin_ph = jnp.sin(jnp.float32(PHI))
    cos_ph = jnp.cos(jnp.float32(PHI))
    start_jax = jnp.array(START, dtype=jnp.float32)
    energy_jax = jnp.float32(ENERGY)

    # Precompute Sobolev weights
    print("  Precomputing Sobolev spectral weights...")
    spec_weights = tuple(
        make_sobolev_weight(*truth_signals[p].shape, s=1.5)
        for p in range(6)
    )

    def loss_fn(dtheta1):
        dtheta2 = jnp.zeros_like(dtheta1)
        pos, de = mcs_cumsum_forward(
            energy_jax, start_jax, sin_th, cos_th, sin_ph, cos_ph,
            dtheta1, dtheta2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de, half_ext)
        sigs = forward(pos, de)
        return sobolev_loss_geomean_log1p(sigs, truth_signals, spec_weights)

    print("  Compiling gradient...")
    t0 = time.time()
    grad_fn = jax.jit(jax.grad(loss_fn))
    dtheta1_zero = jnp.zeros(N_SEGMENTS)
    grads = grad_fn(dtheta1_zero)
    jax.block_until_ready(grads)
    print(f"    Compiled ({time.time()-t0:.1f}s)")

    # Check at k=0, N/4, N/2, 3N/4
    test_indices = [0, N_SEGMENTS // 4, N_SEGMENTS // 2, 3 * N_SEGMENTS // 4]
    all_nonzero = True
    details = []
    for k in test_indices:
        g = float(grads[k])
        details.append(f"k={k}: grad={g:.6e}")
        if abs(g) < 1e-12:
            all_nonzero = False

    test("4b: Wire loss gradients nonzero",
         all_nonzero,
         "; ".join(details))

    # Report gradient magnitude distribution
    grad_np = np.array(grads)
    nonzero_frac = np.mean(np.abs(grad_np) > 1e-10)
    print(f"  Gradient stats: mean|grad|={np.mean(np.abs(grad_np)):.2e}, "
          f"max|grad|={np.max(np.abs(grad_np)):.2e}, "
          f"nonzero fraction={nonzero_frac:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MCS WIRE SENSITIVITY VALIDATION (Level 4)")
    print("=" * 60)

    log_T, dedx = load_dedx_table_jax()

    # Build simulator
    print("Building DetectorSimulator (differentiable=True)...")
    t0 = time.time()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(detector_config)
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    print(f"  Built ({time.time()-t0:.1f}s)")

    sigs_str, sigs_mcs, dt1, dt2 = test_4a(forward, log_T, dedx, half_ext)
    test_4b(forward, log_T, dedx, sigs_mcs, half_ext)

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("ALL TESTS PASSED")
    else:
        print("FAILURES:")
        for name, passed, msg in results:
            if not passed:
                print(f"  {name}: {msg}")
    print("=" * 60)


if __name__ == '__main__':
    main()
