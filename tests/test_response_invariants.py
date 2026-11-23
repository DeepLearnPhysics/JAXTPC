"""Physics INVARIANTS of the response path.

Every other kernel test in this suite is structural (shape, finiteness,
batch == sequential, JIT-compat) or self-referential -- they compare the code to
itself. A full suite of them passed while the collection response was losing 10.6%
of its charge, while the sub-tick shift was injecting phase-dependent pulse width,
and while the dense and bucketed paths disagreed on the time index. These tests
assert properties the PHYSICS requires, so they fail when the physics is wrong
rather than when the code changes shape.

Files under test: tools/kernels.py, tools/wires.py, tools/physics.py
"""

import numpy as np
import jax.numpy as jnp
import pytest

from tools.kernels import (
    generate_dkernel_table,
    interpolate_diffusion_kernel,
    _enforce_wire_tiling,
)


BPW = 10          # sub-pitch bins per wire (wire_spacing = 0.1 pitch)


def _collection_kernel(H=41, npitch=5, half_weight=True):
    """A synthetic COLLECTION response: charge is collected by whichever wire is
    nearest, so R is a boxcar of exactly one pitch, plus a small induction tail.

    The boxcar's edge lands exactly on a sample. With half_weight=True the edge
    sample carries C/2, which is what makes the wires tile; with False it carries
    the non-collecting value, which is the defect the shipped Y kernel had.
    """
    W = 2 * npitch * BPW + 1
    wz = W // 2
    t = np.arange(H) - H // 2
    pulse = np.exp(-0.5 * (t / 3.0) ** 2)
    pulse = pulse / pulse.sum()
    k = np.zeros((H, W))
    for c in range(W):
        d = (c - wz) / BPW                    # Delta in pitch units
        if abs(d) < 0.5 - 1e-9:
            k[:, c] = pulse                   # collected
        elif abs(abs(d) - 0.5) < 1e-9:
            k[:, c] = 0.5 * pulse if half_weight else 0.0
    return k, wz


def _tiling_sums(kernel, wz):
    """Per-residue-class tiling sums: sum_n R(x - n) for each x on the sub-pitch grid."""
    integ = kernel.sum(axis=0)
    out = []
    for r in range(BPW):
        out.append(sum(integ[c] for c in range(len(integ)) if (c - wz) % BPW == r))
    return np.array(out)


class TestWireTilingInvariant:
    """sum_n R(x - n) must be CONSTANT in x -- every drift electron is collected by
    exactly one wire, so the total collected charge cannot depend on where between
    the wires the charge happens to sit."""

    def test_half_weighted_edge_tiles(self):
        k, wz = _collection_kernel(half_weight=True)
        T = _tiling_sums(k, wz)
        assert np.ptp(T) < 1e-9 * abs(T.mean()), (
            f"a correctly half-weighted collection kernel must tile exactly; "
            f"got per-class sums {T}")

    def test_missing_branch_breaks_tiling(self):
        """Guards the test itself: the defect must actually be detectable."""
        k, wz = _collection_kernel(half_weight=False)
        T = _tiling_sums(k, wz)
        assert np.ptp(T) > 0.5 * abs(T.mean()), (
            "storing the non-collecting branch should leave a ~100% hole in the tiling")

    def test_enforce_repairs_and_is_idempotent(self):
        broken, wz = _collection_kernel(half_weight=False)
        fixed = _enforce_wire_tiling(broken, wz, 1.0 / BPW, 'Y')
        T = _tiling_sums(fixed, wz)
        assert np.ptp(T) < 0.05 * abs(T.mean()), f"tiling not repaired: {T}"
        again = _enforce_wire_tiling(fixed, wz, 1.0 / BPW, 'Y')
        np.testing.assert_array_equal(
            again, fixed, err_msg="_enforce_wire_tiling must be idempotent")

    def test_leaves_a_correct_kernel_untouched(self):
        good, wz = _collection_kernel(half_weight=True)
        out = _enforce_wire_tiling(good, wz, 1.0 / BPW, 'Y')
        np.testing.assert_array_equal(
            out, good, err_msg="a kernel that already tiles must not be nudged")

    def test_skips_induction_planes(self):
        """An induction response has no collection bifurcation and no meaningful
        tiling sum; it must not be 'corrected'."""
        H, npitch = 41, 5
        W = 2 * npitch * BPW + 1
        wz = W // 2
        t = np.arange(H) - H // 2
        bipolar = np.gradient(np.exp(-0.5 * (t / 3.0) ** 2))
        k = np.zeros((H, W))
        for c in range(W):
            d = (c - wz) / BPW
            k[:, c] = bipolar * np.exp(-0.5 * (d / 1.5) ** 2)
        out = _enforce_wire_tiling(k, wz, 1.0 / BPW, 'U')
        np.testing.assert_array_equal(
            out, k, err_msg="induction planes must be left alone")


class TestChargeIndependentOfSubPitchOffset:
    """End-to-end consequence of the tiling invariant, through the real runtime:
    the interpolated (num_wires x ntime) patch must carry the same total charge
    wherever the deposit sits between the wires."""

    @pytest.mark.parametrize("sigma_T", [0.05, 0.15, 0.45])
    def test_patch_sum_flat_in_offset(self, sigma_T):
        """Driven from a BROKEN kernel through the load-time repair, so this guards the
        whole chain. Using an already-correct kernel here would test nothing: it passes
        with or without the fix (verified -- unfixed tree gives 25.9% variation, fixed
        gives 0.0%)."""
        broken, wz = _collection_kernel(half_weight=False)
        base = _enforce_wire_tiling(broken, wz, 1.0 / BPW, 'Y')
        dk = generate_dkernel_table(sigma_T, 0.3, jnp.asarray(base, dtype=jnp.float32),
                                    1.0 / BPW, 0.5, jnp.linspace(0, 1, 16))
        sums = [float(jnp.sum(interpolate_diffusion_kernel(
            dk, 1.0, off, 0.0, 1.0 / BPW, 6)))
            for off in (0.0, 0.1, 0.2, 0.3, 0.4, 0.49)]
        rel = (max(sums) - min(sums)) / abs(sums[0])
        assert rel < 0.02, (
            f"collected charge varies {rel*100:.1f}% with the sub-pitch offset at "
            f"sigma_T={sigma_T}; the wires are not tiling ({sums})")


class TestSubTickPhaseDoesNotChangeWidth:
    """The sub-tick shift must PLACE the pulse, not broaden it. A 2-tap linear blend
    is a shift convolved with a triangle and injects a(1-a) ticks^2 of width that
    depends only on where the deposit lands inside its tick -- not on any physics."""

    def test_width_stable_across_phase(self):
        base, wz = _collection_kernel(H=61, half_weight=True)
        dk = generate_dkernel_table(0.2, 1.0, jnp.asarray(base, dtype=jnp.float32),
                                    1.0 / BPW, 0.5, jnp.linspace(0, 1, 16))

        def width(t_off):
            p = np.asarray(interpolate_diffusion_kernel(dk, 1.0, 0.0, t_off,
                                                        1.0 / BPW, 6))
            prof = np.abs(p).sum(axis=0)
            i = np.arange(len(prof))
            mu = (i * prof).sum() / prof.sum()
            return ((i - mu) ** 2 * prof).sum() / prof.sum()

        w = [width(a) for a in (0.0, 0.25, 0.5, 0.75)]
        swing = max(w) - min(w)
        assert swing < 0.10, (
            f"pulse width varies by {swing:.3f} ticks^2 with the sub-tick phase "
            f"(a 2-tap linear blend injects up to 0.25); widths {w}")


class TestIndexConventionsAgree:
    """The dense and bucketed accumulators must agree on a deposit's time index.

    compute_plane_physics computes its own time_map and feeds it to build_bucket_mapping
    (which assigns each deposit's HOME bucket) while the scatter uses the index from
    prepare_deposit_for_response. If the two expressions ever diverge, a deposit whose
    difference straddles a bucket boundary has its ENTIRE patch written one bucket late
    -- B2 = 2 * kernel_height ticks. That happened once in this repo's history and the
    existing test_dense_equals_bucketed could not catch it, because it passes the SAME
    time_indices array to both accumulators, making them equal by construction.
    """

    def test_prepare_matches_the_bucket_mapping_expression(self):
        from tools.wires import prepare_deposit_for_response
        dt = 0.5
        rng = np.random.RandomState(0)
        ticks = rng.uniform(0.0, 1400.0, 500).astype(np.float32)
        for tk in ticks[:50]:
            _idx, _woff, tidx, _toff, _inten = prepare_deposit_for_response(
                jnp.float32(1.0), jnp.float32(tk), jnp.int32(10), jnp.float32(0.0),
                jnp.float32(1.0), jnp.bool_(True), 0.3, dt, 100)
            # physics.compute_plane_physics builds time_map with this expression; the two
            # MUST match or the bucketed home bucket disagrees with the scatter.
            expected = int(np.floor(float(tk) / dt))
            assert int(tidx) == expected, (
                f"prepare_deposit_for_response time_index ({int(tidx)}) disagrees with "
                f"compute_plane_physics' time_map expression ({expected}) at tick_us={tk}. "
                f"Both sites must be changed together -- see tools/physics.py time_map.")
