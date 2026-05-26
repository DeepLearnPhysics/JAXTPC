"""
Methodical comparison: WHY does OTT SW get correct weight gradients
but our jnp.interp-based Huber SW does not?

Isolates the exact gradient failure by testing each component:

1. jnp.interp gradient w.r.t. xp (CDF breakpoints) — the suspect
2. OTT-style diff_q weighting — the working alternative
3. Full 1D Wasserstein: interp-based vs OTT-style

Uses simple synthetic 1D data (no simulation) to eliminate confounders.

Run: python3 closure_analysis/compare_sw_gradient_paths.py
"""

import jax
import jax.numpy as jnp
import numpy as np

np.set_printoptions(precision=6, suppress=False)


# =============================================================================
# Test 1: jnp.interp gradient w.r.t. xp
# =============================================================================

def test_interp_xp_gradient():
    """Does jnp.interp propagate gradients through xp (breakpoints)?"""
    print("=" * 70)
    print("TEST 1: jnp.interp gradient w.r.t. xp (CDF breakpoints)")
    print("=" * 70)

    # Fixed grid and values (fp), vary breakpoints (xp) via a scale factor
    grid = jnp.linspace(0.01, 0.99, 100)
    fp = jnp.linspace(0.0, 1.0, 50)  # 50 sorted position values

    def loss_via_interp(scale):
        """Scale the CDF breakpoints by `scale` and interpolate."""
        xp = jnp.linspace(0.0, scale, 50)  # CDF goes from 0 to `scale`
        quantiles = jnp.interp(grid, xp, fp)
        return jnp.mean(quantiles ** 2)

    # At scale=1.0, CDF and grid are matched
    scale = 1.0
    ad_grad = float(jax.grad(loss_via_interp)(scale))
    eps = 1e-4
    fd_grad = (float(loss_via_interp(scale + eps)) -
               float(loss_via_interp(scale - eps))) / (2 * eps)

    print(f"  scale=1.0:  AD grad = {ad_grad:.6e},  FD grad = {fd_grad:.6e}")
    print(f"  ratio AD/FD = {ad_grad / (fd_grad + 1e-30):.4f}")

    # At scale=2.0, CDF extends beyond grid
    scale = 2.0
    ad_grad = float(jax.grad(loss_via_interp)(scale))
    fd_grad = (float(loss_via_interp(scale + eps)) -
               float(loss_via_interp(scale - eps))) / (2 * eps)

    print(f"  scale=2.0:  AD grad = {ad_grad:.6e},  FD grad = {fd_grad:.6e}")
    print(f"  ratio AD/FD = {ad_grad / (fd_grad + 1e-30):.4f}")

    # At scale=0.5, CDF compressed
    scale = 0.5
    ad_grad = float(jax.grad(loss_via_interp)(scale))
    fd_grad = (float(loss_via_interp(scale + eps)) -
               float(loss_via_interp(scale - eps))) / (2 * eps)

    print(f"  scale=0.5:  AD grad = {ad_grad:.6e},  FD grad = {fd_grad:.6e}")
    print(f"  ratio AD/FD = {ad_grad / (fd_grad + 1e-30):.4f}")
    print()


# =============================================================================
# Test 2: Our interp-based 1D Wasserstein
# =============================================================================

def test_interp_w1d():
    """Our approach: sort, cumsum → CDF, interp(grid, CDF, sorted_vals)."""
    print("=" * 70)
    print("TEST 2: Interp-based 1D quantile distance (our approach)")
    print("=" * 70)

    # Two sets of positions (fixed)
    key = jax.random.PRNGKey(0)
    n = 200
    pos_a = jax.random.uniform(key, (n,), minval=0.0, maxval=1.0)
    pos_b = jax.random.uniform(jax.random.PRNGKey(1), (n,), minval=0.0, maxval=1.0)

    grid = jnp.linspace(1e-6, 1.0 - 1e-6, 500)

    def w1d_interp(wts_a, wts_b):
        """Our interp-based approach — weights NOT normalized."""
        sort_a = jnp.argsort(pos_a)
        sort_b = jnp.argsort(pos_b)
        cdf_a = jnp.cumsum(wts_a[sort_a])
        cdf_b = jnp.cumsum(wts_b[sort_b])
        quant_a = jnp.interp(grid, cdf_a, pos_a[sort_a])
        quant_b = jnp.interp(grid, cdf_b, pos_b[sort_b])
        return jnp.mean((quant_a - quant_b) ** 2)

    def w1d_interp_normalized(wts_a, wts_b):
        """Our interp-based approach — weights normalized."""
        wts_an = wts_a / jnp.sum(wts_a)
        wts_bn = wts_b / jnp.sum(wts_b)
        sort_a = jnp.argsort(pos_a)
        sort_b = jnp.argsort(pos_b)
        cdf_a = jnp.cumsum(wts_an[sort_a])
        cdf_b = jnp.cumsum(wts_bn[sort_b])
        quant_a = jnp.interp(grid, cdf_a, pos_a[sort_a])
        quant_b = jnp.interp(grid, cdf_b, pos_b[sort_b])
        return jnp.mean((quant_a - quant_b) ** 2)

    # Base weights
    wts_a = jnp.ones(n)
    wts_b = jnp.ones(n)

    # Test: scale wts_a by alpha
    def loss_unnorm(alpha):
        return w1d_interp(alpha * wts_a, wts_b)

    def loss_norm(alpha):
        return w1d_interp_normalized(alpha * wts_a, wts_b)

    alpha = 1.0
    eps = 1e-4

    ad_unnorm = float(jax.grad(loss_unnorm)(alpha))
    fd_unnorm = (float(loss_unnorm(alpha + eps)) -
                 float(loss_unnorm(alpha - eps))) / (2 * eps)

    ad_norm = float(jax.grad(loss_norm)(alpha))
    fd_norm = (float(loss_norm(alpha + eps)) -
               float(loss_norm(alpha - eps))) / (2 * eps)

    print(f"  Unnormalized weights:")
    print(f"    AD grad = {ad_unnorm:.6e},  FD grad = {fd_unnorm:.6e}")
    print(f"    ratio AD/FD = {ad_unnorm / (fd_unnorm + 1e-30):.4f}")
    print(f"  Normalized weights:")
    print(f"    AD grad = {ad_norm:.6e},  FD grad = {fd_norm:.6e}")
    print(f"    ratio AD/FD = {ad_norm / (fd_norm + 1e-30):.4f}")

    # Test at alpha=2.0 (energy doubled)
    alpha = 2.0
    ad_unnorm = float(jax.grad(loss_unnorm)(alpha))
    fd_unnorm = (float(loss_unnorm(alpha + eps)) -
                 float(loss_unnorm(alpha - eps))) / (2 * eps)

    print(f"  Unnormalized at alpha=2.0:")
    print(f"    AD grad = {ad_unnorm:.6e},  FD grad = {fd_unnorm:.6e}")
    print(f"    ratio AD/FD = {ad_unnorm / (fd_unnorm + 1e-30):.4f}")
    print()


# =============================================================================
# Test 3: OTT-style 1D Wasserstein (diff_q weighting)
# =============================================================================

def test_ott_style_w1d():
    """OTT approach: merge CDFs → quantile_levels → diff_q → weighted cost."""
    print("=" * 70)
    print("TEST 3: OTT-style 1D distance (diff_q weighting)")
    print("=" * 70)

    key = jax.random.PRNGKey(0)
    n = 200
    pos_a = jax.random.uniform(key, (n,), minval=0.0, maxval=1.0)
    pos_b = jax.random.uniform(jax.random.PRNGKey(1), (n,), minval=0.0, maxval=1.0)

    def w1d_ott_style(wts_a, wts_b):
        """OTT-style: merge, sort, CDF union, diff_q weighting."""
        # Sort each distribution
        i_a = jnp.argsort(pos_a)
        i_b = jnp.argsort(pos_b)
        sorted_a = pos_a[i_a]
        sorted_b = pos_b[i_b]

        # Merge all values
        all_values = jnp.concatenate([sorted_a, sorted_b])
        all_sorter = jnp.argsort(all_values)
        all_sorted = all_values[all_sorter]

        # Build PDFs in merged order
        a_pdf = jnp.concatenate([wts_a[i_a], jnp.zeros(n)])
        a_pdf = a_pdf[all_sorter]
        b_pdf = jnp.concatenate([jnp.zeros(n), wts_b[i_b]])
        b_pdf = b_pdf[all_sorter]

        # CDFs
        a_cdf = jnp.cumsum(a_pdf)
        b_cdf = jnp.cumsum(b_pdf)

        # Quantile levels = sorted union of both CDFs
        all_cdfs = jnp.concatenate([a_cdf, b_cdf])
        quantile_levels = jnp.sort(all_cdfs)

        # Inverse CDFs via searchsorted
        i_a_inv = jnp.searchsorted(a_cdf, quantile_levels)
        i_b_inv = jnp.searchsorted(b_cdf, quantile_levels)
        qa = all_sorted[i_a_inv]
        qb = all_sorted[i_b_inv]

        # Weighted cost
        diff_q = jnp.diff(quantile_levels)
        costs = (qa[1:] - qb[1:]) ** 2
        return jnp.sum(costs * diff_q)

    # Base weights
    wts_a = jnp.ones(n)
    wts_b = jnp.ones(n)

    def loss_ott(alpha):
        return w1d_ott_style(alpha * wts_a, wts_b)

    eps = 1e-4

    for alpha in [1.0, 2.0, 0.5]:
        ad = float(jax.grad(loss_ott)(alpha))
        fd = (float(loss_ott(alpha + eps)) -
              float(loss_ott(alpha - eps))) / (2 * eps)
        print(f"  alpha={alpha}:  AD = {ad:.6e},  FD = {fd:.6e},  "
              f"ratio = {ad / (fd + 1e-30):.4f}")
    print()


# =============================================================================
# Test 4: Isolate jnp.interp xp gradient vs diff_q gradient
# =============================================================================

def test_gradient_paths_isolated():
    """Directly compare the two gradient mechanisms for weight sensitivity."""
    print("=" * 70)
    print("TEST 4: Isolated gradient paths — interp(xp) vs diff_q")
    print("=" * 70)

    n = 100
    positions = jnp.linspace(0.0, 1.0, n)  # fixed sorted positions
    grid = jnp.linspace(1e-6, 1.0 - 1e-6, 500)
    base_weights = jnp.ones(n)

    # Path A: weights → CDF → interp(grid, CDF, positions)
    def path_interp(alpha):
        wts = alpha * base_weights
        cdf = jnp.cumsum(wts)
        quantiles = jnp.interp(grid, cdf, positions)
        return jnp.mean(quantiles ** 2)

    # Path B: weights → CDF → diff(CDF) → weighted sum
    def path_diff_q(alpha):
        wts = alpha * base_weights
        cdf = jnp.cumsum(wts)
        # Use quantile levels from CDF
        quantile_levels = jnp.sort(cdf)
        diff_q = jnp.diff(quantile_levels)
        # searchsorted to get position values at each quantile
        indices = jnp.searchsorted(cdf, quantile_levels)
        q_vals = positions[indices]
        costs = q_vals[1:] ** 2
        return jnp.sum(costs * diff_q)

    eps = 1e-4

    print("  Path A: weights → cumsum → CDF as xp in jnp.interp")
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        ad = float(jax.grad(path_interp)(alpha))
        fd = (float(path_interp(alpha + eps)) -
              float(path_interp(alpha - eps))) / (2 * eps)
        print(f"    alpha={alpha:.1f}:  AD = {ad:+.6e},  FD = {fd:+.6e},  "
              f"ratio = {ad / (fd + 1e-30):.4f}")

    print("  Path B: weights → cumsum → CDF → diff_q multiplier")
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        ad = float(jax.grad(path_diff_q)(alpha))
        fd = (float(path_diff_q(alpha + eps)) -
              float(path_diff_q(alpha - eps))) / (2 * eps)
        print(f"    alpha={alpha:.1f}:  AD = {ad:+.6e},  FD = {fd:+.6e},  "
              f"ratio = {ad / (fd + 1e-30):.4f}")
    print()


# =============================================================================
# Test 5: Full pipeline comparison with actual OTT
# =============================================================================

def test_full_comparison():
    """Full pipeline: sim-like pointclouds, compare OTT vs Huber for dE grad."""
    print("=" * 70)
    print("TEST 5: Full comparison — OTT SW vs Huber (unnormalized) vs Huber (normalized)")
    print("=" * 70)

    from ott.tools import sliced

    key = jax.random.PRNGKey(42)
    n = 500

    # Two 2D pointclouds
    pts_a = jax.random.uniform(key, (n, 2))
    pts_b = jax.random.uniform(jax.random.PRNGKey(1), (n, 2))
    base_wts_a = jax.random.uniform(jax.random.PRNGKey(2), (n,)) + 0.1
    wts_b = jax.random.uniform(jax.random.PRNGKey(3), (n,)) + 0.1

    n_proj = 50
    delta = 0.01

    # Huber interp-based (unnormalized)
    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    grid = jnp.linspace(1e-6, 1.0 - 1e-6, 500)

    def huber_unnorm(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wts_a[sa])
            cdf_b = jnp.cumsum(wts_b[sb])
            qa = jnp.interp(grid, cdf_a, pa[sa])
            qb = jnp.interp(grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            return jnp.mean(jnp.where(ad <= delta, 0.5 * diff**2,
                                       delta * (ad - 0.5 * delta)))
        return jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))

    # Huber interp-based (normalized)
    def huber_norm(alpha):
        wts_a = alpha * base_wts_a
        wa = wts_a / jnp.sum(wts_a)
        wb = wts_b / jnp.sum(wts_b)
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wa[sa])
            cdf_b = jnp.cumsum(wb[sb])
            qa = jnp.interp(grid, cdf_a, pa[sa])
            qb = jnp.interp(grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            return jnp.mean(jnp.where(ad <= delta, 0.5 * diff**2,
                                       delta * (ad - 0.5 * delta)))
        return jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))

    # OTT SW
    def ott_sw(alpha):
        wts_a = alpha * base_wts_a
        loss, _ = sliced.sliced_wasserstein(
            x=pts_a, y=pts_b, a=wts_a, b=wts_b,
            n_proj=n_proj, rng=key)
        return loss

    eps = 1e-4

    # JIT compile all
    _ = jax.grad(huber_unnorm)(1.0)
    _ = jax.grad(huber_norm)(1.0)
    _ = jax.grad(ott_sw)(1.0)

    print(f"  Varying weight scale alpha (simulates energy change):\n")
    print(f"  {'alpha':>6s}  {'Method':>20s}  {'AD grad':>12s}  {'FD grad':>12s}  {'ratio':>8s}")
    print(f"  {'-'*64}")

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for name, fn in [("Huber unnorm", huber_unnorm),
                          ("Huber norm", huber_norm),
                          ("OTT SW", ott_sw)]:
            ad = float(jax.grad(fn)(alpha))
            fd = (float(fn(alpha + eps)) - float(fn(alpha - eps))) / (2 * eps)
            ratio = ad / (fd + 1e-30) if abs(fd) > 1e-12 else float('inf')
            print(f"  {alpha:>6.1f}  {name:>20s}  {ad:>+12.6e}  {fd:>+12.6e}  {ratio:>8.3f}")
        print()


# =============================================================================
# Test 6: OTT-style Huber (diff_q path) — the proposed fix
# =============================================================================

def test_huber_diff_q():
    """Huber SW using OTT-style diff_q weighting instead of jnp.interp."""
    print("=" * 70)
    print("TEST 6: Huber with diff_q path (proposed fix) vs OTT vs interp")
    print("=" * 70)

    from ott.tools import sliced

    key = jax.random.PRNGKey(42)
    n = 500

    pts_a = jax.random.uniform(key, (n, 2))
    pts_b = jax.random.uniform(jax.random.PRNGKey(1), (n, 2))
    base_wts_a = jax.random.uniform(jax.random.PRNGKey(2), (n,)) + 0.1
    wts_b = jax.random.uniform(jax.random.PRNGKey(3), (n,)) + 0.1

    n_proj = 50
    delta = 0.01

    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    # NEW: Huber with diff_q path (OTT-style gradient flow)
    def huber_diff_q(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T   # (n, n_proj)
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            # Sort each distribution
            i_a = jnp.argsort(pa)
            i_b = jnp.argsort(pb)
            sorted_a = pa[i_a]
            sorted_b = pb[i_b]

            # Merge all position values
            all_values = jnp.concatenate([sorted_a, sorted_b])
            all_sorter = jnp.argsort(all_values)
            all_sorted = all_values[all_sorter]

            # Build PDFs in merged order
            a_pdf = jnp.concatenate([wts_a[i_a], jnp.zeros(n)])
            a_pdf = a_pdf[all_sorter]
            b_pdf = jnp.concatenate([jnp.zeros(n), wts_b[i_b]])
            b_pdf = b_pdf[all_sorter]

            # CDFs
            a_cdf = jnp.cumsum(a_pdf)
            b_cdf = jnp.cumsum(b_pdf)

            # Quantile levels = sorted union of both CDFs
            all_cdfs = jnp.concatenate([a_cdf, b_cdf])
            quantile_levels = jnp.sort(all_cdfs)

            # Inverse CDFs via searchsorted → gather positions
            i_a_inv = jnp.searchsorted(a_cdf, quantile_levels)
            i_b_inv = jnp.searchsorted(b_cdf, quantile_levels)
            qa = all_sorted[i_a_inv]
            qb = all_sorted[i_b_inv]

            # Huber cost weighted by quantile spacing
            diff_q = jnp.diff(quantile_levels)
            diff_pos = qa[1:] - qb[1:]
            abs_diff = jnp.abs(diff_pos)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_pos ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.sum(huber * diff_q)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # Old interp-based (unnormalized) for comparison
    grid = jnp.linspace(1e-6, 1.0 - 1e-6, 500)

    def huber_interp(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wts_a[sa])
            cdf_b = jnp.cumsum(wts_b[sb])
            qa = jnp.interp(grid, cdf_a, pa[sa])
            qb = jnp.interp(grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            return jnp.mean(jnp.where(ad <= delta, 0.5 * diff**2,
                                       delta * (ad - 0.5 * delta)))
        return jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))

    # OTT SW
    def ott_sw(alpha):
        wts_a = alpha * base_wts_a
        loss, _ = sliced.sliced_wasserstein(
            x=pts_a, y=pts_b, a=wts_a, b=wts_b,
            n_proj=n_proj, rng=key)
        return loss

    eps = 1e-4

    # JIT warmup
    _ = jax.grad(huber_diff_q)(1.0)
    _ = jax.grad(huber_interp)(1.0)
    _ = jax.grad(ott_sw)(1.0)

    print(f"\n  {'alpha':>6s}  {'Method':>22s}  {'Loss':>12s}  {'AD grad':>12s}  "
          f"{'FD grad':>12s}  {'ratio':>8s}")
    print(f"  {'-'*78}")

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for name, fn in [("Huber diff_q (NEW)", huber_diff_q),
                          ("Huber interp (OLD)", huber_interp),
                          ("OTT SW", ott_sw)]:
            loss_val = float(fn(alpha))
            ad = float(jax.grad(fn)(alpha))
            fd = (float(fn(alpha + eps)) - float(fn(alpha - eps))) / (2 * eps)
            ratio = ad / (fd + 1e-30) if abs(fd) > 1e-12 else float('inf')
            print(f"  {alpha:>6.1f}  {name:>22s}  {loss_val:>12.6e}  "
                  f"{ad:>+12.6e}  {fd:>+12.6e}  {ratio:>8.3f}")
        print()


# =============================================================================
# Test 7: Adaptive grid interp — can we keep interp speed + get weight grads?
# =============================================================================

def test_adaptive_grid():
    """Test interp with adaptive grid that scales with total mass.

    Idea: instead of fixed grid on [0,1] with normalized CDFs, use
    unnormalized CDFs and scale the grid to [0, max_mass]. Weight
    sensitivity enters through:
      1. grid = base_grid * max_mass  →  interp's x argument  (O(1) grad)
      2. loss = mean(huber) * max_mass  →  multiplicative factor
    """
    print("=" * 70)
    print("TEST 7: Adaptive grid interp vs diff_q vs fixed interp")
    print("=" * 70)

    from ott.tools import sliced

    key = jax.random.PRNGKey(42)
    n = 500

    pts_a = jax.random.uniform(key, (n, 2))
    pts_b = jax.random.uniform(jax.random.PRNGKey(1), (n, 2))
    base_wts_a = jax.random.uniform(jax.random.PRNGKey(2), (n,)) + 0.1
    wts_b = jax.random.uniform(jax.random.PRNGKey(3), (n,)) + 0.1

    n_proj = 50
    n_grid = 500
    delta = 0.01

    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    base_grid = jnp.linspace(1e-6, 1.0 - 1e-6, n_grid)

    # ── Method A: Adaptive grid (unnormalized CDF, grid scales with mass) ──

    def huber_adaptive(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        sum_a = jnp.sum(wts_a)
        sum_b = jnp.sum(wts_b)
        max_mass = jnp.maximum(sum_a, sum_b)

        # Grid covers [0, max_mass]
        grid = base_grid * max_mass

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wts_a[sa])
            cdf_b = jnp.cumsum(wts_b[sb])
            qa = jnp.interp(grid, cdf_a, pa[sa])
            qb = jnp.interp(grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            huber = jnp.where(ad <= delta, 0.5 * diff**2,
                              delta * (ad - 0.5 * delta))
            return jnp.mean(huber)

        spatial = jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))
        # Integral over [0, max_mass] = max_mass * mean
        return spatial * max_mass

    # ── Method B: Normalized interp × average mass ──

    def huber_norm_scaled(alpha):
        wts_a = alpha * base_wts_a
        sum_a = jnp.sum(wts_a)
        sum_b = jnp.sum(wts_b)
        wa = wts_a / sum_a
        wb = wts_b / sum_b

        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wa[sa])
            cdf_b = jnp.cumsum(wb[sb])
            qa = jnp.interp(base_grid, cdf_a, pa[sa])
            qb = jnp.interp(base_grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            huber = jnp.where(ad <= delta, 0.5 * diff**2,
                              delta * (ad - 0.5 * delta))
            return jnp.mean(huber)

        spatial = jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))
        return spatial * (sum_a + sum_b) / 2.0

    # ── Method C: diff_q (OTT-style, known working) ──

    def huber_diff_q(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            i_a = jnp.argsort(pa)
            i_b = jnp.argsort(pb)
            sorted_a = pa[i_a]
            sorted_b = pb[i_b]

            all_values = jnp.concatenate([sorted_a, sorted_b])
            all_sorter = jnp.argsort(all_values)
            all_sorted = all_values[all_sorter]

            a_pdf = jnp.concatenate([wts_a[i_a], jnp.zeros(n)])
            a_pdf = a_pdf[all_sorter]
            b_pdf = jnp.concatenate([jnp.zeros(n), wts_b[i_b]])
            b_pdf = b_pdf[all_sorter]

            a_cdf = jnp.cumsum(a_pdf)
            b_cdf = jnp.cumsum(b_pdf)

            all_cdfs = jnp.concatenate([a_cdf, b_cdf])
            quantile_levels = jnp.sort(all_cdfs)

            i_a_inv = jnp.searchsorted(a_cdf, quantile_levels)
            i_b_inv = jnp.searchsorted(b_cdf, quantile_levels)
            qa = all_sorted[i_a_inv]
            qb = all_sorted[i_b_inv]

            diff_q = jnp.diff(quantile_levels)
            diff_pos = qa[1:] - qb[1:]
            abs_diff = jnp.abs(diff_pos)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_pos ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.sum(huber * diff_q)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── Method D: Fixed grid interp unnormalized (old, broken) ──

    def huber_fixed(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wts_a[sa])
            cdf_b = jnp.cumsum(wts_b[sb])
            qa = jnp.interp(base_grid, cdf_a, pa[sa])
            qb = jnp.interp(base_grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            return jnp.mean(jnp.where(ad <= delta, 0.5 * diff**2,
                                       delta * (ad - 0.5 * delta)))
        return jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))

    # ── OTT SW ──

    def ott_sw(alpha):
        wts_a = alpha * base_wts_a
        loss, _ = sliced.sliced_wasserstein(
            x=pts_a, y=pts_b, a=wts_a, b=wts_b,
            n_proj=n_proj, rng=key)
        return loss

    eps = 1e-4

    methods = [
        ("Adaptive grid", huber_adaptive),
        ("Norm × mass", huber_norm_scaled),
        ("diff_q (OTT-style)", huber_diff_q),
        ("Fixed grid (OLD)", huber_fixed),
        ("OTT SW", ott_sw),
    ]

    # JIT warmup
    for name, fn in methods:
        _ = jax.grad(fn)(1.0)

    print(f"\n  {'alpha':>6s}  {'Method':>22s}  {'Loss':>12s}  {'AD grad':>12s}  "
          f"{'FD grad':>12s}  {'AD/FD':>8s}")
    print(f"  {'-'*82}")

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for name, fn in methods:
            loss_val = float(fn(alpha))
            ad = float(jax.grad(fn)(alpha))
            fd = (float(fn(alpha + eps)) - float(fn(alpha - eps))) / (2 * eps)
            ratio = ad / (fd + 1e-30) if abs(fd) > 1e-12 else float('inf')
            print(f"  {alpha:>6.1f}  {name:>22s}  {loss_val:>12.6e}  "
                  f"{ad:>+12.6e}  {fd:>+12.6e}  {ratio:>8.3f}")
        print()


# =============================================================================
# Test 8: CDF-based interp — flip the interpolation
# =============================================================================

def test_cdf_interp():
    """Flip the interp: interpolate CDF at positions instead of positions at CDF.

    Quantile approach: interp(cdf_grid, CDF_breakpoints, positions)
      weights → xp (weak), positions → fp (strong)

    CDF approach: interp(position_grid, sorted_positions, CDF_values)
      weights → fp (STRONG), positions → xp (weak but irrelevant)

    Loss = mean(Huber(F_a(x) - F_b(x))) — a robust Cramér-type distance.
    """
    print("=" * 70)
    print("TEST 8: CDF-based interp (flip: weights in fp, not xp)")
    print("=" * 70)

    from ott.tools import sliced

    key = jax.random.PRNGKey(42)
    n = 500

    pts_a = jax.random.uniform(key, (n, 2))
    pts_b = jax.random.uniform(jax.random.PRNGKey(1), (n, 2))
    base_wts_a = jax.random.uniform(jax.random.PRNGKey(2), (n,)) + 0.1
    wts_b = jax.random.uniform(jax.random.PRNGKey(3), (n,)) + 0.1

    n_proj = 50
    n_grid = 500
    delta_cdf = 1.0  # CDF differences can be O(100), so delta is larger

    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    # Position grid: fixed on [0, 1] since pointcloud coords are normalized
    pos_grid = jnp.linspace(0.0, 1.0, n_grid)

    # ── CDF-based interp (NEW) ──

    def huber_cdf(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T  # (n, n_proj)
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            # Sort each distribution by projected position
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)

            # Evaluate CDFs at position grid using interp
            # interp(x, xp, fp): xp = sorted positions, fp = CDF values
            # Weights flow through fp → STRONG gradient
            cdf_a = jnp.cumsum(wts_a[sa])
            cdf_b = jnp.cumsum(wts_b[sb])

            F_a = jnp.interp(pos_grid, pa[sa], cdf_a)
            F_b = jnp.interp(pos_grid, pb[sb], cdf_b)

            cdf_diff = F_a - F_b
            abs_diff = jnp.abs(cdf_diff)
            huber = jnp.where(abs_diff <= delta_cdf,
                              0.5 * cdf_diff ** 2,
                              delta_cdf * (abs_diff - 0.5 * delta_cdf))
            return jnp.mean(huber)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── Quantile-based normalized interp (OLD) ──

    base_grid = jnp.linspace(1e-6, 1.0 - 1e-6, n_grid)
    delta_quant = 0.01

    def huber_quantile_norm(alpha):
        wts_a = alpha * base_wts_a
        wa = wts_a / jnp.sum(wts_a)
        wb = wts_b / jnp.sum(wts_b)
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wa[sa])
            cdf_b = jnp.cumsum(wb[sb])
            qa = jnp.interp(base_grid, cdf_a, pa[sa])
            qb = jnp.interp(base_grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            return jnp.mean(jnp.where(ad <= delta_quant, 0.5 * diff**2,
                                       delta_quant * (ad - 0.5 * delta_quant)))
        return jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))

    # ── diff_q (known working reference) ──

    def huber_diff_q(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            i_a = jnp.argsort(pa)
            i_b = jnp.argsort(pb)
            sorted_a = pa[i_a]
            sorted_b = pb[i_b]

            all_values = jnp.concatenate([sorted_a, sorted_b])
            all_sorter = jnp.argsort(all_values)
            all_sorted = all_values[all_sorter]

            a_pdf = jnp.concatenate([wts_a[i_a], jnp.zeros(n)])
            a_pdf = a_pdf[all_sorter]
            b_pdf = jnp.concatenate([jnp.zeros(n), wts_b[i_b]])
            b_pdf = b_pdf[all_sorter]

            a_cdf = jnp.cumsum(a_pdf)
            b_cdf = jnp.cumsum(b_pdf)

            all_cdfs = jnp.concatenate([a_cdf, b_cdf])
            quantile_levels = jnp.sort(all_cdfs)

            i_a_inv = jnp.searchsorted(a_cdf, quantile_levels)
            i_b_inv = jnp.searchsorted(b_cdf, quantile_levels)
            qa = all_sorted[i_a_inv]
            qb = all_sorted[i_b_inv]

            diff_q = jnp.diff(quantile_levels)
            diff_pos = qa[1:] - qb[1:]
            abs_diff = jnp.abs(diff_pos)
            huber = jnp.where(abs_diff <= 0.01,
                              0.5 * diff_pos ** 2,
                              0.01 * (abs_diff - 0.5 * 0.01))
            return jnp.sum(huber * diff_q)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── OTT SW ──

    def ott_sw(alpha):
        wts_a = alpha * base_wts_a
        loss, _ = sliced.sliced_wasserstein(
            x=pts_a, y=pts_b, a=wts_a, b=wts_b,
            n_proj=n_proj, rng=key)
        return loss

    eps = 1e-4

    methods = [
        ("CDF interp (NEW)", huber_cdf),
        ("Quantile norm (OLD)", huber_quantile_norm),
        ("diff_q", huber_diff_q),
        ("OTT SW", ott_sw),
    ]

    # JIT warmup
    for name, fn in methods:
        _ = jax.grad(fn)(1.0)

    print(f"\n  {'alpha':>6s}  {'Method':>22s}  {'Loss':>12s}  {'AD grad':>12s}  "
          f"{'FD grad':>12s}  {'AD/FD':>8s}")
    print(f"  {'-'*82}")

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for name, fn in methods:
            loss_val = float(fn(alpha))
            ad = float(jax.grad(fn)(alpha))
            fd = (float(fn(alpha + eps)) - float(fn(alpha - eps))) / (2 * eps)
            ratio = ad / (fd + 1e-30) if abs(fd) > 1e-12 else float('inf')
            print(f"  {alpha:>6.1f}  {name:>22s}  {loss_val:>12.6e}  "
                  f"{ad:>+12.6e}  {fd:>+12.6e}  {ratio:>8.3f}")
        print()


# =============================================================================
# Test 9: Riemann-sum Wasserstein — weights multiply the cost directly
# =============================================================================

def test_riemann_sum():
    """Riemann-sum approach: evaluate b's quantile at a's CDF levels,
    weight each cost by a's weights.

    loss = sum_i  H(pos_a[i] - Q_b(CDF_a[i])) * w_a[i]

    Weight gradients:
      - w_a multiplicative in sum → O(1) STRONG
      - w_a → cumsum → cdf_a as query x of interp → O(1) via slope
    Position gradients:
      - pos_a direct in subtraction → O(1) STRONG
      - pos_b in fp of interp → O(1) STRONG
    """
    print("=" * 70)
    print("TEST 9: Riemann-sum Wasserstein (weights multiply cost)")
    print("=" * 70)

    from ott.tools import sliced

    key = jax.random.PRNGKey(42)
    n = 500

    pts_a = jax.random.uniform(key, (n, 2))
    pts_b = jax.random.uniform(jax.random.PRNGKey(1), (n, 2))
    base_wts_a = jax.random.uniform(jax.random.PRNGKey(2), (n,)) + 0.1
    wts_b = jax.random.uniform(jax.random.PRNGKey(3), (n,)) + 0.1

    n_proj = 50
    delta = 0.01

    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    # ── Riemann-sum: weights enter multiplicatively ──

    def huber_riemann(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sort_a = jnp.argsort(pa)
            sort_b = jnp.argsort(pb)

            pos_a_sorted = pa[sort_a]
            pos_b_sorted = pb[sort_b]
            wts_a_sorted = wts_a[sort_a]

            cdf_a = jnp.cumsum(wts_a_sorted)
            cdf_b = jnp.cumsum(wts_b[sort_b])

            # b's quantile at a's CDF levels
            Q_b = jnp.interp(cdf_a, cdf_b, pos_b_sorted)

            # Weighted Huber transport cost
            diff = pos_a_sorted - Q_b
            abs_diff = jnp.abs(diff)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.sum(huber * wts_a_sorted)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── diff_q (reference) ──

    def huber_diff_q(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            i_a = jnp.argsort(pa)
            i_b = jnp.argsort(pb)
            sorted_a = pa[i_a]
            sorted_b = pb[i_b]

            all_values = jnp.concatenate([sorted_a, sorted_b])
            all_sorter = jnp.argsort(all_values)
            all_sorted = all_values[all_sorter]

            a_pdf = jnp.concatenate([wts_a[i_a], jnp.zeros(n)])
            a_pdf = a_pdf[all_sorter]
            b_pdf = jnp.concatenate([jnp.zeros(n), wts_b[i_b]])
            b_pdf = b_pdf[all_sorter]

            a_cdf = jnp.cumsum(a_pdf)
            b_cdf = jnp.cumsum(b_pdf)

            all_cdfs = jnp.concatenate([a_cdf, b_cdf])
            quantile_levels = jnp.sort(all_cdfs)

            i_a_inv = jnp.searchsorted(a_cdf, quantile_levels)
            i_b_inv = jnp.searchsorted(b_cdf, quantile_levels)
            qa = all_sorted[i_a_inv]
            qb = all_sorted[i_b_inv]

            diff_q = jnp.diff(quantile_levels)
            diff_pos = qa[1:] - qb[1:]
            abs_diff = jnp.abs(diff_pos)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_pos ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.sum(huber * diff_q)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── Normalized quantile interp (old, broken for weights) ──

    grid = jnp.linspace(1e-6, 1.0 - 1e-6, 500)

    def huber_interp_norm(alpha):
        wts_a = alpha * base_wts_a
        wa = wts_a / jnp.sum(wts_a)
        wb = wts_b / jnp.sum(wts_b)
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wa[sa])
            cdf_b = jnp.cumsum(wb[sb])
            qa = jnp.interp(grid, cdf_a, pa[sa])
            qb = jnp.interp(grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            return jnp.mean(jnp.where(ad <= delta, 0.5 * diff**2,
                                       delta * (ad - 0.5 * delta)))
        return jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))

    # ── OTT SW ──

    def ott_sw(alpha):
        wts_a = alpha * base_wts_a
        loss, _ = sliced.sliced_wasserstein(
            x=pts_a, y=pts_b, a=wts_a, b=wts_b,
            n_proj=n_proj, rng=key)
        return loss

    eps = 1e-4

    methods = [
        ("Riemann-sum (NEW)", huber_riemann),
        ("diff_q (reference)", huber_diff_q),
        ("Interp norm (OLD)", huber_interp_norm),
        ("OTT SW", ott_sw),
    ]

    # JIT warmup
    for name, fn in methods:
        _ = jax.grad(fn)(1.0)

    print(f"\n  {'alpha':>6s}  {'Method':>22s}  {'Loss':>12s}  {'AD grad':>12s}  "
          f"{'FD grad':>12s}  {'AD/FD':>8s}")
    print(f"  {'-'*82}")

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for name, fn in methods:
            loss_val = float(fn(alpha))
            ad = float(jax.grad(fn)(alpha))
            fd = (float(fn(alpha + eps)) - float(fn(alpha - eps))) / (2 * eps)
            ratio = ad / (fd + 1e-30) if abs(fd) > 1e-12 else float('inf')
            print(f"  {alpha:>6.1f}  {name:>22s}  {loss_val:>12.6e}  "
                  f"{ad:>+12.6e}  {fd:>+12.6e}  {ratio:>8.3f}")
        print()


# =============================================================================
# Test 10: Position-space Wasserstein — interp(CDF in fp) + diff → integration weights
# =============================================================================

def test_posspace():
    """Position-space Wasserstein integral.

    W = ∫ H(x - T(x)) × f_a(x) dx

    where:
      - f_a(x) = diff(F_a) where F_a = interp(pos_grid, pos_sorted, CDF)
        → CDF in fp of interp → STRONG weight gradient
        → diff preserves gradient (linear)
        → multiplies cost → weight gradient via multiplication
      - T(x) = Q_b(F_a(x)) = interp(SG(F_a), CDF_b, pos_b)
        → stop_gradient decouples cost from weights (like searchsorted in diff_q)
      - H(x - T(x)) = Huber spatial cost, no weight gradient through SG

    Key: uses interp with CDF in fp + diff to get strong weight gradients,
    while computing the correct Wasserstein-Huber distance.
    """
    print("=" * 70)
    print("TEST 10: Position-space Wasserstein (interp fp + diff)")
    print("=" * 70)

    from ott.tools import sliced

    key = jax.random.PRNGKey(42)
    n = 500

    pts_a = jax.random.uniform(key, (n, 2))
    pts_b = jax.random.uniform(jax.random.PRNGKey(1), (n, 2))
    base_wts_a = jax.random.uniform(jax.random.PRNGKey(2), (n,)) + 0.1
    wts_b = jax.random.uniform(jax.random.PRNGKey(3), (n,)) + 0.1

    n_proj = 50
    n_grid = 500
    delta = 0.01

    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    pos_grid = jnp.linspace(0.0, 1.0, n_grid)

    # ── Position-space with stop_gradient ──

    def huber_posspace_sg(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sort_a = jnp.argsort(pa)
            sort_b = jnp.argsort(pb)

            pos_a_sorted = pa[sort_a]
            pos_b_sorted = pb[sort_b]

            cdf_a = jnp.cumsum(wts_a[sort_a])
            cdf_b = jnp.cumsum(wts_b[sort_b])

            # CDF of a at position grid — weights in fp → STRONG gradient
            F_a = jnp.interp(pos_grid, pos_a_sorted, cdf_a)

            # Transport map T(x) = Q_b(F_a(x))
            # stop_gradient on F_a: decouple cost from weights (like searchsorted)
            T = jnp.interp(jax.lax.stop_gradient(F_a), cdf_b, pos_b_sorted)

            # Spatial Huber cost
            diff_val = pos_grid - T
            abs_diff = jnp.abs(diff_val)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_val ** 2,
                              delta * (abs_diff - 0.5 * delta))

            # Integration weight = mass of a per grid bin (via diff of CDF)
            mass_a = jnp.diff(F_a)  # (n_grid-1,)

            return jnp.sum(huber[:-1] * mass_a)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── Position-space WITHOUT stop_gradient ──

    def huber_posspace_nosg(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sort_a = jnp.argsort(pa)
            sort_b = jnp.argsort(pb)

            pos_a_sorted = pa[sort_a]
            pos_b_sorted = pb[sort_b]

            cdf_a = jnp.cumsum(wts_a[sort_a])
            cdf_b = jnp.cumsum(wts_b[sort_b])

            F_a = jnp.interp(pos_grid, pos_a_sorted, cdf_a)
            # NO stop_gradient — cost also carries weight gradient
            T = jnp.interp(F_a, cdf_b, pos_b_sorted)

            diff_val = pos_grid - T
            abs_diff = jnp.abs(diff_val)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_val ** 2,
                              delta * (abs_diff - 0.5 * delta))

            mass_a = jnp.diff(F_a)
            return jnp.sum(huber[:-1] * mass_a)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── diff_q (reference) ──

    def huber_diff_q(alpha):
        wts_a = alpha * base_wts_a
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            i_a = jnp.argsort(pa)
            i_b = jnp.argsort(pb)
            sorted_a = pa[i_a]
            sorted_b = pb[i_b]

            all_values = jnp.concatenate([sorted_a, sorted_b])
            all_sorter = jnp.argsort(all_values)
            all_sorted = all_values[all_sorter]

            a_pdf = jnp.concatenate([wts_a[i_a], jnp.zeros(n)])
            a_pdf = a_pdf[all_sorter]
            b_pdf = jnp.concatenate([jnp.zeros(n), wts_b[i_b]])
            b_pdf = b_pdf[all_sorter]

            a_cdf = jnp.cumsum(a_pdf)
            b_cdf = jnp.cumsum(b_pdf)

            all_cdfs = jnp.concatenate([a_cdf, b_cdf])
            quantile_levels = jnp.sort(all_cdfs)

            i_a_inv = jnp.searchsorted(a_cdf, quantile_levels)
            i_b_inv = jnp.searchsorted(b_cdf, quantile_levels)
            qa = all_sorted[i_a_inv]
            qb = all_sorted[i_b_inv]

            diff_q = jnp.diff(quantile_levels)
            diff_pos = qa[1:] - qb[1:]
            abs_diff = jnp.abs(diff_pos)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_pos ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.sum(huber * diff_q)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    # ── Normalized quantile interp (old, broken) ──

    grid = jnp.linspace(1e-6, 1.0 - 1e-6, n_grid)

    def huber_interp_norm(alpha):
        wts_a = alpha * base_wts_a
        wa = wts_a / jnp.sum(wts_a)
        wb = wts_b / jnp.sum(wts_b)
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(pa, pb):
            sa = jnp.argsort(pa)
            sb = jnp.argsort(pb)
            cdf_a = jnp.cumsum(wa[sa])
            cdf_b = jnp.cumsum(wb[sb])
            qa = jnp.interp(grid, cdf_a, pa[sa])
            qb = jnp.interp(grid, cdf_b, pb[sb])
            diff = qa - qb
            ad = jnp.abs(diff)
            return jnp.mean(jnp.where(ad <= delta, 0.5 * diff**2,
                                       delta * (ad - 0.5 * delta)))
        return jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))

    # ── OTT SW ──

    def ott_sw(alpha):
        wts_a = alpha * base_wts_a
        loss, _ = sliced.sliced_wasserstein(
            x=pts_a, y=pts_b, a=wts_a, b=wts_b,
            n_proj=n_proj, rng=key)
        return loss

    eps = 1e-4

    methods = [
        ("Posspace+SG (NEW)", huber_posspace_sg),
        ("Posspace noSG", huber_posspace_nosg),
        ("diff_q (reference)", huber_diff_q),
        ("Interp norm (OLD)", huber_interp_norm),
        ("OTT SW", ott_sw),
    ]

    # JIT warmup
    for name, fn in methods:
        _ = jax.grad(fn)(1.0)

    print(f"\n  {'alpha':>6s}  {'Method':>22s}  {'Loss':>12s}  {'AD grad':>12s}  "
          f"{'FD grad':>12s}  {'AD/FD':>8s}")
    print(f"  {'-'*82}")

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for name, fn in methods:
            loss_val = float(fn(alpha))
            ad = float(jax.grad(fn)(alpha))
            fd = (float(fn(alpha + eps)) - float(fn(alpha - eps))) / (2 * eps)
            ratio = ad / (fd + 1e-30) if abs(fd) > 1e-12 else float('inf')
            print(f"  {alpha:>6.1f}  {name:>22s}  {loss_val:>12.6e}  "
                  f"{ad:>+12.6e}  {fd:>+12.6e}  {ratio:>8.3f}")
        print()


# =============================================================================

if __name__ == '__main__':
    test_posspace()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
