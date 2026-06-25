"""
Differentiable SCE field via a SIREN distortion representation.

This is the JAX/JIT-side counterpart to ``efield/ElectricDistortion`` (the
first-principles SCE generator). Instead of storing two independent maps
(E-field + drift corrections) and interpolating each with trilinear (C0)
interpolation, we represent the **spatial distortion field** Δ(r) — the only
directly observable SCE quantity — as a single, smooth, infinitely
differentiable SIREN. The electric field is then *derived* from Δ by
automatic differentiation, so Δ is the single source of truth and E and Δ
are guaranteed self-consistent.

Why a SIREN
-----------
A SIREN (Sitzmann et al., NeurIPS 2020) uses ``sin(ω₀·(W·x + b))`` activations,
giving outputs whose spatial derivatives are exact and smooth — exactly what
E-field recovery needs (``∂Δ/∂x``). A trilinear-interpolated map would give
piecewise-constant, discontinuous gradients.

Δ → E recovery (the key physics)
---------------------------------
The longitudinal distortion is defined as ``Δx ≡ v₀·t_drift − x₀`` with
``v₀ = v(E₀)`` the nominal drift velocity and
``t_drift(x₀) = ∫₀^{x₀} dx / v(Eₓ(x))``. Differentiating
``x₀ + Δx = v₀·t_drift`` w.r.t. x₀:

    1 + ∂Δx/∂x = v₀ / v(Eₓ)        ⟹    v(Eₓ) = v₀ / (1 + ∂Δx/∂x)

so Eₓ is obtained by **inverting the Walkowiak v(E) curve** at the target
velocity ``v₀/(1+∂Δx/∂x)``. The naïve linear formula
``Eₓ = E₀·(1 − ∂Δx/∂x)`` is WRONG because v(E) is sublinear — using it
mis-estimates Eₓ (and hence recombination) by several V/cm. Transverse
components are first-order exact:

    Ey = −E₀·∂Δy/∂x ,   Ez = −E₀·∂Δz/∂x

All derivatives are taken **along the drift coordinate x** via ``jax.jvp``
with tangent ``[1, 0, 0]``.

Units
-----
positions : cm ;  Δ : cm ;  E : V/cm ;  v : cm/μs.  The recovered E (V/cm) is
exactly what ``tools.recombination.compute_quanta`` expects (it divides by
1000 internally to get kV/cm for the Box/EMB ξ term).

Frame
-----
The SIREN is defined in the canonical drift frame used throughout JAXTPC's
local geometry: anode at ``x = 0`` and x increasing toward the cathode
(``x ∈ [0, Lx]`` is the drift distance), transverse coordinates carried via
the ``norm_offsets`` / ``norm_scales`` normalisation. The output BC factor
``(x_norm + 1)`` enforces ``Δ = 0`` at the anode (x = 0) exactly.

Parameter pytree
----------------
A trained SIREN is the dict::

    {'weights': [W0, W1, ..., W_out],   # last entry is the linear output layer
     'biases':  [b0, b1, ..., b_out]}

with static metadata (``omega_0``, ``norm_offsets``, ``norm_scales``) carried
alongside. The pytree is a plain nest of ``jnp.ndarray`` so it can be a JIT
argument or closure-captured constant, and is differentiable end-to-end.
"""

import numpy as np
import jax
import jax.numpy as jnp


# =========================================================================
# Walkowiak drift velocity (pure JAX) — NIM A 449 (2000), LArSoft/ICARUS set
# =========================================================================

_WK = dict(P1=-0.04640, P2=0.01712, P3=1.88125,
           P4=0.99408, P5=0.01172, P6=4.20214, T0=105.749)


def drift_velocity_jax(E_Vcm, T=89.0):
    """Electron drift velocity in LAr (cm/μs) from field magnitude (V/cm).

    Pure-JAX port of ``ElectricDistortion.core.drift_velocity`` so it can be
    traced/differentiated inside the sim. Returns 0 where E ≤ 0.
    """
    E_Vcm = jnp.asarray(E_Vcm)
    tshift = T - _WK['T0']
    E_kV = E_Vcm / 1000.0
    safe_E = jnp.where(E_kV > 0, E_kV, 1.0)
    vd_mm_us = (
        (_WK['P1'] * tshift + 1.0)
        * (_WK['P3'] * safe_E * jnp.log(1.0 + _WK['P4'] / safe_E)
           + _WK['P5'] * safe_E ** _WK['P6'])
        + _WK['P2'] * tshift
    )
    vd = vd_mm_us / 10.0  # mm/μs → cm/μs
    return jnp.where(E_kV > 0, vd, 0.0)


def build_vinv_table(T=89.0, E_min=25.0, E_max=2000.0, n=20000):
    """Monotonic (v_table, E_table) lookup for inverting v(E) via ``jnp.interp``.

    ``E = jnp.interp(v_target, v_table, E_table)``. ``jnp.interp`` requires a
    strictly increasing ``xp`` (= ``v_table``) and clamps to the endpoint E
    outside [v_table[0], v_table[-1]].

    ``E_min`` defaults to 25 V/cm, not 1: the Walkowiak parameterisation goes
    *negative* below ~22 V/cm, which would (a) seed the table with nonphysical
    knots and (b) break the monotonicity ``jnp.interp`` relies on. Clamping the
    table to the physical positive-velocity branch keeps every knot meaningful;
    we assert strict monotonicity rather than silently re-sorting (a re-sort
    would pair v with the wrong E if the curve were ever non-monotone).
    """
    E_table = np.linspace(E_min, E_max, n).astype(np.float32)
    v_table = np.asarray(drift_velocity_jax(E_table, T=T), dtype=np.float32)
    if not np.all(np.diff(v_table) > 0):
        raise ValueError(
            "v(E) is not strictly increasing on the requested E range; "
            "the v→E inversion table would be ambiguous. Raise E_min.")
    return jnp.asarray(v_table), jnp.asarray(E_table)


# =========================================================================
# SIREN forward (pure JAX)
# =========================================================================

def siren_forward(weights, biases, omega_0, coords_norm):
    """Evaluate the SIREN at one normalised coordinate ``coords_norm`` (3,).

    Hidden layers use ``sin(ω₀·(W·x + b))``; the final layer is linear. The
    output is scaled by ``(x_norm + 1)`` to enforce Δ = 0 at the anode.
    Returns the distortion vector (3,) in cm.
    """
    x = coords_norm
    for W, b in zip(weights[:-1], biases[:-1]):
        x = jnp.sin(omega_0 * (W @ x + b))
    raw = weights[-1] @ x + biases[-1]
    return raw * (coords_norm[0] + 1.0)


def _to_norm(positions_cm, norm_offsets, norm_scales):
    return (positions_cm - norm_offsets) / norm_scales


def siren_delta(params, positions_cm, norm_offsets, norm_scales, omega_0):
    """Distortion Δ(r) at physical positions (N, 3) cm → (N, 3) cm."""
    w, b = params['weights'], params['biases']
    coords = _to_norm(positions_cm, norm_offsets, norm_scales)
    return jax.vmap(lambda c: siren_forward(w, b, omega_0, c))(coords)


def efield_from_dDdx(dDdx, E0, v0, v_table, E_table):
    """E-field (N, 3) V/cm from the distortion gradient ∂Δ/∂x (N, 3).

    Eₓ via exact Walkowiak inversion ``v(Eₓ) = v₀/(1 + ∂Δx/∂x)``; Ey, Ez via
    the first-order transverse relation ``E⊥ = −E₀·∂Δ⊥/∂x``. Factored out of
    :func:`recover_efield` so the inversion physics is testable independently
    of the SIREN representation.

    Causality bounds ``∂Δx/∂x > −1`` (an electron cannot arrive before it is
    created), so ``1 + ∂Δx/∂x > 0``. A SIREN is unconstrained, so we floor the
    denominator at a small positive value: this keeps ``v_target`` positive and
    finite (no sign flip to a spurious low-E branch), and ``jnp.interp`` then
    saturates Eₓ at the table's E_max for any residual nonphysical region
    rather than returning garbage. Physical SCE has ``|∂Δx/∂x| ≪ 1`` so the
    floor never engages there.
    """
    denom = jnp.maximum(1.0 + dDdx[:, 0], 1e-3)
    v_target = v0 / denom
    Ex = jnp.interp(v_target, v_table, E_table)
    Ey = -E0 * dDdx[:, 1]
    Ez = -E0 * dDdx[:, 2]
    return jnp.stack([Ex, Ey, Ez], axis=-1)


def recover_efield(params, positions_cm, E0, v0, v_table, E_table,
                   norm_offsets, norm_scales, omega_0):
    """Recover the E-field (N, 3) in V/cm from the SIREN distortion field.

    Eₓ via exact Walkowiak inversion ``v(Eₓ) = v₀/(1 + ∂Δx/∂x)``; Ey, Ez via
    the first-order transverse relation ``E⊥ = −E₀·∂Δ⊥/∂x``. Derivatives are
    along physical x (``jax.jvp`` tangent ``[1, 0, 0]``). Fully JIT- and
    grad-compatible.

    Parameters
    ----------
    params : dict             SIREN pytree ({'weights', 'biases'}).
    positions_cm : (N, 3)     Query positions in cm (drift frame).
    E0 : float                Nominal field (V/cm).
    v0 : float                Nominal drift velocity v(E0) (cm/μs).
    v_table, E_table : (M,)   Monotonic inversion table from ``build_vinv_table``.
    norm_offsets, norm_scales : (3,)   Coordinate normalisation.
    omega_0 : float           SIREN frequency.

    Returns
    -------
    E : (N, 3) in V/cm.
    """
    w, b = params['weights'], params['biases']

    def delta_phys(xyz):
        c = (xyz - norm_offsets) / norm_scales
        return siren_forward(w, b, omega_0, c)

    tangent = jnp.array([1.0, 0.0, 0.0])
    dDdx = jax.vmap(lambda xyz: jax.jvp(delta_phys, (xyz,), (tangent,))[1])(
        positions_cm)  # (N, 3) = ∂Δ/∂x
    return efield_from_dDdx(dDdx, E0, v0, v_table, E_table)


# =========================================================================
# Polynomial Δ field — a more expressive / better-conditioned alternative to
# the SIREN (the SIREN is the representation bottleneck; see run_express). Same
# Δ→E pipeline (E derived from ∂Δ/∂x), same anode-BC scaling (Δ=0 at anode via
# the (x_norm+1) factor). `exps` is a static list of integer (a,b,c) tuples.
# =========================================================================

def poly_delta(coeffs, positions_cm, norm_offsets, norm_scales, exps):
    """Polynomial distortion Δ(r) at positions (N,3) cm → (N,3) cm. Δ=0 at anode."""
    xn = (positions_cm - norm_offsets) / norm_scales
    mon = jnp.stack([xn[:, 0] ** a * xn[:, 1] ** b * xn[:, 2] ** c for (a, b, c) in exps], -1)
    return (mon @ coeffs) * (xn[:, 0:1] + 1.0)


def recover_efield_poly(coeffs, positions_cm, E0, v0, v_table, E_table,
                        norm_offsets, norm_scales, exps):
    """E (N,3) V/cm from a polynomial Δ field — same Δ→E inversion as the SIREN."""
    def delta_phys(xyz):
        xn = (xyz - norm_offsets) / norm_scales
        mon = jnp.stack([xn[0] ** a * xn[1] ** b * xn[2] ** c for (a, b, c) in exps])
        return (mon @ coeffs) * (xn[0] + 1.0)
    tangent = jnp.array([1.0, 0.0, 0.0])
    dDdx = jax.vmap(lambda xyz: jax.jvp(delta_phys, (xyz,), (tangent,))[1])(positions_cm)
    return efield_from_dDdx(dDdx, E0, v0, v_table, E_table)


def poly_exps(deg):
    return [(a, b, c) for a in range(deg + 1) for b in range(deg + 1) for c in range(deg + 1) if a + b + c <= deg]


# =========================================================================
# Initialisation / training (pure JAX + optax) — used to fit Δ maps offline
# =========================================================================

def init_siren(key, in_features=3, out_features=3,
               hidden_features=128, hidden_layers=3, omega_0=5.0):
    """SIREN parameter init matching the equinox reference (Sitzmann scheme).

    First layer bound 1/in; subsequent hidden + output bounds √(6/in)/ω₀.
    Returns ``params`` dict ({'weights', 'biases'}).
    """
    import jax.random as jrandom
    keys = jrandom.split(key, hidden_layers + 2)
    weights, biases = [], []

    def _layer(k, out_f, in_f, bound):
        wk, bk = jrandom.split(k)
        W = jrandom.uniform(wk, (out_f, in_f), minval=-bound, maxval=bound)
        bvec = jrandom.uniform(bk, (out_f,), minval=-bound, maxval=bound)
        return W, bvec

    # first hidden layer
    W, bvec = _layer(keys[0], hidden_features, in_features, 1.0 / in_features)
    weights.append(W); biases.append(bvec)
    # remaining hidden layers
    hb = jnp.sqrt(6.0 / hidden_features) / omega_0
    for i in range(1, hidden_layers):
        W, bvec = _layer(keys[i], hidden_features, hidden_features, hb)
        weights.append(W); biases.append(bvec)
    # linear output layer (same bound as hidden)
    W, bvec = _layer(keys[hidden_layers], out_features, hidden_features, hb)
    weights.append(W); biases.append(bvec)
    return {'weights': weights, 'biases': biases}


def train_siren(positions_cm, corrections_cm, norm_offsets, norm_scales,
                omega_0=5.0, hidden_features=128, hidden_layers=3,
                n_epochs=2000, lines_per_batch=256, n_per_line=None,
                peak_lr=1e-3, seed=0, verbose=True):
    """Fit a SIREN to distortion samples (MSE). Returns the params dict.

    If ``n_per_line`` is given the data is treated as ``(n_lines, n_per_line)``
    blocks and batched by line (matching the reference notebook); otherwise it
    is a flat point cloud trained full-batch.
    """
    import optax
    import jax.random as jrandom

    pos = jnp.asarray((positions_cm - norm_offsets) / norm_scales, jnp.float32)
    corr = jnp.asarray(corrections_cm, jnp.float32)
    params = init_siren(jrandom.PRNGKey(seed), hidden_features=hidden_features,
                        hidden_layers=hidden_layers, omega_0=omega_0)

    if n_per_line is not None:
        n_lines = pos.shape[0] // n_per_line
        pos = pos[:n_lines * n_per_line].reshape(n_lines, n_per_line, 3)
        corr = corr[:n_lines * n_per_line].reshape(n_lines, n_per_line, 3)
        n_batches = max(1, n_lines // lines_per_batch)
    else:
        n_lines, n_batches = pos.shape[0], 1

    total_steps = n_epochs * n_batches
    sched = optax.warmup_cosine_decay_schedule(
        init_value=peak_lr * 0.01, peak_value=peak_lr,
        warmup_steps=max(1, 50 * n_batches), decay_steps=total_steps,
        end_value=peak_lr * 0.01)
    opt = optax.adam(sched)
    opt_state = opt.init(params)

    def loss_fn(p, bp, bc):
        pred = jax.vmap(lambda c: siren_forward(p['weights'], p['biases'],
                                                omega_0, c))(bp)
        return jnp.mean((pred - bc) ** 2)

    @jax.jit
    def step(p, opt_state, bp, bc):
        loss, grads = jax.value_and_grad(loss_fn)(p, bp, bc)
        updates, opt_state = opt.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    rng = np.random.RandomState(seed)
    last = None
    for epoch in range(n_epochs):
        if n_per_line is not None:
            perm = rng.permutation(n_lines)
            ep_loss = 0.0
            for bi in range(n_batches):
                idx = perm[bi * lines_per_batch:(bi + 1) * lines_per_batch]
                bp = pos[idx].reshape(-1, 3)
                bc = corr[idx].reshape(-1, 3)
                params, opt_state, loss = step(params, opt_state, bp, bc)
                ep_loss += float(loss)
            last = ep_loss / n_batches
        else:
            params, opt_state, loss = step(params, opt_state, pos, corr)
            last = float(loss)
        if verbose and epoch % 200 == 0:
            print(f"  [siren] epoch {epoch:4d}  loss {last:.6e}")
    if verbose:
        print(f"  [siren] final loss {last:.6e}")
    return params


# =========================================================================
# Serialisation
# =========================================================================

def save_siren_npz(path, params, omega_0, norm_offsets, norm_scales,
                   E0, T=89.0, extra=None):
    """Save a trained SIREN + metadata to a flat ``.npz``."""
    data = {}
    for i, (W, b) in enumerate(zip(params['weights'], params['biases'])):
        data[f'w_{i}'] = np.asarray(W)
        data[f'b_{i}'] = np.asarray(b)
    data['type'] = 'siren'   # self-describing distortion-file type tag
    data['n_layers'] = np.int32(len(params['weights']))
    data['omega_0'] = np.float32(omega_0)
    data['norm_offsets'] = np.asarray(norm_offsets, np.float32)
    data['norm_scales'] = np.asarray(norm_scales, np.float32)
    data['E0'] = np.float32(E0)
    data['T'] = np.float32(T)
    if extra:
        for k, v in extra.items():
            data[k] = np.asarray(v)
    np.savez(path, **data)


def load_siren_npz(path):
    """Load a SIREN saved by :func:`save_siren_npz`.

    Returns ``(params, meta)`` where ``meta`` has omega_0, norm_offsets,
    norm_scales, E0, T (and any extra scalars).
    """
    d = np.load(path, allow_pickle=False)
    n = int(d['n_layers'])
    weights = [jnp.asarray(d[f'w_{i}']) for i in range(n)]
    biases = [jnp.asarray(d[f'b_{i}']) for i in range(n)]
    meta = dict(
        omega_0=float(d['omega_0']),
        norm_offsets=jnp.asarray(d['norm_offsets']),
        norm_scales=jnp.asarray(d['norm_scales']),
        E0=float(d['E0']),
        T=float(d['T']),
    )
    return {'weights': weights, 'biases': biases}, meta
