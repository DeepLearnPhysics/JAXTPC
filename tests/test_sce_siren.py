"""Tests for the differentiable SCE-SIREN field (tools/sce_siren.py) and its
wiring into the forward sim.

Fast unit tests cover the physics (Walkowiak port, v(E) inversion round-trip,
the Δ→E recovery formula on an analytically known field) and JAX hygiene
(JIT + grad). A slow integration test confirms the SIREN actually plumbs into
DetectorSimulator and changes the readout while staying finite.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools import sce_siren as S


# --------------------------------------------------------------------------- #
#  Physics: drift velocity + inversion                                        #
# --------------------------------------------------------------------------- #

def test_walkowiak_matches_numpy_reference():
    """JAX Walkowiak port == the NumPy generator version to float precision."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'efield'))
    np_dv = pytest.importorskip(
        'ElectricDistortion.core.drift_velocity',
        reason="ElectricDistortion package not importable").drift_velocity
    for E in (100.0, 273.0, 500.0, 1000.0):
        assert abs(float(np_dv(E)) - float(S.drift_velocity_jax(E))) < 1e-5


def test_vinv_table_round_trips():
    """Inverting v(E) at the nominal field returns E0."""
    T = 89.0
    v_tab, E_tab = S.build_vinv_table(T)
    for E0 in (273.0, 400.0, 500.0, 800.0):
        v0 = S.drift_velocity_jax(E0, T)
        E_rt = float(jnp.interp(v0, v_tab, E_tab))
        assert abs(E_rt - E0) < 1.0  # within table resolution


def test_efield_from_dDdx_recovers_known_field():
    """The Δ→E inversion recovers a known *nonuniform* field exactly.

    For a constant field Eₓ=E1 the drift is uniform: Δx=x0·(v0/v1−1), so
    ∂Δx/∂x = v0/v(E1)−1. Feeding that gradient back through efield_from_dDdx
    must return E1 — this is the closed-loop test of the v-inversion, and it
    fails for the naïve linear formula Eₓ=E0(1−∂Δx/∂x)."""
    T, E0 = 89.0, 500.0
    v_tab, E_tab = S.build_vinv_table(T)
    v0 = float(S.drift_velocity_jax(E0, T))
    for E1 in (450.0, 480.0, 520.0, 560.0):
        v1 = float(S.drift_velocity_jax(E1, T))
        dDdx_x = v0 / v1 - 1.0
        dDdx = jnp.array([[dDdx_x, 0.0, 0.0]])
        E = S.efield_from_dDdx(dDdx, E0, v0, v_tab, E_tab)
        assert abs(float(E[0, 0]) - E1) < 1.0
        # naïve linear formula would give E0*(1 - dDdx_x); confirm it's wrong
        naive = E0 * (1.0 - dDdx_x)
        if abs(E1 - E0) > 20:
            assert abs(naive - E1) > abs(float(E[0, 0]) - E1)


def test_vinv_table_strictly_increasing_and_positive():
    """Table knots are physical (v>0) and strictly increasing for jnp.interp."""
    v_tab, E_tab = S.build_vinv_table(89.0)
    assert bool(jnp.all(v_tab > 0))
    assert bool(jnp.all(jnp.diff(v_tab) > 0))


def test_efield_from_dDdx_causality_floor():
    """Nonphysical gradient ∂Δx/∂x ≤ −1 stays finite and saturates at high E
    instead of flipping to the spurious low-E branch (causality floor)."""
    T, E0 = 89.0, 500.0
    v_tab, E_tab = S.build_vinv_table(T)
    v0 = float(S.drift_velocity_jax(E0, T))
    for bad in (-1.0, -2.0, -5.0):
        E = S.efield_from_dDdx(jnp.array([[bad, 0.0, 0.0]]), E0, v0, v_tab, E_tab)
        assert bool(jnp.all(jnp.isfinite(E)))
        assert float(E[0, 0]) > E0  # saturates high, not collapsed to E_min


def test_transverse_efield_first_order():
    """Ey, Ez follow the first-order transverse relation E⊥ = −E0·∂Δ⊥/∂x."""
    T, E0 = 89.0, 500.0
    v_tab, E_tab = S.build_vinv_table(T)
    v0 = float(S.drift_velocity_jax(E0, T))
    dDdx = jnp.array([[0.0, 0.01, -0.02]])
    E = S.efield_from_dDdx(dDdx, E0, v0, v_tab, E_tab)
    assert abs(float(E[0, 1]) - (-E0 * 0.01)) < 1e-3
    assert abs(float(E[0, 2]) - (-E0 * -0.02)) < 1e-3


# --------------------------------------------------------------------------- #
#  SIREN forward + recovery: JAX hygiene                                      #
# --------------------------------------------------------------------------- #

def _siren_bundle(seed=0, L=(216.0, 432.0, 432.0), E0=500.0, T=89.0):
    Lx, Ly, Lz = L
    params = S.init_siren(jax.random.PRNGKey(seed))
    norm_off = jnp.array([Lx / 2, 0.0, 0.0])          # local-frame offsets
    norm_sc = jnp.array([Lx / 2, Ly / 2, Lz / 2])
    v_tab, E_tab = S.build_vinv_table(T)
    v0 = float(S.drift_velocity_jax(E0, T))
    return params, norm_off, norm_sc, v_tab, E_tab, v0, E0


def test_recover_efield_jit_finite_and_near_anode_nominal():
    params, off, sc, v_tab, E_tab, v0, E0 = _siren_bundle()
    f = jax.jit(lambda p, x: S.recover_efield(p, x, E0, v0, v_tab, E_tab, off, sc, 5.0))
    pos = jnp.array(np.random.RandomState(1).uniform(
        [0, -200, -200], [200, 200, 200], size=(128, 3)).astype('float32'))
    E = f(params, pos)
    assert E.shape == (128, 3)
    assert bool(jnp.all(jnp.isfinite(E)))
    # BC: Δ→0 at the anode (x=0) ⇒ ∂Δ/∂x there is bounded ⇒ Ex near E0.
    near = jnp.array([[0.5, 0.0, 0.0]])
    assert abs(float(f(params, near)[0, 0]) - E0) < 50.0


def test_recover_efield_differentiable_in_params():
    """∂/∂(SIREN params) of an E-derived loss is finite (needed for closure)."""
    params, off, sc, v_tab, E_tab, v0, E0 = _siren_bundle()
    pos = jnp.array(np.random.RandomState(2).uniform(
        [0, -200, -200], [200, 200, 200], size=(32, 3)).astype('float32'))

    def loss(p):
        E = S.recover_efield(p, pos, E0, v0, v_tab, E_tab, off, sc, 5.0)
        return jnp.sum(E ** 2)
    g = jax.grad(loss)(params)
    assert bool(jnp.all(jnp.isfinite(g['weights'][0])))
    assert float(jnp.sum(jnp.abs(g['weights'][0]))) > 0.0


def test_siren_save_load_round_trip(tmp_path):
    params, off, sc, v_tab, E_tab, v0, E0 = _siren_bundle()
    path = str(tmp_path / 'siren.npz')
    S.save_siren_npz(path, params, 5.0,
                     np.array([108.0, 216.0, 216.0]),
                     np.array([108.0, 216.0, 216.0]), E0, 89.0,
                     extra=dict(Lx=216.0))
    p2, meta = S.load_siren_npz(path)
    assert meta['E0'] == E0 and meta['omega_0'] == 5.0
    pos = jnp.array([[10.0, 5.0, -3.0]])
    d1 = S.siren_delta(params, pos, jnp.array([108., 216., 216.]),
                       jnp.array([108., 216., 216.]), 5.0)
    d2 = S.siren_delta(p2, pos, jnp.array([108., 216., 216.]),
                       jnp.array([108., 216., 216.]), 5.0)
    assert float(jnp.max(jnp.abs(d1 - d2))) < 1e-6


def test_compute_phi_drift_grad_finite_drift_aligned():
    """∂phi_drift/∂θ is finite for a track exactly aligned with the drift
    field (cos_phi → 1) — without the arccos clip this NaN-poisons grads."""
    from tools.physics import compute_phi_drift
    efield_corr = jnp.array([[1.0, 0.0, 0.0]])  # E along +x (drift axis)

    def loss(theta):
        phi_d, _ = compute_phi_drift(efield_corr, theta, jnp.array([0.0]), 500.0)
        return jnp.sum(phi_d)

    g = jax.grad(loss)(jnp.array([jnp.pi / 2]))  # track ∥ x → cos_phi = 1
    assert bool(jnp.all(jnp.isfinite(g)))


def test_anode_bc_zero_distortion():
    """Output BC factor (x_norm+1) forces Δ=0 exactly at the anode (x=0)."""
    params, off, sc, _, _, _, _ = _siren_bundle()
    at_anode = jnp.array([[0.0, 30.0, -40.0], [0.0, -10.0, 5.0]])
    d = S.siren_delta(params, at_anode, off, sc, 5.0)
    assert float(jnp.max(jnp.abs(d))) < 1e-6


# --------------------------------------------------------------------------- #
#  Integration: SIREN plumbs into DetectorSimulator                           #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.requires_kernels
def test_siren_wires_into_simulator(tmp_path):
    """A SIREN field loads via electric_dist_siren_path, runs end-to-end, and
    perturbs the readout vs. the no-SCE baseline while staying finite."""
    import jax
    from tools.simulation import DetectorSimulator
    from tools.config import create_sim_config
    from tools.loader import build_deposit_data

    wire_config = {
        'volumes': [{
            'id': 0,
            'geometry': {'ranges': [[-20.0, 0.0], [-20.0, 20.0], [-20.0, 20.0]],
                         'drift_direction': -1},
            'planes': [
                {'plane_id': 0, 'type': 'first_induction', 'angle': 60.0,
                 'wire_spacing': 0.3, 'distance_from_anode': 0.6, 'bias_voltage': -200.0},
                {'plane_id': 1, 'type': 'second_induction', 'angle': -60.0,
                 'wire_spacing': 0.3, 'distance_from_anode': 0.3, 'bias_voltage': -200.0},
                {'plane_id': 2, 'type': 'collection', 'angle': 0.0,
                 'wire_spacing': 0.3, 'distance_from_anode': 0.0, 'bias_voltage': 500.0},
            ]}],
        'readout': {'sampling_rate': 2.0, 'electrons_per_adc': 182},
        'simulation': {
            'drift': {'velocity': 1.6, 'longitudinal_diffusion': 6.2,
                      'transverse_diffusion': 16.3, 'electron_lifetime': 10.0},
            'charge_recombination': {'model': 'modified_box',
                                     'recomb_parameters': {'alpha': 0.93, 'beta': 0.212}}},
        'medium': {'type': 'liquid_argon',
                   'properties': {'density': 1.396, 'ionization_energy': 23.6,
                                  'excitation_ratio': 0.21},
                   'temperature': 87.0, 'pressure': 1.0},
        'electric_field': {'field_strength': 500.0},
    }
    # Build a SIREN with an inflated output scale so the (untrained) distortion
    # is non-negligible — enough to measurably move the signal.
    Lx, Ly, Lz = 20.0, 40.0, 40.0
    params = S.init_siren(jax.random.PRNGKey(3))
    params['weights'][-1] = params['weights'][-1] * 30.0
    path = str(tmp_path / 'siren_min.npz')
    S.save_siren_npz(path, params, 5.0,
                     np.array([Lx / 2, Ly / 2, Lz / 2]),
                     np.array([Lx / 2, Ly / 2, Lz / 2]), 500.0, 89.0)

    TOTAL_PAD, RESP_CHUNK = 2000, 1000
    rng = np.random.RandomState(7)
    pts = rng.uniform([-18, -18, -18], [-2, 18, 18], size=(200, 3))
    pos = (pts * 10.0).astype(np.float32)
    de = rng.uniform(0.5, 3.0, 200).astype(np.float32)
    dx = rng.uniform(0.05, 0.4, 200).astype(np.float32)
    sc = create_sim_config(wire_config, total_pad=TOTAL_PAD, include_track_hits=False)
    dep = build_deposit_data(pos, de, dx, sc, track_ids=np.zeros(200, np.int32),
                             group_size=5, gap_threshold_mm=5.0)

    def run(**kw):
        sim = DetectorSimulator(wire_config, total_pad=TOTAL_PAD,
                                response_chunk_size=RESP_CHUNK,
                                include_track_hits=False, **kw)
        sig, _, _ = sim.process_event(dep, key=jax.random.PRNGKey(0))
        from tools.output import to_dense
        return to_dense(sig, sim.config)

    base = run()
    sce = run(include_electric_dist=True, electric_dist_siren_path=path)

    for k in base:
        assert bool(jnp.all(jnp.isfinite(sce[k]))), f"non-finite signal in {k}"
    total_diff = sum(float(jnp.sum(jnp.abs(sce[k] - base[k]))) for k in base)
    assert total_diff > 0.0, "SCE-SIREN did not perturb the readout"
