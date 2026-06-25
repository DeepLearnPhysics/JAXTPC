"""Tests for the unified drift-field distortion module (tools/distortion.py).

Covers the parity guarantee (new reps reproduce the sce_siren numerics), the
recover→produce bake bridge, and the design invariants the audits flagged:
single drift_direction flip, anode-BC carried in the baked data, and padding
(NaN) sanitization. CPU-only, no full sim needed.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools import distortion as DI
import tools.sce_siren as S

T = 89.0
E0 = 500.0


@pytest.fixture(scope="module")
def shared():
    vt, et = S.build_vinv_table(T)
    return {'v_table': vt, 'E_table': et, 'nominal_field': jnp.float32(E0)}


@pytest.fixture(scope="module")
def meta():
    v0 = S.drift_velocity_jax(jnp.float32(E0), T)
    return dict(norm_offsets=jnp.array([10., 0., 0.]), norm_scales=jnp.array([10., 20., 20.]),
                E0=jnp.float32(E0), v0=v0, drift_direction=jnp.float32(1.0))


@pytest.fixture(scope="module")
def positions():
    rng = np.random.RandomState(0)
    return jnp.asarray(np.stack([rng.uniform(0.5, 19.5, 256),
                                 rng.uniform(-19, 19, 256),
                                 rng.uniform(-19, 19, 256)], -1), jnp.float32)


@pytest.fixture(scope="module")
def siren_fp(meta):
    p = S.init_siren(jax.random.PRNGKey(1), hidden_features=16, hidden_layers=2, omega_0=2.0)
    return {**meta, 'weights': p['weights'], 'biases': p['biases'], 'omega_0': jnp.float32(2.0)}


@pytest.fixture(scope="module")
def poly_setup(meta):
    exps = S.poly_exps(3)
    coeffs = jnp.asarray(np.random.RandomState(2).randn(len(exps), 3) * 0.02, jnp.float32)
    return exps, {**meta, 'coeffs': coeffs}


def test_siren_delta_matches_sce_siren(siren_fp, positions):
    new = jax.vmap(lambda p: DI.siren_delta(siren_fp, p))(positions)
    old = S.siren_delta({'weights': siren_fp['weights'], 'biases': siren_fp['biases']},
                        positions, siren_fp['norm_offsets'], siren_fp['norm_scales'], siren_fp['omega_0'])
    assert np.max(np.abs(np.asarray(new) - np.asarray(old))) < 1e-6


def test_poly_delta_matches_sce_siren(poly_setup, positions):
    exps, fp = poly_setup
    new = jax.vmap(lambda p: DI.make_poly_delta(exps)(fp, p))(positions)
    old = S.poly_delta(fp['coeffs'], positions, fp['norm_offsets'], fp['norm_scales'], exps)
    assert np.max(np.abs(np.asarray(new) - np.asarray(old))) < 1e-6


def test_apply_distortion_efield_matches_recover_efield(siren_fp, positions, shared):
    out = DI.apply_distortion(DI.siren_delta, siren_fp, positions, float(siren_fp['v0']), shared)
    raw = S.recover_efield({'weights': siren_fp['weights'], 'biases': siren_fp['biases']},
                           positions, siren_fp['E0'], siren_fp['v0'], shared['v_table'],
                           shared['E_table'], siren_fp['norm_offsets'], siren_fp['norm_scales'],
                           siren_fp['omega_0'])
    # |efield_correction|·nominal == |E| (the Ex flip preserves magnitude)
    new_mag = np.asarray(jnp.linalg.norm(out.efield_correction, axis=1)) * E0
    assert np.max(np.abs(new_mag - np.asarray(jnp.linalg.norm(raw, axis=1)))) < 1e-2


def test_drift_direction_single_flip(siren_fp, positions, shared):
    o_pos = DI.apply_distortion(DI.siren_delta, {**siren_fp, 'drift_direction': jnp.float32(1.0)},
                                positions, float(siren_fp['v0']), shared)
    o_neg = DI.apply_distortion(DI.siren_delta, {**siren_fp, 'drift_direction': jnp.float32(-1.0)},
                                positions, float(siren_fp['v0']), shared)
    # |E| invariant; Ex sign flips; transverse unchanged
    assert np.allclose(jnp.linalg.norm(o_pos.efield_correction, axis=1),
                       jnp.linalg.norm(o_neg.efield_correction, axis=1), atol=1e-6)
    assert np.allclose(o_pos.efield_correction[:, 0], -o_neg.efield_correction[:, 0], atol=1e-6)
    assert np.allclose(o_pos.efield_correction[:, 1:], o_neg.efield_correction[:, 1:], atol=1e-6)


def test_bake_grid_reproduces_poly_and_carries_bc(poly_setup, positions):
    exps, fp = poly_setup
    delta_fn = DI.make_poly_delta(exps)
    G = 32
    gfp = DI.bake(delta_fn, fp, [0., -20., -20.], [20./(G-1), 40./(G-1), 40./(G-1)], (G, G, G))
    # anode plane (x index 0) carries Δ=0 in the DATA (BC is rep-internal, not re-applied)
    assert np.max(np.abs(np.asarray(gfp['grid'][0]))) < 1e-5
    # grid_delta reproduces the poly Δ to grid resolution
    poly = jax.vmap(lambda p: delta_fn(fp, p))(positions)
    grid = jax.vmap(lambda p: DI.grid_delta(gfp, p))(positions)
    assert np.max(np.abs(np.asarray(poly) - np.asarray(grid))) < 5e-2  # cm, 32^3 res


def test_none_is_first_class_no_distortion(positions):
    assert 'none' in DI.REPS
    assert DI.make_delta_fn('none') is None              # 'none' → use nominal_outputs
    out = DI.nominal_outputs(positions)
    assert np.allclose(np.asarray(out.efield_correction), np.array([1., 0., 0.]))
    assert np.all(np.asarray(out.drift_time_corr_us) == 0)
    assert np.all(np.asarray(out.drift_yz_corr_cm) == 0)
    with pytest.raises(ValueError):
        DI.make_delta_fn('bogus')


def test_padding_positions_sanitized(siren_fp, shared):
    pos = jnp.array([[5., 0., 0.], [jnp.nan, 0., 0.], [jnp.inf, 1., -1.]], jnp.float32)
    out = DI.apply_distortion(DI.siren_delta, siren_fp, pos, float(siren_fp['v0']), shared)
    assert np.all(np.isfinite(np.asarray(out.efield_correction)))
    assert np.all(np.isfinite(np.asarray(out.drift_time_corr_us)))
    assert np.all(np.isfinite(np.asarray(out.drift_yz_corr_cm)))
