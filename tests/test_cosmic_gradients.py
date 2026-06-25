"""Cosmic-ray-like muon samples (known entrance/exit, high energy) and their
gradients through the differentiable simulator w.r.t. the SCE field.

Fast tests cover the generator + the "outside is zeroed" guarantee (value AND
gradient). A slow test verifies the end-to-end gradient ∂(signal)/∂(SCE field)
on a cosmic track is finite and finite-difference-consistent — the foundation
for recovering the SCE field from cosmic-ray data.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.particle_generator import (
    load_dedx_table_jax, generate_cosmic_chord, sample_surface_endpoints,
    mask_outside_volume)

# The PDG dE/dx table (tools/data/) is required; skip cleanly if absent.
_HAS_DEDX = os.path.exists(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'tools', 'data', 'muon_dedx_lar.csv'))
pytestmark = pytest.mark.skipif(not _HAS_DEDX, reason="muon dE/dx table not present")

HALF = (200.0, 200.0, 200.0)  # mm, a 40 cm cube centred at origin


def test_cosmic_chord_spans_volume_high_energy():
    """A GeV muon between two surface points stays inside, ~MIP, no stopping."""
    logT, dedx = load_dedx_table_jax()
    entr = jnp.array([-200.0, -120.0, 80.0])
    ex = jnp.array([200.0, 100.0, -60.0])
    pos, de, theta, phi, step = generate_cosmic_chord(
        entr, ex, 4000.0, 256, logT, dedx, half_extents_mm=HALF)
    inside = (jnp.abs(pos[:, 0]) < HALF[0]) & (jnp.abs(pos[:, 1]) < HALF[1]) & \
             (jnp.abs(pos[:, 2]) < HALF[2])
    # chord-spanning ⇒ all interior except possibly the endpoint(s) lying
    # exactly ON a surface face (entrance here), which mask_outside_volume
    # conservatively zeros — a measure-zero edge, not a leak.
    assert int(inside.sum()) >= 256 - 2
    assert int((de > 0).sum()) >= 256 - 2                 # high E ⇒ no stopping
    # ~constant dE/dx along the chord (MIP) over the interior segments
    dedx_seg = np.asarray(de / (step / 10.0))[np.asarray(inside)]
    assert float(dedx_seg.std() / dedx_seg.mean()) < 0.05


def test_cosmic_outside_is_zeroed_value_and_gradient():
    """Overshoot the chord so segments leave the box; those must contribute
    zero dE and zero gradient (the explicit outside guard)."""
    logT, dedx = load_dedx_table_jax()
    entr = jnp.array([-200.0, -120.0, 80.0])
    ex = jnp.array([200.0, 100.0, -60.0])
    # n_segments sized for a chord 50% longer than entrance→exit ⇒ overshoot
    L = float(jnp.linalg.norm(ex - entr))
    n = 384
    # generate WITHOUT masking to find the outside set, then WITH masking
    pos, de_raw, *_ = generate_cosmic_chord(entr, ex * 1.5, 4000.0, n, logT, dedx)
    # "outside" = strictly beyond a wall (mask keeps on-face points via <=);
    # those truly-outside segments must contribute zero value and gradient.
    outside = (jnp.abs(pos[:, 0]) > HALF[0] + 0.1) | (jnp.abs(pos[:, 1]) > HALF[1] + 0.1) | \
              (jnp.abs(pos[:, 2]) > HALF[2] + 0.1)
    assert int(outside.sum()) > 0                          # some really are outside
    de_masked = mask_outside_volume(pos, de_raw, HALF)
    assert float(de_masked[outside].sum()) == 0.0

    def loss(d):
        return jnp.sum(mask_outside_volume(pos, d, HALF) ** 2)
    g = jax.grad(loss)(de_raw)
    assert float(jnp.abs(g[outside]).sum()) == 0.0        # zero gradient outside


def test_sample_surface_endpoints_on_faces():
    rng = np.random.RandomState(0)
    for _ in range(20):
        a, b = sample_surface_endpoints(rng, HALF)
        # each endpoint lies on at least one face (|coord| == half extent)
        for p in (a, b):
            on_face = any(abs(abs(p[i]) - HALF[i]) < 1e-3 for i in range(3))
            assert on_face
        assert np.linalg.norm(b - a) >= min(HALF)


@pytest.mark.slow
@pytest.mark.requires_kernels
def test_sce_field_gradient_on_cosmic_finite_difference():
    """∂(signal loss)/∂(SCE field) on a cosmic track, through the differentiable
    sim, is finite and matches a finite-difference check."""
    import tempfile
    import tools.sce_siren as S
    from tools.simulation import DetectorSimulator

    cfg = {'volumes': [{'id': 0,
            'geometry': {'ranges': [[-20., 0.], [-20., 20.], [-20., 20.]],
                         'drift_direction': -1},
            'planes': [{'plane_id': p, 'type': t, 'angle': a, 'wire_spacing': 0.3,
                        'distance_from_anode': d, 'bias_voltage': bv}
                       for p, (t, a, d, bv) in enumerate(
                           [('first_induction', 60., 0.6, -200.),
                            ('second_induction', -60., 0.3, -200.),
                            ('collection', 0., 0., 500.)])]}],
           'readout': {'sampling_rate': 2.0, 'electrons_per_adc': 182},
           'simulation': {'drift': {'velocity': 1.6, 'longitudinal_diffusion': 6.2,
                          'transverse_diffusion': 16.3, 'electron_lifetime': 10.0},
           'charge_recombination': {'model': 'modified_box',
                                    'recomb_parameters': {'alpha': 0.93, 'beta': 0.212}}},
           'medium': {'type': 'liquid_argon',
                      'properties': {'density': 1.396, 'ionization_energy': 23.6,
                                     'excitation_ratio': 0.21},
                      'temperature': 87., 'pressure': 1.0},
           'electric_field': {'field_strength': 500.0}}

    p = S.init_siren(jax.random.PRNGKey(5))
    p['weights'][-1] = p['weights'][-1] * 20.0
    fp = os.path.join(tempfile.mkdtemp(), 's.npz')
    S.save_siren_npz(fp, p, 5.0, np.array([10., 20., 20.]),
                     np.array([10., 20., 20.]), 500.0, 89.0)

    NSEG = 256
    sim = DetectorSimulator(cfg, total_pad=NSEG, response_chunk_size=NSEG,
                            include_track_hits=False, differentiable=True,
                            n_segments=NSEG, iterate_mode='scan', distortion=fp)
    logT, dedx = load_dedx_table_jax()
    pos, de, _, _, step = generate_cosmic_chord(
        jnp.array([-20., -12., 8.]) * 10, jnp.array([0., 10., -6.]) * 10,
        4000.0, NSEG, logT, dedx, half_extents_mm=(200., 200., 200.))
    base = sim._default_sim_params

    def loss(sce):
        sigs = sim.forward_segments(base._replace(distortion_field=sce), pos, de, dx=step)
        return sum(jnp.sum(s ** 2) for s in sigs)

    g = jax.grad(loss)(base.distortion_field)
    gw = g['weights'][-1]
    assert bool(jnp.all(jnp.isfinite(gw)))
    assert float(jnp.linalg.norm(gw)) > 0.0

    eps, idx = 1e-3, (0, 0, 0)
    W = base.distortion_field['weights'][-1]

    def L1(Wx):
        s = dict(base.distortion_field); wl = list(s['weights']); wl[-1] = Wx
        s['weights'] = wl
        return float(loss(s))
    fd = (L1(W.at[idx].add(eps)) - L1(W.at[idx].add(-eps))) / (2 * eps)
    assert abs(fd - float(gw[idx])) / (abs(fd) + 1e-9) < 0.05


@pytest.mark.slow
@pytest.mark.requires_kernels
def test_sce_strength_closure_from_cosmics():
    """Recover the global SCE strength from cosmic-ray signals by gradient
    descent through the differentiable simulator: α: 0.4 → ~1.0, loss ↓ ≫10×."""
    import optax
    from closure.cosmic.recover_field import build, emag_grid

    sim, pos, de, step = build(n_seg=256, truth_scale=1.0)
    base = sim._default_sim_params
    truth = base.distortion_field

    def fwd(stk):
        return sim.forward_segments(base._replace(distortion_field=stk), pos, de, dx=step)
    obs = [jax.lax.stop_gradient(s) for s in fwd(truth)]

    def scaled(alpha):
        s = dict(truth); wl = list(s['weights']); wl[-1] = wl[-1] * alpha
        s['weights'] = wl
        return s

    def loss(alpha):
        sg = fwd(scaled(alpha))
        return sum(jnp.mean((a - b) ** 2) for a, b in zip(sg, obs)) / len(sg)

    Et = emag_grid(sim, truth)
    a = jnp.array(0.4)
    opt = optax.adam(optax.cosine_decay_schedule(0.05, 50))
    st = opt.init(a)
    vg = jax.jit(jax.value_and_grad(loss))
    l0 = float(loss(a))
    for _ in range(50):
        l, g = vg(a); u, st = opt.update(g, st, a); a = optax.apply_updates(a, u)
    e_mae = float(jnp.mean(jnp.abs(emag_grid(sim, scaled(a)) - Et)))
    assert float(loss(a)) < l0 / 10.0       # signal loss collapses
    assert abs(float(a) - 1.0) < 0.1        # strength recovered
    assert e_mae < 2.0                       # |E| within ~2 V/cm of truth
