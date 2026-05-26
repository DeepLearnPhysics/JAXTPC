"""Single run (Init A) for 1000 steps with convergence plot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax, jax.numpy as jnp, numpy as np, optax, time, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight
from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax, generate_muon_segments_trig, mask_outside_volume, build_muon_forward)

N_SEGMENTS = 4000; STEP_SIZE_MM = 0.5
N_STEPS = 600; LR = 0.01; B1 = 0.9; B2 = 0.99
TRUTH_THETA = np.pi / 4; TRUTH_PHI = np.pi / 6
TRUTH_PHYS = np.array([-200, 0, 100, np.sin(TRUTH_THETA), np.cos(TRUTH_THETA),
                        np.sin(TRUTH_PHI), np.cos(TRUTH_PHI), 500.0])
SCALES = np.array([200, 200, 200, 1, 1, 1, 1, 500.0])
SCALES_JAX = jnp.array(SCALES, dtype=jnp.float32)
INIT_PHYS = np.array([-200 + 400, 0 + 300, 100 - 300,
                       np.sin(TRUTH_THETA + 0.4), np.cos(TRUTH_THETA + 0.4),
                       np.sin(TRUTH_PHI + 0.4), np.cos(TRUTH_PHI + 0.4), 600.0])
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

log_T, dedx = load_dedx_table_jax()
det = generate_detector('config/cubic_wireplane_config.yaml')
sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS)
forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

def fwd(phys):
    pos, de = generate_muon_segments_trig(phys[7], jnp.array([phys[0], phys[1], phys[2]]),
        phys[3], phys[4], phys[5], phys[6], STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
    de = mask_outside_volume(pos, de)
    sigs = forward(pos, de)
    active = de > 0
    ep_start = jax.lax.stop_gradient(pos[jnp.argmax(active)])
    ep_end = jax.lax.stop_gradient(pos[N_SEGMENTS - 1 - jnp.argmax(active[::-1])])
    return sigs, ep_start, ep_end

print('Compiling...', flush=True)
t0 = time.time()
truth_sigs, truth_start, truth_end = jax.jit(fwd)(jnp.array(TRUTH_PHYS, dtype=jnp.float32))
for s in truth_sigs: jax.block_until_ready(s)
truth_start = np.array(truth_start); truth_end = np.array(truth_end)
spec_w = tuple(make_sobolev_weight(*truth_sigs[p].shape, s=1.5) for p in range(6))

def loss_fn(n):
    sigs, ep_start, ep_end = fwd(n * SCALES_JAX)
    loss = sobolev_loss_geomean_log1p(sigs, truth_sigs, spec_w)
    return loss, (ep_start, ep_end)

lg = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
init_n = jnp.array(INIT_PHYS / SCALES, dtype=jnp.float32)
(l, _), g = lg(init_n); jax.block_until_ready(g)
print(f'  All compiled ({time.time()-t0:.0f}s)', flush=True)

def project_uc(p):
    st, ct = p[3], p[4]; nt = jnp.maximum(jnp.sqrt(st**2 + ct**2), 1e-8)
    sp, cp = p[5], p[6]; np_ = jnp.maximum(jnp.sqrt(sp**2 + cp**2), 1e-8)
    return p.at[3].set(st/nt).at[4].set(ct/nt).at[5].set(sp/np_).at[6].set(cp/np_)

opt = optax.adam(LR, b1=B1, b2=B2)
state = opt.init(init_n); params = init_n
phist = np.empty((N_STEPS + 1, 8)); lhist = np.empty(N_STEPS + 1)
ep_s = np.empty((N_STEPS + 1, 3)); ep_e = np.empty((N_STEPS + 1, 3))
phist[0] = np.array(params * SCALES_JAX); lhist[0] = float(l)
# Get init endpoints
(_, (init_ep_s, init_ep_e)), _ = lg(init_n)
ep_s[0] = np.array(init_ep_s); ep_e[0] = np.array(init_ep_e)

print(f'Running {N_STEPS} steps...', flush=True)
ts = time.time()
for step in range(N_STEPS):
    (loss, (ep_start, ep_end)), grad = lg(params)
    updates, state = opt.update(grad, state, params)
    params = optax.apply_updates(params, updates)
    params = project_uc(params)
    p = np.array(params * SCALES_JAX)
    phist[step + 1] = p; lhist[step + 1] = float(loss)
    ep_s[step + 1] = np.array(ep_start); ep_e[step + 1] = np.array(ep_end)
    if (step + 1) % 100 == 0:
        th = np.degrees(np.arctan2(p[3], p[4]))
        ph = np.degrees(np.arctan2(p[5], p[6]))
        print(f'  Step {step+1:4d}: loss={float(loss):.2f} x={p[0]:.0f} y={p[1]:.0f} z={p[2]:.0f} '
              f'th={th:.1f} ph={ph:.1f} E={p[7]:.0f} ({time.time()-ts:.0f}s)', flush=True)
print(f'Optimization done ({time.time()-ts:.0f}s)')

# Save NPZ
out = os.path.join(OUT_DIR, 'multi_optimization_history.npz')
np.savez(out, truth_phys=TRUTH_PHYS, truth_start=truth_start, truth_end=truth_end,
    n_steps=N_STEPS, loss_history_0=lhist, param_history_0=phist,
    endpoints_starts_0=ep_s, endpoints_ends_0=ep_e)
print(f'Saved {out}')

# Plot convergence
steps = np.arange(N_STEPS + 1)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].semilogy(steps, lhist, lw=1.5)
axes[0, 0].set_ylabel('Loss'); axes[0, 0].set_title('Loss'); axes[0, 0].grid(True, alpha=0.3)

dr = np.sqrt((phist[:, 0] + 200)**2 + phist[:, 1]**2 + (phist[:, 2] - 100)**2)
axes[0, 1].semilogy(steps, dr + 1e-3, lw=1.5)
axes[0, 1].set_ylabel('|r-r_truth| mm'); axes[0, 1].set_title('Position Diff'); axes[0, 1].grid(True, alpha=0.3)

dth = np.abs(np.degrees(np.arctan2(phist[:, 3], phist[:, 4]) - TRUTH_THETA))
axes[0, 2].semilogy(steps, dth + 1e-4, lw=1.5)
axes[0, 2].set_ylabel('deg'); axes[0, 2].set_title('Theta Diff'); axes[0, 2].grid(True, alpha=0.3)

dph = np.abs(np.degrees(np.arctan2(phist[:, 5], phist[:, 6]) - TRUTH_PHI))
axes[1, 0].semilogy(steps, dph + 1e-4, lw=1.5)
axes[1, 0].set_ylabel('deg'); axes[1, 0].set_title('Phi Diff'); axes[1, 0].grid(True, alpha=0.3)

dE = np.abs(phist[:, 7] - 500)
axes[1, 1].semilogy(steps, dE + 1e-3, lw=1.5)
axes[1, 1].set_ylabel('MeV'); axes[1, 1].set_title('Energy Diff'); axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].axis('off')
for ax in axes.flat:
    if ax.get_visible():
        ax.set_xlabel('Step')

fig.suptitle(f'Init A: {N_STEPS} steps, Adam LR={LR} b1={B1} b2={B2}\n'
             f'Truth: x=-200, y=0, z=100, th=45, ph=30, E=500', fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT_DIR, 'single_run_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved convergence plot')
print('All done!')
