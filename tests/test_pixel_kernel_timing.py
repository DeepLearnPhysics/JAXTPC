"""
Sub-tick timing of the pixel response kernel.

The pixel response NPZ is at a finer time resolution (0.1 us) than the
simulation tick, so ``interpolate_pixel_response_kernel`` shifts the trace by
the deposit's sub-tick offset and then rebins it. These tests place the kernel
output on the global time axis exactly as the simulator does
(``global_tick = time_index - time_zero_bin + j``) and check it against the
response delayed to the true arrival time.

Files under test: tools/kernels.py (load_pixel_response_kernel,
interpolate_pixel_response_kernel)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tools.kernels import load_pixel_response_kernel, interpolate_pixel_response_kernel


NPZ_DT = 0.1      # us, NPZ time bin
N_GLOBAL = 64     # global ticks used to place the kernel output


def _write_npz(path, arrival_bin, n_time=48, shape="ramp"):
    """Synthetic distance-indexed pixel response, 10 bins per pixel.

    Centre-pixel current over the 20 NPZ bins before arrival (integral 1
    electron per electron), zero after arrival: an asymmetric ramp, or a
    smooth Gaussian (sigma 0.3 us) for which the tick-centre centroid of the
    rebinned signal has no visible binning error.
    """
    resp = np.zeros((15, 15, n_time))
    k = np.arange(arrival_bin - 20, arrival_bin)
    if shape == "ramp":
        pulse = (k - k[0] + 1.0) ** 2
    else:
        pulse = np.exp(-0.5 * ((k - k.mean()) / 3.0) ** 2)
    pulse /= pulse.sum() * NPZ_DT
    resp[:5, :5, k] = pulse
    np.savez(path, response=resp, pixel_bin_size=0.04, time_bin_size=NPZ_DT,
             drift_time=arrival_bin * NPZ_DT)
    return pulse, k


def _place(pk, tick_us, dt):
    """Centre-pixel kernel output placed on the global tick axis (simulator rule)."""
    time_index = jnp.floor(tick_us / dt).astype(jnp.int32)
    t_offset = tick_us / dt - time_index
    out = interpolate_pixel_response_kernel(
        pk.DKernel, 0.0, 0.0, 0.0, t_offset, pk.pixel_spacing,
        3, 3, pk.rebin_factor)[1, 1]
    idx = time_index - pk.time_zero_bin + jnp.arange(out.shape[0])
    return jnp.zeros(N_GLOBAL).at[idx].add(out)


def _expected(pulse, k, arrival_bin, tick_us, dt):
    """Exact tick average of the piecewise-constant response delayed to arrive at tick_us."""
    edges = np.arange(N_GLOBAL + 1) * dt
    out = np.zeros(N_GLOBAL)
    for v, kk in zip(pulse, k):
        lo = tick_us + (kk - arrival_bin) * NPZ_DT
        overlap = np.clip(np.minimum(edges[1:], lo + NPZ_DT) - np.maximum(edges[:-1], lo), 0, None)
        out += v * overlap / dt
    return out


@pytest.mark.parametrize("arrival_bin", [33, 36])
@pytest.mark.parametrize("rebin", [1, 5])
def test_kernel_matches_delayed_response(tmp_path, arrival_bin, rebin):
    """Placed kernel output equals the tick-averaged response arriving at tick_us.

    Sweeps the arrival over three ticks. arrival_bin 33 and 36 give the two
    phases (arrival_bin - 1) % 5 = 2 and 0 of the NPZ time origin.
    """
    path = tmp_path / "pixel_response.npz"
    pulse, k = _write_npz(path, arrival_bin)
    dt = NPZ_DT * rebin
    pk = load_pixel_response_kernel(str(path), num_s=4, time_spacing=dt, pixel_pitch_cm=0.4)
    for tick_us in 20 * dt + np.linspace(0.0, 3 * dt, 31):
        got = np.asarray(_place(pk, float(tick_us), dt))
        want = _expected(pulse, k, arrival_bin, tick_us, dt)
        np.testing.assert_allclose(got, want, atol=1e-5 * want.max(),
                                   err_msg=f"tick_us={tick_us:.3f}")


def test_centroid_follows_arrival(tmp_path):
    """Signal centroid minus arrival time is constant across a tick (0.5 us tick).

    The expected offset is the centroid of the response relative to its
    arrival. Tolerance 0.005 us (1 % of a tick).
    """
    arrival_bin = 33
    path = tmp_path / "pixel_response.npz"
    pulse, k = _write_npz(path, arrival_bin, shape="gauss")
    dt = 0.5
    pk = load_pixel_response_kernel(str(path), num_s=4, time_spacing=dt, pixel_pitch_cm=0.4)
    c_resp = np.sum(pulse * ((k + 0.5 - arrival_bin) * NPZ_DT)) / pulse.sum()
    t_mid = (np.arange(N_GLOBAL) + 0.5) * dt
    resid = []
    for tick_us in 10.0 + np.linspace(0.0, 2 * dt, 41):
        sig = np.asarray(_place(pk, float(tick_us), dt))
        resid.append(np.sum(sig * t_mid) / sig.sum() - (tick_us + c_resp))
    resid = np.array(resid)
    assert np.abs(resid).max() < 0.005, resid


class TestTimingGradient:
    """d(signal)/d(arrival time) is finite and matches finite differences."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Kernel built in float32 as in production; evaluated in float64 so the
        finite differences converge (see test_pipeline_forward.py)."""
        path = tmp_path / "pixel_response.npz"
        _write_npz(path, 33)
        dt = 0.5
        pk = load_pixel_response_kernel(str(path), num_s=4, time_spacing=dt, pixel_pitch_cm=0.4)
        was_enabled = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        w = jnp.asarray(np.random.RandomState(0).normal(size=N_GLOBAL))
        yield (lambda t: jnp.sum(w * _place(pk, jnp.float64(t), dt))), dt
        jax.config.update("jax_enable_x64", was_enabled)

    def test_continuous_across_tick_boundary(self, setup):
        f, dt = setup
        t_b = 21 * dt
        for eps in (1e-4, 1e-6):
            assert abs(float(f(t_b - eps)) - float(f(t_b + eps))) < 100 * eps

    def test_grad_matches_fd_near_boundaries(self, setup):
        f, dt = setup
        h = 1e-5
        # either side of a tick boundary and of an NPZ-bin boundary inside a tick
        for t in (21 * dt - 3 * h, 21 * dt + 3 * h, 21 * dt + 0.2 - 3 * h, 21 * dt + 0.2 + 3 * h,
                  21 * dt + 0.137):
            g = float(jax.grad(f)(jnp.float64(t)))
            fd = (float(f(t + h)) - float(f(t - h))) / (2 * h)
            assert np.isfinite(g)
            assert g == pytest.approx(fd, rel=1e-4, abs=1e-6), (t, g, fd)
