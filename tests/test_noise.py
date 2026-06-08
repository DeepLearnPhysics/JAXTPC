"""
Tests for noise generation.

File under test: tools/noise.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.noise import (
    load_noise_params,
    _get_noise_spectrum_shape,
    _generate_noise_for_plane,
)


@pytest.fixture
def noise_params():
    """Load noise model parameters."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'config', 'noise_spectrum.npz')
    if not os.path.exists(path):
        pytest.skip("noise_spectrum.npz not found")
    return load_noise_params(path)


class TestNoiseGeneration:
    """Tests for _generate_noise_for_plane."""

    @pytest.fixture
    def noise_setup(self, noise_params):
        noise_x, noise_y, noise_z, emp_freqs, emp_shape = noise_params
        num_wires = 50
        num_time_ticks = 500
        wire_length_m = 3.0

        spectrum = _get_noise_spectrum_shape(num_time_ticks, emp_freqs, emp_shape)
        series_rms = np.full(num_wires, noise_y + noise_z * wire_length_m, dtype=np.float32)
        target_rms = np.sqrt(noise_x**2 + (noise_y + noise_z * wire_length_m)**2)

        return {
            'num_wires': num_wires,
            'num_time_ticks': num_time_ticks,
            'spectrum': jnp.array(spectrum),
            'series_rms': jnp.array(series_rms),
            'white_rms': noise_x,
            'target_rms': target_rms,
        }

    def test_correct_rms(self, noise_setup, jax_key):
        """Generated noise RMS should be within 15% of target."""
        noise = _generate_noise_for_plane(
            jax_key,
            noise_setup['num_wires'],
            noise_setup['num_time_ticks'],
            noise_setup['spectrum'],
            noise_setup['series_rms'],
            noise_setup['white_rms'])
        jax.block_until_ready(noise)

        measured_rms = float(jnp.std(noise))
        target = noise_setup['target_rms']
        np.testing.assert_allclose(measured_rms, target, rtol=0.15)

    def test_correct_shape(self, noise_setup, jax_key):
        """Output shape should be (num_wires, num_time_ticks)."""
        noise = _generate_noise_for_plane(
            jax_key,
            noise_setup['num_wires'],
            noise_setup['num_time_ticks'],
            noise_setup['spectrum'],
            noise_setup['series_rms'],
            noise_setup['white_rms'])
        assert noise.shape == (50, 500)

    def test_shaped_spectrum(self, noise_setup, jax_key):
        """Noise should have shaped spectrum: power at 100kHz >> 900kHz."""
        noise = _generate_noise_for_plane(
            jax_key,
            noise_setup['num_wires'],
            noise_setup['num_time_ticks'],
            noise_setup['spectrum'],
            noise_setup['series_rms'],
            noise_setup['white_rms'])
        jax.block_until_ready(noise)

        spectrum = np.abs(np.fft.rfft(np.array(noise[0])))
        freqs = np.fft.rfftfreq(noise_setup['num_time_ticks'], d=0.5e-6)

        mask_100k = (freqs > 50e3) & (freqs < 150e3)
        mask_900k = (freqs > 850e3) & (freqs < 950e3)

        if np.any(mask_100k) and np.any(mask_900k):
            power_100k = np.mean(spectrum[mask_100k]**2)
            power_900k = np.mean(spectrum[mask_900k]**2)
            assert power_100k > power_900k

    def test_reproducibility(self, noise_setup):
        """Same PRNG key should produce identical noise."""
        key = jax.random.PRNGKey(99)
        args = (key, noise_setup['num_wires'], noise_setup['num_time_ticks'],
                noise_setup['spectrum'], noise_setup['series_rms'],
                noise_setup['white_rms'])

        noise1 = _generate_noise_for_plane(*args)
        noise2 = _generate_noise_for_plane(*args)
        np.testing.assert_array_equal(np.array(noise1), np.array(noise2))

    def test_white_only_flat(self, noise_params, jax_key):
        """With series_rms=0, noise should be approximately flat spectrum."""
        noise_x, _, _, emp_freqs, emp_shape = noise_params
        num_wires = 50
        num_time_ticks = 500
        spectrum = jnp.array(_get_noise_spectrum_shape(num_time_ticks, emp_freqs, emp_shape))
        series_rms = jnp.zeros(num_wires)

        noise = _generate_noise_for_plane(
            jax_key, num_wires, num_time_ticks, spectrum, series_rms, noise_x)

        fft = np.abs(np.fft.rfft(np.array(noise[0])))[1:-1]
        cv = np.std(fft) / np.mean(fft)
        assert cv < 1.5, f"White-only noise should be flat, CV={cv:.2f}"


class TestStandaloneNoise:
    """Tests for standalone generate_noise / add_noise."""

    @pytest.fixture
    def sim_config(self):
        """Build a SimConfig for noise tests."""
        import os
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'config', 'cubic_wireplane_config.yaml')
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")
        from tools.geometry import generate_detector
        from tools.simulation import DetectorSimulator
        det = generate_detector(config_path)
        sim = DetectorSimulator(det, include_noise=False, include_electronics=False,
                                include_track_hits=False)
        return sim.config

    def test_generate_noise_shapes(self, sim_config):
        """generate_noise should return correct shapes for all 6 planes."""
        from tools.noise import generate_noise
        noise = generate_noise(sim_config, key=jax.random.PRNGKey(0))

        assert len(noise) == 6
        for (s, p), arr in noise.items():
            nw = sim_config.volumes[s].num_wires[p]
            nt = sim_config.num_time_steps
            assert arr.shape == (nw, nt), f"({s},{p}): expected ({nw},{nt}), got {arr.shape}"

    def test_add_noise_changes_signal(self, sim_config):
        """add_noise should modify the input signals."""
        from tools.noise import add_noise
        fake = {(s, p): jnp.zeros((sim_config.volumes[s].num_wires[p],
                                    sim_config.num_time_steps))
                for s in range(sim_config.n_volumes) for p in range(sim_config.volumes[s].n_planes)}

        noisy = add_noise(fake, sim_config, key=jax.random.PRNGKey(42))
        for key, arr in noisy.items():
            assert float(jnp.std(arr)) > 0, f"{key}: noise should be nonzero"

    def test_add_noise_incoherent_reproducible(self, sim_config):
        """Default (incoherent) add_noise is deterministic for a fixed key —
        the in-JIT incoherent kernel is unchanged by the tagged refactor."""
        from tools.noise import add_noise
        fake = {(s, p): jnp.zeros((sim_config.volumes[s].num_wires[p],
                                    sim_config.num_time_steps))
                for s in range(sim_config.n_volumes)
                for p in range(sim_config.volumes[s].n_planes)}
        a = add_noise(fake, sim_config, key=jax.random.PRNGKey(5))
        b = add_noise(fake, sim_config, key=jax.random.PRNGKey(5))
        for k in a:
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


class TestCoherentNoise:
    """Tests for the tagged coherent component (tools/coherent_noise.py)."""

    def test_add_coherent_noise_matches_per_plane(self):
        """The dict-level applier == per-plane generate_coherent_noise drawn
        from the same Generator, in insertion order."""
        from tools.coherent_noise import (add_coherent_noise,
                                           generate_coherent_noise)
        sigs = {(0, 0): np.zeros((100, 256), np.float32),
                (0, 1): np.zeros((50, 256), np.float32)}
        nwk = {(0, 0): 100, (0, 1): 50}
        cfg = dict(group_size=32, beta=0.15, rms_adc=2.5)

        out = add_coherent_noise({k: v.copy() for k, v in sigs.items()},
                                 nwk, 256, coherent_cfg=cfg,
                                 rng=np.random.default_rng(7))

        rng = np.random.default_rng(7)
        exp00 = generate_coherent_noise(100, 256, group_size=32, beta=0.15,
                                        rms_adc=2.5, rng=rng)
        exp01 = generate_coherent_noise(50, 256, group_size=32, beta=0.15,
                                        rms_adc=2.5, rng=rng)
        np.testing.assert_allclose(np.asarray(out[(0, 0)]), exp00, atol=1e-5)
        np.testing.assert_allclose(np.asarray(out[(0, 1)]), exp01, atol=1e-5)

    def test_coherent_shared_within_group_and_no_root_n(self):
        """Wires in a group are identical, and per-channel RMS ~ rms_adc
        (the shared waveform is NOT averaged down by 1/sqrt(group_size))."""
        from tools.coherent_noise import generate_coherent_noise
        gs, rms = 64, 2.5
        noise = generate_coherent_noise(256, 4096, group_size=gs, rms_adc=rms,
                                        rng=np.random.default_rng(3))
        for g0 in range(0, 256, gs):
            assert np.allclose(noise[g0:g0 + gs], noise[g0][None, :])
        per_channel_rms = noise.std(axis=1).mean()
        assert 0.75 * rms < per_channel_rms < 1.5 * rms

    def test_add_noise_coherent_tag(self, ):
        """Host add_noise(coherent=True) adds a per-group shared waveform."""
        import os
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'config', 'cubic_wireplane_config.yaml')
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")
        from tools.geometry import generate_detector
        from tools.simulation import DetectorSimulator
        from tools.noise import add_noise
        det = generate_detector(config_path)
        sim = DetectorSimulator(det, include_noise=False,
                                include_electronics=False,
                                include_track_hits=False)
        cfg = sim.config
        fake = {(s, p): jnp.zeros((cfg.volumes[s].num_wires[p],
                                    cfg.num_time_steps))
                for s in range(cfg.n_volumes)
                for p in range(cfg.volumes[s].n_planes)}
        out = add_noise(fake, cfg, key=jax.random.PRNGKey(1),
                        incoherent=False, coherent=True,
                        coherent_cfg=dict(group_size=64))
        arr = np.asarray(out[(0, 0)])
        assert float(arr.std()) > 0
        assert np.allclose(arr[0], arr[1]), "wires 0,1 share a coherent group"
