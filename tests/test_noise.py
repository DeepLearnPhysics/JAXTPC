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
