"""
Tests for detector geometry parsing and parameter calculation.

File under test: tools/geometry.py
"""

import os
import numpy as np
import pytest
from tools.geometry import (
    get_drift_velocity,
    get_single_plane_wire_params,
    get_plane_geometry_for_volume,
    calculate_max_diffusion_sigmas,
    generate_detector,
)
from tools.config import create_sim_config


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'config', 'cubic_wireplane_config.yaml')


class TestDriftVelocity:
    """Tests for get_drift_velocity."""

    def test_velocity_unit_conversion(self, minimal_detector_config):
        """1.6 mm/us should convert to 0.16 cm/us."""
        velocity = get_drift_velocity(minimal_detector_config)
        np.testing.assert_allclose(velocity, 0.16, rtol=1e-6)


class TestSimConfig:
    """Tests for create_sim_config with new volumes schema."""

    def test_time_parameter_calculation(self, minimal_detector_config):
        """Verify num_time_steps analytically."""
        cfg = create_sim_config(minimal_detector_config, include_track_hits=False)

        # max_drift = 20cm, v=0.16 → max_drift_time=125us
        # sampling_rate=2MHz → step=0.5us
        np.testing.assert_allclose(cfg.time_step_us, 0.5, rtol=1e-6)
        # num = ceil(125/0.5)+1 = 251
        assert cfg.num_time_steps == 251

    def test_volume_geometry(self, minimal_detector_config):
        """Verify volume geometry is correct."""
        cfg = create_sim_config(minimal_detector_config, include_track_hits=False)
        assert cfg.n_volumes == 2
        assert cfg.volumes[0].max_drift_cm == 20.0
        assert cfg.volumes[0].drift_direction == -1
        assert cfg.volumes[1].drift_direction == 1
        assert cfg.volumes[0].x_anode_cm == -20.0
        assert cfg.volumes[1].x_anode_cm == 20.0


class TestWireParams:
    """Tests for wire parameter calculations."""

    def test_y_plane_wire_count(self, minimal_detector_config):
        """Y-plane (angle=0): should have expected number of wires for 40cm detector."""
        vol_cfg = minimal_detector_config['volumes'][0]
        ranges = vol_cfg['geometry']['ranges']
        dims_cm = {
            'y': ranges[1][1] - ranges[1][0],
            'z': ranges[2][1] - ranges[2][0],
        }
        plane_config = vol_cfg['planes'][2]  # Y-plane

        angle, spacing, offset, n_wires, max_idx = get_single_plane_wire_params(
            plane_config, dims_cm)

        np.testing.assert_allclose(float(angle), 0.0, atol=1e-6)
        np.testing.assert_allclose(float(spacing), 0.3, rtol=1e-6)
        assert n_wires > 100, f"Expected >100 wires, got {n_wires}"
        assert n_wires < 200, f"Expected <200 wires, got {n_wires}"

    def test_uv_plane_symmetry(self, minimal_detector_config):
        """U and V planes should have the same number of wires in symmetric geometry."""
        cfg = create_sim_config(minimal_detector_config, include_track_hits=False)
        vol = cfg.volumes[0]
        assert vol.num_wires[0] == vol.num_wires[1], \
            f"U={vol.num_wires[0]} != V={vol.num_wires[1]}"

    def test_plane_geometry(self, minimal_detector_config):
        """Verify plane distances and furthest plane."""
        vol_cfg = minimal_detector_config['volumes'][0]
        distances, furthest = get_plane_geometry_for_volume(vol_cfg['planes'])
        assert furthest == 0  # plane 0 has distance_from_anode=0.6 (furthest)
        np.testing.assert_allclose(distances[0], 0.6)
        np.testing.assert_allclose(distances[2], 0.0)


class TestDiffusionSigmas:
    """Tests for calculate_max_diffusion_sigmas."""

    def test_sigma_calculation(self):
        """Verify sigma calculation analytically."""
        max_drift = 200.0
        velocity = 0.16
        D_trans = 16.3e-6
        D_long = 6.2e-6
        wire_spacing = 0.3
        time_spacing = 0.5

        max_drift_time = max_drift / velocity  # 1250 us
        expected_sigma_trans_cm = np.sqrt(2.0 * D_trans * max_drift_time)
        expected_D_long_temporal = D_long / (velocity ** 2)
        expected_sigma_long_us = np.sqrt(2.0 * expected_D_long_temporal * max_drift_time)

        sigma_t_cm, sigma_l_us, sigma_t_ul, sigma_l_ul = calculate_max_diffusion_sigmas(
            max_drift, velocity, D_trans, D_long, wire_spacing, time_spacing)

        np.testing.assert_allclose(sigma_t_cm, expected_sigma_trans_cm, rtol=1e-4)
        np.testing.assert_allclose(sigma_l_us, expected_sigma_long_us, rtol=1e-4)
        np.testing.assert_allclose(sigma_t_ul, expected_sigma_trans_cm / wire_spacing, rtol=1e-4)
        np.testing.assert_allclose(sigma_l_ul, expected_sigma_long_us / time_spacing, rtol=1e-4)


class TestGenerateDetector:
    """Tests for generate_detector."""

    def test_missing_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_detector(str(tmp_path / "nonexistent.yaml"))

    def test_incomplete_yaml_raises(self, tmp_path):
        yaml_file = tmp_path / "incomplete.yaml"
        yaml_file.write_text("simulation:\n  drift:\n    velocity: 1.6\n")
        with pytest.raises(KeyError):
            generate_detector(str(yaml_file))

    @pytest.mark.requires_config
    def test_full_config_loads(self):
        if not os.path.exists(CONFIG_PATH):
            pytest.skip("Config file not found")
        detector = generate_detector(CONFIG_PATH)
        assert 'volumes' in detector
        assert len(detector['volumes']) == 2
