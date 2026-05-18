"""
Shared fixtures for JAXTPC test suite.
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from tools.config import VolumeDeposits, DepositData, ModifiedBoxParams


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_config: test needs the YAML config file")
    config.addinivalue_line("markers", "requires_kernels: test needs response kernel NPZ files")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def jax_key():
    """Reproducible JAX PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def recomb_params():
    """Standard ArgoNeuT recombination parameters as ModifiedBoxParams."""
    return ModifiedBoxParams(
        density=jnp.array(1.396),
        w_value=jnp.array(23.6),
        excitation_ratio=jnp.array(0.21),
        field_strength_Vcm=jnp.array(500.0),
        alpha=jnp.array(0.93),
        beta=jnp.array(0.212),
    )


@pytest.fixture
def minimal_detector_config():
    """Small detector configuration dict mimicking generate_detector() output.

    40x40x40 cm detector with 3 planes per volume, suitable for fast tests.
    Uses the new 'volumes' YAML schema.
    """
    config = {
        'volumes': [
            {
                'id': 0,
                'description': 'Test East',
                'geometry': {
                    'ranges': [[-20.0, 0.0], [-20.0, 20.0], [-20.0, 20.0]],
                    'drift_direction': -1,
                },
                'planes': [
                    {'plane_id': 0, 'type': 'first_induction', 'angle': 60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.6, 'bias_voltage': -200.0},
                    {'plane_id': 1, 'type': 'second_induction', 'angle': -60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.3, 'bias_voltage': -200.0},
                    {'plane_id': 2, 'type': 'collection', 'angle': 0.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.0, 'bias_voltage': 500.0},
                ]
            },
            {
                'id': 1,
                'description': 'Test West',
                'geometry': {
                    'ranges': [[0.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]],
                    'drift_direction': 1,
                },
                'planes': [
                    {'plane_id': 3, 'type': 'first_induction', 'angle': 60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.6, 'bias_voltage': -200.0},
                    {'plane_id': 4, 'type': 'second_induction', 'angle': -60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.3, 'bias_voltage': -200.0},
                    {'plane_id': 5, 'type': 'collection', 'angle': 0.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.0, 'bias_voltage': 500.0},
                ]
            }
        ],
        'readout': {'sampling_rate': 2.0, 'electrons_per_adc': 182},
        'simulation': {
            'drift': {
                'velocity': 1.6,
                'longitudinal_diffusion': 6.2,
                'transverse_diffusion': 16.3,
                'electron_lifetime': 10.0,
            },
            'charge_recombination': {
                'model': 'modified_box',
                'recomb_parameters': {'alpha': 0.93, 'beta': 0.212},
            },
        },
        'medium': {
            'type': 'liquid_argon',
            'properties': {'density': 1.396, 'ionization_energy': 23.6, 'excitation_ratio': 0.21},
            'temperature': 87.0,
            'pressure': 1.0,
        },
        'electric_field': {'field_strength': 500.0},
    }

    return config


@pytest.fixture
def small_deposit_data():
    """DepositData with ~100 synthetic particles spanning both volumes.

    Built via build_deposit_data with the minimal detector config.
    """
    from tools.config import create_sim_config
    from tools.loader import build_deposit_data

    rng = np.random.RandomState(123)
    n = 100

    # Positions in mm: x spans [-100, 100], y and z span [-100, 100]
    x_mm = rng.uniform(-100, 100, size=n).astype(np.float32)
    y_mm = rng.uniform(-100, 100, size=n).astype(np.float32)
    z_mm = rng.uniform(-100, 100, size=n).astype(np.float32)
    positions_mm = np.stack([x_mm, y_mm, z_mm], axis=1)

    de = rng.uniform(0.5, 5.0, size=n).astype(np.float32)
    dx = rng.uniform(0.01, 0.5, size=n).astype(np.float32)
    theta = rng.uniform(0, np.pi, size=n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    track_ids = rng.randint(0, 5, size=n).astype(np.int32)

    # Build with minimal config to get proper volume splitting
    # Use a config that covers the positions range
    config = {
        'volumes': [
            {
                'id': 0,
                'geometry': {
                    'ranges': [[-20.0, 0.0], [-20.0, 20.0], [-20.0, 20.0]],
                    'drift_direction': -1,
                },
                'planes': [
                    {'plane_id': 0, 'type': 'first_induction', 'angle': 60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.6, 'bias_voltage': -200.0},
                    {'plane_id': 1, 'type': 'second_induction', 'angle': -60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.3, 'bias_voltage': -200.0},
                    {'plane_id': 2, 'type': 'collection', 'angle': 0.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.0, 'bias_voltage': 500.0},
                ]
            },
            {
                'id': 1,
                'geometry': {
                    'ranges': [[0.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]],
                    'drift_direction': 1,
                },
                'planes': [
                    {'plane_id': 3, 'type': 'first_induction', 'angle': 60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.6, 'bias_voltage': -200.0},
                    {'plane_id': 4, 'type': 'second_induction', 'angle': -60.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.3, 'bias_voltage': -200.0},
                    {'plane_id': 5, 'type': 'collection', 'angle': 0.0,
                     'wire_spacing': 0.3, 'distance_from_anode': 0.0, 'bias_voltage': 500.0},
                ]
            }
        ],
        'readout': {'sampling_rate': 2.0, 'electrons_per_adc': 182},
        'simulation': {
            'drift': {
                'velocity': 1.6,
                'longitudinal_diffusion': 6.2,
                'transverse_diffusion': 16.3,
                'electron_lifetime': 10.0,
            },
            'charge_recombination': {
                'model': 'modified_box',
                'recomb_parameters': {'alpha': 0.93, 'beta': 0.212},
            },
        },
        'medium': {
            'type': 'liquid_argon',
            'properties': {'density': 1.396, 'ionization_energy': 23.6, 'excitation_ratio': 0.21},
            'temperature': 87.0,
            'pressure': 1.0,
        },
        'electric_field': {'field_strength': 500.0},
    }

    sim_config = create_sim_config(config, total_pad=200, include_track_hits=False)

    return build_deposit_data(
        positions_mm, de, dx, sim_config,
        theta=theta, phi=phi, track_ids=track_ids,
    )
