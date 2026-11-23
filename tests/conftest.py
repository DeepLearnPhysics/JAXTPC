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


# ---------------------------------------------------------------------------
# Compact wire detector + synthetic tracks (pipeline tests)
# ---------------------------------------------------------------------------

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'config', 'cubic_wireplane_config.yaml')


@pytest.fixture(scope="session")
def compact_detector():
    """Factory: cubic_wireplane with the full 216 cm drift but a 60 x 60 cm readout.

    ``pitches`` overrides the U, V, Y wire spacing (cm). The drift length is kept so
    that diffusion and attenuation are as large as in the real detector.
    """
    from tools.geometry import generate_detector

    def make(pitches=(0.3, 0.3, 0.3)):
        det = generate_detector(_CONFIG)
        for vol in det['volumes']:
            ranges = vol['geometry']['ranges']
            ranges[1], ranges[2] = [-30.0, 30.0], [-30.0, 30.0]
            for plane, pitch in zip(vol['planes'], pitches):
                plane['wire_spacing'] = pitch
        return det
    return make


@pytest.fixture(scope="session")
def synthetic_tracks():
    """Factory: deterministic deposits of three tracks, each with charge in both volumes.

    Tracks 1 and 2 span 5-210 cm of |x| with one deposit per group; track 3 is two short
    stubs with deposits 3 mm apart, i.e. groups of five. dE is non-uniform, angles are
    the true ones. Leaves padding entries in every volume for total_pad >= 64.
    Coordinates are irregular on purpose: a deposit arriving exactly on a tick edge sits
    on a kink of the sub-tick interpolation, where finite differences are one-sided.
    """
    from tools.loader import build_deposit_data

    def make(sim_config):
        ends = [((-2103.3, -201.3, -151.9), (2097.1, 148.3, 221.7)),
                ((-51.3, 152.1, -218.9), (-2047.9, -251.7, 101.3)),
                ((2001.9, -218.3, 181.7), (83.3, 191.9, -58.7)),
                ((-903.1, 40.3, 60.7), (-890.2, 62.9, 69.4)),
                ((1203.3, -70.9, -30.1), (1219.7, -52.8, -18.3))]
        counts = [40, 20, 20, 10, 10]
        tracks = [1, 2, 2, 3, 3]
        pos, theta, phi, tid = [], [], [], []
        for (a, b), n, t in zip(ends, counts, tracks):
            a, b = np.array(a), np.array(b)
            d = (b - a) / np.linalg.norm(b - a)
            pos.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
            theta.append(np.full(n, np.arccos(d[2])))
            phi.append(np.full(n, np.arctan2(d[1], d[0])))
            tid.append(np.full(n, t, dtype=np.int32))
        n = sum(counts)
        return build_deposit_data(
            np.concatenate(pos).astype(np.float32),
            np.linspace(0.5, 1.5, n, dtype=np.float32),
            np.full(n, 3.0, dtype=np.float32), sim_config,
            theta=np.concatenate(theta).astype(np.float32),
            phi=np.concatenate(phi).astype(np.float32),
            track_ids=np.concatenate(tid))
    return make


@pytest.fixture
def count_compiles():
    """Factory: number of XLA compilations triggered while running ``fn``."""
    def count(fn):
        events = []

        def listen(event, duration, **kwargs):
            if event.endswith('backend_compile_duration'):
                events.append(event)
        jax.monitoring.register_event_duration_secs_listener(listen)
        fn()
        # Listeners cannot be removed through the public API in every supported
        # JAX; copy the count so later events are ignored.
        return len(events)
    return count
