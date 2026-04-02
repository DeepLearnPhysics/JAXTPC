"""
JAXTPC Tools Package

Simulation tools for LArTPC detector simulation.

Main entry points:
- generate_detector: Parse YAML detector configuration
- DetectorSimulator: Main simulation class
  - process_event(): Production path (fori_loop batching, post-processing)
  - forward(): Differentiable path (gradients through SimParams)
- load_particle_step_data: Load HDF5 particle step data
- generate_noise / add_noise: Standalone noise generation (accepts SimConfig)
"""

from tools.config import (
    # Core types
    DepositData,
    VolumeDeposits,
    SimParams,
    SimConfig,
    ModifiedBoxParams,
    EMBParams,
    SCEOutputs,
    VolumeGeometry,
    VolumeIntermediates,
    PlaneIntermediates,
    DiffusionConfig,
    TrackHitsConfig,
    DigitizationConfig,
    ResponseKernel,
    # Helpers
    get_volume_deposits,
    # Factories
    create_sim_params,
    create_sim_config,
    create_deposit_data,
    pad_deposit_data,
    create_track_hits_config,
    create_digitization_config,
)

from tools.geometry import generate_detector
from tools.loader import load_particle_step_data, build_deposit_data, load_event
from tools.simulation import DetectorSimulator
from tools.recombination import RECOMB_MODELS, compute_quanta, XI_FN
from tools.particle_generator import (
    generate_muon_segments, generate_muon_segments_trig,
    load_dedx_table_jax, mask_outside_volume,
)

from tools.wires import sparse_buckets_to_dense

from tools.noise import (
    add_noise,
    generate_noise,
    generate_noise_bucketed,
)

__all__ = [
    # Core types
    'DepositData', 'VolumeDeposits', 'SimParams', 'SimConfig',
    'ModifiedBoxParams', 'EMBParams',
    'SCEOutputs', 'VolumeGeometry', 'VolumeIntermediates', 'PlaneIntermediates',
    'DiffusionConfig', 'TrackHitsConfig', 'DigitizationConfig', 'ResponseKernel',
    # Recombination
    'RECOMB_MODELS', 'compute_quanta', 'XI_FN',
    # Entry points
    'generate_detector', 'load_particle_step_data', 'build_deposit_data',
    'load_event', 'get_volume_deposits', 'DetectorSimulator',
    # Factories
    'create_sim_params', 'create_sim_config', 'create_deposit_data',
    'pad_deposit_data',
    'create_track_hits_config', 'create_digitization_config',
    # Particle generation (differentiable)
    'generate_muon_segments', 'generate_muon_segments_trig',
    'load_dedx_table_jax', 'mask_outside_volume',
    # Sparse utilities
    'sparse_buckets_to_dense',
    # Noise post-processing
    'add_noise', 'generate_noise', 'generate_noise_bucketed',
]
