"""
JAXTPC Refactored Tools Package

This package contains the refactored simulation tools for LArTPC detector simulation.
See REFACTORING_PLAN.md for details on the changes from the original tools package.

Main entry points:
- DetectorSimulator: Main simulation class
- run_simulation: Convenience function for single-event processing

Configuration classes:
- DepositData: Input data container
- DriftParams: Drift physics parameters
- TimeParams: Time discretization parameters
- PlaneGeometry: Wire plane geometry
- DiffusionParams: Diffusion parameters
- TrackHitsConfig: Track labeling configuration
"""

from tools.config import (
    DepositData,
    DriftParams,
    TimeParams,
    PlaneGeometry,
    DiffusionParams,
    TrackHitsConfig,
    create_diffusion_params,
    create_drift_params,
    create_time_params,
    create_plane_geometry,
    create_track_hits_config,
)

from tools.simulation import (
    DetectorSimulator,
    run_simulation,
)

from tools.wires import (
    sparse_buckets_to_dense,
    accumulate_response_signals_sparse_bucketed,
)

__all__ = [
    # Config classes
    'DepositData',
    'DriftParams',
    'TimeParams',
    'PlaneGeometry',
    'DiffusionParams',
    'TrackHitsConfig',
    # Factory functions
    'create_diffusion_params',
    'create_drift_params',
    'create_time_params',
    'create_plane_geometry',
    'create_track_hits_config',
    # Main simulation
    'DetectorSimulator',
    'run_simulation',
    # Sparse utilities
    'sparse_buckets_to_dense',
    'accumulate_response_signals_sparse_bucketed',
]
