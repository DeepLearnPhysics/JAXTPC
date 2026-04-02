"""
Particle step data loading and preprocessing for LArTPC simulation.

Handles HDF5 I/O, group ID assignment for segment correspondence,
and side-splitting with fixed-size padding for JIT compilation.
"""

import h5py
import jax.numpy as jnp
import numpy as np

from tools.config import DepositData, VolumeDeposits


class ParticleStepExtractor:
    """
    Simplified extractor for particle steps from HDF5 files.
    
    This class extracts particle step data from an HDF5 file and converts it to JAX arrays.
    It handles the standard structure of particle physics HDF5 files with steps, particles,
    and associations between them.
    
    Attributes
    ----------
    file_path : str
        Path to the HDF5 file.
    file : h5py.File or None
        The opened HDF5 file or None if not opened.
    verbose : bool
        Whether to print verbose information.
    pstep_path : str or None
        The actual path to the particle step dataset.
    particle_path : str or None
        The actual path to the particle dataset.
    association_path : str or None
        The actual path to the association dataset.
    """

    def __init__(self, file_path: str, verbose: bool = False):
        """
        Initialize the extractor with the path to an HDF5 file.
        
        Parameters
        ----------
        file_path : str
            Path to the HDF5 file.
        verbose : bool, optional
            Whether to print verbose information, by default False.
        """
        self.file_path = file_path
        self.file = None
        self.verbose = verbose

        # Common paths in particle physics HDF5 files
        self.pstep_paths = ['pstep/lar_vol']
        self.particle_paths = ['particle/geant4']
        self.association_paths = [
            'ass/particle_pstep_lar_vol'
        ]

        # The actual paths found in this file
        self.pstep_path = None
        self.particle_path = None
        self.association_path = None

        # Open the file and find paths
        self.open_file()
        self._find_dataset_paths()

        if verbose:
            print(f"Loaded file: {file_path}")
            print(f"Step path: {self.pstep_path}")
            print(f"Particle path: {self.particle_path}")
            print(f"Association path: {self.association_path}")

    def _find_dataset_paths(self):
        """
        Find the actual dataset paths in the file.
        
        Checks for the existence of common paths and sets the actual paths
        found in the file.
        """
        # Find step path
        for path in self.pstep_paths:
            if path in self.file:
                self.pstep_path = path
                break

        # Find particle path
        for path in self.particle_paths:
            if path in self.file:
                self.particle_path = path
                break

        # Find association path
        for path in self.association_paths:
            if path in self.file:
                self.association_path = path
                break

    def open_file(self):
        """
        Open the HDF5 file.
        
        Opens the file in read mode and stores the file object.
        """
        self.file = h5py.File(self.file_path, 'r')

    def close(self):
        """
        Close the HDF5 file.
        
        Closes the file if it is open and sets the file object to None.
        """
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        """
        Context manager entry method.
        
        Returns
        -------
        ParticleStepExtractor
            The extractor object.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method.
        
        Closes the file when exiting the context.
        
        Parameters
        ----------
        exc_type : type
            Exception type if an exception was raised.
        exc_val : Exception
            Exception value if an exception was raised.
        exc_tb : traceback
            Exception traceback if an exception was raised.
        """
        self.close()

    def _get_numeric_fields(self, dataset, event_idx=0):
        """
        Extract numeric fields from a dataset for a specific event.
        Skips string fields that JAX doesn't support.
        
        Parameters
        ----------
        dataset : str
            Path to the dataset in the HDF5 file.
        event_idx : int, optional
            Index of the event to extract, by default 0.
            
        Returns
        -------
        dict
            Dictionary mapping field names to numpy arrays.
        """
        if dataset not in self.file:
            if self.verbose:
                print(f"Dataset {dataset} not found")
            return {}

        try:
            data = self.file[dataset][event_idx]
        except Exception as e:
            if self.verbose:
                print(f"Error accessing {dataset}[{event_idx}]: {e}")
            return {}

        result = {}

        # Check if data has fields (structured array)
        if hasattr(data, 'dtype') and data.dtype.names:
            for field in data.dtype.names:
                try:
                    field_data = data[field]

                    # Skip string fields - JAX doesn't support these
                    if isinstance(field_data, (bytes, np.bytes_)) or (
                            isinstance(field_data, np.ndarray) and field_data.dtype.kind in ('S', 'U', 'O')):
                        if self.verbose:
                            print(f"Skipping string field: {field}")
                        continue

                    result[field] = np.asarray(field_data)
                except Exception as e:
                    if self.verbose:
                        print(f"Error extracting field {field}: {e}")

        return result

    def get_step_to_particle_mapping(self, event_idx=0):
        """
        Create a mapping from each step to its parent particle.
        
        Parameters
        ----------
        event_idx : int, optional
            Index of the event to extract, by default 0.
            
        Returns
        -------
        np.ndarray or None
            Array where index[i] gives the particle index for step i,
            or None if the mapping could not be created.
        """
        if not self.association_path or self.association_path not in self.file:
            if self.verbose:
                print(f"Association dataset {self.association_path} not found")
            return None

        try:
            mapping_data = self.file[self.association_path][event_idx]

            # Check how the mapping is stored
            if hasattr(mapping_data, 'dtype') and mapping_data.dtype.names and 'start' in mapping_data.dtype.names:
                # Format with start/end indices
                starts = mapping_data['start']
                ends = mapping_data['end']

                # Create a mapping from step index to particle index
                num_steps = np.max(ends) if len(ends) > 0 else 0
                step_to_particle = np.zeros(num_steps, dtype=np.int32)

                for i in range(len(starts)):
                    start_idx = starts[i]
                    end_idx = ends[i]
                    step_to_particle[start_idx:end_idx] = i

                return step_to_particle
            else:
                # Other format - this would need to be adapted based on the specific file structure
                if self.verbose:
                    print("Unsupported association format")
                return None

        except Exception as e:
            if self.verbose:
                print(f"Error getting step-to-particle mapping: {e}")
            return None

    def extract_step_arrays(self, event_idx=0):
        """
        Extract step data as JAX arrays.
        For each property, returns an array of shape (N, ...) where N is the number of steps.
        
        Parameters
        ----------
        event_idx : int, optional
            Index of the event to extract, by default 0.
            
        Returns
        -------
        dict
            Dictionary mapping property names to JAX arrays.
        """
        # Get step data
        step_data = self._get_numeric_fields(self.pstep_path, event_idx)
        if not step_data:
            if self.verbose:
                print("No step data found")
            return {}

        # Get particle data
        particle_data = self._get_numeric_fields(self.particle_path, event_idx)
        if not particle_data:
            if self.verbose:
                print("No particle data found")
            return step_data  # Return just step data if no particle data

        # Get mapping from steps to particles
        step_to_particle = self.get_step_to_particle_mapping(event_idx)
        if step_to_particle is None:
            if self.verbose:
                print("No step-to-particle mapping found")
            return step_data  # Return just step data if no mapping

        # Initialize result with step properties
        result = dict(step_data)

        # For each particle property, add it to each step
        for key, value in particle_data.items():
            # Skip if already exists in step data
            if key in result:
                continue

            try:
                # Get the property for each step's particle
                result[f"particle_{key}"] = value[step_to_particle]
            except Exception as e:
                if self.verbose:
                    print(f"Error mapping particle property {key} to steps: {e}")

        # Add some convenience properties
        if 'x' in result and 'y' in result and 'z' in result:
            result['position'] = np.stack([result['x'], result['y'], result['z']], axis=1)

        return result


def load_particle_step_data(file_path, event_idx=0, verbose=False):
    """
    Load particle step data from an HDF5 file and return as DepositData.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    event_idx : int, optional
        Index of the event to extract, by default 0.
    verbose : bool, optional
        Whether to print verbose information, by default False.

    Returns
    -------
    deposit_data : DepositData
        Namedtuple with positions_mm, de, dx, valid_mask, theta, phi,
        track_ids, group_ids.
    group_to_track : np.ndarray
        Lookup array: group_to_track[group_id] = track_id.
    """
    with ParticleStepExtractor(file_path, verbose=verbose) as extractor:
        step_data = extractor.extract_step_arrays(event_idx)

    # Get positions and determine array size (numpy — no XLA compilation)
    positions_mm = np.asarray(
        step_data.get('position', np.empty((0, 3))), dtype=np.float32
    )
    n = positions_mm.shape[0]

    track_ids = np.asarray(step_data.get('track_id', np.ones((n,))), dtype=np.int32)

    # GEANT4 stores time in nanoseconds; convert to microseconds
    t_ns = np.asarray(step_data.get('t', np.zeros((n,))), dtype=np.float32)
    t0_us = t_ns / 1000.0

    return {
        'positions_mm': positions_mm,
        'de': np.asarray(step_data.get('de', np.zeros((n,))), dtype=np.float32),
        'dx': np.asarray(step_data.get('dx', np.zeros((n,))), dtype=np.float32),
        'theta': np.asarray(step_data.get('theta', np.zeros((n,))), dtype=np.float32),
        'phi': np.asarray(step_data.get('phi', np.zeros((n,))), dtype=np.float32),
        'track_ids': track_ids,
        't0_us': t0_us,
    }


def load_event(file_path, sim_config, event_idx=0, verbose=False,
               group_size=5, gap_threshold_mm=5.0):
    """Load an event from HDF5 and build simulation-ready DepositData.

    Convenience function combining load_particle_step_data + build_deposit_data.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    sim_config : SimConfig
        Simulation config with volume definitions and total_pad.
    event_idx : int
        Event index in HDF5 file. Default 0.
    verbose : bool
        Print loading info. Default False.
    group_size : int
        Consecutive deposits per group. Default 5.
    gap_threshold_mm : float
        Spatial gap threshold for group splitting. Default 5.0.

    Returns
    -------
    DepositData
        Multi-volume, padded, grouped, ready for process_event.
    """
    raw = load_particle_step_data(file_path, event_idx=event_idx, verbose=verbose)
    return build_deposit_data(
        raw['positions_mm'], raw['de'], raw['dx'], sim_config,
        theta=raw['theta'], phi=raw['phi'], track_ids=raw['track_ids'],
        t0_us=raw['t0_us'],
        group_size=group_size, gap_threshold_mm=gap_threshold_mm)


def main():
    """
    Example usage of the particle step extractor.

    This function demonstrates how to use the particle step extractor
    from the command line with various options.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Extract particle step data from HDF5 files')
    parser.add_argument('file_path', help='Path to the HDF5 file')
    parser.add_argument('--event', '-e', type=int, default=0, help='Event index (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose information')

    args = parser.parse_args()

    # Extract step data as DepositData
    deposit_data, group_to_track = load_particle_step_data(args.file_path, args.event, args.verbose)

    # Print summary
    n_segments = deposit_data.positions_mm.shape[0]
    n_tracks = len(np.unique(deposit_data.track_ids))
    total_de = float(np.sum(deposit_data.de))

    print(f"\nLoaded DepositData:")
    print(f"  Segments: {n_segments:,}")
    print(f"  Unique tracks: {n_tracks}")
    print(f"  Total dE: {total_de:.2f} MeV")
    print(f"\nFields:")
    for field in deposit_data._fields:
        arr = getattr(deposit_data, field)
        print(f"  {field}: shape={arr.shape}, dtype={arr.dtype}")


# =============================================================================
# GROUP ID ASSIGNMENT (numpy host-side, before split)
# =============================================================================

def compute_group_ids(positions_mm, track_ids, valid_mask,
                      group_size=5, gap_threshold_mm=5.0):
    """Assign group IDs for segment correspondence: N consecutive deposits per track.

    Groups are split on large spatial gaps (neutrons/gammas) to avoid
    grouping physically distant deposits.

    Parameters
    ----------
    positions_mm : np.ndarray, shape (N, 3)
    track_ids : np.ndarray, shape (N,), int32
    valid_mask : np.ndarray, shape (N,), bool
    group_size : int
        Consecutive deposits per group. Default 5.
    gap_threshold_mm : float
        Start new group if gap exceeds this. Default 5.0.

    Returns
    -------
    group_ids : np.ndarray, shape (N,), int32
        Group ID per deposit (0 = invalid/padding).
    group_to_track : np.ndarray, shape (n_groups,), int32
        Lookup: group_to_track[group_id] = track_id.
    n_groups : int
        Total number of groups (including invalid group 0).
    """
    n = len(track_ids)
    group_ids = np.zeros(n, dtype=np.int32)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return group_ids, np.array([0], dtype=np.int32), 1

    v_tids = track_ids[valid_idx]
    v_pos = positions_mm[valid_idx]

    # Stable sort by track_id preserves trajectory order within each track
    sort_order = np.argsort(v_tids, kind='stable')
    sorted_idx = valid_idx[sort_order]
    sorted_tids = v_tids[sort_order]
    sorted_pos = v_pos[sort_order]
    n_valid = len(sorted_idx)

    # Track boundaries
    track_change = np.zeros(n_valid, dtype=bool)
    track_change[1:] = sorted_tids[1:] != sorted_tids[:-1]

    # Spatial gap boundaries (within same track)
    gaps = np.zeros(n_valid)
    gaps[1:] = np.linalg.norm(sorted_pos[1:] - sorted_pos[:-1], axis=1)
    gap_break = gaps > gap_threshold_mm

    # Contiguous segment starts: track change or spatial gap
    # (Volume boundary splitting is handled by split_by_volume — each volume's
    #  deposits are already separated before compute_group_ids is called.)
    seg_start = track_change | gap_break
    seg_start[0] = True

    # Within-segment position via forward-filled segment start indices
    seg_start_positions = np.where(seg_start, np.arange(n_valid), 0)
    seg_start_positions = np.maximum.accumulate(seg_start_positions)
    within_seg = np.arange(n_valid) - seg_start_positions

    # Group boundaries: segment start or every N deposits within a segment
    group_start = seg_start.copy()
    group_start |= (within_seg % group_size == 0) & (within_seg > 0)

    # Consecutive group IDs (1-based; 0 reserved for invalid)
    group_labels = np.cumsum(group_start)

    # Write back to original deposit positions
    group_ids[sorted_idx] = group_labels

    # Build group_to_track lookup
    n_groups = int(group_labels.max()) + 1
    group_to_track = np.zeros(n_groups, dtype=np.int32)
    group_to_track[group_labels] = sorted_tids

    return group_ids, group_to_track, n_groups


def build_deposit_data(positions_mm, de, dx, sim_config,
                       theta=None, phi=None, track_ids=None,
                       group_ids=None, t0_us=None,
                       group_size=5, gap_threshold_mm=5.0):
    """Build simulation-ready DepositData from flat deposit arrays.

    Assigns deposits to volumes by position, computes segment groups
    per-volume (or uses pre-computed group_ids), pads each volume to
    total_pad, and converts to JAX arrays.

    Parameters
    ----------
    positions_mm : np.ndarray, shape (N, 3)
        Deposit positions in mm.
    de : np.ndarray, shape (N,)
        Energy deposits in MeV.
    dx : np.ndarray or float, shape (N,) or scalar
        Step lengths in mm.
    sim_config : SimConfig
        Simulation config with volume definitions and total_pad.
    theta : np.ndarray, shape (N,), optional
        Polar angles. Default zeros.
    phi : np.ndarray, shape (N,), optional
        Azimuthal angles. Default zeros.
    track_ids : np.ndarray, shape (N,), optional
        Particle track IDs. Default zeros (no track info).
    group_ids : np.ndarray, shape (N,), optional
        Pre-computed group IDs. If provided, used directly (no auto-grouping).
        Must not span volume boundaries.
    group_size : int
        Consecutive deposits per group for auto-grouping. Default 5.
    gap_threshold_mm : float
        Spatial gap threshold for group splitting. Default 5.0.

    Returns
    -------
    DepositData
        Multi-volume deposit data, padded, ready for process_event.
    """
    positions_mm = np.asarray(positions_mm, dtype=np.float32)
    de = np.asarray(de, dtype=np.float32)
    N = positions_mm.shape[0]

    if np.ndim(dx) == 0:
        dx = np.full(N, float(dx), dtype=np.float32)
    else:
        dx = np.asarray(dx, dtype=np.float32)

    theta = np.asarray(theta, dtype=np.float32) if theta is not None else np.zeros(N, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32) if phi is not None else np.zeros(N, dtype=np.float32)
    track_ids = np.asarray(track_ids, dtype=np.int32) if track_ids is not None else np.zeros(N, dtype=np.int32)
    t0_us = np.asarray(t0_us, dtype=np.float32) if t0_us is not None else np.zeros(N, dtype=np.float32)

    has_precomputed_groups = group_ids is not None
    if has_precomputed_groups:
        group_ids = np.asarray(group_ids, dtype=np.int32)

    x_cm = positions_mm[:, 0] / 10.0
    valid_mask = np.ones(N, dtype=bool)
    total_pad = sim_config.total_pad

    all_fields = {
        'positions_mm': positions_mm,
        'de': de,
        'dx': dx,
        'theta': theta,
        'phi': phi,
        'track_ids': track_ids,
        't0_us': t0_us,
    }

    vol_arrays = {field: [] for field in all_fields}
    vol_arrays['group_ids'] = []
    vol_n_actuals = []
    vol_group_to_track = []
    vol_original_indices = []

    for vol_idx in range(sim_config.n_volumes):
        vol = sim_config.volumes[vol_idx]
        x_min, x_max = vol.ranges_cm[0]
        vol_mask = valid_mask & (x_cm >= x_min) & (x_cm < x_max)

        n_actual = int(np.sum(vol_mask))
        if n_actual > total_pad:
            print(f"WARNING: volume {vol_idx} has {n_actual:,} deposits "
                  f"> total_pad ({total_pad:,}). Truncating!")
        n_use = min(n_actual, total_pad)

        # Track which original deposits go into this volume
        vol_original_indices.append(np.where(vol_mask)[0][:n_use])

        # Extract this volume's deposits
        for field, arr in all_fields.items():
            vol_arrays[field].append(arr[vol_mask][:n_use])

        # Groups: use pre-computed or compute per-volume
        if has_precomputed_groups:
            v_gids = group_ids[vol_mask][:n_use]
            # Build group_to_track from pre-computed groups
            v_tids = track_ids[vol_mask][:n_use]
            max_gid = int(v_gids.max()) if n_use > 0 else 0
            g2t = np.zeros(max_gid + 1, dtype=np.int32)
            if n_use > 0:
                g2t[v_gids] = v_tids
        else:
            v_pos = positions_mm[vol_mask][:n_use]
            v_tids = track_ids[vol_mask][:n_use]
            v_valid = np.ones(n_use, dtype=bool)
            v_gids, g2t, _ = compute_group_ids(
                v_pos, v_tids, v_valid,
                group_size=group_size, gap_threshold_mm=gap_threshold_mm)

        vol_arrays['group_ids'].append(v_gids)

        vol_n_actuals.append(n_use)
        vol_group_to_track.append(g2t)

    # Pad and convert to JAX
    return _build_padded_deposit_data(
        vol_arrays, vol_n_actuals, vol_group_to_track, vol_original_indices, total_pad)


def _build_padded_deposit_data(vol_arrays, vol_n_actuals, vol_group_to_track,
                                vol_original_indices, total_pad):
    """Pad per-volume arrays to total_pad and construct DepositData.

    Builds VolumeDeposits for each volume, wraps in DepositData.
    """

    def _pad(arr, n_use, pad_val=0):
        pad_size = total_pad - n_use
        if pad_size <= 0:
            return jnp.asarray(arr[:total_pad])
        if arr.ndim == 2:
            return jnp.asarray(np.pad(arr, ((0, pad_size), (0, 0)),
                                       constant_values=pad_val))
        return jnp.asarray(np.pad(arr, (0, pad_size), constant_values=pad_val))

    n_volumes = len(vol_n_actuals)
    volumes = tuple(
        VolumeDeposits(
            positions_mm=_pad(vol_arrays['positions_mm'][v], vol_n_actuals[v]),
            de=_pad(vol_arrays['de'][v], vol_n_actuals[v]),
            dx=_pad(vol_arrays['dx'][v], vol_n_actuals[v], pad_val=1.0),
            theta=_pad(vol_arrays['theta'][v], vol_n_actuals[v]),
            phi=_pad(vol_arrays['phi'][v], vol_n_actuals[v]),
            track_ids=_pad(vol_arrays['track_ids'][v], vol_n_actuals[v]),
            group_ids=_pad(vol_arrays['group_ids'][v], vol_n_actuals[v]),
            t0_us=_pad(vol_arrays['t0_us'][v], vol_n_actuals[v]),
            charge=jnp.zeros(total_pad, dtype=jnp.float32),
            photons=jnp.zeros(total_pad, dtype=jnp.float32),
            qs_fractions=jnp.zeros(total_pad, dtype=jnp.float32),
            n_actual=vol_n_actuals[v],
        )
        for v in range(n_volumes)
    )

    return DepositData(
        volumes=volumes,
        group_to_track=tuple(vol_group_to_track),
        original_indices=tuple(vol_original_indices),
    )


if __name__ == "__main__":
    main()