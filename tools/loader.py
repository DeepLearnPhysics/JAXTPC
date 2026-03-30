"""
Particle step data loading and preprocessing for LArTPC simulation.

Handles HDF5 I/O, group ID assignment for segment correspondence,
and side-splitting with fixed-size padding for JIT compilation.
"""

import h5py
import jax.numpy as jnp
import numpy as np

from tools.config import DepositData


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
    valid_mask = np.ones(n, dtype=bool)

    # Compute group IDs for segment correspondence
    group_ids, group_to_track, n_groups = compute_group_ids(
        positions_mm, track_ids, valid_mask)

    deposit_data = DepositData(
        positions_mm=positions_mm,
        de=np.asarray(step_data.get('de', np.zeros((n,))), dtype=np.float32),
        dx=np.asarray(step_data.get('dx', np.zeros((n,))), dtype=np.float32),
        valid_mask=valid_mask,
        theta=np.asarray(step_data.get('theta', np.zeros((n,))), dtype=np.float32),
        phi=np.asarray(step_data.get('phi', np.zeros((n,))), dtype=np.float32),
        track_ids=track_ids,
        group_ids=group_ids,
    )
    return deposit_data, group_to_track


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

    # Side change boundary (x < 0 → east, x >= 0 → west)
    side = (sorted_pos[:, 0] >= 0).astype(np.int8)
    side_change = np.zeros(n_valid, dtype=bool)
    side_change[1:] = side[1:] != side[:-1]

    # Contiguous segment starts: track change, spatial gap, or side change
    seg_start = track_change | gap_break | side_change
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


# =============================================================================
# SPLIT & PAD (numpy host-side, no JIT)
# =============================================================================

def _extract_and_pad(deposit_data, mask, n_valid, pad_size):
    """Extract masked deposits and pad to fixed size.

    All operations in numpy. Returns DepositData with JAX arrays
    at the fixed shape (no XLA recompilation).
    """
    # Extract valid entries per field
    arrays = {}
    for field in deposit_data._fields:
        arr = np.asarray(getattr(deposit_data, field))
        extracted = arr[mask]

        if field == 'valid_mask':
            # Recompute valid_mask for the padded output
            continue

        pad_width = pad_size - n_valid
        if pad_width > 0:
            # dx=1.0 for padding avoids division by zero in recombination
            pad_val = 1.0 if field == 'dx' else 0
            if arr.ndim == 2:
                extracted = np.pad(extracted, ((0, pad_width), (0, 0)),
                                   constant_values=pad_val)
            else:
                extracted = np.pad(extracted, (0, pad_width), constant_values=pad_val)
        else:
            extracted = extracted[:pad_size]
        arrays[field] = extracted

    valid_out = np.arange(pad_size) < n_valid

    return DepositData(
        positions_mm=jnp.asarray(arrays['positions_mm']),
        de=jnp.asarray(arrays['de']),
        dx=jnp.asarray(arrays['dx']),
        valid_mask=jnp.asarray(valid_out),
        theta=jnp.asarray(arrays['theta']),
        phi=jnp.asarray(arrays['phi']),
        track_ids=jnp.asarray(arrays['track_ids']),
        group_ids=jnp.asarray(arrays['group_ids']),
    )


def split_and_pad_data(deposit_data, total_pad):
    """Split deposit data by side and pad to total_pad.

    All operations use numpy to avoid XLA recompilation on variable-length
    intermediates. The final padded arrays are converted to JAX at the end
    (fixed total_pad shape -> single JIT compilation for all events).

    Parameters
    ----------
    deposit_data : DepositData
        Input data (can be any size, will be split and padded).
    total_pad : int
        Fixed pad size per side.

    Returns
    -------
    east_data : DepositData
        Padded data for east side (x < 0).
    west_data : DepositData
        Padded data for west side (x >= 0).
    counts : dict
        Actual counts: n_east, n_west, n_tracks.
    """
    positions_mm = np.asarray(deposit_data.positions_mm)
    valid_mask = np.asarray(deposit_data.valid_mask)
    track_ids = np.asarray(deposit_data.track_ids)

    x_mm = positions_mm[:, 0]
    east_mask = valid_mask & (x_mm < 0)
    west_mask = valid_mask & (x_mm >= 0)

    n_east = int(np.sum(east_mask))
    n_west = int(np.sum(west_mask))
    n_tracks = int(len(np.unique(track_ids[valid_mask])))

    if n_east > total_pad:
        print(f"WARNING: n_east ({n_east:,}) > total_pad ({total_pad:,}). Truncating!")
    if n_west > total_pad:
        print(f"WARNING: n_west ({n_west:,}) > total_pad ({total_pad:,}). Truncating!")

    east_data = _extract_and_pad(
        deposit_data, east_mask, min(n_east, total_pad), total_pad)
    west_data = _extract_and_pad(
        deposit_data, west_mask, min(n_west, total_pad), total_pad)

    counts = {'n_east': min(n_east, total_pad),
              'n_west': min(n_west, total_pad),
              'n_tracks': n_tracks}
    return east_data, west_data, counts


if __name__ == "__main__":
    main()