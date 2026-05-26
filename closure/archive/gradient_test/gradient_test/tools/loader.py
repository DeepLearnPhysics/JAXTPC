import h5py
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json


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
            Dictionary mapping field names to JAX arrays.
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

                    # Convert to JAX array for numeric data
                    result[field] = jnp.array(field_data)
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
        jnp.ndarray or None
            JAX array where index[i] gives the particle index for step i,
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

                return jnp.array(step_to_particle)
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
            result['position'] = jnp.stack([result['x'], result['y'], result['z']], axis=1)

        return result


def load_particle_step_data(file_path, event_idx=0, verbose=False):
    """
    Convenience function to extract particle step data from an HDF5 file.
    
    This function creates a ParticleStepExtractor, extracts the data,
    and closes the file, all in one call.
    
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
    dict
        Dictionary mapping property names to JAX arrays.
    """
    with ParticleStepExtractor(file_path, verbose=verbose) as extractor:
        return extractor.extract_step_arrays(event_idx)


def main():
    """
    Example usage of the particle step extractor.
    
    This function demonstrates how to use the particle step extractor
    from the command line with various options.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Extract particle step data from HDF5 files')
    parser.add_argument('file_path', help='Path to the HDF5 file')
    parser.add_argument('--event', '-e', type=int, default=0, help='Event index (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose information')
    parser.add_argument('--output', '-o', help='Save data to JSON file (will convert JAX arrays to lists)')

    args = parser.parse_args()

    # Extract step data
    step_data = load_particle_step_data(args.file_path, args.event, args.verbose)

    # Print summary of the extracted data
    print("\nExtracted step arrays:")
    for key, value in step_data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

        for pdg in particle_types:
            # Get steps for this particle type
            mask = step_data['particle_pdg'] == pdg
            energy = jnp.sum(step_data['de'][mask])
            print(f"  PDG code {pdg}: total energy = {energy}")

    # Save to file if requested
    if args.output:
        try:
            # Convert JAX arrays to lists for JSON serialization
            json_data = {}
            for key, value in step_data.items():
                if hasattr(value, 'shape'):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value

            with open(args.output, 'w') as f:
                json.dump(json_data, f)

            print(f"\nData saved to {args.output}")
        except Exception as e:
            print(f"Error saving data: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()