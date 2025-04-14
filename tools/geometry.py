import yaml
import os


def generate_detector(config_file_path):
    """
    Reads a JAXTPC detector configuration YAML file and returns a detector dictionary.

    Parameters:
        config_file_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing all detector properties from the configuration file.
    """
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at {config_file_path}")
        return None

    try:
        with open(config_file_path, 'r') as file:
            detector_config = yaml.safe_load(file)

        # Basic validation to ensure the config has the expected structure
        required_keys = ['detector', 'wire_planes', 'readout', 'simulation', 'medium', 'electric_field']
        for key in required_keys:
            if key not in detector_config:
                print(f"Error: Missing required section '{key}' in configuration file")
                return None

        # Calculate total number of wire planes
        total_planes = sum(len(side['planes']) for side in detector_config['wire_planes']['sides'])
        detector_config['total_planes'] = total_planes

        # Calculate maximum drift time based on x dimension (drift direction)
        dimensions = detector_config['detector']['dimensions']
        drift_velocity_cm_per_us = detector_config['simulation']['drift']['velocity'] / 10  # convert mm/μs to cm/μs
        max_drift_time_us = dimensions[
                                'x'] / drift_velocity_cm_per_us / 2  # divide by 2 because drift is from center to edge
        detector_config['max_drift_time_us'] = max_drift_time_us
        detector_config['drift_direction'] = 'x'

        return detector_config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    except Exception as e:
        print(f"Error loading detector configuration: {e}")
        return None


if __name__ == "__main__":
    # Path to your detector configuration file
    config_path = "config/cubic_wireplane_config.yaml"

    # Generate the detector dictionary
    detector = generate_detector(config_path)

    if detector:
        print("Detector Configuration Successfully Loaded")
        print("==========================================")

        # Access detector properties
        print(f"Detector name: {detector['detector']['name']}")
        dimensions = detector['detector']['dimensions']
        print(f"Dimensions: {dimensions['x']} × {dimensions['y']} × {dimensions['z']} cm³")

        # Access wire planes information
        print("\nWire Plane Configuration:")
        print("------------------------")
        for side in detector['wire_planes']['sides']:
            side_id = side['side_id']
            print(f"Side {side_id}: {side['description']}")

            for plane in side['planes']:
                plane_id = plane['plane_id']
                plane_type = plane['type']
                print(f"  Plane {plane_id} ({plane_type}):")
                print(f"    Angle: {plane['angle']} degrees")
                print(f"    Wire spacing: {plane['wire_spacing']} cm")
                print(f"    Bias voltage: {plane['bias_voltage']} V")

        # Access electric field and medium properties
        print(f"\nElectric field strength: {detector['electric_field']['field_strength']} V/cm")
        print(f"Medium: {detector['medium']['type']} at {detector['medium']['temperature']} K")

        # Access calculated properties
        print("\nCalculated Properties:")
        print("--------------------")
        print(f"Total wire planes: {detector['total_planes']}")
        print(f"Drift direction: {detector['drift_direction']}")
        print(f"Maximum drift time: {detector['max_drift_time_us']:.2f} μs")
    else:
        print("Failed to load detector configuration.")