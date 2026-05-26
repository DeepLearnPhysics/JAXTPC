"""
Example usage of the diffusion kernel system.

This demonstrates how to:
1. Load pre-extracted kernel NPZ files
2. Create diffusion kernel arrays
3. Perform runtime interpolation

NOTE: This assumes you have already run extract_responses_from_images.py
to create the kernel NPZ files from the response images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
# Handle both module and script execution
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools_old.responses import (
        create_diffusion_kernel_array,
        interpolate_diffusion_kernel_batch,
        visualize_kernel,
        calculate_wire_count
    )
else:
    from . import (
        create_diffusion_kernel_array,
        interpolate_diffusion_kernel_batch,
        visualize_kernel,
        calculate_wire_count
    )
import jax
import jax.numpy as jnp


def main():
    # Parameters
    planes = ['U', 'V', 'Y']
    time_bin_size = 0.5
    wire_spacing = 0.1
    wire_stride = 10
    num_s = 16
    
    # Step 1: Check that kernel files exist
    print("="*60)
    print("STEP 1: Checking for kernel NPZ files")
    print("="*60)
    
    kernel_files_exist = True
    for plane in planes:
        kernel_file = f'{plane}_plane_kernel.npz'
        if os.path.exists(kernel_file):
            print(f"  ✓ Found {kernel_file}")
        else:
            print(f"  ✗ Missing {kernel_file}")
            kernel_files_exist = False
    
    if not kernel_files_exist:
        print("\nERROR: Kernel files not found!")
        print("Please run 'python extract_responses_from_images.py' first to create them.")
        return
    
    # Step 2: Create diffusion kernel arrays from loaded kernels
    print("\n" + "="*60)
    print("STEP 2: Creating diffusion kernel arrays from NPZ files")
    print("="*60)
    
    DKernels = create_diffusion_kernel_array(
        planes=planes,
        num_s=num_s,
        kernel_dir='.',  # Current directory since we're in tools/responses
        wire_spacing=wire_spacing,
        time_spacing=time_bin_size
    )
    
    # Step 4: Demonstrate runtime interpolation
    print("\n" + "="*60)
    print("STEP 4: Runtime interpolation example")
    print("="*60)
    
    # Use first available plane
    plane = list(DKernels.keys())[0]
    DKernel, linear_s, kernel_shape, x_coords, y_coords = DKernels[plane]
    num_wires = calculate_wire_count(kernel_shape[1], wire_spacing)
    
    print(f"\nUsing {plane} plane for interpolation example")
    print(f"Number of wires: {num_wires}")
    print(f"Output shape per segment: ({num_wires}, {kernel_shape[0]-1})")
    
    # Example: Interpolate for a batch of segments
    batch_size = 10000
    
    # Generate random parameters for demonstration
    key = jax.random.PRNGKey(42)
    key, subkey1 = jax.random.split(key)
    s_batch = jax.random.uniform(subkey1, (batch_size,), minval=0.0, maxval=1.0)
    
    key, subkey2 = jax.random.split(key)
    w_batch = jax.random.uniform(subkey2, (batch_size,), minval=0.0, maxval=0.99)
    
    key, subkey3 = jax.random.split(key)
    t_batch = jax.random.uniform(subkey3, (batch_size,), minval=0.0, maxval=0.49)
    
    # Perform batch interpolation
    print(f"\nInterpolating {batch_size:,} segments...")
    start_time = time.time()
    
    results = interpolate_diffusion_kernel_batch(
        DKernel, s_batch, w_batch, t_batch,
        wire_stride, wire_spacing, time_bin_size, num_wires
    )
    results.block_until_ready()
    
    end_time = time.time()
    
    print(f"Interpolation complete!")
    print(f"  Time: {(end_time - start_time)*1000:.1f} ms")
    print(f"  Output shape: {results.shape}")
    print(f"  Throughput: {batch_size/(end_time - start_time):.0f} segments/second")


if __name__ == "__main__":
    import time
    main()