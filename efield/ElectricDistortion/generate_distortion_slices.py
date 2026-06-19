#!/usr/bin/env python3
"""Regenerate distortion_slices/ plots with correct anode/cathode labels.

Usage
-----
    python3 -m ElectricDistortion.generate_distortion_slices
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io.config_loader import build_params
from .run_sce import run
from .plotting.visualize import plot_distortion_slice, _slice_3d
from .plotting.advanced import plot_indicator


PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots", "distortion_slices")


def _nearest_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Run simulation
    params = build_params(preset="jaxtpc", overrides={
        "Nx_output": 21, "Ny_output": 21, "Nz_output": 21,
    })
    Lx = params["Lx"]
    Ly = params["Ly"]
    Lz = params["Lz"]

    print(f"Running SCE simulation (Lx={Lx}, Ly={Ly}, Lz={Lz}) ...")
    r = run(params)

    ox = r["output_x"]
    oy = r["output_y"]
    oz = r["output_z"]
    dx = r["delta_x"]
    dy = r["delta_y"]
    dz = r["delta_z"]

    # Slice fractions: 20%, 50%, 80%
    fracs = [0.2, 0.5, 0.8]

    components = {"dx": dx, "dy": dy, "dz": dz}

    # Plane configs: (plane, normal_axis, normal_grid, slice_values)
    plane_configs = [
        ("xz", "y", oy, [_nearest_index(oy, f * Ly) for f in fracs]),
        ("xy", "z", oz, [_nearest_index(oz, f * Lz) for f in fracs]),
        ("yz", "x", ox, [_nearest_index(ox, f * Lx) for f in fracs]),
    ]

    n_plots = 0
    for plane, axis, grid, indices in plane_configs:
        for comp_name, comp_data in components.items():
            for idx in indices:
                val = grid[idx]
                fname = f"{plane}_{comp_name}_{axis}{int(round(val))}.png"
                fpath = os.path.join(PLOT_DIR, fname)

                # Create figure: heatmap left, indicator right
                fig = plt.figure(figsize=(16, 6))
                ax_heat = fig.add_axes([0.05, 0.08, 0.55, 0.85])
                ax_ind = fig.add_axes([0.65, 0.05, 0.33, 0.90],
                                      projection="3d")

                # Heatmap
                plot_distortion_slice(ox, oy, oz, comp_data, comp_name,
                                      plane=plane, index=idx, ax=ax_heat)

                # 3D indicator
                slice_frac = val / {"x": Lx, "y": Ly, "z": Lz}[axis]
                plot_indicator(Lx, Ly, Lz, plane, slice_frac, comp_name,
                               ax=ax_ind)

                fig.savefig(fpath, dpi=150, bbox_inches="tight")
                plt.close(fig)
                n_plots += 1

    print(f"Done! {n_plots} plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
