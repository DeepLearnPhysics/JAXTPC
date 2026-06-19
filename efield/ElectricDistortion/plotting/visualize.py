"""Diagnostic plots for SCE simulation results."""

import numpy as np
import matplotlib.pyplot as plt


def _mid_index(arr):
    """Index closest to the midpoint of *arr*."""
    return int(np.argmin(np.abs(arr - (arr[0] + arr[-1]) / 2.0)))


def _slice_3d(data, x_grid, y_grid, z_grid, plane, index=None):
    """Extract a 2-D slice from a 3-D array.

    Returns (data_2d, extent, xlabel, ylabel, slice_label).
    """
    if plane == "xz":
        idx = index if index is not None else _mid_index(y_grid)
        return (data[:, idx, :].T,
                [x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
                "x (cm)", "z (cm)", f"y = {y_grid[idx]:.1f} cm")
    elif plane == "xy":
        idx = index if index is not None else _mid_index(z_grid)
        return (data[:, :, idx].T,
                [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                "x (cm)", "y (cm)", f"z = {z_grid[idx]:.1f} cm")
    elif plane == "yz":
        idx = index if index is not None else _mid_index(x_grid)
        return (data[idx, :, :].T,
                [y_grid[0], y_grid[-1], z_grid[0], z_grid[-1]],
                "y (cm)", "z (cm)", f"x = {x_grid[idx]:.1f} cm")
    raise ValueError(f"Unknown plane '{plane}'")


def plot_efield_slice(x_grid, y_grid, z_grid, E_mag, E0,
                      plane="xz", index=None, ax=None, save_path=None):
    """Plot a 2-D colour map of |E|/E0."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    data, extent, xl, yl, sl = _slice_3d(E_mag / E0, x_grid, y_grid, z_grid,
                                          plane, index)
    im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="|E| / E0")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"E-field ratio  ({sl})")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_distortion_slice(x_grid, y_grid, z_grid, delta, component,
                          plane="xz", index=None, ax=None, save_path=None):
    """Plot a 2-D colour map of one distortion component (cm)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    data, extent, xl, yl, sl = _slice_3d(delta, x_grid, y_grid, z_grid,
                                          plane, index)
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"{component} (cm)").ax.tick_params(labelsize=13)
    ax.set_xlabel(xl, fontsize=15)
    ax.set_ylabel(yl, fontsize=15)
    ax.set_title(f"{component}  ({sl})", fontsize=16)
    ax.tick_params(labelsize=13)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_efield_profile(x_grid, E_mag, E0, y_grid, z_grid,
                        iy=None, iz=None, ax=None, save_path=None):
    """Plot |E|/E0 vs x along a single (y, z) line (defaults to centre)."""
    if iy is None:
        iy = _mid_index(y_grid)
    if iz is None:
        iz = _mid_index(z_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(x_grid, E_mag[:, iy, iz] / E0)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("|E| / E0")
    ax.set_title(f"|E|/E0 along x  (y={y_grid[iy]:.1f}, z={z_grid[iz]:.1f} cm)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_distortion_profiles(output_x, output_y, output_z,
                             delta_x, delta_y, delta_z,
                             save_path=None):
    """Plot all three distortion components vs x at the centre of y-z."""
    iy = _mid_index(output_y)
    iz = _mid_index(output_z)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, delta, label in zip(axes,
                                 [delta_x, delta_y, delta_z],
                                 ["delta_x", "delta_y", "delta_z"]):
        ax.plot(output_x, delta[:, iy, iz])
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel(f"{label} (cm)")
        ax.set_title(f"{label} vs x  (y={output_y[iy]:.1f}, z={output_z[iz]:.1f})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_summary(x_poisson, y_poisson, z_poisson, E_mag, E0,
                 output_x, output_y, output_z,
                 delta_x, delta_y, delta_z,
                 save_path=None):
    """2x3 summary: E-field slices + distortion maps."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_efield_slice(x_poisson, y_poisson, z_poisson, E_mag, E0,
                      plane="xz", ax=axes[0, 0])
    plot_efield_slice(x_poisson, y_poisson, z_poisson, E_mag, E0,
                      plane="xy", ax=axes[0, 1])
    plot_efield_profile(x_poisson, E_mag, E0, y_poisson, z_poisson,
                        ax=axes[0, 2])

    for col, (delta, label) in enumerate(
        [(delta_x, "delta_x"), (delta_y, "delta_y"), (delta_z, "delta_z")]
    ):
        plot_distortion_slice(output_x, output_y, output_z,
                              delta, label, plane="xz", ax=axes[1, col])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
