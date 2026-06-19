"""Advanced SCE visualizations: warped grids, vector fields, streamlines, etc."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class _Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch projected into 3D — gives clean arrowheads."""

    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._pts = (posA, posB)

    def do_3d_projection(self, renderer=None):
        pA, pB = self._pts
        xA, yA, zA = proj3d.proj_transform(*pA, self.axes.M)
        xB, yB, zB = proj3d.proj_transform(*pB, self.axes.M)
        self.set_positions((xA, yA), (xB, yB))
        return min(zA, zB)


def _mid_index(arr):
    """Index closest to the midpoint of *arr*."""
    return int(np.argmin(np.abs(arr - (arr[0] + arr[-1]) / 2.0)))


# ========================================================================== #
#  1. Warped Grid                                                            #
# ========================================================================== #

def plot_warped_grid(output_x, output_y, output_z,
                     delta_x, delta_y, delta_z,
                     plane="xz", index=None, skip=1,
                     ax=None, save_path=None):
    """Warped grid showing how SCE distorts a regular Cartesian mesh.

    Gray lines = undistorted grid.  Coloured lines = distorted
    (reconstructed) positions, coloured by total in-plane displacement.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    if plane == "xz":
        idx = index if index is not None else _mid_index(output_y)
        h, v = output_x, output_z
        dh, dv = delta_x[:, idx, :], delta_z[:, idx, :]
        xlabel, ylabel = "x (cm)", "z (cm)"
        sl = f"y = {output_y[idx]:.1f} cm"
    elif plane == "xy":
        idx = index if index is not None else _mid_index(output_z)
        h, v = output_x, output_y
        dh, dv = delta_x[:, :, idx], delta_y[:, :, idx]
        xlabel, ylabel = "x (cm)", "y (cm)"
        sl = f"z = {output_z[idx]:.1f} cm"
    elif plane == "yz":
        idx = index if index is not None else _mid_index(output_x)
        h, v = output_y, output_z
        dh, dv = delta_y[idx, :, :], delta_z[idx, :, :]
        xlabel, ylabel = "y (cm)", "z (cm)"
        sl = f"x = {output_x[idx]:.1f} cm"
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    H, V = np.meshgrid(h, v, indexing='ij')
    H_w, V_w = H + dh, V + dv
    disp = np.sqrt(dh**2 + dv**2)

    cmap = plt.cm.inferno
    vmax = np.max(disp) or 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    # Undistorted grid (light gray)
    for i in range(0, len(h), skip):
        ax.plot([h[i], h[i]], [v[0], v[-1]], color='#d0d0d0', lw=0.5, zorder=1)
    for j in range(0, len(v), skip):
        ax.plot([h[0], h[-1]], [v[j], v[j]], color='#d0d0d0', lw=0.5, zorder=1)

    # Warped grid — constant-v lines (horizontal)
    for j in range(0, len(v), skip):
        pts = np.column_stack([H_w[:, j], V_w[:, j]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = cmap(norm(0.5 * (disp[:-1, j] + disp[1:, j])))
        ax.add_collection(LineCollection(segs, colors=cols, lw=1.2, zorder=2))

    # Warped grid — constant-h lines (vertical)
    for i in range(0, len(h), skip):
        pts = np.column_stack([H_w[i, :], V_w[i, :]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = cmap(norm(0.5 * (disp[i, :-1] + disp[i, 1:])))
        ax.add_collection(LineCollection(segs, colors=cols, lw=1.2, zorder=2))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="|displacement| (cm)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Warped grid  ({sl})")
    margin = max(h[-1] - h[0], v[-1] - v[0]) * 0.03
    ax.set_xlim(min(h[0], H_w.min()) - margin, max(h[-1], H_w.max()) + margin)
    ax.set_ylim(min(v[0], V_w.min()) - margin, max(v[-1], V_w.max()) + margin)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ========================================================================== #
#  2. Total Distortion Magnitude                                             #
# ========================================================================== #

def plot_total_distortion(output_x, output_y, output_z,
                          delta_x, delta_y, delta_z,
                          plane="xz", index=None,
                          contour_levels=None,
                          ax=None, save_path=None):
    """|delta| = sqrt(dx^2 + dy^2 + dz^2) with optional contour overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    delta_mag = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    if plane == "xz":
        idx = index if index is not None else _mid_index(output_y)
        data = delta_mag[:, idx, :].T
        extent = [output_x[0], output_x[-1], output_z[0], output_z[-1]]
        h_coords, v_coords = output_x, output_z
        xlabel, ylabel = "x (cm)", "z (cm)"
        sl = f"y = {output_y[idx]:.1f} cm"
    elif plane == "xy":
        idx = index if index is not None else _mid_index(output_z)
        data = delta_mag[:, :, idx].T
        extent = [output_x[0], output_x[-1], output_y[0], output_y[-1]]
        h_coords, v_coords = output_x, output_y
        xlabel, ylabel = "x (cm)", "y (cm)"
        sl = f"z = {output_z[idx]:.1f} cm"
    elif plane == "yz":
        idx = index if index is not None else _mid_index(output_x)
        data = delta_mag[idx, :, :].T
        extent = [output_y[0], output_y[-1], output_z[0], output_z[-1]]
        h_coords, v_coords = output_y, output_z
        xlabel, ylabel = "y (cm)", "z (cm)"
        sl = f"x = {output_x[idx]:.1f} cm"
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="|delta| (cm)")

    if contour_levels is not None:
        cs = ax.contour(h_coords, v_coords, data,
                        levels=contour_levels,
                        colors='white', linewidths=0.8, linestyles='--')
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Total distortion |delta|  ({sl})")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ========================================================================== #
#  3. Quiver / Transverse Vector Field                                       #
# ========================================================================== #

def plot_quiver_transverse(output_x, output_y, output_z,
                           delta_y, delta_z,
                           x_index=None, skip=1,
                           ax=None, save_path=None):
    """Quiver plot of transverse distortion (delta_y, delta_z) on a y-z face."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if x_index is None:
        x_index = len(output_x) - 1  # cathode

    Y, Z = np.meshgrid(output_y, output_z, indexing='ij')
    dy = delta_y[x_index, :, :]
    dz = delta_z[x_index, :, :]
    mag = np.sqrt(dy**2 + dz**2)

    q = ax.quiver(Y[::skip, ::skip], Z[::skip, ::skip],
                  dy[::skip, ::skip], dz[::skip, ::skip],
                  mag[::skip, ::skip],
                  cmap='inferno', angles='xy', scale_units='xy')
    plt.colorbar(q, ax=ax, label="|transverse displacement| (cm)").ax.tick_params(labelsize=13)
    ax.set_xlabel("y (cm)", fontsize=15)
    ax.set_ylabel("z (cm)", fontsize=15)
    ax.set_title(f"Transverse distortion  (x = {output_x[x_index]:.1f} cm)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ========================================================================== #
#  4. Face Maps                                                              #
# ========================================================================== #

def plot_face_maps(output_x, output_y, output_z,
                   delta_x, delta_y, delta_z,
                   save_path=None):
    """2x2 panel: distortions on cathode, top, and side faces."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    delta_mag = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # (0,0): Cathode face — |delta|(y, z)
    ax = axes[0, 0]
    data = delta_mag[-1, :, :].T
    extent = [output_y[0], output_y[-1], output_z[0], output_z[-1]]
    im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="|delta| (cm)").ax.tick_params(labelsize=12)
    ax.set_xlabel("y (cm)", fontsize=14)
    ax.set_ylabel("z (cm)", fontsize=14)
    ax.set_title(f"Cathode face  (x = {output_x[-1]:.1f} cm)", fontsize=15)
    ax.tick_params(labelsize=12)

    # (0,1): Cathode face quiver
    ax = axes[0, 1]
    Y, Z = np.meshgrid(output_y, output_z, indexing='ij')
    dy = delta_y[-1, :, :]
    dz = delta_z[-1, :, :]
    mag = np.sqrt(dy**2 + dz**2)
    q = ax.quiver(Y, Z, dy, dz, mag, cmap='inferno', angles='xy',
                  scale_units='xy')
    plt.colorbar(q, ax=ax, label="|transverse| (cm)").ax.tick_params(labelsize=12)
    ax.set_xlabel("y (cm)", fontsize=14)
    ax.set_ylabel("z (cm)", fontsize=14)
    ax.set_title(f"Cathode transverse vectors  (x = {output_x[-1]:.1f} cm)", fontsize=15)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=12)

    # (1,0): Top face — |delta|(x, z) at y = Ly
    ax = axes[1, 0]
    data = delta_mag[:, -1, :].T
    extent = [output_x[0], output_x[-1], output_z[0], output_z[-1]]
    im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="|delta| (cm)").ax.tick_params(labelsize=12)
    ax.set_xlabel("x (cm)", fontsize=14)
    ax.set_ylabel("z (cm)", fontsize=14)
    ax.set_title(f"Top face  (y = {output_y[-1]:.1f} cm)", fontsize=15)
    ax.tick_params(labelsize=12)

    # (1,1): Side face — |delta|(x, y) at z = Lz
    ax = axes[1, 1]
    data = delta_mag[:, :, -1].T
    extent = [output_x[0], output_x[-1], output_y[0], output_y[-1]]
    im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="|delta| (cm)").ax.tick_params(labelsize=12)
    ax.set_xlabel("x (cm)", fontsize=14)
    ax.set_ylabel("y (cm)", fontsize=14)
    ax.set_title(f"Side face  (z = {output_z[-1]:.1f} cm)", fontsize=15)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


# ========================================================================== #
#  5. Profile Fan                                                            #
# ========================================================================== #

def plot_profile_fan(coords, profiles, labels, colors=None,
                     xlabel="", ylabel="", title="",
                     ax=None, save_path=None):
    """Overlaid 1D profiles coloured by position.

    Parameters
    ----------
    coords : 1D array
        Common x-axis for all profiles.
    profiles : list of 1D arrays
        Y-values for each profile.
    labels : list of str
        Legend label for each profile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    n = len(profiles)
    if colors is None:
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for prof, lab, col in zip(profiles, labels, colors):
        ax.plot(coords, prof, color=col, lw=1.5, label=lab)

    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ========================================================================== #
#  6. E-field Streamlines                                                    #
# ========================================================================== #

def plot_efield_streamlines(x_grid, y_grid, z_grid,
                            Ex, Ey, Ez, E0,
                            plane="xz", index=None,
                            ax=None, save_path=None):
    """Streamlines of the E-field coloured by |E|/E0.

    Electrons drift approximately opposite to the plotted streamlines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    if plane == "xz":
        idx = index if index is not None else _mid_index(y_grid)
        U = Ex[:, idx, :].T
        V = Ez[:, idx, :].T
        speed = (E_mag[:, idx, :] / E0).T
        h_grid, v_grid = x_grid, z_grid
        xlabel, ylabel = "x (cm)", "z (cm)"
        sl = f"y = {y_grid[idx]:.1f} cm"
    elif plane == "xy":
        idx = index if index is not None else _mid_index(z_grid)
        U = Ex[:, :, idx].T
        V = Ey[:, :, idx].T
        speed = (E_mag[:, :, idx] / E0).T
        h_grid, v_grid = x_grid, y_grid
        xlabel, ylabel = "x (cm)", "y (cm)"
        sl = f"z = {z_grid[idx]:.1f} cm"
    else:
        raise ValueError(f"Streamlines support 'xz' and 'xy' planes")

    strm = ax.streamplot(h_grid, v_grid, U, V,
                         color=speed, cmap='coolwarm',
                         density=1.5, linewidth=1.0, arrowsize=1.2)
    plt.colorbar(strm.lines, ax=ax, label="|E| / E0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"E-field streamlines  ({sl})")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ========================================================================== #
#  7. Convergence Diagnostics                                                #
# ========================================================================== #

def plot_convergence(history, save_path=None):
    """Convergence diagnostics for self-consistent v_ion iteration.

    Parameters
    ----------
    history : dict
        Keys: ``rel_change`` (list), ``rho_profiles`` (list of 1D),
        ``x_grid`` (1D), ``v_ion_profiles`` (list of 1D),
        optional ``tol`` (float).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    iters = np.arange(1, len(history['rel_change']) + 1)

    # (0): Convergence curve
    ax = axes[0]
    ax.semilogy(iters, history['rel_change'], 'o-', color='C0', lw=2, ms=6)
    if 'tol' in history:
        ax.axhline(history['tol'], ls='--', color='red', lw=1,
                   label=f"tol = {history['tol']:.0e}")
        ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative change in rho")
    ax.set_title("Convergence")
    ax.grid(True, alpha=0.3)

    # (1): rho(x) evolution
    ax = axes[1]
    x = history['x_grid']
    n = len(history['rho_profiles'])
    cmap = plt.cm.viridis
    for i, rho_1d in enumerate(history['rho_profiles']):
        color = cmap(i / max(n - 1, 1))
        label = "initial" if i == 0 else f"iter {i}"
        ax.plot(x, rho_1d * 1e9, color=color, lw=1.5, label=label)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel(r"$\rho$ (nC/m$^3$)")
    ax.set_title("Charge density evolution")
    ax.legend(fontsize=7, loc='upper left')

    # (2): v_ion(x) evolution
    ax = axes[2]
    for i, v_prof in enumerate(history['v_ion_profiles']):
        color = cmap(i / max(n - 1, 1))
        label = "initial (constant)" if i == 0 else f"iter {i}"
        ax.plot(x, v_prof * 1e3, color=color, lw=1.5, label=label)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel(r"$v_\mathrm{ion}$ (mm/s)")
    ax.set_title("Ion drift speed evolution")
    ax.legend(fontsize=7, loc='best')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


# ========================================================================== #
#  8. Multi-Detector Comparison                                              #
# ========================================================================== #

def plot_detector_comparison(detector_results, save_path=None):
    """Compare SCE across multiple detectors.

    Parameters
    ----------
    detector_results : dict
        ``{name: {'x_norm': 1D, 'E_ratio_profile': 1D,
                  'max_delta_y_frac': float, 'max_delta': float}}``
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = list(detector_results.keys())
    colors = [plt.cm.Set1(i / 9.0) for i in range(len(names))]

    # (0): |E|/E0 profiles
    ax = axes[0]
    for name, col in zip(names, colors):
        r = detector_results[name]
        ax.plot(r['x_norm'], r['E_ratio_profile'], color=col, lw=2,
                label=name)
    ax.axhline(1.0, ls='--', color='gray', lw=0.8)
    ax.set_xlabel("x / Lx")
    ax.set_ylabel("|E| / E0")
    ax.set_title("E-field ratio along drift")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1): Fractional transverse distortion
    ax = axes[1]
    vals = [detector_results[n]['max_delta_y_frac'] * 100 for n in names]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel("max |delta_y| / Ly  (%)")
    ax.set_title("Fractional transverse distortion")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va='center', fontsize=9)

    # (2): Absolute total distortion
    ax = axes[2]
    vals = [detector_results[n]['max_delta'] for n in names]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel("max |delta| (cm)")
    ax.set_title("Total distortion magnitude")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va='center', fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


# ========================================================================== #
#  9. Self-Consistent Correction Map                                         #
# ========================================================================== #

# ========================================================================== #
#  Indicator Diagram (3D cube with slice + distortion arrow)                 #
# ========================================================================== #

def plot_indicator(Lx, Ly, Lz, plane, slice_frac, component,
                   ax=None, save_path=None):
    """Draw a 3D wireframe cube showing detector orientation, slice plane,
    and distortion component arrow.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Detector dimensions (cm).
    plane : str
        Slice plane: ``"xz"``, ``"xy"``, or ``"yz"``.
    slice_frac : float
        Fractional position of the slice along the normal axis (0-1).
    component : str
        Distortion component being plotted: ``"dx"``, ``"dy"``, ``"dz"``.
    """
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # -- Draw 12 cube edges (consistent weight, butt caps for clean corners) --
    corners = np.array([[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
                        [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]])
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        xs, ys, zs = zip(corners[i], corners[j])
        ax.plot3D(xs, ys, zs, color='k', lw=1.0, alpha=0.5,
                  solid_capstyle='butt')

    # -- Highlight anode (x=0) and cathode (x=Lx) faces ----------------------
    anode_face = [[0, 0, 0], [0, Ly, 0], [0, Ly, Lz], [0, 0, Lz]]
    cathode_face = [[Lx, 0, 0], [Lx, Ly, 0], [Lx, Ly, Lz], [Lx, 0, Lz]]
    ax.add_collection3d(Poly3DCollection([anode_face], alpha=0.08,
                                         facecolor='#2196F3',
                                         edgecolor='none'))
    ax.add_collection3d(Poly3DCollection([cathode_face], alpha=0.08,
                                         facecolor='#E53935',
                                         edgecolor='none'))

    # Anode (x=0) on the left, Cathode (x=Lx) on the right
    ax.text(0, Ly * 0.25, Lz * 0.15, "Anode", fontsize=14, zdir='y',
            color='#1565C0', ha='center', va='center', fontweight='bold')
    ax.text(Lx, Ly * 0.25, Lz * 0.15, "Cathode", fontsize=14, zdir='y',
            color='#C62828', ha='center', va='center', fontweight='bold')

    # -- Slice plane ----------------------------------------------------------
    gap = 12.0  # cm offset above the top edge of each slice
    if plane == "xz":
        y_val = slice_frac * Ly
        verts = [[0, y_val, 0], [Lx, y_val, 0],
                 [Lx, y_val, Lz], [0, y_val, Lz]]
        slice_label = f"y = {y_val:.0f} cm"
        # Screen-space vertical text (2D) to match z(cm) axis label
        label_pos = None  # handled separately as 2D text
        label_zdir = None
        label_rotation = 90
        slice_label_2d = slice_label  # flag for 2D placement
    elif plane == "xy":
        z_val = slice_frac * Lz
        verts = [[0, 0, z_val], [Lx, 0, z_val],
                 [Lx, Ly, z_val], [0, Ly, z_val]]
        slice_label = f"z = {z_val:.0f} cm"
        # Back edge center
        label_pos = (Lx / 2, Ly + gap, z_val)
        label_zdir = 'x'
        label_rotation = 0
    elif plane == "yz":
        x_val = slice_frac * Lx
        verts = [[x_val, 0, 0], [x_val, Ly, 0],
                 [x_val, Ly, Lz], [x_val, 0, Lz]]
        slice_label = f"x = {x_val:.0f} cm"
        # Top edge center, shifted left
        label_pos = (x_val, Ly / 2 - 80, Lz - 5)
        label_zdir = 'y'
        label_rotation = 0
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    ax.add_collection3d(Poly3DCollection([verts], alpha=0.22,
                                         facecolor='#FF9800',
                                         edgecolor='#E65100',
                                         linewidth=1.5))
    if label_pos is not None:
        kw = dict(fontsize=13, color='#E65100', ha='center', fontweight='bold',
                  rotation=label_rotation)
        if label_zdir is not None:
            kw['zdir'] = label_zdir
        ax.text(*label_pos, slice_label, **kw)

    # -- Distortion arrow — FancyArrowPatch for clean arrowhead ----------------
    if component not in ("dx", "dy", "dz"):
        raise ValueError(f"Unknown component '{component}'")
    arrow_len = min(Lx, Ly, Lz) * 0.30
    # dy is longer to compensate for perspective foreshortening along y
    dirs = {"dx": (arrow_len, 0, 0),
            "dy": (0, arrow_len * 1.8, 0),
            "dz": (0, 0, arrow_len)}
    da = dirs[component]
    # Above the top-left corner (x=0, y=0, z=Lz) — same height for all
    cx, cy, cz = Lx * 0.12, Ly * 0.15, Lz * 1.14
    # Center the arrow at (cx, cy, cz)
    start = (cx - da[0] / 2, cy - da[1] / 2, cz - da[2] / 2)
    tip = (cx + da[0] / 2, cy + da[1] / 2, cz + da[2] / 2)
    arrow = _Arrow3D(start, tip,
                     arrowstyle='-|>', color='#D32F2F',
                     lw=2.5, mutation_scale=12, shrinkA=0, shrinkB=0)
    ax.add_artist(arrow)
    # Label well to the left of the arrow
    ax.text(cx - Lx * 0.35, cy, cz,
            f"$\\delta_{component[1]}$", fontsize=19, color='#D32F2F',
            fontweight='bold', ha='center', va='center')

    # -- Clean up axes: no grid, no panes, minimal ticks ----------------------
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    ax.grid(False)

    ax.set_xlabel("x (cm)", fontsize=13, labelpad=-6)
    ax.set_ylabel("y (cm)", fontsize=13, labelpad=-2)
    ax.set_zlabel("z (cm)", fontsize=13, labelpad=-2)

    ax.set_xticks([0, Lx])
    ax.set_yticks([0, Ly])
    ax.set_zticks([0, Lz])
    ax.tick_params(labelsize=11, pad=0)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect([Lx, Ly, Lz])  # true proportions
    ax.view_init(elev=22, azim=-55)

    # -- Deferred 2D label for XZ slice (screen-space vertical) ----------------
    if label_pos is None:
        # Project the original 3D position to 2D axes coordinates
        fig.canvas.draw()
        x3d, y3d, z3d = -gap * 2, slice_frac * Ly, Lz / 2
        x2, y2, _ = proj3d.proj_transform(x3d, y3d, z3d, ax.get_proj())
        # Convert projection coords to axes-fraction coords
        x_disp, y_disp = ax.transData.transform((x2, y2))
        x_ax, y_ax = ax.transAxes.inverted().transform((x_disp, y_disp))
        ax.text2D(x_ax, y_ax, slice_label, transform=ax.transAxes,
                  fontsize=9, color='#E65100', ha='center', va='center',
                  fontweight='bold', rotation=90)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_sc_correction(output_x, output_y, output_z,
                       delta_y_base, delta_y_sc,
                       plane="xz", index=None, save_path=None):
    """Three-panel comparison: base, self-consistent, and correction."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if plane == "xz":
        idx = index if index is not None else _mid_index(output_y)
        base_2d = delta_y_base[:, idx, :].T
        sc_2d = delta_y_sc[:, idx, :].T
        extent = [output_x[0], output_x[-1], output_z[0], output_z[-1]]
        xlabel, ylabel = "x (cm)", "z (cm)"
        sl = f"y = {output_y[idx]:.1f} cm"
    elif plane == "xy":
        idx = index if index is not None else _mid_index(output_z)
        base_2d = delta_y_base[:, :, idx].T
        sc_2d = delta_y_sc[:, :, idx].T
        extent = [output_x[0], output_x[-1], output_y[0], output_y[-1]]
        xlabel, ylabel = "x (cm)", "y (cm)"
        sl = f"z = {output_z[idx]:.1f} cm"
    else:
        raise ValueError(f"Supports 'xz' and 'xy' planes")

    diff_2d = sc_2d - base_2d

    # Shared scale for base and SC
    vmax = max(abs(base_2d.min()), abs(base_2d.max()),
               abs(sc_2d.min()), abs(sc_2d.max()))

    ax = axes[0]
    im = ax.imshow(base_2d, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="cm")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"delta_y (constant v_ion)  ({sl})")

    ax = axes[1]
    im = ax.imshow(sc_2d, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="cm")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"delta_y (self-consistent)  ({sl})")

    ax = axes[2]
    dmax = max(abs(diff_2d.min()), abs(diff_2d.max())) or 0.01
    im = ax.imshow(diff_2d, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    plt.colorbar(im, ax=ax, label="cm")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"Correction (SC - base)  ({sl})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
