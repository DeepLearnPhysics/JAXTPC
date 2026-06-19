"""
Tests for electric field distortion physics and SCE utilities.

Files under test:
    tools/efield_distortions.py  — toy E-field, drift corrections, interpolation
    tools/utils.py               — save_sce_data / load_sce_data
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.efield_distortions import (
    generate_toy_efield_map,
    compute_drift_corrections,
    interpolate_map_3d,
    create_single_interpolation_fn,
)
from tools.utils import save_sce_data, load_sce_data


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
HALF_X = 100.0   # cm — smaller than production for fast tests
HALF_Y = 100.0
HALF_Z = 100.0
E0 = 200.0       # V/cm
EPS_MAX = 0.10    # 10 % longitudinal
EPS_TRANS = 0.05  # 5 % transverse
GRID = (21, 11, 11)   # odd so grid nodes land exactly on boundaries


@pytest.fixture
def toy_sides():
    """Toy E-field maps (east, west) with moderate distortions."""
    return generate_toy_efield_map(
        HALF_X, HALF_Y, HALF_Z, E0,
        grid_shape=GRID,
        epsilon_max=EPS_MAX,
        epsilon_trans=EPS_TRANS,
    )


@pytest.fixture
def uniform_sides():
    """Uniform E-field maps (zero distortion) for baseline tests."""
    return generate_toy_efield_map(
        HALF_X, HALF_Y, HALF_Z, E0,
        grid_shape=GRID,
        epsilon_max=0.0,
        epsilon_trans=0.0,
    )


# =========================================================================
# generate_toy_efield_map
# =========================================================================

class TestGenerateToyEfieldMap:
    """Tests for generate_toy_efield_map (per-side output)."""

    def test_output_shape_and_dtype(self, toy_sides):
        east_side, west_side = toy_sides
        Nx, Ny, Nz = GRID
        for efield, origin, spacing in [east_side, west_side]:
            assert efield.shape == (Nx, Ny, Nz, 3)
            assert efield.dtype == np.float32
            assert origin.shape == (3,)
            assert spacing.shape == (3,)

    def test_grid_origin_and_spacing(self, toy_sides):
        (east_ef, east_o, east_s), (west_ef, west_o, west_s) = toy_sides

        # East: origin at (-L, -Ly, -Lz), x covers [-L, 0]
        np.testing.assert_allclose(east_o, [-HALF_X, -HALF_Y, -HALF_Z])
        np.testing.assert_allclose(east_s[0], HALF_X / (GRID[0] - 1), rtol=1e-10)

        # West: origin at (0, -Ly, -Lz), x covers [0, +L]
        np.testing.assert_allclose(west_o, [0.0, -HALF_Y, -HALF_Z])
        np.testing.assert_allclose(west_s[0], HALF_X / (GRID[0] - 1), rtol=1e-10)

        # y, z spacing shared
        expected_sy = 2 * HALF_Y / (GRID[1] - 1)
        expected_sz = 2 * HALF_Z / (GRID[2] - 1)
        np.testing.assert_allclose(east_s[1], expected_sy, rtol=1e-10)
        np.testing.assert_allclose(west_s[2], expected_sz, rtol=1e-10)

    def test_zero_distortion_recovers_uniform_field(self, uniform_sides):
        """eps=0 should give a perfectly uniform field on each side."""
        (east_ef, _, _), (west_ef, _, _) = uniform_sides

        # East: Ex = +E0 everywhere
        np.testing.assert_allclose(east_ef[..., 0], E0, rtol=1e-5)
        # West: Ex = -E0 everywhere
        np.testing.assert_allclose(west_ef[..., 0], -E0, rtol=1e-5)
        # Transverse = 0
        np.testing.assert_allclose(east_ef[..., 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(east_ef[..., 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(west_ef[..., 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(west_ef[..., 2], 0.0, atol=1e-6)

    def test_ex_positive_east_negative_west(self, toy_sides):
        """Ex > 0 on east, Ex < 0 on west (everywhere within each side)."""
        (east_ef, _, _), (west_ef, _, _) = toy_sides
        assert np.all(east_ef[..., 0] > 0), "East Ex should be positive"
        assert np.all(west_ef[..., 0] < 0), "West Ex should be negative"

    def test_field_magnitude_symmetric_between_sides(self, toy_sides):
        """|Ex| at mirror positions across cathode should be equal."""
        (east_ef, _, _), (west_ef, _, _) = toy_sides
        Nx = GRID[0]
        # East index i (x from anode toward cathode) corresponds to
        # West index (Nx-1-i) (x from cathode toward anode)
        for i in range(Nx):
            j = Nx - 1 - i
            east_mag = np.abs(east_ef[i, :, :, 0])
            west_mag = np.abs(west_ef[j, :, :, 0])
            np.testing.assert_allclose(east_mag, west_mag, rtol=1e-5,
                                       err_msg=f"Asymmetry at east[{i}] vs west[{j}]")

    def test_longitudinal_distortion_at_boundaries(self, toy_sides):
        """Ex at anode = E0*(1-eps), at cathode = E0*(1+eps)."""
        (east_ef, _, _), _ = toy_sides
        cy, cz = GRID[1] // 2, GRID[2] // 2

        # East anode (index 0, x = -L)
        Ex_anode = float(east_ef[0, cy, cz, 0])
        np.testing.assert_allclose(Ex_anode, E0 * (1 - EPS_MAX), rtol=1e-4)

        # East cathode (index -1, x = 0)
        Ex_cathode = float(east_ef[-1, cy, cz, 0])
        np.testing.assert_allclose(Ex_cathode, E0 * (1 + EPS_MAX), rtol=1e-4)

    def test_voltage_integral_preserved(self, toy_sides):
        """Integral of Ex from anode to cathode should equal E0 * L."""
        (east_ef, east_o, east_s), (west_ef, west_o, west_s) = toy_sides
        cy, cz = GRID[1] // 2, GRID[2] // 2
        expected = E0 * HALF_X

        # East side: integral of Ex dx from -L to 0
        integral_east = float(np.trapezoid(east_ef[:, cy, cz, 0], dx=east_s[0]))
        np.testing.assert_allclose(integral_east, expected, rtol=1e-3,
                                   err_msg="Voltage integral not preserved on east side")

        # West side: integral of -Ex dx from 0 to +L
        integral_west = float(-np.trapezoid(west_ef[:, cy, cz, 0], dx=west_s[0]))
        np.testing.assert_allclose(integral_west, expected, rtol=1e-3,
                                   err_msg="Voltage integral not preserved on west side")

    def test_gauss_law_dex_dx_constant(self, toy_sides):
        """dEx/dx should be approximately constant (uniform space charge)."""
        (east_ef, _, east_s), _ = toy_sides
        dx = east_s[0]
        cy, cz = GRID[1] // 2, GRID[2] // 2

        east_Ex = east_ef[:, cy, cz, 0]
        dEdx = np.diff(east_Ex) / dx
        expected_slope = E0 * EPS_MAX * 2.0 / HALF_X
        np.testing.assert_allclose(dEdx, expected_slope, rtol=1e-4)

    def test_transverse_zero_at_anode(self, toy_sides):
        """Ey and Ez should be zero at the anode planes."""
        (east_ef, _, _), (west_ef, _, _) = toy_sides

        # East anode (index 0)
        np.testing.assert_allclose(east_ef[0, :, :, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(east_ef[0, :, :, 2], 0.0, atol=1e-6)
        # West anode (index -1)
        np.testing.assert_allclose(west_ef[-1, :, :, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(west_ef[-1, :, :, 2], 0.0, atol=1e-6)

    def test_transverse_antisymmetric_in_y(self, toy_sides):
        """Ey(y) = -Ey(-y) (odd symmetry about y=0)."""
        (east_ef, _, _), _ = toy_sides
        cy = GRID[1] // 2
        cx = GRID[0] // 2  # mid-drift on east side
        cz = GRID[2] // 2
        for dy in range(1, cy):
            Ey_plus = float(east_ef[cx, cy + dy, cz, 1])
            Ey_minus = float(east_ef[cx, cy - dy, cz, 1])
            np.testing.assert_allclose(Ey_plus, -Ey_minus, atol=1e-5,
                                       err_msg=f"Ey antisymmetry violated at dy={dy}")

    def test_transverse_antisymmetric_in_z(self, toy_sides):
        """Ez(z) = -Ez(-z)."""
        (east_ef, _, _), _ = toy_sides
        cz = GRID[2] // 2
        cx = GRID[0] // 2
        cy = GRID[1] // 2
        for dz in range(1, cz):
            Ez_plus = float(east_ef[cx, cy, cz + dz, 2])
            Ez_minus = float(east_ef[cx, cy, cz - dz, 2])
            np.testing.assert_allclose(Ez_plus, -Ez_minus, atol=1e-5,
                                       err_msg=f"Ez antisymmetry violated at dz={dz}")

    def test_transverse_grows_toward_cathode(self, toy_sides):
        """|Ey| should increase from anode toward cathode."""
        (east_ef, _, _), _ = toy_sides
        cy_off = GRID[1] // 2 + 2  # y > 0
        cz = GRID[2] // 2

        Ey_near_anode = abs(float(east_ef[1, cy_off, cz, 1]))
        Ey_near_cathode = abs(float(east_ef[-2, cy_off, cz, 1]))

        assert Ey_near_cathode > Ey_near_anode, (
            f"|Ey| should grow toward cathode: "
            f"near_anode={Ey_near_anode:.4f}, near_cathode={Ey_near_cathode:.4f}"
        )

    def test_transverse_zero_when_epsilon_trans_zero(self):
        """eps_trans=0 should give Ey=Ez=0 everywhere."""
        (east_ef, _, _), (west_ef, _, _) = generate_toy_efield_map(
            HALF_X, HALF_Y, HALF_Z, E0,
            grid_shape=GRID, epsilon_max=0.1, epsilon_trans=0.0,
        )
        np.testing.assert_allclose(east_ef[..., 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(east_ef[..., 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(west_ef[..., 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(west_ef[..., 2], 0.0, atol=1e-6)


# =========================================================================
# compute_drift_corrections (per-side)
# =========================================================================

class TestComputeDriftCorrections:
    """Tests for compute_drift_corrections."""

    @pytest.fixture
    def east_corrections(self, toy_sides):
        """Drift corrections for the east side."""
        east_ef, east_o, east_s = toy_sides[0]
        return compute_drift_corrections(
            east_ef, east_o, east_s,
            anode_x_cm=-HALF_X, nominal_field_Vcm=E0,
            drift_velocity_cm_us=0.11, dt_us=2.0,
        ), east_o, east_s

    @pytest.fixture
    def west_corrections(self, toy_sides):
        """Drift corrections for the west side."""
        west_ef, west_o, west_s = toy_sides[1]
        return compute_drift_corrections(
            west_ef, west_o, west_s,
            anode_x_cm=HALF_X, nominal_field_Vcm=E0,
            drift_velocity_cm_us=0.11, dt_us=2.0,
        ), west_o, west_s

    @pytest.fixture
    def east_uniform_corrections(self, uniform_sides):
        """Corrections from uniform east-side field (should be ~zero).

        Uses a fine dt to keep discretisation error < atol.
        """
        east_ef, east_o, east_s = uniform_sides[0]
        return compute_drift_corrections(
            east_ef, east_o, east_s,
            anode_x_cm=-HALF_X, nominal_field_Vcm=E0,
            drift_velocity_cm_us=0.11, dt_us=0.2,
        )

    @pytest.fixture
    def west_uniform_corrections(self, uniform_sides):
        """Corrections from uniform west-side field (should be ~zero).

        Uses a fine dt to keep discretisation error < atol.
        """
        west_ef, west_o, west_s = uniform_sides[1]
        return compute_drift_corrections(
            west_ef, west_o, west_s,
            anode_x_cm=HALF_X, nominal_field_Vcm=E0,
            drift_velocity_cm_us=0.11, dt_us=0.2,
        )

    def test_output_shape(self, east_corrections):
        corr, _, _ = east_corrections
        assert corr.shape == (*GRID, 3)
        assert corr.dtype == np.float32

    def test_zero_distortion_gives_near_zero_corrections(
        self, east_uniform_corrections, west_uniform_corrections
    ):
        """Uniform field should produce near-zero corrections on both sides."""
        np.testing.assert_allclose(east_uniform_corrections, 0.0, atol=0.05,
                                   err_msg="East uniform corrections should be ~0")
        np.testing.assert_allclose(west_uniform_corrections, 0.0, atol=0.05,
                                   err_msg="West uniform corrections should be ~0")

    def test_anode_points_zero_correction(self, east_corrections, west_corrections):
        """Points at the anode (zero drift distance) should have zero correction."""
        east_corr, _, _ = east_corrections
        west_corr, _, _ = west_corrections

        # East anode = index 0, West anode = index -1
        np.testing.assert_allclose(east_corr[0, :, :, :], 0.0, atol=0.01,
                                   err_msg="East anode corrections should be zero")
        np.testing.assert_allclose(west_corr[-1, :, :, :], 0.0, atol=0.01,
                                   err_msg="West anode corrections should be zero")

    def test_transverse_squeeze_sign(self, east_corrections):
        """Positive y deposits should have negative Dy (image squeezed inward)."""
        corr, _, _ = east_corrections
        mid_z = GRID[2] // 2

        # Pick a mid-drift east point
        ix = GRID[0] // 2
        for iy in range(GRID[1] // 2 + 1, GRID[1]):
            dy_val = float(corr[ix, iy, mid_z, 1])
            assert dy_val <= 0.01, (
                f"Dy should be <= 0 for y>0, got {dy_val:.4f} at grid ({ix}, {iy})"
            )

    def test_dy_antisymmetric_in_y(self, east_corrections):
        """Dy at +y should be approximately -Dy at -y."""
        corr, _, _ = east_corrections
        cy = GRID[1] // 2
        cz = GRID[2] // 2
        ix = GRID[0] // 2

        for dy in range(1, cy):
            val_plus = float(corr[ix, cy + dy, cz, 1])
            val_minus = float(corr[ix, cy - dy, cz, 1])
            np.testing.assert_allclose(val_plus, -val_minus, atol=0.1,
                                       err_msg=f"Dy antisymmetry violated at dy={dy}")

    def test_corrections_bounded_by_detector_size(self, east_corrections):
        """Corrections must stay within physical limits.

        Note the units: channel 0 is a drift-TIME correction (μs), channels
        1-2 are transverse spatial displacements (cm) — the SCE-rework
        convention (see SCEOutputs.drift_time_corr_us / drift_yz_corr_cm and
        the consumer at tools/simulation.py). So bound each in its own units:
        the time excess cannot exceed the nominal full-drift time, and a
        transverse displacement cannot exceed the detector half-width.
        """
        corr, _, _ = east_corrections
        dt_corr_us = np.abs(corr[..., 0])
        yz_corr_cm = np.abs(corr[..., 1:])

        drift_velocity_cm_us = 0.11  # matches the east_corrections fixture
        nominal_full_drift_time_us = HALF_X / drift_velocity_cm_us

        assert dt_corr_us.max() < nominal_full_drift_time_us, (
            f"Max time correction {dt_corr_us.max():.2f} us exceeds nominal "
            f"full-drift time {nominal_full_drift_time_us:.2f} us"
        )
        assert yz_corr_cm.max() < HALF_X, (
            f"Max transverse correction {yz_corr_cm.max():.2f} cm exceeds "
            f"detector half-width {HALF_X:.2f} cm"
        )

    def test_cathode_has_largest_transverse_correction(self, east_corrections):
        """Points near cathode (max drift) should have the largest |Dy|."""
        corr, _, _ = east_corrections
        cy_off = GRID[1] // 2 + 2  # off-center y
        cz = GRID[2] // 2

        dy_near_anode = abs(float(corr[1, cy_off, cz, 1]))
        dy_near_cathode = abs(float(corr[-2, cy_off, cz, 1]))

        assert dy_near_cathode > dy_near_anode, (
            f"|Dy| near cathode ({dy_near_cathode:.4f}) should exceed "
            f"near anode ({dy_near_anode:.4f})"
        )

    def test_transverse_zero_at_center(self, east_corrections):
        """Dy and Dz should be zero at y=0, z=0 (transverse center).

        The transverse E-field is proportional to y/half_y and z/half_z,
        so electrons starting at the center have no transverse push.
        """
        corr, _, _ = east_corrections
        cy = GRID[1] // 2
        cz = GRID[2] // 2

        for ix in range(GRID[0]):
            np.testing.assert_allclose(
                float(corr[ix, cy, cz, 1]), 0.0, atol=0.05,
                err_msg=f"Dy should be ~0 at center, ix={ix}",
            )
            np.testing.assert_allclose(
                float(corr[ix, cy, cz, 2]), 0.0, atol=0.05,
                err_msg=f"Dz should be ~0 at center, ix={ix}",
            )

    def test_transverse_corrections_push_inward(self, east_corrections):
        """SCE transverse corrections must push electrons toward the center.

        For positive space charge, E_y > 0 at y > 0, so electrons (v = -mu*E)
        drift in -y direction. The correction Dy should be negative for y > 0
        and positive for y < 0.
        """
        corr, east_o, east_s = east_corrections
        cy = GRID[1] // 2
        cz = GRID[2] // 2

        # Check across all x positions (skip anode where correction ~0)
        for ix in range(2, GRID[0]):
            for iy in range(cy + 1, GRID[1]):
                dy_val = float(corr[ix, iy, cz, 1])
                assert dy_val <= 0.02, (
                    f"Dy should be <= 0 for y>0 (inward push), "
                    f"got {dy_val:.4f} at ({ix}, {iy})"
                )
            for iy in range(0, cy):
                dy_val = float(corr[ix, iy, cz, 1])
                assert dy_val >= -0.02, (
                    f"Dy should be >= 0 for y<0 (inward push), "
                    f"got {dy_val:.4f} at ({ix}, {iy})"
                )

    def test_east_west_correction_symmetry(self, east_corrections, west_corrections):
        """Mirror positions across cathode should have the same |corrections|.

        Dy, Dz should mirror in sign for y/z but have equal magnitude.
        Dx should have equal magnitude (both sides experience the same
        distortion strength).
        """
        east_corr, _, _ = east_corrections
        west_corr, _, _ = west_corrections
        Nx = GRID[0]

        for i in range(Nx):
            j = Nx - 1 - i
            # |Dx| should match
            np.testing.assert_allclose(
                np.abs(east_corr[i, :, :, 0]),
                np.abs(west_corr[j, :, :, 0]),
                atol=0.3,
                err_msg=f"|Dx| asymmetry at east[{i}] vs west[{j}]",
            )
            # |Dy| should match
            np.testing.assert_allclose(
                np.abs(east_corr[i, :, :, 1]),
                np.abs(west_corr[j, :, :, 1]),
                atol=0.3,
                err_msg=f"|Dy| asymmetry at east[{i}] vs west[{j}]",
            )

    def test_dx_positive_everywhere(self, east_corrections):
        """Delta_x should be positive (longer apparent drift) at all points.

        The linear distortion preserves the voltage integral (avg field = E0)
        but 1/E is convex, so by Jensen's inequality the time integral
        int(dx/v(x)) > L/v_nom. Electrons spend extra time in the weak-field
        region near the anode, making total drift time longer than nominal
        regardless of starting position.
        """
        corr, _, _ = east_corrections
        cy = GRID[1] // 2
        cz = GRID[2] // 2

        # Skip anode (index 0, ~zero correction) — check interior points
        for ix in range(2, GRID[0]):
            dx_val = float(corr[ix, cy, cz, 0])
            assert dx_val >= -0.05, (
                f"Dx should be >= 0 (slower net drift), got {dx_val:.4f} at ix={ix}"
            )

    def test_dx_peaks_in_interior(self, east_corrections):
        """Max delta_x should occur in the interior, not at the boundaries.

        The anode boundary has zero drift → zero correction. The cathode
        boundary traverses both weak-field (slow) and strong-field (fast)
        regions, which partially cancel. Mid-drift deposits only traverse
        the weak-field region and accumulate the largest time excess.
        """
        corr, _, _ = east_corrections
        cy = GRID[1] // 2
        cz = GRID[2] // 2

        dx_profile = np.array([float(corr[ix, cy, cz, 0]) for ix in range(GRID[0])])
        argmax = int(np.argmax(dx_profile))

        # Peak should not be at the anode (index 0) or cathode (index -1)
        assert 0 < argmax < GRID[0] - 1, (
            f"Dx peak at index {argmax}, expected interior. "
            f"Profile: anode={dx_profile[0]:.3f}, peak={dx_profile[argmax]:.3f}, "
            f"cathode={dx_profile[-1]:.3f}"
        )


# =========================================================================
# interpolate_map_3d
# =========================================================================

class TestInterpolateMap3d:
    """Tests for interpolate_map_3d."""

    @pytest.fixture
    def east_field_jax(self, toy_sides):
        """Channel-first JAX field and grid metadata for east side."""
        east_ef, east_o, east_s = toy_sides[0]
        field_T = jnp.moveaxis(jnp.array(east_ef), -1, 0)  # (3, Nx, Ny, Nz)
        return field_T, jnp.array(east_o, dtype=jnp.float32), jnp.array(east_s, dtype=jnp.float32)

    def test_output_shape(self, east_field_jax):
        field, origin, spacing = east_field_jax
        positions = jnp.array([[-50.0, 0.0, 0.0], [-80.0, 20.0, -30.0]])
        result = interpolate_map_3d(positions, field, origin, spacing)
        jax.block_until_ready(result)
        assert result.shape == (2, 3)

    def test_exact_grid_point_recovery(self, toy_sides):
        """Interpolation at exact grid nodes should recover stored values."""
        east_ef, east_o, east_s = toy_sides[0]
        field_T = jnp.moveaxis(jnp.array(east_ef), -1, 0)
        origin_j = jnp.array(east_o, dtype=jnp.float32)
        spacing_j = jnp.array(east_s, dtype=jnp.float32)

        test_indices = [(2, 3, 4), (5, 5, 5), (10, 0, 0), (18, 8, 9)]
        for ix, iy, iz in test_indices:
            pos_cm = east_o + np.array([ix, iy, iz]) * east_s
            result = interpolate_map_3d(
                jnp.array(pos_cm[None, :], dtype=jnp.float32),
                field_T, origin_j, spacing_j,
            )
            jax.block_until_ready(result)
            expected = east_ef[ix, iy, iz, :]
            np.testing.assert_allclose(
                np.array(result[0]), expected, rtol=1e-4,
                err_msg=f"Grid point ({ix},{iy},{iz}) mismatch",
            )

    def test_midpoint_linear_interpolation(self):
        """Querying halfway between two grid nodes should give their average."""
        field = np.zeros((2, 1, 1, 3), dtype=np.float32)
        field[0, 0, 0, 0] = 10.0
        field[1, 0, 0, 0] = 30.0

        field_T = jnp.moveaxis(jnp.array(field), -1, 0)
        origin = jnp.array([0.0, 0.0, 0.0])
        spacing = jnp.array([1.0, 1.0, 1.0])

        pos = jnp.array([[0.5, 0.0, 0.0]])
        result = interpolate_map_3d(pos, field_T, origin, spacing)
        jax.block_until_ready(result)
        np.testing.assert_allclose(float(result[0, 0]), 20.0, rtol=1e-5)

    def test_boundary_clamping(self, east_field_jax):
        """Positions far outside should clamp to boundary values."""
        field, origin, spacing = east_field_jax

        far_east = jnp.array([[-9999.0, 0.0, 0.0]])
        boundary = jnp.array([[-HALF_X, 0.0, 0.0]])

        result_far = interpolate_map_3d(far_east, field, origin, spacing)
        result_bound = interpolate_map_3d(boundary, field, origin, spacing)
        jax.block_until_ready(result_far)
        jax.block_until_ready(result_bound)

        np.testing.assert_allclose(
            np.array(result_far), np.array(result_bound), rtol=1e-5,
            err_msg="Out-of-bounds should clamp to nearest boundary",
        )

    def test_boundary_clamping_all_axes(self, east_field_jax):
        """Out-of-bounds in y and z should also clamp to edge values."""
        field, origin, spacing = east_field_jax

        # Far out in +y, should match the +y boundary
        far_y = jnp.array([[-50.0, 9999.0, 0.0]])
        at_y_edge = jnp.array([[-50.0, HALF_Y, 0.0]])
        np.testing.assert_allclose(
            np.array(interpolate_map_3d(far_y, field, origin, spacing)),
            np.array(interpolate_map_3d(at_y_edge, field, origin, spacing)),
            rtol=1e-5,
        )

        # Far out in -z, should match the -z boundary
        far_z = jnp.array([[-50.0, 0.0, -9999.0]])
        at_z_edge = jnp.array([[-50.0, 0.0, -HALF_Z]])
        np.testing.assert_allclose(
            np.array(interpolate_map_3d(far_z, field, origin, spacing)),
            np.array(interpolate_map_3d(at_z_edge, field, origin, spacing)),
            rtol=1e-5,
        )

    def test_clamping_returns_nonzero(self, east_field_jax):
        """Out-of-bounds queries should return the edge field value, not zero.

        This is the key property of mode='nearest' vs mode='constant'.
        A zero return would silently break recombination (division by ~0).
        """
        field, origin, spacing = east_field_jax
        far_out = jnp.array([[-9999.0, 9999.0, -9999.0]])
        result = interpolate_map_3d(far_out, field, origin, spacing)
        jax.block_until_ready(result)

        # Ex should be nonzero (it's the E-field at the corner of the grid)
        assert float(jnp.abs(result[0, 0])) > 0.1, (
            f"Out-of-bounds interpolation returned near-zero Ex={float(result[0, 0]):.4f}; "
            "mode='nearest' should return edge value, not zero"
        )

    def test_trilinear_not_nearest_neighbor(self):
        """Verify order=1 gives linear blending, not piecewise-constant.

        A 2-point grid with values 0 and 100: at x=0.25 trilinear gives 25,
        nearest-neighbor would give 0 (round to index 0).
        """
        field = np.zeros((2, 1, 1, 3), dtype=np.float32)
        field[0, 0, 0, 0] = 0.0
        field[1, 0, 0, 0] = 100.0

        field_T = jnp.moveaxis(jnp.array(field), -1, 0)
        origin = jnp.array([0.0, 0.0, 0.0])
        spacing = jnp.array([1.0, 1.0, 1.0])

        pos = jnp.array([[0.25, 0.0, 0.0]])
        result = interpolate_map_3d(pos, field_T, origin, spacing)
        jax.block_until_ready(result)

        # Trilinear: 25.0; nearest-neighbor would give 0.0
        np.testing.assert_allclose(float(result[0, 0]), 25.0, rtol=1e-5)

    def test_uniform_field_constant_everywhere(self, uniform_sides):
        """Interpolation of a uniform east field should return E0 everywhere."""
        east_ef, east_o, east_s = uniform_sides[0]
        field_T = jnp.moveaxis(jnp.array(east_ef), -1, 0)
        origin_j = jnp.array(east_o, dtype=jnp.float32)
        spacing_j = jnp.array(east_s, dtype=jnp.float32)

        rng = np.random.RandomState(42)
        positions = rng.uniform(
            [-HALF_X + 1, -HALF_Y + 1, -HALF_Z + 1],
            [-1.0, HALF_Y - 1, HALF_Z - 1],
            size=(50, 3),
        ).astype(np.float32)

        result = interpolate_map_3d(jnp.array(positions), field_T, origin_j, spacing_j)
        jax.block_until_ready(result)

        np.testing.assert_allclose(np.array(result[:, 0]), E0, rtol=1e-3)
        np.testing.assert_allclose(np.array(result[:, 1]), 0.0, atol=0.1)
        np.testing.assert_allclose(np.array(result[:, 2]), 0.0, atol=0.1)

    def test_single_point(self, east_field_jax):
        """N=1 query should work without error."""
        field, origin, spacing = east_field_jax
        pos = jnp.array([[-50.0, 0.0, 0.0]])
        result = interpolate_map_3d(pos, field, origin, spacing)
        jax.block_until_ready(result)
        assert result.shape == (1, 3)

    def test_jit_compatible(self, east_field_jax):
        """Should produce identical results when wrapped in jax.jit."""
        field, origin, spacing = east_field_jax
        positions = jnp.array([[-50.0, 20.0, -30.0], [-80.0, -10.0, 40.0]])

        result_eager = interpolate_map_3d(positions, field, origin, spacing)
        jax.block_until_ready(result_eager)

        @jax.jit
        def jitted_interp(pos):
            return interpolate_map_3d(pos, field, origin, spacing)

        result_jit = jitted_interp(positions)
        jax.block_until_ready(result_jit)

        np.testing.assert_allclose(
            np.array(result_jit), np.array(result_eager), rtol=1e-6,
        )




# =========================================================================
# create_single_interpolation_fn
# =========================================================================

class TestCreateSingleInterpolationFn:
    """Tests for create_single_interpolation_fn (per-volume)."""

    @pytest.fixture
    def vol0_fn(self, toy_sides):
        east_side, _ = toy_sides
        efield, origin, spacing = east_side
        efield_jax = jnp.moveaxis(jnp.array(efield), -1, 0)
        return create_single_interpolation_fn(
            efield_jax, jnp.array(origin, dtype=jnp.float32),
            jnp.array(spacing, dtype=jnp.float32))

    def test_east_positions_get_positive_ex(self, vol0_fn):
        pos = jnp.array([[-50.0, 0.0, 0.0]])
        result = vol0_fn(pos)
        assert float(result[0, 0]) > 0

    def test_jit_compatible(self, vol0_fn):
        pos = jnp.array([[-50.0, 0.0, 0.0]])
        eager = vol0_fn(pos)
        jitted = jax.jit(vol0_fn)
        result = jitted(pos)
        np.testing.assert_allclose(np.array(result), np.array(eager), rtol=1e-6)


# =========================================================================
# save_sce_data / load_sce_data (volume-based format)
# =========================================================================

class TestSCESaveLoad:
    """Tests for per-volume SCE save/load."""

    def test_round_trip(self, toy_sides, tmp_path):
        east_side, west_side = toy_sides
        east_e, east_o, east_s = east_side
        west_e, west_o, west_s = west_side

        path = str(tmp_path / "sce_test.h5")
        save_sce_data(path, [
            {"efield_map": east_e, "drift_correction_map": east_e * 0.01,
             "origin_cm": east_o, "spacing_cm": east_s},
            {"efield_map": west_e, "drift_correction_map": west_e * 0.01,
             "origin_cm": west_o, "spacing_cm": west_s},
        ])

        loaded = load_sce_data(path)
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded[0]["efield_map"], east_e, rtol=1e-6)
        np.testing.assert_allclose(loaded[1]["efield_map"], west_e, rtol=1e-6)

    def test_hdf5_groups_named_volume(self, toy_sides, tmp_path):
        east_side, west_side = toy_sides
        east_e, east_o, east_s = east_side
        west_e, west_o, west_s = west_side

        path = str(tmp_path / "sce_groups.h5")
        save_sce_data(path, [
            {"efield_map": east_e, "drift_correction_map": east_e * 0.01,
             "origin_cm": east_o, "spacing_cm": east_s},
            {"efield_map": west_e, "drift_correction_map": west_e * 0.01,
             "origin_cm": west_o, "spacing_cm": west_s},
        ])

        import h5py
        with h5py.File(path, "r") as f:
            assert "volume_0" in f
            assert "volume_1" in f
            assert "efield_map" in f["volume_0"]


# =========================================================================
# End-to-end: generate → save → load → interpolate
# =========================================================================

class TestEndToEnd:
    """Generate toy maps, save, reload, query via single interpolation fn."""

    def test_generate_save_load_query(self, toy_sides, tmp_path):
        east_side, west_side = toy_sides
        east_e, east_o, east_s = east_side
        west_e, west_o, west_s = west_side

        path = str(tmp_path / "sce_e2e.h5")
        save_sce_data(path, [
            {"efield_map": east_e, "drift_correction_map": east_e * 0.01,
             "origin_cm": east_o, "spacing_cm": east_s},
            {"efield_map": west_e, "drift_correction_map": west_e * 0.01,
             "origin_cm": west_o, "spacing_cm": west_s},
        ])

        loaded = load_sce_data(path)

        # Build per-volume interpolation functions
        for v in range(2):
            d = loaded[v]
            efield_jax = jnp.moveaxis(jnp.array(d["efield_map"]), -1, 0)
            fn = create_single_interpolation_fn(
                efield_jax,
                jnp.array(d["origin_cm"], dtype=jnp.float32),
                jnp.array(d["spacing_cm"], dtype=jnp.float32))

            # Query a point inside this volume
            if v == 0:
                pos = jnp.array([[-50.0, 0.0, 0.0]])
                E = fn(pos)
                assert float(E[0, 0]) > 0, "East Ex should be positive"
            else:
                pos = jnp.array([[50.0, 0.0, 0.0]])
                E = fn(pos)
                assert float(E[0, 0]) < 0, "West Ex should be negative"
