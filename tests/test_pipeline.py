"""
Production pipeline integration tests (synthetic deposits).

One dense + track-hits simulator on a compact detector is shared by the module,
plus one bucketed simulator for the dense == bucketed comparison.

Tests: output layout, track-hit charge and labels, qs_fractions, SimParams response,
recompilation, warm_up, volume mirror symmetry, bucketed equivalence,
hand-placed deposits on the full-size detector.
"""

import os
import numpy as np
import jax.numpy as jnp
import pytest

from tools.config import create_track_hits_config
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import build_deposit_data
from tools.output import to_dense


CONFIG_PATH = 'config/cubic_wireplane_config.yaml'
PAD = 64
PLANES = [(v, p) for v in range(2) for p in range(3)]


# ---------------------------------------------------------------------------
# Module-scoped fixtures (1 JIT compile for main simulator)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim(compact_detector):
    """Dense + track_hits simulator. Shared across all tests."""
    return DetectorSimulator(
        compact_detector(), total_pad=PAD, response_chunk_size=PAD // 2,
        include_track_hits=True,
        track_config=create_track_hits_config(
            max_keys=50_000, hits_chunk_size=PAD // 2, maxg=64, inter_thresh=0.01))


@pytest.fixture(scope="module")
def deposits(sim, synthetic_tracks):
    return synthetic_tracks(sim.config)


@pytest.fixture(scope="module")
def sim_results(sim, deposits):
    """Run once, reuse signals/track_hits/qs across tests."""
    return sim.process_event(deposits)


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

class TestOutputLayout:

    def test_planes_and_shapes(self, sim, sim_results):
        """Exactly one (num_wires, num_time) signal and one hit tuple per plane."""
        signals, hits, _ = sim_results
        cfg = sim.config
        assert cfg.volumes[0].num_wires[0] != cfg.volumes[0].num_wires[2]
        assert set(signals) == set(PLANES)
        assert set(hits) == set(PLANES) | {'group_to_track'}
        for v, p in PLANES:
            assert signals[(v, p)].shape == (
                cfg.volumes[v].num_wires[p], cfg.num_time_steps)


# ---------------------------------------------------------------------------
# Track hits
# ---------------------------------------------------------------------------

class TestTrackHits:

    @pytest.mark.parametrize("vol,plane", PLANES)
    def test_charge_per_track(self, sim, sim_results, vol, plane):
        """Hits of each track on each plane sum to its attenuated charge."""
        _, hits, filled = sim_results
        sk, tk, gk, ch, count, _ = hits[(vol, plane)]
        count = int(count)
        assert 0 < count < sim.config.track_hits.max_keys
        hit_tracks = np.asarray(hits['group_to_track'][vol])[np.asarray(gk[:count])]
        ch = np.asarray(ch[:count], dtype=np.float64)

        d = filled.volumes[vol]
        n = int(d.n_actual)
        geom, params = sim.config.volumes[vol], sim.default_sim_params
        dist = np.asarray(d.positions_mm[:n, 0]) / 10 - geom.plane_distances_cm[plane]
        arriving = np.asarray(d.charge[:n], np.float64) * np.exp(
            -dist / float(params.velocity_cm_us) / float(params.lifetime_us))
        tracks = np.asarray(d.track_ids[:n])
        assert set(hit_tracks) == set(tracks) == {1, 2, 3}
        for t in (1, 2, 3):
            assert ch[hit_tracks == t].sum() == pytest.approx(
                arriving[tracks == t].sum(), rel=1e-2)

    def test_qs_fractions(self, sim, sim_results):
        """qs_fractions is each deposit's share of its group's charge, 0 on padding."""
        _, _, filled = sim_results
        for d in filled.volumes:
            n = int(d.n_actual)
            assert 0 < n < PAD
            q, g = np.asarray(d.charge[:n]), np.asarray(d.group_ids[:n])
            assert np.all(q > 0)
            total = np.bincount(g, weights=q)
            assert np.sum(np.bincount(g) > 1) >= 2     # several multi-deposit groups
            np.testing.assert_allclose(np.asarray(d.qs_fractions[:n]), q / total[g],
                                       rtol=1e-5)
            assert np.all(np.asarray(d.qs_fractions[n:]) == 0)


# ---------------------------------------------------------------------------
# SimParams
# ---------------------------------------------------------------------------

class TestSimParams:

    def test_velocity_shifts_arrival_time(self, sim):
        """A slower drift delays the Y pulse by d(1/v2 - 1/v1) and attenuates it."""
        params = sim.default_sim_params
        x_mm, v2 = -500.0, 0.17
        deps = build_deposit_data(
            np.array([[x_mm, 0.0, 0.0]], np.float32), np.array([2.0], np.float32),
            np.array([3.0], np.float32), sim.config)
        pulse = []
        for p in (params, params._replace(velocity_cm_us=jnp.array(v2))):
            y = np.asarray(sim.process_event(deps, sim_params=p)[0][(0, 2)], np.float64)
            t = np.arange(y.shape[1])
            pulse.append((y.sum(), (y.sum(0) * t).sum() / y.sum()))
        (q1, t1), (q2, t2) = pulse
        dt_us = (216.0 + x_mm / 10) * (1 / v2 - 1 / float(params.velocity_cm_us))
        assert (t2 - t1) * sim.config.time_step_us == pytest.approx(dt_us, abs=0.05)
        assert q2 / q1 == pytest.approx(
            np.exp(-dt_us / float(params.lifetime_us)), rel=1e-4)

    def test_defaults_match_config(self, sim):
        """Default SimParams are the YAML values in cm, us units."""
        p = sim.default_sim_params
        got = [p.velocity_cm_us, p.lifetime_us, p.diffusion_trans_cm2_us,
               p.diffusion_long_cm2_us, p.recomb_params.alpha,
               p.recomb_params.beta_90, p.recomb_params.R]
        np.testing.assert_allclose(
            np.array(got, np.float64),
            [0.16, 1.0e4, 12.0e-6, 7.2e-6, 0.904, 0.204, 1.25], rtol=1e-6)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

class TestCompilation:

    def test_params_change_without_recompile(self, sim, deposits, sim_results):
        n = sim._calculator_jit._cache_size()
        assert n == 1
        p = sim.default_sim_params._replace(velocity_cm_us=jnp.array(0.17),
                                            lifetime_us=jnp.array(5000.0))
        sim.process_event(deposits, sim_params=p)
        assert sim._calculator_jit._cache_size() == n

    def test_warm_up_covers_process_event(self, compact_detector, synthetic_tracks):
        """After warm_up() the first real event triggers no compilation."""
        fresh = DetectorSimulator(
            compact_detector(), total_pad=PAD, response_chunk_size=PAD,
            include_track_hits=False)
        assert fresh._calculator_jit._cache_size() == 0
        fresh.warm_up()
        assert fresh._calculator_jit._cache_size() == 1
        fresh.process_event(synthetic_tracks(fresh.config))
        assert fresh._calculator_jit._cache_size() == 1


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

class TestVolumes:

    def test_mirrored_deposits(self, sim):
        """x -> -x moves the signal to the other volume unchanged, the rest is zero."""
        rng = np.random.RandomState(5)
        pos = rng.uniform([200, -250, -250], [2000, 250, 250], (60, 3)).astype(np.float32)
        out = {}
        for sign in (-1, 1):
            p = pos * np.array([sign, 1, 1], np.float32)
            deps = build_deposit_data(p, np.full(60, 2.0, np.float32),
                                      np.full(60, 3.0, np.float32), sim.config)
            out[sign], _, _ = sim.process_event(deps)
        for p in range(3):
            east, west = np.asarray(out[-1][(0, p)]), np.asarray(out[1][(1, p)])
            assert np.abs(east).max() > 1.0
            # x_local = +-(x_anode - x) rounds identically for both signs
            np.testing.assert_array_equal(east, west)
            assert not np.any(np.asarray(out[-1][(1, p)]))
            assert not np.any(np.asarray(out[1][(0, p)]))


# ---------------------------------------------------------------------------
# Bucketed equivalence (2nd JIT compile)
# ---------------------------------------------------------------------------

class TestBucketed:

    def test_dense_equals_bucketed(self, compact_detector, deposits, sim_results):
        signals_dense, _, _ = sim_results
        sim_b = DetectorSimulator(
            compact_detector(), total_pad=PAD, response_chunk_size=PAD // 2,
            use_bucketed=True, max_active_buckets=1000, include_track_hits=False)
        signals_b, _, _ = sim_b.process_event(deposits)
        dense_b = to_dense(signals_b, sim_b.config)
        for key in PLANES:
            d, b = np.asarray(signals_dense[key]), np.asarray(dense_b[key])
            peak = np.abs(d).max()
            assert peak > 1.0
            assert np.abs(d - b).max() < 1e-5 * peak, key


# ---------------------------------------------------------------------------
# Synthetic deterministic pipeline (no external data files)
# ---------------------------------------------------------------------------

RESPONSE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'tools', 'responses')
has_kernels = all(
    os.path.exists(os.path.join(RESPONSE_PATH, f'{p}_plane_kernel.npz'))
    for p in ['U', 'V', 'Y'])


@pytest.mark.requires_kernels
@pytest.mark.skipif(not has_kernels, reason="Response kernel files not found")
class TestSyntheticPipeline:
    """Full pipeline test with hand-placed deposits.

    Five deposits placed at known positions in volume 0 (east).
    Validates physics invariants rather than comparing to frozen snapshots.
    """

    @pytest.fixture(scope="class")
    def synthetic_sim(self):
        det = generate_detector(CONFIG_PATH)
        return DetectorSimulator(
            det, total_pad=500, response_chunk_size=500,
            include_track_hits=False, include_intrinsic_noise=False,
            include_electronics=False, include_digitize=False)

    @pytest.fixture(scope="class")
    def synthetic_results(self, synthetic_sim):
        positions_mm = np.array([
            [-500.0, 0.0, 0.0],
            [-800.0, 0.0, 0.0],
            [-1200.0, 0.0, 0.0],
            [-500.0, 100.0, 100.0],
            [-500.0, -100.0, -100.0],
        ], dtype=np.float32)
        de = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        dx = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)

        deposits = build_deposit_data(positions_mm, de, dx, synthetic_sim.config)
        signals, _, _ = synthetic_sim.process_event(deposits)
        return signals, deposits

    def test_east_has_signal(self, synthetic_results, synthetic_sim):
        """Volume 0 (east) should have nonzero signal — deposits are at x<0."""
        signals, _ = synthetic_results
        cfg = synthetic_sim.config
        for pi in range(cfg.volumes[0].n_planes):
            total = float(jnp.sum(jnp.abs(signals[(0, pi)])))
            assert total > 0, f"East plane {pi} should have signal"

    def test_west_is_empty(self, synthetic_results, synthetic_sim):
        """Volume 1 (west) should have zero signal — no deposits at x>0."""
        signals, _ = synthetic_results
        cfg = synthetic_sim.config
        for pi in range(cfg.volumes[1].n_planes):
            total = float(jnp.sum(jnp.abs(signals[(1, pi)])))
            assert total < 1e-3, f"West plane {pi} should be empty"

    def test_collection_plane_positive(self, synthetic_results):
        """Y-plane (collection, plane 2) signal peak should be positive."""
        signals, _ = synthetic_results
        y_signal = np.asarray(signals[(0, 2)])
        assert float(np.max(y_signal)) > 0, "Collection plane peak should be positive"

    def test_induction_plane_bipolar(self, synthetic_results):
        """U-plane (induction, plane 0) signal should have both signs."""
        signals, _ = synthetic_results
        u_signal = np.asarray(signals[(0, 0)])
        assert float(np.max(u_signal)) > 0, "Induction should have positive values"
        assert float(np.min(u_signal)) < 0, "Induction should have negative values"

    def test_far_deposits_more_diffused(self, synthetic_results, synthetic_sim):
        """Deposits further from anode should produce broader signals on Y-plane."""
        signals, _ = synthetic_results
        y_signal = np.asarray(signals[(0, 2)])

        # Find the wire with peak signal — should be near wire for z=0
        wire_sums = np.sum(np.abs(y_signal), axis=1)
        peak_wire = int(np.argmax(wire_sums))

        # Time profile at peak wire should span multiple bins (diffusion)
        time_profile = y_signal[peak_wire]
        active_bins = np.sum(np.abs(time_profile) > 0.01 * np.max(np.abs(time_profile)))
        assert active_bins > 3, "Signal should be spread across multiple time bins"

    def test_deterministic(self, synthetic_sim):
        """Same deposits should produce identical output (no stochastic noise)."""
        positions_mm = np.array([[-500.0, 0.0, 0.0]], dtype=np.float32)
        de = np.array([2.0], dtype=np.float32)
        dx = np.array([0.3], dtype=np.float32)

        deps = build_deposit_data(positions_mm, de, dx, synthetic_sim.config)
        sigs1, _, _ = synthetic_sim.process_event(deps)
        sigs2, _, _ = synthetic_sim.process_event(deps)

        for key in sigs1:
            np.testing.assert_array_equal(
                np.asarray(sigs1[key]), np.asarray(sigs2[key]),
                err_msg=f"Plane {key} not deterministic")
