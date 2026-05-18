"""Tests for post-response pixel truth labeling.

Validates that:
1. The pixel path produces signal and truth from a single response pass
2. Signal values are physically reasonable (positive, bounded)
3. Per-group truth sums to the total signal
4. Track hits finalization produces valid labeled hits
5. Output formats (sparse, dense) work correctly
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import gc

from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import load_event
from tools.config import create_track_hits_config
from tools.output import to_sparse, to_dense, _detect_format
from tools.track_hits import finalize_track_hits


CONFIG_PATH = "config/cubic_pixel_config.yaml"
DATA_PATH = "out.h5"
EVENT_IDX = 7
TOTAL_PAD = 200_000
RESPONSE_CHUNK_SIZE = 50_000


@pytest.fixture(scope="module")
def simulation_result():
    """Run one pixel simulation and cache the result for all tests."""
    detector_config = generate_detector(CONFIG_PATH)
    jax.clear_caches()
    gc.collect()

    track_config = create_track_hits_config()
    simulator = DetectorSimulator(
        detector_config,
        include_track_hits=True,
        total_pad=TOTAL_PAD,
        response_chunk_size=RESPONSE_CHUNK_SIZE,
        track_config=track_config,
    )
    cfg = simulator.config

    deposits = load_event(DATA_PATH, cfg, event_idx=EVENT_IDX)
    simulator.warm_up()

    response_signals, track_hits_raw, deposits = simulator.process_event(
        deposits, key=jax.random.PRNGKey(42))

    return {
        "simulator": simulator,
        "cfg": cfg,
        "response_signals": response_signals,
        "track_hits_raw": track_hits_raw,
        "deposits": deposits,
    }


class TestPixelSignalFormat:
    def test_signal_is_pixel_sparse(self, simulation_result):
        for key, sig in simulation_result["response_signals"].items():
            assert isinstance(sig, dict), f"{key}: expected dict, got {type(sig)}"
            assert "pixel_y" in sig
            assert "pixel_z" in sig
            assert "time" in sig
            assert "values" in sig

    def test_detect_format(self, simulation_result):
        for sig in simulation_result["response_signals"].values():
            assert _detect_format(sig) == "pixel_sparse"

    def test_signal_nonempty(self, simulation_result):
        for key, sig in simulation_result["response_signals"].items():
            assert len(sig["values"]) > 0, f"{key}: empty signal"

    def test_signal_values_nonzero(self, simulation_result):
        """Response kernel can produce negative values (bipolar induction)."""
        for key, sig in simulation_result["response_signals"].items():
            vals = np.asarray(sig["values"])
            assert np.all(vals != 0), f"{key}: has zero values"

    def test_pixel_indices_in_bounds(self, simulation_result):
        cfg = simulation_result["cfg"]
        for (v, p), sig in simulation_result["response_signals"].items():
            vol = cfg.volumes[v]
            num_py, num_pz = vol.pixel_shape
            py = np.asarray(sig["pixel_y"])
            pz = np.asarray(sig["pixel_z"])
            t = np.asarray(sig["time"])
            assert np.all(py >= 0) and np.all(py < num_py), f"({v},{p}): py out of bounds"
            assert np.all(pz >= 0) and np.all(pz < num_pz), f"({v},{p}): pz out of bounds"
            assert np.all(t >= 0) and np.all(t < cfg.num_time_steps), f"({v},{p}): time out of bounds"


class TestTrackHits:
    def test_track_hits_present(self, simulation_result):
        th = simulation_result["track_hits_raw"]
        assert len(th) > 0

    def test_track_hits_have_group_to_track(self, simulation_result):
        th = simulation_result["track_hits_raw"]
        assert "group_to_track" in th

    def test_track_hits_count_positive(self, simulation_result):
        th = simulation_result["track_hits_raw"]
        for key, raw in th.items():
            if not isinstance(key, tuple):
                continue
            count = int(raw[4])
            assert count > 0, f"{key}: zero track hits"

    def test_finalize_track_hits(self, simulation_result):
        sim = simulation_result["simulator"]
        th_raw = simulation_result["track_hits_raw"]

        # Copy since finalize_track_hits pops 'group_to_track'
        th_copy = dict(th_raw)
        result = sim.finalize_track_hits(th_copy)

        for key, labeled in result.items():
            assert "labeled_hits" in labeled
            assert "num_labeled" in labeled
            assert labeled["num_labeled"] > 0, f"{key}: no labeled hits"


class TestSignalTruthConsistency:
    def test_group_sum_equals_signal(self, simulation_result):
        """Per-group truth charges at each (pixel, time) should sum to the
        total signal. This is the fundamental invariant of the unified path."""
        cfg = simulation_result["cfg"]
        response_signals = simulation_result["response_signals"]
        track_hits_raw = simulation_result["track_hits_raw"]

        num_pz = cfg.volumes[0].pixel_shape[1]

        for key, raw in track_hits_raw.items():
            if not isinstance(key, tuple):
                continue
            v, p = key
            state_sk, state_tk, state_gk, state_ch, state_count, _ = raw
            count = int(state_count)
            if count == 0:
                continue

            sks = np.asarray(state_sk[:count])
            tks = np.asarray(state_tk[:count])
            chs = np.asarray(state_ch[:count])

            # Sum over groups at each (pixel, time) — same logic as process_event
            max_t = cfg.num_time_steps
            composite = sks.astype(np.int64) * max_t + tks.astype(np.int64)
            order = np.argsort(composite, kind="stable")
            s_comp = composite[order]
            s_chs = chs[order]
            s_sks = sks[order]
            s_tks = tks[order]
            boundaries = np.ones(count, dtype=bool)
            boundaries[1:] = s_comp[1:] != s_comp[:-1]
            starts = np.where(boundaries)[0]
            summed = np.add.reduceat(s_chs, starts)
            out_sk = s_sks[starts]
            out_tk = s_tks[starts]

            mask = summed != 0
            truth_py = (out_sk[mask] // num_pz).astype(np.int32)
            truth_pz = (out_sk[mask] % num_pz).astype(np.int32)
            truth_t = out_tk[mask].astype(np.int32)
            truth_v = summed[mask].astype(np.float32)

            # Compare with response_signals
            sig = response_signals[(v, p)]
            sig_py = np.asarray(sig["pixel_y"])
            sig_pz = np.asarray(sig["pixel_z"])
            sig_t = np.asarray(sig["time"])
            sig_v = np.asarray(sig["values"])

            assert len(truth_py) == len(sig_py), \
                f"({v},{p}): truth has {len(truth_py)} entries, signal has {len(sig_py)}"

            # Sort both by (py, pz, t) for comparison
            truth_order = np.lexsort((truth_t, truth_pz, truth_py))
            sig_order = np.lexsort((sig_t, sig_pz, sig_py))

            np.testing.assert_array_equal(truth_py[truth_order], sig_py[sig_order])
            np.testing.assert_array_equal(truth_pz[truth_order], sig_pz[sig_order])
            np.testing.assert_array_equal(truth_t[truth_order], sig_t[sig_order])
            np.testing.assert_allclose(
                truth_v[truth_order], sig_v[sig_order], rtol=1e-4, atol=1e-5,
                err_msg=f"({v},{p}): signal values don't match group sums")


class TestNegativeValues:
    def test_signal_has_negatives(self, simulation_result):
        """Response kernel produces bipolar induction — signal should have negatives."""
        for key, sig in simulation_result["response_signals"].items():
            vals = np.asarray(sig["values"])
            n_neg = np.sum(vals < 0)
            assert n_neg > 0, f"{key}: no negative signal values (bipolar response expected)"

    def test_truth_has_negatives(self, simulation_result):
        """Per-group entries should include negative response contributions."""
        th = simulation_result["track_hits_raw"]
        for key, raw in th.items():
            if not isinstance(key, tuple):
                continue
            ch = np.asarray(raw[3][:int(raw[4])])
            n_neg = np.sum(ch < 0)
            assert n_neg > 0, f"{key}: no negative truth entries"


class TestOutputConversions:
    def test_to_sparse_passthrough(self, simulation_result):
        cfg = simulation_result["cfg"]
        response_signals = simulation_result["response_signals"]
        sparse = to_sparse(response_signals, cfg, threshold_adc=0.0)

        for key in response_signals:
            assert key in sparse
            orig = response_signals[key]
            sp = sparse[key]
            assert len(sp["values"]) == len(orig["values"])

    def test_to_sparse_threshold(self, simulation_result):
        cfg = simulation_result["cfg"]
        response_signals = simulation_result["response_signals"]
        threshold = 500.0 / cfg.electrons_per_adc
        sparse = to_sparse(response_signals, cfg, threshold_adc=threshold)

        for key, sp in sparse.items():
            assert np.all(np.abs(sp["values"]) >= threshold)

    def test_to_dense_roundtrip(self, simulation_result):
        """sparse → dense → sparse should preserve nonzero entries."""
        cfg = simulation_result["cfg"]
        response_signals = simulation_result["response_signals"]
        dense = to_dense(response_signals, cfg)

        for (v, p), d in dense.items():
            vol = cfg.volumes[v]
            num_py, num_pz = vol.pixel_shape
            assert d.shape == (num_py, num_pz, cfg.num_time_steps)
            assert d.dtype == np.float32
            n_nonzero = np.count_nonzero(d)
            n_sparse = len(response_signals[(v, p)]["values"])
            assert n_nonzero == n_sparse, \
                f"({v},{p}): dense has {n_nonzero} nonzero, sparse has {n_sparse}"


class TestDepositsOutput:
    def test_qs_fractions_filled(self, simulation_result):
        deposits = simulation_result["deposits"]
        for v, vol in enumerate(deposits.volumes):
            qs = np.asarray(vol.qs_fractions)
            n = vol.n_actual
            valid_qs = qs[:n]
            assert np.all(valid_qs >= 0), f"vol {v}: negative qs_fractions"
            assert np.all(valid_qs <= 1.0 + 1e-6), f"vol {v}: qs_fractions > 1"

    def test_charges_filled(self, simulation_result):
        deposits = simulation_result["deposits"]
        for v, vol in enumerate(deposits.volumes):
            charges = np.asarray(vol.charge)
            n = vol.n_actual
            assert np.any(charges[:n] > 0), f"vol {v}: no nonzero charges"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
