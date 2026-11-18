"""Smoke tests for examples/ (run as subprocesses from the repo root).

The MCS forward-model checks run by default (no detector). The end-to-end example
runs are slow (minutes, GPU recommended) and run only with JAXTPC_SLOW_TESTS=1.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slow = pytest.mark.skipif(os.environ.get('JAXTPC_SLOW_TESTS') != '1',
                          reason='slow example run; set JAXTPC_SLOW_TESTS=1')


def _run(args, tmp_path, timeout=1800):
    env = dict(os.environ, JAXTPC_EXAMPLES_OUT=str(tmp_path), MPLBACKEND='Agg')
    r = subprocess.run([sys.executable, '-m', *args], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, (r.stdout[-3000:] + r.stderr[-3000:])
    return r.stdout


def test_mcs_forward_model(tmp_path):
    _run(['examples.mcs.validate_forward'], tmp_path)


@slow
def test_muon_track_fit(tmp_path):
    _run(['examples.muon.run', 'A', '--steps', '20'], tmp_path)
    assert (tmp_path / 'optimization_A.npz').exists()


@slow
def test_muon_event_display(tmp_path):
    _run(['examples.muon.event_display'], tmp_path)


@slow
def test_mcs_globals_fit(tmp_path):
    _run(['examples.mcs.run', '--mode', 'globals-only', '--steps', '5'], tmp_path)


@slow
def test_segments_full_mode_small(tmp_path):
    # n_seg below the default max_reloc (1000) used to crash in relocate_segments
    _run(['examples.segments.run', '--data', 'muon.h5', '--n-seg', '500',
          '--steps', '30', '--mode', 'full', '--plot-every', '1000'], tmp_path)
