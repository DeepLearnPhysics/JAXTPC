"""
Measure charge/signal loss from thresholding at various levels.

Two independent thresholds:
  1. corr_threshold (electrons): filters correspondence entries in CSR encoding.
     Charge loss = sum(dropped charge) / sum(total charge).
  2. threshold_adc (ADC): filters sparse signal output.
     Signal loss = sum(|dropped signal|) / sum(|total signal|).

Neither requires re-running the simulation — one sim run, then sweep in post-processing.

Usage:
    python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml
    python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml --events 5
    python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml --mode corr
    python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml --mode adc
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np

from tools.geometry import generate_detector
from tools.config import create_track_hits_config
from tools.simulation import DetectorSimulator
from tools.loader import load_event
from tools.output import to_dense

from profiler.timing import sync_result


CORR_THRESHOLDS = [0, 1, 5, 10, 25, 50, 100, 200, 500]
ADC_THRESHOLDS = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]


def analyze_corr_threshold(track_hits_raw, cfg, thresholds):
    """Sweep corr_threshold values on raw track_hits output.

    For each threshold, measure how much charge is dropped from correspondence.
    """
    results = []
    for thresh in thresholds:
        total_charge = 0.0
        kept_charge = 0.0
        total_entries = 0
        kept_entries = 0

        for (vol_idx, plane_idx), raw in track_hits_raw.items():
            if not isinstance((vol_idx, plane_idx), tuple):
                continue
            sk, tk, gk, ch, count, _ = raw
            P = int(count)
            if P == 0:
                continue
            chs = np.asarray(ch[:P], dtype=np.float32)

            total_charge += float(np.sum(chs))
            total_entries += P

            if thresh > 0:
                keep = chs > thresh
                kept_charge += float(np.sum(chs[keep]))
                kept_entries += int(np.sum(keep))
            else:
                kept_charge += float(np.sum(chs))
                kept_entries += P

        lost_charge = total_charge - kept_charge
        frac_lost = lost_charge / total_charge if total_charge > 0 else 0
        frac_entries = 1.0 - (kept_entries / total_entries) if total_entries > 0 else 0

        results.append({
            'threshold': thresh,
            'total_charge': total_charge,
            'kept_charge': kept_charge,
            'charge_lost_frac': frac_lost,
            'total_entries': total_entries,
            'kept_entries': kept_entries,
            'entries_dropped_frac': frac_entries,
        })

    return results


def analyze_adc_threshold(response_signals, cfg, thresholds):
    """Sweep threshold_adc values on dense signals.

    For each threshold, measure how much signal is lost.
    """
    # Convert to dense once
    dense = to_dense(response_signals, cfg)

    results = []
    for thresh in thresholds:
        total_signal = 0.0
        kept_signal = 0.0
        total_nonzero = 0
        kept_nonzero = 0

        for (vol_idx, plane_idx), arr in dense.items():
            arr_np = np.asarray(arr)
            abs_arr = np.abs(arr_np)

            total_signal += float(np.sum(abs_arr))
            nz = abs_arr > 0
            total_nonzero += int(np.sum(nz))

            if thresh > 0:
                keep = abs_arr >= thresh
                kept_signal += float(np.sum(abs_arr[keep]))
                kept_nonzero += int(np.sum(keep))
            else:
                kept_signal += float(np.sum(abs_arr[nz]))
                kept_nonzero += int(np.sum(nz))

        lost_signal = total_signal - kept_signal
        frac_lost = lost_signal / total_signal if total_signal > 0 else 0
        frac_entries = 1.0 - (kept_nonzero / total_nonzero) if total_nonzero > 0 else 0

        results.append({
            'threshold': thresh,
            'total_signal': total_signal,
            'kept_signal': kept_signal,
            'signal_lost_frac': frac_lost,
            'total_bins': total_nonzero,
            'kept_bins': kept_nonzero,
            'bins_dropped_frac': frac_entries,
        })

    return results


def print_corr_results(results):
    print(f'\n  {"Thresh (e-)":>12} {"Kept Charge":>14} {"Lost %":>8} '
          f'{"Kept Entries":>14} {"Dropped %":>10}')
    print(f'  {"─" * 62}')
    for r in results:
        print(f'  {r["threshold"]:>12.0f} {r["kept_charge"]:>14,.0f} '
              f'{r["charge_lost_frac"]*100:>7.3f}% '
              f'{r["kept_entries"]:>14,} {r["entries_dropped_frac"]*100:>9.1f}%')


def print_adc_results(results):
    print(f'\n  {"Thresh (ADC)":>12} {"Kept Signal":>14} {"Lost %":>8} '
          f'{"Kept Bins":>14} {"Dropped %":>10}')
    print(f'  {"─" * 62}')
    for r in results:
        print(f'  {r["threshold"]:>12.1f} {r["kept_signal"]:>14,.0f} '
              f'{r["signal_lost_frac"]*100:>7.3f}% '
              f'{r["kept_bins"]:>14,} {r["bins_dropped_frac"]*100:>9.1f}%')


def main():
    parser = argparse.ArgumentParser(
        description='Analyze charge/signal loss from thresholding')
    parser.add_argument('--data', required=True, help='Input HDF5 file')
    parser.add_argument('--config', required=True, help='Detector geometry YAML')
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--events', type=int, default=1,
                        help='Number of events to average over (default: 1)')
    parser.add_argument('--total-pad', type=int, default=500_000)
    parser.add_argument('--response-chunk', type=int, default=50_000)
    parser.add_argument('--hits-chunk', type=int, default=25_000)
    parser.add_argument('--max-keys', type=int, default=4_000_000)
    parser.add_argument('--mode', choices=['both', 'corr', 'adc'], default='both',
                        help='Which threshold to sweep (default: both)')
    parser.add_argument('--corr-values', type=float, nargs='+', default=None,
                        help='Custom corr_threshold values to test')
    parser.add_argument('--adc-values', type=float, nargs='+', default=None,
                        help='Custom threshold_adc values to test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-config', default=None,
                        help='Save chosen thresholds to production config YAML')
    parser.add_argument('--save-corr', type=float, default=None,
                        help='corr_threshold value to save (must be in tested values)')
    parser.add_argument('--save-adc', type=float, default=None,
                        help='threshold_adc value to save (must be in tested values)')

    args = parser.parse_args()

    corr_thresholds = args.corr_values if args.corr_values else CORR_THRESHOLDS
    adc_thresholds = args.adc_values if args.adc_values else ADC_THRESHOLDS

    detector_config = generate_detector(args.config)

    do_corr = args.mode in ('both', 'corr')
    do_adc = args.mode in ('both', 'adc')

    track_config = create_track_hits_config(
        max_keys=args.max_keys, hits_chunk_size=args.hits_chunk
    ) if do_corr else None

    sim = DetectorSimulator(
        detector_config,
        total_pad=args.total_pad,
        response_chunk_size=args.response_chunk,
        include_track_hits=do_corr,
        track_config=track_config,
    )

    print('=' * 70)
    print(' JAXTPC — Threshold Analysis')
    print('=' * 70)
    print(f'  Data:     {args.data}')
    print(f'  Config:   {args.config}')
    print(f'  Events:   {args.events} (starting at {args.event})')
    print(f'  Mode:     {args.mode}')
    print(f'  Device:   {jax.devices()[0]}')

    sim.warm_up()

    # Accumulate results across events
    all_corr = None
    all_adc = None

    key = jax.random.PRNGKey(args.seed)
    for i in range(args.events):
        evt_idx = args.event + i
        key, subkey = jax.random.split(key)

        deposits = load_event(args.data, sim.config, event_idx=evt_idx)
        n_deps = sum(v.n_actual for v in deposits.volumes)
        print(f'\n  Event {evt_idx}: {n_deps:,} deposits')

        response_signals, track_hits_raw, deposits = sim.process_event(
            deposits, key=subkey)
        sync_result(response_signals)

        if do_corr:
            corr_results = analyze_corr_threshold(
                track_hits_raw, sim.config, corr_thresholds)
            if all_corr is None:
                all_corr = [{k: 0.0 for k in r} for r in corr_results]
                for j, r in enumerate(corr_results):
                    all_corr[j]['threshold'] = r['threshold']
            for j, r in enumerate(corr_results):
                all_corr[j]['total_charge'] += r['total_charge']
                all_corr[j]['kept_charge'] += r['kept_charge']
                all_corr[j]['total_entries'] += r['total_entries']
                all_corr[j]['kept_entries'] += r['kept_entries']

        if do_adc:
            adc_results = analyze_adc_threshold(
                response_signals, sim.config, adc_thresholds)
            if all_adc is None:
                all_adc = [{k: 0.0 for k in r} for r in adc_results]
                for j, r in enumerate(adc_results):
                    all_adc[j]['threshold'] = r['threshold']
            for j, r in enumerate(adc_results):
                all_adc[j]['total_signal'] += r['total_signal']
                all_adc[j]['kept_signal'] += r['kept_signal']
                all_adc[j]['total_bins'] += r['total_bins']
                all_adc[j]['kept_bins'] += r['kept_bins']

    # Recompute fractions from accumulated totals
    if all_corr:
        for r in all_corr:
            tc = r['total_charge']
            r['charge_lost_frac'] = (tc - r['kept_charge']) / tc if tc > 0 else 0
            te = r['total_entries']
            r['entries_dropped_frac'] = 1.0 - (r['kept_entries'] / te) if te > 0 else 0

    if all_adc:
        for r in all_adc:
            ts = r['total_signal']
            r['signal_lost_frac'] = (ts - r['kept_signal']) / ts if ts > 0 else 0
            tb = r['total_bins']
            r['bins_dropped_frac'] = 1.0 - (r['kept_bins'] / tb) if tb > 0 else 0

    # Print results
    if all_corr:
        print(f'\n  Correspondence Threshold (corr_threshold, electrons)')
        print(f'  Accumulated over {args.events} event(s)')
        print_corr_results(all_corr)

    if all_adc:
        print(f'\n  ADC Threshold (threshold_adc)')
        print(f'  Accumulated over {args.events} event(s)')
        print_adc_results(all_adc)

    if args.save_config:
        from profiler.production_config import update_config
        updates = {}
        if args.save_corr is not None:
            updates['corr_threshold'] = args.save_corr
        if args.save_adc is not None:
            updates['threshold_adc'] = args.save_adc
        if updates:
            update_config(args.save_config, updates,
                          detector_config_path=args.config)
            print(f'  Saved to {args.save_config}')

    print()


if __name__ == '__main__':
    main()
