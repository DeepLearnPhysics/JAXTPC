"""
Profiler figure generation.

Each function takes data arrays and saves a figure to profiler/figures/.
All figures use a consistent style and are designed to answer specific
parameter-sizing questions.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def _savefig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def plot_deposit_distribution(all_counts, total_pad, tag=''):
    """Histogram of max-across-volumes deposit count per event.

    Parameters
    ----------
    all_counts : np.ndarray, shape (n_events, n_volumes)
    total_pad : int
    """
    max_per_event = all_counts.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(max_per_event, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(total_pad, color='red', linestyle='--', linewidth=1.5,
               label=f'total_pad = {total_pad:,}')
    pcts = np.percentile(max_per_event, [50, 99.9])
    ax.axvline(pcts[0], color='gray', linestyle=':', linewidth=1, label=f'P50 = {int(pcts[0]):,}')
    ax.axvline(pcts[1], color='orange', linestyle=':', linewidth=1, label=f'P99.9 = {int(pcts[1]):,}')
    ax.set_xlabel('Max deposits per volume')
    ax.set_ylabel('Events')
    ax.set_title('Deposit Count Distribution')
    ax.legend(fontsize=8)
    name = f'deposit_distribution{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)


def plot_keys_vs_deposits(deps, keys, total_pad, max_keys, upper_max_ratio, tag=''):
    """Scatter of estimated keys vs deposit count per volume.

    Parameters
    ----------
    deps, keys : np.ndarray, shape (n_points,)
    total_pad : int
    max_keys : int
    upper_max_ratio : float
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(deps, keys, s=4, alpha=0.4, color='steelblue', label='per-volume estimates')

    x_line = np.array([0, total_pad])
    ax.plot(x_line, upper_max_ratio * x_line, 'r--', linewidth=1.2,
            label=f'upper ratio = {upper_max_ratio:.2f}')
    ax.axvline(total_pad, color='gray', linestyle=':', linewidth=1,
               label=f'total_pad = {total_pad:,}')
    ax.axhline(max_keys, color='green', linestyle='--', linewidth=1,
               label=f'max_keys = {max_keys:,}')

    ax.set_xlabel('Deposits per volume')
    ax.set_ylabel('Estimated keys per plane')
    ax.set_title('Keys vs Deposits')
    ax.legend(fontsize=8)
    name = f'keys_vs_deposits{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)


def plot_keys_ratio(deps, keys, tag=''):
    """Keys/deposits ratio vs deposit count, binned."""
    ratio = keys / np.maximum(deps, 1)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(deps, ratio, s=4, alpha=0.3, color='steelblue')

    n_bins = min(10, len(deps) // 5)
    if n_bins >= 3:
        bin_edges = np.percentile(deps, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        for i in range(len(bin_edges) - 1):
            m = (deps >= bin_edges[i]) & (deps < bin_edges[i + 1])
            if m.sum() > 0:
                cx = (bin_edges[i] + bin_edges[i + 1]) / 2
                ax.errorbar(cx, ratio[m].mean(), yerr=ratio[m].std(),
                            fmt='o', color='red', markersize=5, capsize=3)

    ax.set_xlabel('Deposits per volume')
    ax.set_ylabel('Keys / deposits ratio')
    ax.set_title('Keys/Deposits Ratio vs Event Size')
    name = f'keys_ratio{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)


def plot_chunk_timing(candidates, times_ms, param_name, best_value=None, tag=''):
    """Line plot of chunk size vs timing.

    Parameters
    ----------
    candidates : list of int
    times_ms : list of (mean, std) tuples
    param_name : str
    best_value : int, optional
    """
    means = [t[0] for t in times_ms]
    stds = [t[1] for t in times_ms]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(candidates, means, yerr=stds, fmt='o-', color='steelblue',
                markersize=4, capsize=3, linewidth=1.2)
    if best_value is not None:
        ax.axvline(best_value, color='red', linestyle='--', linewidth=1,
                   label=f'best = {best_value:,}')
        ax.legend(fontsize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Time per event (ms)')
    ax.set_title(f'Timing vs {param_name}')
    name = f'chunk_timing_{param_name}{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)


def plot_corr_threshold(thresholds, charge_kept_frac, tag=''):
    """Charge kept fraction vs corr_threshold.

    Parameters
    ----------
    thresholds : array-like
    charge_kept_frac : array-like (fraction 0-1)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, np.array(charge_kept_frac) * 100, 'o-',
            color='steelblue', markersize=5, linewidth=1.2)
    ax.set_xlabel('corr_threshold (electrons)')
    ax.set_ylabel('Charge kept (%)')
    ax.set_title('Track Hits Charge Retention vs Threshold')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    name = f'corr_threshold{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)


def plot_adc_threshold(thresholds, signal_kept_frac, tag=''):
    """Signal kept fraction vs threshold_adc.

    Parameters
    ----------
    thresholds : array-like
    signal_kept_frac : array-like (fraction 0-1)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, np.array(signal_kept_frac) * 100, 'o-',
            color='steelblue', markersize=5, linewidth=1.2)
    ax.set_xlabel('threshold_adc (ADC)')
    ax.set_ylabel('Signal kept (%)')
    ax.set_title('Sensor Signal Retention vs ADC Threshold')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    name = f'adc_threshold{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)


def plot_feature_timing(labels, means_ms, baseline_ms=None, tag=''):
    """Bar chart of feature combination timings.

    Parameters
    ----------
    labels : list of str
    means_ms : list of float
    baseline_ms : float, optional
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    colors = ['steelblue'] * len(labels)
    ax.barh(x, means_ms, color=colors, edgecolor='white', linewidth=0.5)
    if baseline_ms is not None:
        ax.axvline(baseline_ms, color='red', linestyle='--', linewidth=1,
                   label=f'Baseline = {baseline_ms:.0f} ms')
        ax.legend(fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Time per event (ms)')
    ax.set_title('Feature Timing Breakdown')
    ax.invert_yaxis()
    name = f'feature_timing{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)


def plot_param_sweep(values, means_ms, stds_ms, param_name, tag=''):
    """Line plot of parameter sweep timing.

    Parameters
    ----------
    values : list of numeric
    means_ms, stds_ms : lists of float
    param_name : str
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(values, means_ms, yerr=stds_ms, fmt='o-',
                color='steelblue', markersize=5, capsize=3, linewidth=1.2)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Time per event (ms)')
    ax.set_title(f'Timing vs {param_name}')
    ax.grid(True, alpha=0.3)
    name = f'param_sweep_{param_name}{"_" + tag if tag else ""}.png'
    return _savefig(fig, name)
