"""
Coherent noise generation for JAXTPC detector simulation.

Models correlated noise shared across wire groups in liquid argon TPCs.
All wires within a group share the same coherent waveform. Adjacent groups
are anti-correlated with coupling coefficient beta.

Noise model:
    - Spectral shape: A(f) = 1 / (1 + f/f_corner)^(slope/2), no DC
    - Inter-group coupling: w'(g) = w(g) - beta * w(g-1) - beta * w(g+1)
    - Per-group RMS scaled to target value

Based on MicroBooNE coherent noise observations (arXiv:1705.07341).

NumPy-only implementation — does not require JAX. The coherent noise is
generated per-group (not per-wire), so it runs on CPU before being added
to the GPU response signals.
"""

import numpy as np


def coherent_spectrum(n_ticks, corner_freq_hz=20000.0, spectral_slope=1.5,
                      sampling_rate_hz=2e6):
    """Compute the coherent noise amplitude spectrum.

    A(f) = 1 / (1 + f/f_corner)^(slope/2), with A(0) = 0 (no DC).

    Parameters
    ----------
    n_ticks : int
        Number of time samples.
    corner_freq_hz : float
        Corner frequency in Hz.
    spectral_slope : float
        Power-law rolloff exponent.
    sampling_rate_hz : float
        Sampling rate in Hz.

    Returns
    -------
    spectrum : ndarray, shape (n_ticks // 2 + 1,)
        Amplitude spectrum for rfft.
    """
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz)
    spectrum = 1.0 / (1.0 + freqs / corner_freq_hz) ** (spectral_slope / 2.0)
    spectrum[0] = 0.0
    return spectrum.astype(np.float32)


def _expected_rms(spectrum, n_ticks):
    """Parseval RMS of a signal with the given amplitude spectrum."""
    S = np.asarray(spectrum, dtype=np.float64)
    N = n_ticks
    var = (S[0]**2 + 4.0 * np.sum(S[1:-1]**2) + S[-1]**2) / N**2
    return float(np.sqrt(max(var, 0.0)))


def generate_group_waveforms(n_groups, n_ticks, beta=0.15, rms_adc=2.5,
                             corner_freq_hz=20000.0, spectral_slope=1.5,
                             sampling_rate_hz=2e6, rng=None):
    """Generate coherent noise waveforms per group with neighbor coupling.

    Parameters
    ----------
    n_groups : int
        Number of wire groups.
    n_ticks : int
        Number of time samples.
    beta : float
        Anti-correlation coefficient between adjacent groups.
    rms_adc : float
        Target RMS amplitude per group in ADC.
    corner_freq_hz : float
        Spectral corner frequency in Hz.
    spectral_slope : float
        Spectral rolloff exponent.
    sampling_rate_hz : float
        Sampling rate in Hz.
    rng : numpy.random.Generator, optional
        Random number generator. If None, uses default_rng().

    Returns
    -------
    waveforms : ndarray, shape (n_groups, n_ticks)
        Coherent noise waveform per group.
    """
    if rng is None:
        rng = np.random.default_rng()

    spectrum = coherent_spectrum(n_ticks, corner_freq_hz, spectral_slope,
                                sampling_rate_hz)
    n_freq = len(spectrum)

    base = np.empty((n_groups, n_ticks), dtype=np.float32)
    for g in range(n_groups):
        real = rng.standard_normal(n_freq) * spectrum
        imag = rng.standard_normal(n_freq) * spectrum
        cpx = real + 1j * imag
        cpx[0] = cpx[0].real
        if n_ticks % 2 == 0:
            cpx[-1] = cpx[-1].real
        base[g] = np.fft.irfft(cpx, n=n_ticks)

    left = np.concatenate([np.zeros((1, n_ticks), dtype=np.float32), base[:-1]], axis=0)
    right = np.concatenate([base[1:], np.zeros((1, n_ticks), dtype=np.float32)], axis=0)
    waveforms = base - beta * (left + right)

    expected = _expected_rms(spectrum, n_ticks)
    if expected > 0:
        waveforms *= rms_adc / expected

    return waveforms


def broadcast_to_wires(group_waveforms, n_wires, group_size):
    """Broadcast (n_groups, n_ticks) group waveforms to (n_wires, n_ticks).

    Each wire gets the waveform of its group.
    """
    wire_to_group = np.arange(n_wires) // group_size
    return group_waveforms[wire_to_group]


def generate_coherent_noise(n_wires, n_ticks, group_size=64, beta=0.15,
                            rms_adc=2.5, corner_freq_hz=20000.0,
                            spectral_slope=1.5, sampling_rate_hz=2e6,
                            rng=None):
    """Generate coherent noise for a wire plane.

    Convenience function combining waveform generation and broadcasting.

    Parameters
    ----------
    n_wires : int
        Number of wires in the plane.
    n_ticks : int
        Number of time samples.
    group_size : int
        Wires per coherent group.
    beta : float
        Inter-group anti-correlation coefficient.
    rms_adc : float
        Target RMS per group in ADC.
    corner_freq_hz : float
        Spectral corner frequency.
    spectral_slope : float
        Spectral rolloff exponent.
    sampling_rate_hz : float
        Sampling rate in Hz.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    noise : ndarray, shape (n_wires, n_ticks)
        Coherent noise in ADC.
    """
    n_groups = (n_wires + group_size - 1) // group_size
    waveforms = generate_group_waveforms(
        n_groups, n_ticks, beta=beta, rms_adc=rms_adc,
        corner_freq_hz=corner_freq_hz, spectral_slope=spectral_slope,
        sampling_rate_hz=sampling_rate_hz, rng=rng,
    )
    return broadcast_to_wires(waveforms, n_wires, group_size)
