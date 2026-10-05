"""Per-window spectral analysis in the order domain."""

from __future__ import annotations

import numpy as np

from .mapping import WINDOW_REVS


def periodic_hann(n: int) -> np.ndarray:
    """Periodic Hann window (denominator n, not n-1)."""
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


def order_spectrum(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (orders, peak amplitudes) for one angle-domain window.

    Amplitudes are normalised by the window weight sum; positive-frequency
    bins are doubled except DC and Nyquist, giving single-sided peak
    amplitude. Order of bin i is i / WINDOW_REVS.
    """
    x = samples - samples.mean()
    w = periodic_hann(samples.size)
    spectrum = np.fft.rfft(x * w)
    amplitude = np.abs(spectrum) / w.sum()
    if samples.size % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    orders = np.arange(spectrum.size) / WINDOW_REVS
    return orders, amplitude
