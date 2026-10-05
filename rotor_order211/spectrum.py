"""Per-window order spectrum computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anglemap import AngleWindow, WINDOW_REVS


@dataclass(frozen=True)
class OrderSpectrum:
    orders: np.ndarray
    amplitudes: np.ndarray


def periodic_hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def order_spectrum(window: AngleWindow) -> OrderSpectrum:
    """Real FFT of the demeaned, Hann-windowed equi-angular samples.

    Amplitudes are divided by the window weight sum; positive frequencies are
    doubled except DC and Nyquist, giving peak amplitudes. Order of bin k is
    k / WINDOW_REVS.
    """
    x = window.samples - float(np.mean(window.samples))
    w = periodic_hann(x.size)
    spec = np.abs(np.fft.rfft(x * w)) / float(np.sum(w))
    if x.size % 2 == 0:
        spec[1:-1] *= 2.0
    else:
        spec[1:] *= 2.0
    orders = np.arange(spec.size, dtype=np.float64) / WINDOW_REVS
    return OrderSpectrum(orders=orders, amplitudes=spec)


def extract_orders(spectrum: OrderSpectrum, targets: list[int]) -> dict[int, float]:
    """Amplitude at each integer target order (bin = order * WINDOW_REVS)."""
    out: dict[int, float] = {}
    for order in targets:
        idx = order * WINDOW_REVS
        if idx >= spectrum.amplitudes.size:
            raise ValueError(f"order {order} exceeds spectrum range")
        out[order] = float(spectrum.amplitudes[idx])
    return out
