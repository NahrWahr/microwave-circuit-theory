"""Pure-Python helpers for noise-signal simulation and spectral estimation.

Used by notebook 04 (Noise and LNA Design). No marimo imports — functions
are unit-testable via plain `python -c`.
"""

from __future__ import annotations

import numpy as np


def generate_white(n: int, variance: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return n samples of zero-mean Gaussian white noise with the given variance."""
    rng = rng if rng is not None else np.random.default_rng()
    return rng.normal(0.0, np.sqrt(variance), size=n)


def generate_flicker(n: int, variance: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return n samples of 1/f noise via spectral shaping of white noise.

    Shape a white-noise FFT by 1/sqrt(f), inverse transform, rescale
    to the requested variance.
    """
    rng = rng if rng is not None else np.random.default_rng()
    white = rng.normal(0.0, 1.0, size=n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    spec = spec / np.sqrt(freqs)
    shaped = np.fft.irfft(spec, n=n)
    shaped = shaped - shaped.mean()
    return shaped * np.sqrt(variance / shaped.var())


def generate_band_limited(n: int, f_s: float, f_hi: float, variance: float = 1.0,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Return n samples of zero-mean band-limited (0..f_hi) Gaussian noise."""
    rng = rng if rng is not None else np.random.default_rng()
    white = rng.normal(0.0, 1.0, size=n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / f_s)
    spec[freqs > f_hi] = 0.0
    shaped = np.fft.irfft(spec, n=n)
    shaped = shaped - shaped.mean()
    return shaped * np.sqrt(variance / shaped.var())


def generate_shot(n: int, rate: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return n samples of a Poisson-counted pulse train, mean-subtracted."""
    rng = rng if rng is not None else np.random.default_rng()
    counts = rng.poisson(rate, size=n).astype(float)
    return counts - rate


def estimate_autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation estimate R̂(τ) for τ = 0..max_lag-1."""
    n = x.size
    x = x - x.mean()
    full = np.correlate(x, x, mode="full") / n
    mid = full.size // 2
    return full[mid : mid + max_lag]


def estimate_psd(x: np.ndarray, f_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch-like PSD: segment the signal, FFT each segment, average."""
    n = x.size
    seg = min(1024, n)
    hop = seg // 2
    window = np.hanning(seg)
    segments = []
    for start in range(0, n - seg + 1, hop):
        chunk = x[start : start + seg] * window
        segments.append(np.abs(np.fft.rfft(chunk)) ** 2)
    if not segments:
        segments = [np.abs(np.fft.rfft(x * np.hanning(n))) ** 2]
    psd = np.mean(segments, axis=0) / (f_s * (window ** 2).sum())
    freqs = np.fft.rfftfreq(seg, d=1.0 / f_s)
    return freqs, psd


def friis_cascade(F: list[float], G_A: list[float]) -> float:
    """Friis cascade: F_total given per-stage noise figure and available-gain (linear, not dB).

    F and G_A must be the same length. Returns F_total (linear).
    """
    assert len(F) == len(G_A), "F and G_A must match"
    assert all(g > 0 for g in G_A), "available gains must be positive linear"
    f_tot = F[0]
    cum_gain = 1.0
    for i in range(1, len(F)):
        cum_gain *= G_A[i - 1]
        f_tot += (F[i] - 1.0) / cum_gain
    return f_tot


def noise_figure_from_Gamma(Gamma_s: complex, F_min: float, R_n_norm: float,
                            Gamma_opt: complex) -> float:
    """F(Γ_s) from the four noise parameters. F_min, R_n_norm linear; Γ's are complex."""
    denom = (1.0 - abs(Gamma_s) ** 2) * abs(1.0 + Gamma_opt) ** 2
    return F_min + 4.0 * R_n_norm * abs(Gamma_s - Gamma_opt) ** 2 / denom
