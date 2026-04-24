"""Pure-Python helpers for Γ-plane circles used in notebook 04 Part III.

Noise, available-gain, and source/load stability circles — all return
(center, radius) in the Γ-plane unit disk. Pure functions, no marimo
imports. Conventions follow Pozar, *Microwave Engineering* 4th ed, §12.3.
"""

from __future__ import annotations

import numpy as np


def noise_circle(F_target: float, F_min: float, R_n_norm: float,
                 Gamma_opt: complex) -> tuple[complex, float]:
    """Noise figure circle in the Γ_s plane for F = F_target.

    All arguments linear; returns (center, radius) with center complex.
    """
    assert F_target >= F_min, "F_target must be >= F_min"
    N = (F_target - F_min) / (4.0 * R_n_norm) * abs(1.0 + Gamma_opt) ** 2
    center = Gamma_opt / (1.0 + N)
    radius = np.sqrt(N ** 2 + N * (1.0 - abs(Gamma_opt) ** 2)) / (1.0 + N)
    return center, float(radius)


def available_gain_circle(G_target_dB: float, S11: complex, S12: complex,
                          S21: complex, S22: complex) -> tuple[complex, float]:
    """Available-gain circle in the Γ_s plane for G_A = G_target_dB.

    Pozar eqs. (12.76)–(12.78):
        g_a = G_A / |S21|²
        C_1 = S11 − Δ · S22*
        center = g_a · C_1* / (1 + g_a · (|S11|² − |Δ|²))
        radius = √(1 − 2K|S12 S21|g_a + |S12 S21|² g_a²) / |1 + g_a · (|S11|² − |Δ|²)|
    """
    g_a = 10 ** (G_target_dB / 10) / (abs(S21) ** 2)
    Delta = S11 * S22 - S12 * S21
    K = (1.0 - abs(S11) ** 2 - abs(S22) ** 2 + abs(Delta) ** 2) / (2.0 * abs(S12 * S21))
    C1 = S11 - Delta * np.conj(S22)
    denom_factor = 1.0 + g_a * (abs(S11) ** 2 - abs(Delta) ** 2)

    center = g_a * np.conj(C1) / denom_factor
    discrim = 1.0 - 2.0 * K * abs(S12 * S21) * g_a + (abs(S12 * S21) * g_a) ** 2
    radius = np.sqrt(max(discrim, 0.0)) / abs(denom_factor)
    return center, float(radius)


def source_stability_circle(S11: complex, S12: complex,
                            S21: complex, S22: complex) -> tuple[complex, float]:
    """Source stability circle — locus of Γ_s where |Γ_out| = 1."""
    Delta = S11 * S22 - S12 * S21
    denom = abs(S11) ** 2 - abs(Delta) ** 2
    assert abs(denom) > 1e-20, "degenerate source stability circle"
    center = np.conj(S11 - Delta * np.conj(S22)) / denom
    radius = abs(S12 * S21) / abs(denom)
    return center, float(radius)


def load_stability_circle(S11: complex, S12: complex,
                          S21: complex, S22: complex) -> tuple[complex, float]:
    """Load stability circle — locus of Γ_L where |Γ_in| = 1."""
    Delta = S11 * S22 - S12 * S21
    denom = abs(S22) ** 2 - abs(Delta) ** 2
    assert abs(denom) > 1e-20, "degenerate load stability circle"
    center = np.conj(S22 - Delta * np.conj(S11)) / denom
    radius = abs(S12 * S21) / abs(denom)
    return center, float(radius)


def rollett_K(S11: complex, S12: complex,
              S21: complex, S22: complex) -> tuple[float, complex]:
    """Return (K, Δ) — Rollett stability factor and determinant."""
    Delta = S11 * S22 - S12 * S21
    K = (1.0 - abs(S11) ** 2 - abs(S22) ** 2 + abs(Delta) ** 2) / (2.0 * abs(S12 * S21))
    return float(K), Delta


def MAG_dB(S11: complex, S12: complex,
           S21: complex, S22: complex) -> float:
    """Maximum available gain (dB) for an unconditionally-stable device.

    For K <= 1 (conditionally stable), returns the |S21/S12| ceiling (MSG).
    """
    K, _ = rollett_K(S11, S12, S21, S22)
    if K > 1.0:
        return 10.0 * np.log10(abs(S21) / abs(S12) * (K - np.sqrt(K ** 2 - 1.0)))
    return 10.0 * np.log10(abs(S21) / abs(S12))
