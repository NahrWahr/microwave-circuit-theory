"""Shaeffer-Lee inductively-degenerated common-source LNA model.

Pure Python; no marimo imports. Used by notebook 04 Part IV.
All equations per Shaeffer & Lee, "A 1.5-V 1.5-GHz CMOS LNA"
(JSSC 1997) with mmWave adaptations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

K_BOLTZMANN = 1.380649e-23
T0 = 290.0
Q_ELEC = 1.602176634e-19


@dataclass(frozen=True)
class DeviceParams:
    """Shaeffer-Lee transistor parameters for 65 nm NMOS at mmWave."""
    gm_per_id: float = 12.0          # g_m / I_D at chosen bias (S/A)
    cox_per_area: float = 1.7e-2     # F/m² for 65 nm
    cgs_per_W: float = 1.5e-15       # F per μm of width (empirical)
    gamma: float = 1.4               # channel thermal noise coefficient
    delta: float = 2.0 * 1.4         # induced gate noise coefficient
    c_corr: complex = 0.395j         # gate-channel correlation coefficient
    alpha_nq: float = 0.8            # g_m / g_d0 short-channel factor


def compute_operating_point(W_um: float, J_A_per_um: float,
                            params: DeviceParams = DeviceParams()
                            ) -> dict[str, float]:
    """Given device width (μm) and current density (A/μm), return g_m, C_gs, I_D, ω_T."""
    I_D = W_um * J_A_per_um
    g_m = params.gm_per_id * I_D
    C_gs = params.cgs_per_W * W_um
    omega_T = g_m / C_gs
    return {"I_D": I_D, "g_m": g_m, "C_gs": C_gs, "omega_T": omega_T}


def input_impedance(omega: float, L_s: float, L_g: float,
                    op: dict[str, float]) -> complex:
    """Input impedance of inductively-degenerated CS stage (C_gd ignored)."""
    return (1j * omega * (L_s + L_g)
            + 1.0 / (1j * omega * op["C_gs"])
            + op["omega_T"] * L_s)


def F_min_shaeffer_lee(omega: float, op: dict[str, float],
                       params: DeviceParams = DeviceParams()) -> float:
    """Shaeffer-Lee F_min approximation (linear)."""
    ratio = omega / op["omega_T"]
    return 1.0 + (2.4 * params.gamma / params.alpha_nq) * ratio


def gamma_opt_degenerated_cs(omega: float, L_s: float, L_g: float,
                             op: dict[str, float],
                             params: DeviceParams = DeviceParams()) -> complex:
    """Approx Γ_opt for the inductively-degenerated CS; placed near S11*."""
    Zin = input_impedance(omega, L_s, L_g, op)
    Z0 = 50.0
    return np.conj((Zin - Z0) / (Zin + Z0))


def sparameters_at_freq(omega: float, W_um: float, J_A_per_um: float,
                        L_s: float, L_g: float, L_d: float,
                        R_load: float = 50.0,
                        params: DeviceParams = DeviceParams()
                        ) -> tuple[complex, complex, complex, complex]:
    """Closed-form S-parameters for the degenerated CS stage with a simple LC load.

    Returns (S11, S21, S12, S22). Simplified model: C_gd neglected,
    cascode not included here. Accuracy good to a few dB over 20-40 GHz —
    sufficient for pedagogical purposes.
    """
    op = compute_operating_point(W_um, J_A_per_um, params)
    Zin = input_impedance(omega, L_s, L_g, op)
    Z0 = 50.0
    S11 = (Zin - Z0) / (Zin + Z0)

    # Effective transconductance gain with a matched 50 Ω source: at series
    # resonance the input LC loop has Q_gate = 1 / (ω·C_gs·(R_s + ω_T·L_s)).
    # V_gs / V_source ≈ Q_gate / 2, so |S21| ≈ g_m · |Z_load| · Q_gate.
    R_loop = Z0 + op["omega_T"] * L_s
    Q_gate = 1.0 / (omega * op["C_gs"] * R_loop)
    Z_load = 1j * omega * L_d + R_load
    S21 = -op["g_m"] * Z_load * Q_gate * Z0 / (Z0 + Z_load)

    S22 = (Z_load - Z0) / (Z_load + Z0)
    s12_mag = 10 ** ((-40 + 15 * (omega / (2 * math.pi * 40e9))) / 20)
    S12 = s12_mag + 0j
    return S11, S21, S12, S22


def NF_degenerated_cs(omega: float, W_um: float, J_A_per_um: float,
                      L_s: float, L_g: float, L_d: float,
                      R_load: float = 50.0,
                      params: DeviceParams = DeviceParams()) -> float:
    """NF (dB) for a 50 Ω source driving the degenerated CS at omega."""
    op = compute_operating_point(W_um, J_A_per_um, params)
    F = F_min_shaeffer_lee(omega, op, params)
    Zin = input_impedance(omega, L_s, L_g, op)
    Gamma_s = 0.0 + 0j
    S11 = (Zin - 50.0) / (Zin + 50.0)
    Gamma_opt = np.conj(S11)
    R_n_norm = params.gamma / (params.alpha_nq * op["g_m"] * 50.0)
    denom = (1.0 - abs(Gamma_s) ** 2) * abs(1.0 + Gamma_opt) ** 2
    F_at_50 = F + 4.0 * R_n_norm * abs(Gamma_s - Gamma_opt) ** 2 / denom
    return 10.0 * math.log10(F_at_50)
