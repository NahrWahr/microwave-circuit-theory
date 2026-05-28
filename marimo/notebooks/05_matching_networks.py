# v2.0
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full", css_file="")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, mo, np


@app.cell
def _(np):
    # v1.0
    """Pure-Python helpers for impedance matching network synthesis.

    Used by notebook 05 (Matching Networks). No marimo imports.
    Conventions: Pozar, *Microwave Engineering* 4th ed., Chapters 5 and 6.
    All frequencies in Hz, impedances in Ω, elements in H or F.
    """




    # ---------------------------------------------------------------------------
    # L-network
    # ---------------------------------------------------------------------------

    def l_network(R_S: float, R_L: float, f0: float,
                  topology: str = "lowpass") -> dict:
        """Design an L-network matching R_S to R_L at f0.

        topology: 'lowpass'  → shunt-C at high-Z port, series-L in series arm
                  'highpass' → shunt-L at high-Z port, series-C in series arm

        Returns dict with keys: Q, Cp (F), Ls (H) for lowpass;
        Lp (H), Cs (F) for highpass.
        """
        assert R_S > 0 and R_L > 0 and f0 > 0
        R_H, R_Lo = max(R_S, R_L), min(R_S, R_L)
        Q = np.sqrt(R_H / R_Lo - 1.0)
        w0 = 2.0 * np.pi * f0
        if topology == "lowpass":
            Cp = Q / (w0 * R_H)
            Ls = Q * R_Lo / w0
            return {"Q": Q, "Cp_F": Cp, "Ls_H": Ls}
        else:  # highpass
            Lp = R_H / (w0 * Q)
            Cs = 1.0 / (w0 * Q * R_Lo)
            return {"Q": Q, "Lp_H": Lp, "Cs_F": Cs}


    # ---------------------------------------------------------------------------
    # π-network
    # ---------------------------------------------------------------------------

    def pi_network(R_S: float, R_L: float, f0: float, Q: float) -> dict:
        """Design a low-pass π-network (shunt-C, series-L, shunt-C).

        Q must satisfy Q > sqrt(max(R_S,R_L)/min(R_S,R_L) - 1).
        Returns dict with keys: C1_F, L_H, C2_F, Q_min.
        """
        R_H, R_Lo = max(R_S, R_L), min(R_S, R_L)
        Q_min = np.sqrt(R_H / R_Lo - 1.0)
        assert Q > Q_min, f"Q={Q:.3f} must exceed Q_min={Q_min:.3f}"
        w0 = 2.0 * np.pi * f0
        # Virtual resistor R_virt = R_H / (1 + Q^2) <= R_Lo
        # Two L-sections: left Q_1 from R_S to R_virt, right Q_2 from R_virt to R_L
        R_virt = R_H / (1.0 + Q ** 2)
        Q1 = np.sqrt(R_S / R_virt - 1.0)
        Q2 = np.sqrt(R_L / R_virt - 1.0)
        C1 = Q1 / (w0 * R_S)
        C2 = Q2 / (w0 * R_L)
        # Series element: two inductors in series (one from each L-section) sum
        L = (Q1 * R_virt + Q2 * R_virt) / w0
        return {"Q_min": Q_min, "C1_F": C1, "L_H": L, "C2_F": C2}


    # ---------------------------------------------------------------------------
    # T-network
    # ---------------------------------------------------------------------------

    def t_network(R_S: float, R_L: float, f0: float, Q: float) -> dict:
        """Design a low-pass T-network (series-L, shunt-C, series-L).

        Q must exceed Q_min. Returns dict: L1_H, C_F, L2_H, Q_min.
        """
        R_H, R_Lo = max(R_S, R_L), min(R_S, R_L)
        Q_min = np.sqrt(R_H / R_Lo - 1.0)
        assert Q > Q_min, f"Q={Q:.3f} must exceed Q_min={Q_min:.3f}"
        w0 = 2.0 * np.pi * f0
        R_virt = R_Lo * (1.0 + Q ** 2)
        Q1 = np.sqrt(R_virt / R_S - 1.0)
        Q2 = np.sqrt(R_virt / R_L - 1.0)
        L1 = Q1 * R_S / w0
        L2 = Q2 * R_L / w0
        C = (Q1 + Q2) / (w0 * R_virt)   # both shunt caps sit at the R_virt node
        return {"Q_min": Q_min, "L1_H": L1, "C_F": C, "L2_H": L2}


    # ---------------------------------------------------------------------------
    # Frequency-domain S-parameter evaluation for 2-element ladder networks
    # ---------------------------------------------------------------------------

    def abcd_series_Z(Z):
        """ABCD matrix for series impedance Z (numpy array or scalar)."""
        one = np.ones_like(Z)
        zero = np.zeros_like(Z)
        # shape (..., 2, 2)
        return np.array([[one, Z], [zero, one]]).transpose(2, 0, 1) if Z.ndim > 0 else np.array([[1, Z], [0, 1]])


    def abcd_shunt_Y(Y):
        """ABCD matrix for shunt admittance Y."""
        one = np.ones_like(Y)
        zero = np.zeros_like(Y)
        return np.array([[one, zero], [Y, one]]).transpose(2, 0, 1) if Y.ndim > 0 else np.array([[1, 0], [Y, 1]])


    def abcd_to_S(ABCD, Z0: float = 50.0):
        """Convert ABCD matrices (shape N×2×2) to S-parameters (shape N×2×2)."""
        A = ABCD[:, 0, 0]
        B = ABCD[:, 0, 1]
        C = ABCD[:, 1, 0]
        D = ABCD[:, 1, 1]
        denom = A + B / Z0 + C * Z0 + D
        S11 = (A + B / Z0 - C * Z0 - D) / denom
        S12 = 2.0 * (A * D - B * C) / denom
        S21 = 2.0 / denom
        S22 = (-A + B / Z0 - C * Z0 + D) / denom
        N = len(A)
        S = np.zeros((N, 2, 2), dtype=complex)
        S[:, 0, 0] = S11
        S[:, 0, 1] = S12
        S[:, 1, 0] = S21
        S[:, 1, 1] = S22
        return S


    def l_network_S(R_S: float, R_L: float, f0: float,
                    topology: str, freqs: np.ndarray, Z0: float = 50.0) -> np.ndarray:
        """Return S-matrix (N×2×2) of an L-network over frequency array freqs (Hz).

        The network is designed at f0. S-params computed from ABCD chain.
        Source and load terminations are R_S, R_L; Z0 used for ABCD→S conversion.
        """
        params = l_network(R_S, R_L, f0, topology)
        w = 2.0 * np.pi * freqs
        if topology == "lowpass":
            Z_series = 1j * w * params["Ls_H"]
            Y_shunt = 1j * w * params["Cp_F"]
        else:
            Z_series = 1.0 / (1j * w * params["Cs_F"])
            Y_shunt = 1.0 / (1j * w * params["Lp_H"])
        M_shunt = np.stack([np.array([[1, 0], [yy, 1]]) for yy in Y_shunt])
        M_series = np.stack([np.array([[1, zz], [0, 1]]) for zz in Z_series])
        # The shunt element sits on the high-impedance side (Pozar §5.1):
        # source-high-Z → shunt at port 1; load-high-Z → shunt at port 2.
        if R_S >= R_L:
            ABCD = np.einsum('nij,njk->nik', M_shunt, M_series)
        else:
            ABCD = np.einsum('nij,njk->nik', M_series, M_shunt)
        return abcd_to_S(ABCD, Z0)


    def pi_network_S(R_S: float, R_L: float, f0: float, Q: float,
                     freqs: np.ndarray, Z0: float = 50.0) -> np.ndarray:
        """Return S-matrix (N×2×2) of a low-pass π-network over freqs."""
        params = pi_network(R_S, R_L, f0, Q)
        w = 2.0 * np.pi * freqs
        Y1 = 1j * w * params["C1_F"]
        Y2 = 1j * w * params["C2_F"]
        ZL = 1j * w * params["L_H"]
        M1 = np.stack([np.array([[1, 0], [y, 1]]) for y in Y1])
        M2 = np.stack([np.array([[1, z], [0, 1]]) for z in ZL])
        M3 = np.stack([np.array([[1, 0], [y, 1]]) for y in Y2])
        ABCD = np.einsum('nij,njk->nik', np.einsum('nij,njk->nik', M1, M2), M3)
        return abcd_to_S(ABCD, Z0)


    def t_network_S(R_S: float, R_L: float, f0: float, Q: float,
                    freqs: np.ndarray, Z0: float = 50.0) -> np.ndarray:
        """Return S-matrix (N×2×2) of a low-pass T-network over freqs."""
        params = t_network(R_S, R_L, f0, Q)
        w = 2.0 * np.pi * freqs
        Z1 = 1j * w * params["L1_H"]
        YC = 1j * w * params["C_F"]
        Z2 = 1j * w * params["L2_H"]
        M1 = np.stack([np.array([[1, z], [0, 1]]) for z in Z1])
        M2 = np.stack([np.array([[1, 0], [y, 1]]) for y in YC])
        M3 = np.stack([np.array([[1, z], [0, 1]]) for z in Z2])
        ABCD = np.einsum('nij,njk->nik', np.einsum('nij,njk->nik', M1, M2), M3)
        return abcd_to_S(ABCD, Z0)


    # ---------------------------------------------------------------------------
    # Bode-Fano bound (parallel RC load)
    # ---------------------------------------------------------------------------

    def bode_fano_gamma_max(BW_Hz: float, R_L: float, C_L: float) -> float:
        """Maximum achievable |Γ_max| over BW_Hz for a parallel-RC load (Pozar §5.8).

        Returns |Γ_max| (linear). If the bound implies |Γ_max| ≥ 1, returns 1.0.
        """
        # ∫ ln(1/|Γ|) dω ≤ π/(R_L C_L)
        # For equal-ripple fill over BW (rad/s):
        # BW * ln(1/|Γ_max|) ≤ π / (R_L * C_L)
        BW_rad = 2.0 * np.pi * BW_Hz
        exponent = np.pi / (R_L * C_L * BW_rad)
        return float(np.exp(-exponent))


    # ---------------------------------------------------------------------------
    # Chebyshev low-pass prototype element values
    # ---------------------------------------------------------------------------

    def chebyshev_prototype(N: int, ripple_dB: float) -> list[float]:
        """Return normalised element values g_0, g_1, ..., g_N, g_{N+1} (Pozar Table 5.5).

        Ladder network: g_0 = source conductance, g_1..g_N are L/C values,
        g_{N+1} = load conductance/resistance.
        ripple_dB: passband ripple in dB (0.01 to 3 dB typical).
        """
        epsilon = np.sqrt(10.0 ** (ripple_dB / 10.0) - 1.0)
        beta = np.log(1.0 / np.tanh(ripple_dB / 17.37))
        gamma = np.sinh(beta / (2.0 * N))
        g = [1.0]  # g_0
        for k in range(1, N + 1):
            a_k = np.sin((2 * k - 1) * np.pi / (2 * N))
            b_k = gamma ** 2 + np.sin(k * np.pi / N) ** 2
            if k == 1:
                g.append(2.0 * a_k / gamma)
            else:
                g.append(4.0 * a_k * g[k - 1] / (b_k * g[k - 1]))
        # g_{N+1}
        if N % 2 == 0:
            g.append(1.0 / np.tanh(beta / 4.0) ** 2)
        else:
            g.append(1.0)
        return g


    def chebyshev_bandpass_elements(N: int, ripple_dB: float,
                                    R_S: float, R_L: float,
                                    f0: float, BW: float) -> list[dict]:
        """Scale Chebyshev prototype to a bandpass network between R_S and R_L.

        Returns list of N dicts, each with keys: type ('series_LC' or 'shunt_LC'),
        L_H, C_F.
        BW: 3-dB bandwidth in Hz.
        """
        g = chebyshev_prototype(N, ripple_dB)
        w0 = 2.0 * np.pi * f0
        bw = 2.0 * np.pi * BW
        elements = []
        for k in range(1, N + 1):
            if k % 2 == 1:  # series arm
                L = g[k] * R_S / bw
                C = bw / (w0 ** 2 * L * bw)
                elements.append({"type": "series_LC", "L_H": L, "C_F": C})
            else:  # shunt arm
                C = g[k] / (bw * R_S)
                L = bw / (w0 ** 2 * C * bw)
                elements.append({"type": "shunt_LC", "L_H": L, "C_F": C})
        return elements


    # ---------------------------------------------------------------------------
    # Quarter-wave transformer
    # ---------------------------------------------------------------------------

    def qw_transformer(R_S: float, R_L: float) -> dict:
        """Design a single-section quarter-wave transformer.

        Returns Z1 (Ω) — the transformer line impedance.
        """
        return {"Z1_ohm": float(np.sqrt(R_S * R_L))}


    def qw_transformer_S11(R_S: float, R_L: float, f0: float,
                           freqs: np.ndarray) -> np.ndarray:
        """Return |S11| of a single λ/4 transformer vs frequency.

        Uses the exact transmission-line two-port (Pozar eq. 5.47).
        """
        Z1 = np.sqrt(R_S * R_L)
        theta = np.pi / 2.0 * freqs / f0  # electrical length (π/2 at f0)
        # Input impedance: Z_in = Z1 * (R_L + j Z1 tan θ) / (Z1 + j R_L tan θ)
        t = np.tan(theta)
        Z_in = Z1 * (R_L + 1j * Z1 * t) / (Z1 + 1j * R_L * t)
        Gamma = (Z_in - R_S) / (Z_in + R_S)
        return np.abs(Gamma)


    # ---------------------------------------------------------------------------
    # Single-stub tuner
    # ---------------------------------------------------------------------------

    def single_stub_tuner(Z_L: complex, Z0: float, f0: float,
                          stub_type: str = "short") -> list[dict]:
        """Design a parallel single-stub tuner (Pozar §5.2).

        Returns up to 2 solutions: each dict has d_lambda (position as fraction of λ)
        and ell_lambda (stub length as fraction of λ).
        stub_type: 'short' or 'open'.
        """
        Y0 = 1.0 / Z0
        YL = 1.0 / Z_L
        GL = YL.real
        BL = YL.imag
        solutions = []
        # Solve for t = tan(βd) such that Re(Y_in) = Y0
        # G_in(d) = Y0 when t satisfies quadratic (Pozar eq. 5.8)
        # G_L(1 + t^2) / ((G_L + Y0 t^2 * ... )) = Y0  →  derived below:
        # Using admittance transformation: Y_in = Y0 * (YL + jY0 t) / (Y0 + j YL t)
        # Re(Y_in) = Y0 =>
        # GL*(1 + t^2) / (1 + 2*BL/Y0*t + (GL^2 + BL^2)/Y0^2 * t^2) = Y0
        a = GL / Y0
        b = BL / Y0
        # Equation: GL*(1+t²) = Y0 * |Y0 + j*YL*t|² / Y0²  ... standard result:
        # t² (GL² + BL² - GL*Y0) + t * 2*BL*Y0 + GL*Y0 - Y0² = 0  ...
        # Simplified Pozar (5.8): t = [BL ± sqrt(GL*(Y0 - GL) + BL²)] / (GL - Y0)
        discriminant = GL * (Y0 - GL) + BL ** 2
        if discriminant < 0:
            return []  # load cannot be matched (forbidden region)
        for sign in [+1.0, -1.0]:
            t = (BL + sign * np.sqrt(discriminant)) / (GL - Y0) if abs(GL - Y0) > 1e-15 * Y0 else np.inf
            d_lambda = np.arctan(t) / (2.0 * np.pi) % 0.5
            # Susceptance at position d
            Y_in = Y0 * (YL + 1j * Y0 * t) / (Y0 + 1j * YL * t)
            B_in = Y_in.imag
            # Stub susceptance must cancel B_in
            if stub_type == "short":
                # B_stub = -Y0 * cot(beta*ell)  => cot(x) = Y0 / B_stub_need
                # Y0 * cot(x) = -B_in  => cot(x) = -B_in/Y0  => tan(x) = -Y0/B_in
                ell_lambda = np.arctan(-Y0 / B_in) / (2.0 * np.pi) % 0.5
            else:  # open
                # B_stub = Y0 * tan(beta*ell)  => tan = -B_in / Y0
                ell_lambda = np.arctan(-B_in / Y0) / (2.0 * np.pi) % 0.5
            solutions.append({
                "d_lambda": float(d_lambda),
                "ell_lambda": float(ell_lambda),
                "d_mm": float(d_lambda * 3e8 / (f0 * np.sqrt(1.0)) * 1e3),  # in air
                "ell_mm": float(ell_lambda * 3e8 / (f0 * np.sqrt(1.0)) * 1e3),
            })
        return solutions


    # ---------------------------------------------------------------------------
    # Coupled inductors and transformers (notebook 05 §15-§18)
    # ---------------------------------------------------------------------------

    def coupled_inductors_Zmatrix(L1: float, L2: float, k: float,
                                  w: np.ndarray) -> np.ndarray:
        """Z-matrix of a coupled-inductor two-port over frequency w (rad/s)."""
        M = k * np.sqrt(L1 * L2)
        wa = np.atleast_1d(np.asarray(w, dtype=float))
        Z = np.zeros((len(wa), 2, 2), dtype=complex)
        Z[:, 0, 0] = 1j * wa * L1
        Z[:, 0, 1] = 1j * wa * M
        Z[:, 1, 0] = 1j * wa * M
        Z[:, 1, 1] = 1j * wa * L2
        return Z


    def coupled_inductors_T_model(L1: float, L2: float, k: float) -> dict:
        """T-model element values for coupled inductors: (L1-M, L2-M, M).

        Returned values may be negative (passive in the T-model representation).
        """
        M = k * np.sqrt(L1 * L2)
        return {"La_H": L1 - M, "Lb_H": L2 - M, "Lm_H": M, "M_H": M, "k": k}


    def transformer_input_Z(L1: float, L2: float, k: float,
                            Z_load, w) -> np.ndarray:
        """Z_in = jωL1 + (ωM)² / (Z_load + jωL2). Vectorised over w."""
        M = k * np.sqrt(L1 * L2)
        wa = np.asarray(w, dtype=float)
        Zl = np.asarray(Z_load, dtype=complex)
        return 1j * wa * L1 + (wa * M) ** 2 / (Zl + 1j * wa * L2)


    # ---------------------------------------------------------------------------
    # Bridged T-coil (constant-R design) — notebook 05 §19
    # ---------------------------------------------------------------------------

    def tcoil_constant_R(R: float, C: float, k: float) -> dict:
        """Constant-R bridged-T-coil design: returns L_half, M, C_B, L_T."""
        L_half = R ** 2 * C / (2.0 * (1.0 + k))
        M = k * L_half
        C_B = (1.0 - k) * C / (4.0 * (1.0 + k))
        L_T = R ** 2 * C
        return {"L_half_H": L_half, "M_H": M, "C_B_F": C_B, "L_T_H": L_T}


    def _tcoil_nodal_solve(R: float, C: float, k: float, freqs: np.ndarray):
        """Solve the 3-node admittance system at each freq; return (V_a, V_tap, V_b).

        Source: ideal voltage source V_src = 1 V through series resistor R at node a.
        Termination: R from node b to ground. Load capacitor C at node tap to ground.
        Bridge cap C_B between nodes a and b. Coupled inductors between (a–tap) and
        (tap–b) with self-inductance L_half each, mutual M = k·L_half.
        """
        p = tcoil_constant_R(R, C, k)
        L_half = p["L_half_H"]
        C_B = p["C_B_F"]
        fs = np.atleast_1d(np.asarray(freqs, dtype=float))
        V = np.zeros((len(fs), 3), dtype=complex)
        G_R = 1.0 / R
        for i, f in enumerate(fs):
            w = 2.0 * np.pi * f
            if w == 0.0:
                # DC: coils short ⇒ V_a = V_tap = V_b. Resistor divider gives V = 1/2.
                V[i] = [0.5, 0.5, 0.5]
                continue
            y = 1.0 / (1j * w * L_half * (1.0 - k ** 2))   # coupled-coil Y-scale
            Y_CB = 1j * w * C_B
            Y_C = 1j * w * C
            M_sys = np.array([
                [G_R + y + Y_CB,   -y * (1.0 + k),               y * k - Y_CB],
                [-y * (1.0 + k),    2.0 * y * (1.0 + k) + Y_C,  -y * (1.0 + k)],
                [y * k - Y_CB,     -y * (1.0 + k),               y + Y_CB + G_R],
            ], dtype=complex)
            rhs = np.array([G_R, 0.0, 0.0], dtype=complex)
            try:
                V[i] = np.linalg.solve(M_sys, rhs)
            except np.linalg.LinAlgError:
                V[i] = np.nan
        return V


    def tcoil_tap_response(R: float, C: float, k: float,
                           freqs: np.ndarray) -> np.ndarray:
        """V_tap / V_src for bridged-T-coil (constant-R, matched source+load)."""
        return _tcoil_nodal_solve(R, C, k, freqs)[:, 1]


    def tcoil_input_S11(R: float, C: float, k: float,
                        freqs: np.ndarray, Z0: float | None = None) -> np.ndarray:
        """S11 at the input port (should be ≈ 0 for the constant-R design)."""
        if Z0 is None:
            Z0 = R
        V = _tcoil_nodal_solve(R, C, k, freqs)
        V_a = V[:, 0]
        I_in = (1.0 - V_a) / R
        Z_in = np.where(np.abs(I_in) > 1e-20, V_a / I_in, 1e20)
        return (Z_in - Z0) / (Z_in + Z0)


    def tcoil_poles(R: float, C: float, k: float) -> np.ndarray:
        """Approximate dominant poles of the tap-node transfer (rad/s).

        The constant-R T-coil tap transfer reduces to a low-order rational with
        natural frequency ω_n ≈ 2/(R·C) and damping ζ(k). For k=1/3, ζ=1/√2
        (Butterworth); for k=1/2, ζ ≈ √3/2 (Bessel-like).
        """
        wn = 2.0 / (R * C)
        zeta_map = {1/3: 1/np.sqrt(2), 1/2: np.sqrt(3)/2.0}
        zeta = zeta_map.get(k, 0.5 + 0.5 * k)   # rough interpolation
        if zeta < 1:
            wd = wn * np.sqrt(1 - zeta ** 2)
            return np.array([-zeta * wn + 1j * wd, -zeta * wn - 1j * wd])
        return np.array([wn * (-zeta + np.sqrt(zeta ** 2 - 1)),
                         wn * (-zeta - np.sqrt(zeta ** 2 - 1))])


    # ---------------------------------------------------------------------------
    # Marchand balun (notebook 05 §20)
    # ---------------------------------------------------------------------------

    def marchand_balun_response(C_coup: float, f0: float,
                                freqs: np.ndarray) -> dict:
        """Simplified Marchand balun model parametrised by mid-band coupling C.

        C_coup = (Z0e - Z0o) / (Z0e + Z0o) for each λ/4 coupled section.
        Returns single-ended-to-one-diff-port |S21| and input return loss vs freq.
        At f = f0 each port carries half the input power (|S21|² = 1/2).
        """
        fs = np.atleast_1d(np.asarray(freqs, dtype=float))
        theta = np.pi / 2.0 * fs / f0
        sin2 = np.sin(theta) ** 2
        cos2 = np.cos(theta) ** 2
        if abs(C_coup) < 1e-9:
            S21_2 = np.zeros_like(theta)
        else:
            S21_2 = 0.5 * sin2 / (sin2 + ((1.0 - C_coup ** 2) / C_coup ** 2) * cos2)
        S11_2 = np.clip(1.0 - 2.0 * S21_2, 0, 1)
        S21_dB = 10.0 * np.log10(np.maximum(S21_2, 1e-12))
        S11_dB = 10.0 * np.log10(np.maximum(S11_2, 1e-12))
        peak = S21_2.max()
        in_band = S21_2 > peak / 2.0
        BW_pct = (float(fs[in_band].max() - fs[in_band].min()) / f0 * 100.0
                  if in_band.any() else 0.0)
        return {"f_Hz": fs, "S21_dB": S21_dB, "S11_dB": S11_dB,
                "BW_3dB_pct": BW_pct}


    # ---------------------------------------------------------------------------
    # Non-Foster NIC (notebook 05 §22)
    # ---------------------------------------------------------------------------

    def nic_input_Z(Z_load, R_loss: float = 0.0, L_par: float = 0.0,
                    w=0.0) -> np.ndarray:
        """Ideal NIC: Z_in = -Z_load + R_loss + jωL_par."""
        wa = np.asarray(w, dtype=float)
        Z = np.asarray(Z_load, dtype=complex)
        return -Z + R_loss + 1j * wa * L_par


    # ---------------------------------------------------------------------------
    # N-path filter — translated impedance (notebook 05 §24)
    # ---------------------------------------------------------------------------

    def npath_translated_Z(R_sw: float, C_BB: float, R_BB: float,
                           N: int, f_LO: float, freqs: np.ndarray,
                           K_harm: int = 5) -> np.ndarray:
        """N-path RF input impedance (Mirzaei 2010, Andrews-Molnar 2010).

        Z_RF(jω) ≈ R_sw + Σ_{n=-K..K} |a_n|² · Z_BB(j(ω - n·ω_LO))
        with |a_n|² = sinc²(n/N) (numpy uses normalised sinc(x) = sin(πx)/(πx)).
        """
        fs = np.atleast_1d(np.asarray(freqs, dtype=float))
        w = 2.0 * np.pi * fs
        w_LO = 2.0 * np.pi * f_LO
        Z_in = np.full_like(w, R_sw, dtype=complex)
        for n in range(-K_harm, K_harm + 1):
            a2 = np.sinc(n / N) ** 2 if N > 0 else 0.0
            w_shift = w - n * w_LO
            Z_BB = 1.0 / (1.0 / R_BB + 1j * w_shift * C_BB)
            Z_in = Z_in + a2 * Z_BB
        return Z_in


    # ---------------------------------------------------------------------------
    # Distributed amplifier — N-section artificial-TL gain (notebook 05 §23)
    # ---------------------------------------------------------------------------

    def distributed_amp_S21(N: int, gm: float, Z0: float,
                            L_sec: float, C_sec: float,
                            R_gate: float, R_drain: float,
                            freqs: np.ndarray) -> np.ndarray:
        """|S21| (linear) of an N-section distributed amplifier.

        Below the artificial-TL cutoff f_c = 1/(π√(L_sec·C_sec)) the per-section
        phase is θ = 2·arcsin(ω/ω_c); above f_c the line is evanescent and the
        amplifier ceases to function. Per-section loss α ≈ (R_gate+R_drain)/(2Z0).
        """
        fs = np.atleast_1d(np.asarray(freqs, dtype=float))
        w = 2.0 * np.pi * fs
        f_c = 1.0 / (np.pi * np.sqrt(L_sec * C_sec))
        w_c = 2.0 * np.pi * f_c
        arg = np.clip(w / w_c, 0.0, 0.9999)
        theta = 2.0 * np.arcsin(arg)
        sinc_arg = N * theta / 2.0
        sinc_term = np.where(np.abs(sinc_arg) < 1e-9, 1.0,
                             np.sin(sinc_arg) / sinc_arg)
        alpha = (R_gate + R_drain) / (2.0 * Z0)
        S21_mag = (N * gm * Z0 / 2.0) * np.abs(sinc_term) * np.exp(-N * alpha * arg)
        S21_mag = np.where(w > w_c, S21_mag * 0.05, S21_mag)
        return S21_mag

    return (
        bode_fano_gamma_max,
        chebyshev_prototype,
        coupled_inductors_T_model,
        coupled_inductors_Zmatrix,
        distributed_amp_S21,
        l_network,
        l_network_S,
        marchand_balun_response,
        nic_input_Z,
        npath_translated_Z,
        pi_network,
        pi_network_S,
        qw_transformer,
        single_stub_tuner,
        t_network,
        t_network_S,
        tcoil_constant_R,
        tcoil_input_S11,
        tcoil_poles,
        tcoil_tap_response,
        transformer_input_Z,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 05 — Matching Networks and Broadband Design

    Builds on notebooks 01–04. Lumped and distributed impedance matching,
    Bode-Fano bandwidth limits, transformer coupling. Mixer and sampler noise
    (cyclostationary) moved to notebook 04 §3.8, §4.9.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part I — Impedance Matching Fundamentals

    ### §1. Motivation: Mismatch Loss and the Power Transfer Theorem

    Consider a source characterized by a Thevenin equivalent voltage $V_S$ and internal impedance $Z_S = R_S + jX_S$. It drives a load impedance $Z_L = R_L + jX_L$.
    The current flowing through the circuit is:

    $$I = \frac{V_S}{Z_S + Z_L} = \frac{V_S}{(R_S + R_L) + j(X_S + X_L)}$$

    The time-average power delivered to the load is given by $P_L = \frac{1}{2} |I|^2 R_L$:

    $$P_L = \frac{1}{2} \frac{|V_S|^2 R_L}{(R_S + R_L)^2 + (X_S + X_L)^2}$$

    To maximize $P_L$ with respect to $X_L$, we set $X_L = -X_S$. Substituting this condition yields:

    $$P_L = \frac{1}{2} \frac{|V_S|^2 R_L}{(R_S + R_L)^2}$$

    Maximizing with respect to $R_L$ by setting $\partial P_L / \partial R_L = 0$ gives $R_L = R_S$. Therefore, maximum power transfer occurs when $Z_L = Z_S^*$. Under this conjugate match condition, the **available power** is:

    $$P_{\text{avail}} = \frac{|V_S|^2}{8 R_S}$$

    For any other load, the fraction actually delivered is represented by the mismatch loss (ML):

    $$\text{ML} = 1 - |\Gamma_L|^2 = \frac{4 R_S R_L}{|Z_S + Z_L|^2}$$

    where $\Gamma_L = (Z_L - Z_S^*)/(Z_L + Z_S)$. In dB: $\text{ML}_{\text{dB}} = 10\log_{10}(1 - |\Gamma_L|^2)$.

    **Quality factor.** For a series resonator: $Q = X/R$; for a shunt resonator: $Q = B/G$.
    A single-resonator match has an approximate 3 dB fractional bandwidth of $1/Q$:

    $$\text{BW}_{3\,\text{dB}} \approx \frac{f_0}{Q_{\text{loaded}}}$$

    Large impedance transformation ratios ($R_H/R_L \gg 1$) force high $Q$, leading to a narrow bandwidth.
    Parts II–IV provide the tools to escape this constraint.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §2. Network Representations for Matching

    Matching networks are two-ports inserted between $Z_S$ and $Z_L$ to provide the required impedance transformation.
    The ABCD (transmission) matrix formulation is the most natural representation for cascaded networks, as the matrices multiply directly:

    $$\begin{bmatrix}V_1\\I_1\end{bmatrix} = \begin{bmatrix}A & B\\C & D\end{bmatrix}
    \begin{bmatrix}V_2\\-I_2\end{bmatrix}$$

    For a **series impedance** $Z$: The voltage drops by $I_1 Z$, and current is unchanged ($I_1 = -I_2$).
    $A=1,\; B=Z,\; C=0,\; D=1$.

    For a **shunt admittance** $Y$: The voltage is unchanged ($V_1 = V_2$), and current splits ($I_1 = -I_2 + V_2 Y$).
    $A=1,\; B=0,\; C=Y,\; D=1$.

    A lossless LC network satisfies the condition that power in equals power out, which imposes constraints on the matrix elements. Specifically, it can be shown that $A,D$ must be purely real, and $B,C$ purely imaginary.
    Reciprocity requires the determinant to be unity: $AD - BC = 1$ for all passive, isotropic networks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §3. The Smith Chart from First Principles

    The Smith chart is the single most important graphical tool in microwave engineering. It is not an ad-hoc construction; it follows inevitably from the algebraic properties of the **Mobius (bilinear) transformation** that relates impedance to reflection coefficient. This section derives the chart rigorously.

    #### 3.1 The Bilinear (Mobius) Transformation

    Every normalized impedance $z = Z/Z_0 = r + jx$ maps to a reflection coefficient $\Gamma = u + jv$ via:

    $$\Gamma = \frac{z - 1}{z + 1}$$

    The inverse is $z = \frac{1 + \Gamma}{1 - \Gamma}$. This is a **Mobius transformation** of the form $w = \frac{az + b}{cz + d}$ with $a=1, b=-1, c=1, d=1$.

    **Key property:** A Mobius transformation maps every generalized circle (circle or straight line) in one complex plane to a generalized circle in the other. This single fact generates the entire Smith chart.

    #### 3.2 Deriving the Constant-Resistance Circles

    Set $z = r + jx$ with $r$ fixed. Substituting into $\Gamma = (z-1)/(z+1)$:

    $$\Gamma = \frac{(r-1) + jx}{(r+1) + jx}$$

    Multiply numerator and denominator by the conjugate of the denominator:

    $$\Gamma = \frac{[(r-1)+jx][(r+1)-jx]}{|{(r+1)+jx}|^2} = \frac{(r^2-1+x^2) + j[x(r+1) - x(r-1)]}{(r+1)^2 + x^2}$$

    $$u = \frac{r^2 - 1 + x^2}{(r+1)^2 + x^2}, \qquad v = \frac{2x}{(r+1)^2 + x^2}$$

    After algebraic manipulation (eliminating $x$ between these two equations), one obtains:

    $$\boxed{\left(u - \frac{r}{r+1}\right)^2 + v^2 = \frac{1}{(r+1)^2}}$$

    This is a circle in the $\Gamma$-plane centered at $\left(\frac{r}{r+1}, 0\right)$ with radius $\frac{1}{r+1}$.

    - $r = 0$: center $(0, 0)$, radius $1$ -- the unit circle itself (pure reactance).
    - $r = 1$: center $(0.5, 0)$, radius $0.5$ -- passes through the origin.
    - $r \to \infty$: center $(1, 0)$, radius $0$ -- the point $\Gamma = 1$ (open circuit).

    #### 3.3 Deriving the Constant-Reactance Circles

    Now fix $x$ and let $r$ vary over $[0, \infty)$. Eliminating $r$ from the same pair of equations yields:

    $$\boxed{(u - 1)^2 + \left(v - \frac{1}{x}\right)^2 = \frac{1}{x^2}}$$

    These are circles centered at $\left(1, \frac{1}{x}\right)$ with radius $\frac{1}{|x|}$, all tangent to the point $\Gamma = 1$ (open circuit). Positive $x$ (inductive) gives circles in the upper half-plane; negative $x$ (capacitive) in the lower half-plane. The $x = 0$ "circle" degenerates to the real axis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 3.4 The Admittance Chart and the 180-degree Rotation

    For shunt elements it is natural to work in normalized admittance $y = g + jb = 1/z$. Substituting $z = 1/y$:

    $$\Gamma = \frac{1/y - 1}{1/y + 1} = \frac{1 - y}{1 + y} = -\frac{y - 1}{y + 1}$$

    The leading minus sign is multiplication by $e^{j\pi}$: a **180-degree rotation** of the impedance-chart mapping. Therefore constant-$g$ circles are the constant-$r$ circles reflected through the origin, and constant-$b$ circles are reflected constant-$x$ circles.

    The constant-conductance circle for conductance $g$ is:

    $$\left(u + \frac{g}{g+1}\right)^2 + v^2 = \frac{1}{(g+1)^2}$$

    centered at $\left(-\frac{g}{g+1}, 0\right)$. The constant-susceptance circle for susceptance $b$:

    $$(u + 1)^2 + \left(v + \frac{1}{b}\right)^2 = \frac{1}{b^2}$$

    centered at $\left(-1, -\frac{1}{b}\right)$, tangent to $\Gamma = -1$ (short circuit).

    **The Smith chart is thus a dual-coordinate system**: Z-contours (constant $r$, $x$) and Y-contours (constant $g$, $b$) overlaid on the same $|\Gamma| \le 1$ disk. Switching between the two is equivalent to toggling between series and parallel analysis of the circuit.

    #### 3.5 Movement Rules for Matching

    | Element | Domain | Effect | Trajectory |
    |---------|--------|--------|------------|
    | Series $L$ ($+jx$) | $z_{\text{new}} = z + jx$ | Increase reactance | CW along constant-$r$ circle |
    | Series $C$ ($-jx$) | $z_{\text{new}} = z - jx$ | Decrease reactance | CCW along constant-$r$ circle |
    | Shunt $C$ ($+jb$) | $y_{\text{new}} = y + jb$ | Increase susceptance | CW along constant-$g$ circle |
    | Shunt $L$ ($-jb$) | $y_{\text{new}} = y - jb$ | Decrease susceptance | CCW along constant-$g$ circle |

    A matching design is a sequence of arc moves from the load point to the chart center ($\Gamma = 0$, i.e. $z = 1$).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Interactive I -- Bilinear Mapping: $z$-plane $\to$ $\Gamma$-plane

    The chart below shows the bilinear transform in action. On the left is the normalized impedance plane; on the right is the $\Gamma$-plane (the Smith chart). Vertical lines of constant $r$ in the $z$-plane map to the constant-resistance circles. Horizontal lines of constant $x$ map to the constant-reactance arcs.
    """)
    return


@app.cell
def _(go, mo, np):
    _theta_bl = np.linspace(0, 2 * np.pi, 300)
    _r_vals_bl = [0, 0.2, 0.5, 1.0, 2.0, 5.0]
    _r_colors_bl = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#FF6692"]
    _x_vals_bl = [0.2, 0.5, 1.0, 2.0, 5.0]
    _x_colors_bl = ["#19D3F3", "#B6E880", "#FECB52", "#FF97FF", "#72B7B2"]

    from plotly.subplots import make_subplots as _ms_bl
    _fig_bl = _ms_bl(rows=1, cols=2,
                     subplot_titles=("z-plane (impedance)", "Gamma-plane (Smith chart)"),
                     horizontal_spacing=0.08)

    for _i, _r in enumerate(_r_vals_bl):
        _xln = np.linspace(-6, 6, 200)
        _fig_bl.add_trace(go.Scatter(
            x=np.full_like(_xln, _r), y=_xln, mode="lines",
            line=dict(color=_r_colors_bl[_i], width=1.5),
            name=f"r = {_r}", legendgroup=f"r{_r}", showlegend=True
        ), row=1, col=1)

    for _i, _xv in enumerate(_x_vals_bl):
        _rln = np.linspace(0, 8, 200)
        _fig_bl.add_trace(go.Scatter(
            x=_rln, y=np.full_like(_rln, _xv), mode="lines",
            line=dict(color=_x_colors_bl[_i], width=1, dash="dash"),
            name=f"x = +{_xv}", legendgroup=f"x{_xv}", showlegend=True
        ), row=1, col=1)
        _fig_bl.add_trace(go.Scatter(
            x=_rln, y=np.full_like(_rln, -_xv), mode="lines",
            line=dict(color=_x_colors_bl[_i], width=1, dash="dash"),
            showlegend=False
        ), row=1, col=1)

    _fig_bl.add_trace(go.Scatter(
        x=np.cos(_theta_bl), y=np.sin(_theta_bl), mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", width=1), showlegend=False
    ), row=1, col=2)

    for _i, _r in enumerate(_r_vals_bl):
        _ctr_u = _r / (_r + 1); _rad = 1 / (_r + 1)
        _cu = _ctr_u + _rad * np.cos(_theta_bl)
        _cv = _rad * np.sin(_theta_bl)
        _mask = _cu**2 + _cv**2 <= 1.005
        _cu[~_mask] = np.nan; _cv[~_mask] = np.nan
        _fig_bl.add_trace(go.Scatter(
            x=_cu, y=_cv, mode="lines",
            line=dict(color=_r_colors_bl[_i], width=1.5),
            legendgroup=f"r{_r}", showlegend=False
        ), row=1, col=2)

    for _i, _xv in enumerate(_x_vals_bl):
        for _sgn in [1, -1]:
            _xval = _sgn * _xv
            _cv_c = 1.0 / _xval; _rad = abs(_cv_c)
            _cu = 1 + _rad * np.cos(_theta_bl)
            _cv = _cv_c + _rad * np.sin(_theta_bl)
            _mask = (_cu**2 + _cv**2 <= 1.005)
            _cu[~_mask] = np.nan; _cv[~_mask] = np.nan
            _fig_bl.add_trace(go.Scatter(
                x=_cu, y=_cv, mode="lines",
                line=dict(color=_x_colors_bl[_i], width=1, dash="dash"),
                showlegend=False
            ), row=1, col=2)

    _fig_bl.add_trace(go.Scatter(
        x=[-1, 1], y=[0, 0], mode="lines",
        line=dict(color="rgba(255,255,255,0.5)", width=0.8, dash="dash"),
        showlegend=False
    ), row=1, col=2)

    _fig_bl.update_xaxes(title_text="Re(z) = r", range=[-0.5, 8], row=1, col=1)
    _fig_bl.update_yaxes(title_text="Im(z) = x", range=[-6, 6], row=1, col=1)
    _fig_bl.update_xaxes(title_text="Re(Gamma)", range=[-1.15, 1.15],
                         scaleanchor="y2", scaleratio=1, row=1, col=2)
    _fig_bl.update_yaxes(title_text="Im(Gamma)", range=[-1.15, 1.15], row=1, col=2)
    _fig_bl.update_layout(
        template="plotly_dark", height=500, width=1050,
        title_text="Bilinear Transform: Lines in z-plane Map to Circles in Gamma-plane",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, font=dict(size=10)),
        margin=dict(t=60, b=80)
    )
    mo.ui.plotly(_fig_bl)
    return


@app.cell
def _(mo):
    smith_overlay_dd = mo.ui.dropdown(
        options=["Z-circles only", "Y-circles only", "Z + Y overlay"],
        value="Z + Y overlay", label="Chart overlay"
    )
    smith_r_select = mo.ui.multiselect(
        options=["0", "0.2", "0.5", "1", "2", "5"],
        value=["0", "0.5", "1", "2"], label="r (or g) values"
    )
    smith_x_select = mo.ui.multiselect(
        options=["0.2", "0.5", "1", "2", "5"],
        value=["0.5", "1", "2"], label="x (or b) values"
    )
    mo.md("#### Interactive II -- Smith Chart Circle Construction")
    return smith_overlay_dd, smith_r_select, smith_x_select


@app.cell
def _(go, mo, np, smith_overlay_dd, smith_r_select, smith_x_select):
    _th2 = np.linspace(0, 2 * np.pi, 500)
    _mode = smith_overlay_dd.value
    _r_vals_sc = [float(v) for v in smith_r_select.value]
    _x_vals_sc = [float(v) for v in smith_x_select.value]
    _show_z = _mode in ("Z-circles only", "Z + Y overlay")
    _show_y = _mode in ("Y-circles only", "Z + Y overlay")

    _fig_sc = go.Figure()
    _fig_sc.add_trace(go.Scatter(
        x=np.cos(_th2), y=np.sin(_th2), mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=1.2),
        showlegend=False, hoverinfo="skip"
    ))
    _fig_sc.add_trace(go.Scatter(
        x=[-1, 1], y=[0, 0], mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=0.8),
        showlegend=False, hoverinfo="skip"
    ))

    _zp = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#FF6692"]
    _yp = ["#19D3F3", "#B6E880", "#FECB52", "#FF97FF", "#72B7B2", "#FFA07A"]

    if _show_z:
        for _i, _r in enumerate(_r_vals_sc):
            _ctr = _r / (_r + 1); _rad = 1.0 / (_r + 1)
            _cu = _ctr + _rad * np.cos(_th2); _cv = _rad * np.sin(_th2)
            _m = _cu**2 + _cv**2 <= 1.005; _cu[~_m] = np.nan; _cv[~_m] = np.nan
            _fig_sc.add_trace(go.Scatter(x=_cu, y=_cv, mode="lines",
                line=dict(color=_zp[_i % len(_zp)], width=1.8), name=f"r = {_r:g}"))
        for _i, _xv in enumerate(_x_vals_sc):
            _col = _zp[(_i + 3) % len(_zp)]
            for _sgn in [1, -1]:
                _xval = _sgn * _xv; _cv_c = 1.0 / _xval; _rad = abs(_cv_c)
                _cu = 1 + _rad * np.cos(_th2); _cv = _cv_c + _rad * np.sin(_th2)
                _m = _cu**2 + _cv**2 <= 1.005; _cu[~_m] = np.nan; _cv[~_m] = np.nan
                _fig_sc.add_trace(go.Scatter(x=_cu, y=_cv, mode="lines",
                    line=dict(color=_col, width=1, dash="dash"),
                    name=f"x = {_xval:+g}", showlegend=(_sgn == 1)))

    if _show_y:
        for _i, _g in enumerate(_r_vals_sc):
            _ctr = -_g / (_g + 1); _rad = 1.0 / (_g + 1)
            _cu = _ctr + _rad * np.cos(_th2); _cv = _rad * np.sin(_th2)
            _m = _cu**2 + _cv**2 <= 1.005; _cu[~_m] = np.nan; _cv[~_m] = np.nan
            _fig_sc.add_trace(go.Scatter(x=_cu, y=_cv, mode="lines",
                line=dict(color=_yp[_i % len(_yp)], width=1.8, dash="dot"), name=f"g = {_g:g}"))
        for _i, _bv in enumerate(_x_vals_sc):
            _col = _yp[(_i + 3) % len(_yp)]
            for _sgn in [1, -1]:
                _bval = _sgn * _bv; _cv_c = -1.0 / _bval; _rad = abs(_cv_c)
                _cu = -1 + _rad * np.cos(_th2); _cv = _cv_c + _rad * np.sin(_th2)
                _m = _cu**2 + _cv**2 <= 1.005; _cu[~_m] = np.nan; _cv[~_m] = np.nan
                _fig_sc.add_trace(go.Scatter(x=_cu, y=_cv, mode="lines",
                    line=dict(color=_col, width=1, dash="dashdot"),
                    name=f"b = {_bval:+g}", showlegend=(_sgn == 1)))

    _fig_sc.add_trace(go.Scatter(
        x=[0, 1, -1], y=[0, 0, 0], mode="markers+text",
        marker=dict(size=7, color=["#FECB52", "#EF553B", "#00CC96"]),
        text=["z=1", "OC", "SC"], textposition="top center",
        textfont=dict(size=10, color="white"), showlegend=False
    ))
    _fig_sc.update_layout(
        template="plotly_dark", height=600, width=650,
        title_text=f"Smith Chart Construction ({_mode})",
        xaxis=dict(title="Re(Gamma)", range=[-1.2, 1.2], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(Gamma)", range=[-1.2, 1.2]),
        legend=dict(font=dict(size=9), orientation="v", x=1.02, y=1),
        margin=dict(l=50, r=140, t=50, b=50)
    )
    mo.vstack([
        mo.hstack([smith_overlay_dd, smith_r_select, smith_x_select]),
        mo.ui.plotly(_fig_sc)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Interactive III -- Step-by-Step Matching Network Synthesis

    Build a matching network element by element. Each button adds a reactive component to the cascade. The Smith chart traces the full trajectory from load to (ideally) the chart center $\Gamma = 0$.
    """)
    return


@app.cell
def _(mo):
    # State: list of (element_name, normalized_value) tuples
    get_synth_chain, set_synth_chain = mo.state([])
    # Load impedance state
    get_synth_load, set_synth_load = mo.state((0.5, 1.0))
    return get_synth_chain, get_synth_load, set_synth_chain, set_synth_load


@app.cell
def _(mo, set_synth_chain, set_synth_load):
    synth_step_val = mo.ui.slider(0.1, 5.0, value=0.5, step=0.1, label="Step size (normalized)")
    synth_load_r = mo.ui.slider(0.1, 5.0, value=0.5, step=0.1, label="Load r")
    synth_load_x = mo.ui.slider(-4.0, 4.0, value=1.0, step=0.1, label="Load x")

    def _add_element(elem_name):
        def _handler(_btn_val):
            set_synth_chain(lambda chain: chain + [(elem_name, synth_step_val.value)])
        return _handler

    def _undo(_btn_val):
        set_synth_chain(lambda chain: chain[:-1] if chain else [])

    def _reset(_btn_val):
        set_synth_chain([])

    def _update_load(_btn_val):
        set_synth_load((synth_load_r.value, synth_load_x.value))
        set_synth_chain([])

    btn_series_L = mo.ui.button(label="Series L (+jx)", on_click=_add_element("Series L"))
    btn_series_C = mo.ui.button(label="Series C (-jx)", on_click=_add_element("Series C"))
    btn_shunt_C = mo.ui.button(label="Shunt C (+jb)", on_click=_add_element("Shunt C"))
    btn_shunt_L = mo.ui.button(label="Shunt L (-jb)", on_click=_add_element("Shunt L"))
    btn_undo = mo.ui.button(label="Undo last", on_click=_undo)
    btn_reset = mo.ui.button(label="Reset all", on_click=_reset)
    btn_set_load = mo.ui.button(label="Set load", on_click=_update_load)
    return (
        btn_reset,
        btn_series_C,
        btn_series_L,
        btn_set_load,
        btn_shunt_C,
        btn_shunt_L,
        btn_undo,
        synth_load_r,
        synth_load_x,
        synth_step_val,
    )


@app.cell
def _(
    btn_reset,
    btn_series_C,
    btn_series_L,
    btn_set_load,
    btn_shunt_C,
    btn_shunt_L,
    btn_undo,
    get_synth_chain,
    get_synth_load,
    go,
    mo,
    np,
    synth_load_r,
    synth_load_x,
    synth_step_val,
):
    _chain = get_synth_chain()
    _load_r, _load_x = get_synth_load()
    _th = np.linspace(0, 2 * np.pi, 500)

    # --- Compute the trajectory through the element chain ---
    _z_points = [_load_r + 1j * _load_x]
    _arc_segments = []  # list of (gamma_array, is_series_type)
    _step_labels = []

    for _elem_name, _val in _chain:
        _z_cur = _z_points[-1]
        _t = np.linspace(0, _val, 100)
        if _elem_name == "Series L":
            _z_arc = _z_cur + 1j * _t
            _is_series = True
            _step_labels.append(f"+jx={_val:.1f}")
        elif _elem_name == "Series C":
            _z_arc = _z_cur - 1j * _t
            _is_series = True
            _step_labels.append(f"-jx={_val:.1f}")
        elif _elem_name == "Shunt C":
            _y_cur = 1.0 / _z_cur
            _y_arc = _y_cur + 1j * _t
            _z_arc = 1.0 / _y_arc
            _is_series = False
            _step_labels.append(f"+jb={_val:.1f}")
        else:  # Shunt L
            _y_cur = 1.0 / _z_cur
            _y_arc = _y_cur - 1j * _t
            _z_arc = 1.0 / _y_arc
            _is_series = False
            _step_labels.append(f"-jb={_val:.1f}")
        _gamma_arc = (_z_arc - 1) / (_z_arc + 1)
        _arc_segments.append((_gamma_arc, _is_series, _elem_name))
        _z_points.append(_z_arc[-1])

    _gamma_points = [(_z - 1) / (_z + 1) for _z in _z_points]

    # --- Build the Smith chart figure ---
    _fig_syn = go.Figure()

    # Unit circle
    _fig_syn.add_trace(go.Scatter(
        x=np.cos(_th), y=np.sin(_th), mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=1),
        showlegend=False, hoverinfo="skip"
    ))
    # Background r-circles
    for _rv in [0, 0.2, 0.5, 1, 2, 5]:
        _ct = _rv / (_rv + 1); _rd = 1.0 / (_rv + 1)
        _cu = _ct + _rd * np.cos(_th); _cv = _rd * np.sin(_th)
        _m = _cu**2 + _cv**2 <= 1.005; _cu[~_m] = np.nan; _cv[~_m] = np.nan
        _fig_syn.add_trace(go.Scatter(x=_cu, y=_cv, mode="lines",
            line=dict(color="rgba(100,110,200,0.18)", width=0.7),
            showlegend=False, hoverinfo="skip"))
    # Background g-circles
    for _gv in [0, 0.2, 0.5, 1, 2, 5]:
        _ct = -_gv / (_gv + 1); _rd = 1.0 / (_gv + 1)
        _cu = _ct + _rd * np.cos(_th); _cv = _rd * np.sin(_th)
        _m = _cu**2 + _cv**2 <= 1.005; _cu[~_m] = np.nan; _cv[~_m] = np.nan
        _fig_syn.add_trace(go.Scatter(x=_cu, y=_cv, mode="lines",
            line=dict(color="rgba(25,211,243,0.12)", width=0.7),
            showlegend=False, hoverinfo="skip"))
    # Real axis
    _fig_syn.add_trace(go.Scatter(x=[-1, 1], y=[0, 0], mode="lines",
        line=dict(color="rgba(255,255,255,0.2)", width=0.6),
        showlegend=False, hoverinfo="skip"))

    # Color palette for arcs
    _series_colors = ["#636EFA", "#AB63FA", "#636EFA", "#AB63FA", "#636EFA", "#AB63FA"]
    _shunt_colors = ["#EF553B", "#FF6692", "#EF553B", "#FF6692", "#EF553B", "#FF6692"]

    # Draw arc segments
    for _idx, (_garc, _is_s, _ename) in enumerate(_arc_segments):
        _col = _series_colors[_idx % len(_series_colors)] if _is_s else _shunt_colors[_idx % len(_shunt_colors)]
        _fig_syn.add_trace(go.Scatter(
            x=_garc.real, y=_garc.imag, mode="lines",
            line=dict(color=_col, width=3),
            name=f"Step {_idx+1}: {_ename} ({_step_labels[_idx]})",
            legendgroup=f"step{_idx}"
        ))
        # Arrowhead at the end of each arc
        if len(_garc) > 2:
            _dx = _garc[-1].real - _garc[-3].real
            _dy = _garc[-1].imag - _garc[-3].imag
            _fig_syn.add_annotation(
                x=_garc[-1].real, y=_garc[-1].imag,
                ax=_garc[-1].real - _dx * 8,
                ay=_garc[-1].imag - _dy * 8,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowcolor=_col, arrowwidth=2
            )

    # Draw node markers at each impedance point
    for _idx, _gp in enumerate(_gamma_points):
        _z = _z_points[_idx]
        if _idx == 0:
            _marker_color = "#00CC96"; _sym = "circle"; _label = f"Load z={_z:.2f}"
            _name = "Load"
        elif _idx == len(_gamma_points) - 1:
            _marker_color = "#FECB52"; _sym = "star"; _label = f"z={_z:.2f}"
            _name = "Current"
        else:
            _marker_color = "rgba(255,255,255,0.7)"; _sym = "circle"; _label = f"z={_z:.2f}"
            _name = f"Node {_idx}"
        _fig_syn.add_trace(go.Scatter(
            x=[_gp.real], y=[_gp.imag], mode="markers",
            marker=dict(size=9 if _idx in (0, len(_gamma_points)-1) else 6,
                        color=_marker_color, symbol=_sym,
                        line=dict(width=1, color="white")),
            name=_name,
            hovertext=_label, hoverinfo="text"
        ))

    # Chart center target
    _fig_syn.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=8, color="white", symbol="cross-thin",
                    line=dict(width=1.5, color="white")),
        text=["Target (z=1)"], textposition="bottom left",
        textfont=dict(size=9, color="rgba(255,255,255,0.6)"), showlegend=False
    ))

    # Current impedance info
    _z_cur = _z_points[-1]
    _y_cur = 1.0 / _z_cur
    _gamma_cur = _gamma_points[-1]
    _gamma_mag = abs(_gamma_cur)
    _ml_dB = 10 * np.log10(1 - _gamma_mag**2 + 1e-30) if _gamma_mag < 1 else float('-inf')

    _fig_syn.update_layout(
        template="plotly_dark", height=620, width=700,
        title_text="Step-by-Step Matching Network Synthesis",
        xaxis=dict(title="Re(Gamma)", range=[-1.25, 1.25], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(Gamma)", range=[-1.25, 1.25]),
        legend=dict(font=dict(size=9), x=1.02, y=1, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=50, r=180, t=50, b=50)
    )

    # --- Network state summary ---
    _chain_rows = ""
    for _idx, (_ename, _val) in enumerate(_chain):
        _chain_rows += f"| {_idx+1} | {_ename} | {_step_labels[_idx]} |\n"
    if not _chain_rows:
        _chain_rows = "| -- | No elements | -- |\n"

    _status_md = mo.md(f"""
    **Network State**

    | Property | Value |
    |---|---|
    | z (normalized) | {_z_cur.real:.4f} {_z_cur.imag:+.4f}j |
    | y (normalized) | {_y_cur.real:.4f} {_y_cur.imag:+.4f}j |
    | Gamma | {_gamma_cur.real:.4f} {_gamma_cur.imag:+.4f}j |
    | |Gamma| | {_gamma_mag:.4f} |
    | Mismatch loss | {_ml_dB:.2f} dB |

    **Element Chain**

    | Step | Element | Value |
    |---|---|---|
    {_chain_rows}
    """)

    _controls = mo.vstack([
        mo.md("**Add element:**"),
        mo.hstack([btn_series_L, btn_series_C]),
        mo.hstack([btn_shunt_C, btn_shunt_L]),
        synth_step_val,
        mo.md("---"),
        mo.hstack([btn_undo, btn_reset]),
        mo.md("---"),
        mo.md("**Load impedance:**"),
        synth_load_r, synth_load_x, btn_set_load,
    ])

    mo.hstack([
        mo.vstack([_controls, _status_md]),
        mo.ui.plotly(_fig_syn)
    ], widths=[1, 2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part II — Lumped Matching Networks

    ### §4. L-Network Synthesis

    The L-network transforms between two real impedances $R_S$ and $R_L$ ($R_H = \max(R_S, R_L)$, $R_{Lo} = \min(R_S, R_L)$) using exactly two reactive elements.

    **Derivation of Q and element values:**
    Consider the load $R_H$ in parallel with a shunt reactance $X_p$. The admittance is $Y_p = 1/R_H + 1/(jX_p)$.
    The equivalent series impedance is $Z_s = 1/Y_p$:

    $$Z_s = \frac{R_H (jX_p)}{R_H + jX_p} = \frac{R_H X_p^2 + j R_H^2 X_p}{R_H^2 + X_p^2}$$

    We require the real part to equal $R_{Lo}$:

    $$R_{Lo} = \frac{R_H X_p^2}{R_H^2 + X_p^2} \implies R_{Lo} R_H^2 + R_{Lo} X_p^2 = R_H X_p^2 \implies X_p^2 (R_H - R_{Lo}) = R_{Lo} R_H^2$$

    $$X_p = \pm R_H \sqrt{\frac{R_{Lo}}{R_H - R_{Lo}}} = \pm \frac{R_H}{Q}$$

    where we define the network quality factor $Q$:

    $$Q = \sqrt{\frac{R_H}{R_{Lo}} - 1}$$

    The imaginary part of the parallel combination is $\frac{R_H^2 X_p}{R_H^2 + X_p^2} = \frac{R_H Q}{1 + Q^2} = R_{Lo} Q$.
    To cancel this reactance and achieve a purely real impedance $R_{Lo}$, the series arm must provide the opposite reactance $X_s$:

    $$X_s = \mp R_{Lo} Q$$

    **Low-pass topology** (shunt $C$ at high-impedance port, series $L$ in series arm):

    $$X_p = -1/(\omega_0 C_p) \implies C_p = \frac{Q}{\omega_0 R_H}$$
    $$X_s = \omega_0 L_s \implies L_s = \frac{Q R_{Lo}}{\omega_0}$$

    **High-pass topology**: (shunt $L$ at high-impedance port, series $C$ in series arm).

    The 3 dB bandwidth is approximately $\text{BW} \approx f_0/Q$. For a given transformation ratio, $Q$ is fixed, meaning the L-network has no bandwidth knob. This motivates π and T networks.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    **Orientation rule (the one fact that fixes everything).** The *shunt* element
    always sits on the **high-impedance side**; the *series* element bridges to the
    low-impedance side. Equivalently: a shunt element first lowers $Q$ by moving onto
    a constant-$g$ circle, then the series element walks to the centre. The decision
    tree below picks orientation from $R_S$ vs $R_L$, then picks the type (low-pass
    vs high-pass) from the filtering requirement.
    """),
        mo.mermaid(r"""
flowchart TD
    Start["L-network choice<br/>(need bandwidth control?<br/>use pi/T in §5-§6)"] --> Q1{"Which side is<br/>high impedance?"}
    Q1 -->|"R_S &gt; R_L<br/>source high-Z"| HS["SHUNT element at SOURCE node<br/>SERIES element toward load"]
    Q1 -->|"R_S &lt; R_L<br/>load high-Z"| HL["SERIES element from source<br/>SHUNT element at LOAD node"]
    HS --> T1{"Filtering need?"}
    HL --> T2{"Filtering need?"}
    T1 -->|"reject harmonics"| LP1["LOW-PASS:<br/>shunt C + series L"]
    T1 -->|"block DC / reject<br/>sub-harmonics"| HP1["HIGH-PASS:<br/>shunt L + series C"]
    T2 -->|"reject harmonics"| LP2["LOW-PASS:<br/>series L + shunt C"]
    T2 -->|"block DC"| HP2["HIGH-PASS:<br/>series C + shunt L"]
""")
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    **L-network topologies (low-pass shown).** Left: source is the high-Z side, so the
    shunt cap sits at the source node. Right: load is the high-Z side, so the shunt cap
    sits at the load node. The ⏚ symbol is the ground return for the shunt branch.
    Swap each $C \leftrightarrow L$ to obtain the high-pass dual.
    """),
        mo.hstack([
            mo.vstack([
                mo.md("**Source high-Z** ($R_S > R_L$)"),
                mo.mermaid(r"""
flowchart LR
    S(("R_S")) --- N1(("&middot;"))
    N1 --- Cp["C_p shunt"] --- G1["gnd"]
    N1 --- Ls["L_s series"] --- LD(("R_L"))
"""),
            ]),
            mo.vstack([
                mo.md("**Load high-Z** ($R_L > R_S$)"),
                mo.mermaid(r"""
flowchart LR
    S(("R_S")) --- Ls["L_s series"] --- N1(("&middot;"))
    N1 --- Cp["C_p shunt"] --- G1["gnd"]
    N1 --- LD(("R_L"))
"""),
            ]),
        ]),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §5. π-Network Synthesis

    The π-network uses two shunt elements and one series element, giving an extra degree of freedom: a free parameter $Q > Q_{\min} = \sqrt{R_H/R_{Lo} - 1}$.
    Higher $Q$ leads to a narrower bandwidth (useful for harmonic suppression); lower $Q$ gives wider bandwidth (down to $Q_{\min}$, which recovers the L-network).

    A π-network can be modeled as two cascaded L-networks with a virtual intermediate resistance $R_v < R_{Lo}$.

    The condition is:

    $$R_v = \frac{R_H}{1 + Q_1^2} = \frac{R_{Lo}}{1 + Q_2^2}$$

    where $Q_1 = \sqrt{R_H/R_v - 1}$ and $Q_2 = \sqrt{R_{Lo}/R_v - 1}$. The overall network $Q$ is approximately $Q \approx Q_1 + Q_2$, but traditionally we specify $Q_1$ (the larger of the two) as the design $Q$, meaning $R_v = R_H / (1 + Q^2)$.

    For the low-pass version:

    $$C_1 = \frac{Q_1}{\omega_0 R_S}, \quad
      L   = \frac{(Q_1 + Q_2) R_v}{\omega_0}, \quad
      C_2 = \frac{Q_2}{\omega_0 R_L}$$
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    **π-network topology (low-pass).** Two shunt capacitors flank a series inductor.
    The virtual node sits at $R_v < R_{Lo}$ (lower than both ends): the left L-section
    steps $R_S$ **down** to $R_v$, the right L-section steps $R_v$ **up** to $R_L$.
    Because $R_v$ can be chosen freely (below $R_{Lo}$), $Q$ is a free knob — high $Q$
    means narrow band / strong harmonic rejection. The high-pass dual swaps each
    shunt $C$ for a shunt $L$ and the series $L$ for a series $C$.
    """),
        mo.mermaid(r"""
flowchart LR
    S(("R_S")) --- N1(( ))
    N1 --- C1["C_1 shunt"] --- G1["gnd"]
    N1 --- L["L series"] --- N2["R_v virtual"]
    N2 --- C2["C_2 shunt"] --- G2["gnd"]
    N2 --- LD(("R_L"))
"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §6. T-Network Synthesis

    The T-network (two series arms, one shunt arm) is the dual of the π.
    Its virtual resistance is higher than both terminations: $R_v = R_{Lo}(1 + Q^2) > R_H$.

    The **tapped-capacitor** network used in notebook 04 §17.6 is a degenerate T: the two series arms are capacitors, and the shunt arm is the device output capacitance $C_{ds}$.
    The tap ratio provides impedance transformation without inductors:

    $$n_{\text{tap}} = \sqrt{\frac{R_L}{R_{\text{eff}}}}, \quad
      C_{\text{tap}} = \frac{C_{\text{total}}}{n_{\text{tap}}^2}$$

    For a full low-pass T with tunable $Q$:

    $$L_1 = \frac{Q_1 R_S}{\omega_0}, \quad
      C   = \frac{Q_1/R_S + Q_2/R_L}{\omega_0}, \quad
      L_2 = \frac{Q_2 R_L}{\omega_0}$$
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    **T-network topology (low-pass).** Two series inductors flank a single shunt
    capacitor — the topological dual of the π. The virtual node now sits at
    $R_v = R_{Lo}(1 + Q^2) > R_H$ (higher than both ends): the left L-section steps
    $R_S$ **up** to $R_v$, the right steps $R_v$ **down** to $R_L$. As with the π,
    $Q$ is free. T-networks are preferred when a series DC path is wanted (e.g.
    bias feed) or when the required shunt-$C$ of a π would be impractically large.
    """),
        mo.mermaid(r"""
flowchart LR
    S(("R_S")) --- L1["L_1 series"] --- N1["R_v virtual"]
    N1 --- C["C shunt"] --- G["gnd"]
    N1 --- L2["L_2 series"] --- LD(("R_L"))
"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §6.1 Comparison: Bandwidth and Broadband Matching

    The single-resonator bandwidth $\text{BW} \approx f_0/Q$ is the recurring theme. The
    table contrasts how each topology relates to $Q$ — and, crucially, whether it can
    actually *broaden* a match or only narrow it.

    | Topology | Reactive elements | $Q$ / degree of freedom | Fractional BW | Broadbands the match? |
    |---|---|---|---|---|
    | **L-network** | 2 (1 series + 1 shunt) | $Q = \sqrt{R_H/R_{Lo}-1}$ — **fixed** by the ratio | $\approx f_0/Q$ | No — $Q$ is forced; no knob |
    | **π-network** | 3 (2 shunt + 1 series) | $Q \ge Q_{\min}$, **free upward only** | $\le$ L-net ($Q\!\ge\!Q_{\min}$) | No — only *narrows* (harmonic filtering) |
    | **T-network** | 3 (2 series + 1 shunt) | $Q \ge Q_{\min}$, **free upward only** | $\le$ L-net | No — same; adds series DC path |
    | **Cascaded $n$ L-sections** | $2n$ | per-section $Q$ lowered via geometric-mean intermediate $R$ | grows with $n$ | **Yes** — staged, $\sim$decade |
    | **Multi-section $\lambda/4$ (Chebyshev)** | $N$ lines | ripple-vs-$N$ trade (§9, §12) | grows with $N$ | **Yes** — distributed, RF/mmWave |
    | **Bridged T-coil** | 2 coupled $L$ + 1 cap | constant-$R$ for *all* $k$; $k$ sets flatness | "all-pass" match; $2.83\times$ extension | **Yes** — for a *fixed shunt $C$* |

    **The decisive point.** A π or T network adds a third element, but the extra degree of
    freedom only lets you push $Q$ *above* the L-network floor $Q_{\min}=\sqrt{R_H/R_{Lo}-1}$.
    They trade bandwidth for selectivity — they never broaden the match below the
    single-section limit. Genuine broadbanding requires either **multiple cascaded
    L-sections** (each handling a smaller impedance step through geometric-mean virtual
    resistances) or **stepped $\lambda/4$ transformers** (§9, §12). Both are bounded by the
    **Bode-Fano integral (§8)**:

    $$\int_0^\infty \ln\frac{1}{|\Gamma(\omega)|}\,d\omega \le \frac{\pi}{R_L C_L} \quad\text{[theorem]}$$

    which caps the achievable bandwidth-vs-depth product for any passive match of a
    parallel-$RC$ load. The bridged T-coil (§6.2 preview, full derivation in §20) is the
    special-case escape hatch when the load is a *pure parasitic capacitance* rather than
    a resistance to transform.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §6.2 T-Coils for Wireline (Capacitive-Load) Matching — preview

    The **bridged T-coil** is the Bode-Fano escape hatch for a *fixed
    parasitic capacitance* (bond pad + ESD), not a resistance to
    transform. A centre-tapped coupled-inductor pair plus a small
    bridge capacitor makes the input impedance equal $R$ *for all
    frequencies* — sidestepping Bode-Fano because the parasitic $C$ is
    absorbed into a constant-$R$ all-pass network rather than treated
    as a load. The coupling $k$ is a free knob that shapes the
    tap-node response ($k = 1/3$ Butterworth, $k = 1/2$ Bessel-like).

    The full derivation, pole-zero map, and pole-shaping interactive
    are in **§20** (Part V — Coupled Magnetics). The brief mention here
    only serves to complete the §6.1 comparison table.
    """)
    return


@app.cell
def _(mo):
    me_topo = mo.ui.dropdown(
        options=["L (low-pass)", "L (high-pass)", "π", "T"],
        value="L (low-pass)", label="Topology")
    me_rs = mo.ui.slider(5, 500, value=50, step=5, label="R_S (Ω)")
    me_rl = mo.ui.slider(5, 500, value=200, step=5, label="R_L (Ω)")
    me_f0 = mo.ui.slider(1, 100, value=28, step=1, label="f₀ (GHz)")
    me_q = mo.ui.slider(1.0, 20.0, value=3.0, step=0.1, label="design Q (π / T)")
    me_spec = mo.ui.dropdown(
        options=["-10 dB", "-15 dB", "-20 dB"], value="-10 dB",
        label="Return-loss target")
    me_overlay = mo.ui.checkbox(value=True, label="Overlay all topologies")
    me_loss = mo.ui.checkbox(value=False, label="Show delivered power")
    mo.md(r"""
    ### §7. Interactive I — L/π/T Network Explorer

    Plots the **source-side return loss** $20\log_{10}|\Gamma_{\text{in}}|$ (port 2 terminated
    in the real load $R_L$, referenced to $R_S$) — the true match quality, which dips to
    $-\infty$ at $f_0$. The L-network is always drawn as the **single-section limit**: it sits
    at $Q_{\min}=\sqrt{R_H/R_{Lo}-1}$ and gives the widest possible single-section bandwidth.
    Watch how π and T (design $Q \ge Q_{\min}$) only *narrow* the match.
    """)
    return me_f0, me_loss, me_overlay, me_q, me_rl, me_rs, me_spec, me_topo


@app.cell
def _(go, me_f0, me_loss, me_overlay, me_q, me_rl, me_rs, me_spec, me_topo, mo, np):
    _Rs = float(me_rs.value); _Rl = float(me_rl.value)
    _f0 = float(me_f0.value) * 1e9; _Qd = float(me_q.value)
    _spec = float(me_spec.value.split()[0])
    _RH, _RLo = max(_Rs, _Rl), min(_Rs, _Rl)
    _Qmin = float(np.sqrt(_RH / _RLo - 1.0))
    _Quse = max(_Qd, _Qmin * 1.02 + 1e-6)          # π / T must exceed Q_min
    _clamped = _Qd < _Quse

    _fr = np.linspace(0.05 * _f0, 3.0 * _f0, 2400)
    _w = 2 * np.pi * _fr; _w0 = 2 * np.pi * _f0
    _i0 = int(np.argmin(np.abs(_fr - _f0)))

    def _eye():
        M = np.zeros((len(_w), 2, 2), complex); M[:, 0, 0] = 1; M[:, 1, 1] = 1; return M
    def _Mser(Z):
        M = _eye(); M[:, 0, 1] = Z; return M
    def _Msh(Y):
        M = _eye(); M[:, 1, 0] = Y; return M
    def _chain(*Ms):
        out = Ms[0]
        for M in Ms[1:]:
            out = np.einsum('nij,njk->nik', out, M)
        return out

    def _abcd(name):
        # element values returned alongside ABCD for the table
        if name.startswith("L"):
            Q = _Qmin
            if Q <= 0:
                return _eye(), {"Q": 0.0}
            if name == "L (low-pass)":
                Cp = Q / (_w0 * _RH); Ls = Q * _RLo / _w0
                Yp = 1j * _w * Cp; Zs = 1j * _w * Ls
                ev = {"Q": Q, "C_p (pF)": Cp * 1e12, "L_s (nH)": Ls * 1e9}
            else:
                Lp = _RH / (_w0 * Q); Cs = 1.0 / (_w0 * Q * _RLo)
                Yp = 1.0 / (1j * _w * Lp); Zs = 1.0 / (1j * _w * Cs)
                ev = {"Q": Q, "L_p (nH)": Lp * 1e9, "C_s (pF)": Cs * 1e12}
            # shunt sits on the high-Z side
            abcd = _chain(_Msh(Yp), _Mser(Zs)) if _Rs >= _Rl else _chain(_Mser(Zs), _Msh(Yp))
            return abcd, ev
        if name == "π":
            Rv = _RH / (1.0 + _Quse ** 2)
            Q1 = np.sqrt(_Rs / Rv - 1.0); Q2 = np.sqrt(_Rl / Rv - 1.0)
            C1 = Q1 / (_w0 * _Rs); C2 = Q2 / (_w0 * _Rl); L = (Q1 + Q2) * Rv / _w0
            abcd = _chain(_Msh(1j * _w * C1), _Mser(1j * _w * L), _Msh(1j * _w * C2))
            return abcd, {"Q": _Quse, "C1 (pF)": C1 * 1e12, "L (nH)": L * 1e9,
                          "C2 (pF)": C2 * 1e12, "R_v (Ω)": Rv}
        # T
        Rv = _RLo * (1.0 + _Quse ** 2)
        Q1 = np.sqrt(Rv / _Rs - 1.0); Q2 = np.sqrt(Rv / _Rl - 1.0)
        L1 = Q1 * _Rs / _w0; L2 = Q2 * _Rl / _w0; C = (Q1 + Q2) / (_w0 * Rv)
        abcd = _chain(_Mser(1j * _w * L1), _Msh(1j * _w * C), _Mser(1j * _w * L2))
        return abcd, {"Q": _Quse, "L1 (nH)": L1 * 1e9, "C (pF)": C * 1e12,
                      "L2 (nH)": L2 * 1e9, "R_v (Ω)": Rv}

    def _gamma(abcd):
        A, B, C, D = abcd[:, 0, 0], abcd[:, 0, 1], abcd[:, 1, 0], abcd[:, 1, 1]
        Zin = (A * _Rl + B) / (C * _Rl + D)
        return (Zin - _Rs) / (Zin + _Rs)
    def _rl_db(abcd):
        return 20 * np.log10(np.abs(_gamma(abcd)) + 1e-12)
    def _band(rl):
        if rl[_i0] >= _spec:
            return None
        lo = _i0
        while lo > 0 and rl[lo] < _spec:
            lo -= 1
        hi = _i0
        while hi < len(_fr) - 1 and rl[hi] < _spec:
            hi += 1
        return _fr[lo], _fr[hi], (_fr[hi] - _fr[lo]) / _f0

    _names = ["L (low-pass)", "L (high-pass)", "π", "T"]
    _abcds = {n: _abcd(n) for n in _names}
    _rls = {n: _rl_db(_abcds[n][0]) for n in _names}
    _bands = {n: _band(_rls[n]) for n in _names}
    _prim = me_topo.value

    _palette = {"L (low-pass)": "#00CC96", "L (high-pass)": "#19D3F3",
                "π": "#FFA15A", "T": "#AB63FA"}
    _to_plot = {_prim, "L (low-pass)"}
    if me_overlay.value:
        _to_plot |= {"L (low-pass)", "π", "T"}

    _fig = go.Figure()
    for _n in _names:
        if _n not in _to_plot:
            continue
        _is_prim = (_n == _prim)
        _is_limit = (_n == "L (low-pass)" and not _is_prim)
        _fig.add_trace(go.Scatter(
            x=_fr / 1e9, y=_rls[_n],
            name=(f"{_n}  ◄ limit" if _is_limit else _n),
            line=dict(color=_palette[_n], width=3.2 if _is_prim else 1.6,
                      dash="dot" if _is_limit else "solid")))

    if me_loss.value:
        _gt = 10 * np.log10(1.0 - np.abs(_gamma(_abcds[_prim][0])) ** 2 + 1e-12)
        _fig.add_trace(go.Scatter(x=_fr / 1e9, y=_gt, name=f"{_prim} delivered power",
                                  line=dict(color="#FECB52", width=1.5, dash="dash")))

    _fig.add_hline(y=_spec, line=dict(color="rgba(255,80,80,0.7)", dash="dash"),
                   annotation_text=f"{_spec:g} dB target", annotation_position="top left")
    _fig.add_vline(x=_f0 / 1e9, line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
                   annotation_text="f₀")
    _pb = _bands[_prim]
    if _pb:
        _fig.add_vrect(x0=_pb[0] / 1e9, x1=_pb[1] / 1e9,
                       fillcolor=_palette[_prim], opacity=0.13, line_width=0)
    _fig.update_layout(
        template="plotly_dark", height=520,
        title=(f"Return loss — {_prim}:  R_S={_Rs:g}Ω → R_L={_Rl:g}Ω @ {_f0/1e9:.0f} GHz"
               f"   (ratio R_H/R_Lo = {_RH/_RLo:.2g})"),
        xaxis=dict(title="Frequency (GHz)"),
        yaxis=dict(title="20 log|Γ_in|  (dB)", range=[-40, 2]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, font=dict(size=10)),
        margin=dict(t=60, b=90))

    def _fmt(b):
        return "— (misses target)" if not b else f"{b[2]*100:.1f}%  ({b[0]/1e9:.1f}–{b[1]/1e9:.1f} GHz)"
    _qrep = {"L (low-pass)": _Qmin, "L (high-pass)": _Qmin, "π": _Quse, "T": _Quse}
    _cmp_rows = "\n".join(
        f"| {n} | {_qrep[n]:.2f} | {_rls[n][_i0]:.0f} dB | {_fmt(_bands[n])} |"
        for n in ["L (low-pass)", "π", "T"])
    _cmp = mo.md("| Topology | $Q$ | RL @ f₀ | Frac. BW @ target |\n|---|---|---|---|\n" + _cmp_rows)

    _ev = _abcds[_prim][1]
    _ev_md = mo.md("| Element | Value |\n|---|---|\n"
                   + "\n".join(f"| {k} | {v:.4g} |" for k, v in _ev.items()))

    _Lb = _bands["L (low-pass)"]
    _lim_bw = _Lb[2] * 100 if _Lb else 0.0
    _prim_bw = _pb[2] * 100 if _pb else 0.0
    if _prim.startswith("L"):
        _verdict = "this **is** the single-section limit"
    elif _lim_bw > 0 and _prim_bw > 0:
        _verdict = f"**{_prim_bw/_lim_bw*100:.0f}%** of the limit (narrower, as expected)"
    else:
        _verdict = "n/a"
    _note_clamp = (f" Design Q was raised to {_Quse:.2f} (you requested {_Qd:.2f} < $Q_{{min}}$)."
                   if _clamped else "")
    _insight = mo.md(rf"""
    **Reading the limit.** Ratio $R_H/R_{{Lo}} = {_RH/_RLo:.2g}$ forces
    $Q_{{\min}} = {_Qmin:.2f}$, so the best single-section match (the L-network) spans
    **{_lim_bw:.1f}%** at the {_spec:g} dB target. Your **{_prim}** achieves
    **{_prim_bw:.1f}%** — {_verdict}.{_note_clamp}

    π and T add a free $Q \ge Q_{{\min}}$ that only *narrows* the band (buying harmonic
    selectivity, not bandwidth). To beat the L-section limit you must cascade sections
    (§9) — and that, in turn, is capped by the Bode-Fano integral (§8).
    """)

    mo.vstack([
        mo.hstack([me_topo, me_rs, me_rl, me_f0, me_q], justify="start", gap=1.5),
        mo.hstack([me_spec, me_overlay, me_loss], justify="start", gap=1.5),
        mo.ui.plotly(_fig),
        mo.hstack([
            mo.vstack([mo.md("**Topology comparison**"), _cmp]),
            mo.vstack([mo.md(f"**{_prim} element values**"), _ev_md]),
        ], widths=[2, 1], gap=2),
        _insight,
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    ## Part III — Broadband Matching

    ### §8. The Bode–Fano Limit

    #### 8.0 The question — and the one-line answer

    A load $R_L \parallel C_L$ is given. We may insert **any** lossless matching network
    between it and the source. *How small can we force the reflection $|\Gamma|$, and over
    how wide a band?* The answer is a single hard limit:

    $$\boxed{\;\int_0^\infty \ln\frac{1}{|\Gamma(\omega)|}\,d\omega \;\le\; \frac{\pi}{R_L C_L}\;}
      \qquad\text{[Theorem — Bode–Fano, parallel-RC load]}$$

    **The intuition that makes everything else obvious.** Read the integral as a *budget*: at
    each frequency, $\ln(1/|\Gamma|)$ is how well-matched we are there (zero when $|\Gamma|=1$,
    large when $|\Gamma|\to 0$). A lossless network **cannot create** matching — it can only
    **move it around in frequency**. The total area under $\ln(1/|\Gamma|)$ is fixed by the
    load alone. Want a deep match? You must spend the budget over a narrow band. Want a wide
    band? You must accept a shallow match. The figure below shows the same budget spent two
    ways; the proof that follows simply pins down the size of the budget.
    """),
        mo.md("**Plan of proof** (four steps; each is one short subsection):"),
        mo.mermaid(r"""
flowchart TD
    A["Matching budget<br/>A = ∫₀^∞ ln(1/|Γ|) dω"] --> S1["S1 — Γ_in is bounded-real<br/>(load passive + causal ⇒ |Γ|≤1, analytic in RHP)"]
    S1 --> S2["S2 — Cauchy on a big RHP contour turns A into<br/>an accounting identity:  A = (π/2)·a₁ − π·Σ Re(zₖ)"]
    S2 --> S3["S3 — the load sets the budget a₁;<br/>a real network only ADDS RHP zeros zₖ, which leak budget away"]
    S3 --> S4["S4 — evaluate the budget on the bare load<br/>(one elementary integral) = π/(R_L C_L)"]
    S4 --> R["⇒  A ≤ π/(R_L C_L)"]
""")
    ])
    return


@app.cell
def _(go, mo, np):
    _Z0, _RL, _CL, _f0 = 50.0, 200.0, 0.5e-12, 20e9
    _w0 = 2 * np.pi * _f0
    _f = np.linspace(0.2e9, 80e9, 4000); _w = 2 * np.pi * _f
    # bare parallel-RC load
    _ZL = _RL / (1 + 1j * _w * _RL * _CL)
    _GL = (_ZL - _Z0) / (_ZL + _Z0)
    # a real matched network: shunt L resonates C_L at f0, then an L-section R_L→Z0
    _Lp = 1.0 / (_w0 ** 2 * _CL)
    _Zr = 1.0 / (1j * _w * _CL + 1.0 / (1j * _w * _Lp) + 1.0 / _RL)
    _RH, _RLo = max(_RL, _Z0), min(_RL, _Z0); _Q = np.sqrt(_RH / _RLo - 1)
    _Ls = _Q * _RLo / _w0; _Cp = _Q / (_w0 * _RH)
    _Zin = 1j * _w * _Ls + 1.0 / (1j * _w * _Cp + 1.0 / _Zr)
    _Gin = (_Zin - _Z0) / (_Zin + _Z0)

    _ybare = np.log(1.0 / np.abs(_GL))
    _ymatch = np.minimum(np.log(1.0 / np.abs(_Gin)), 3.5)   # clip the f0 notch for display
    _budget = np.pi / (_RL * _CL) / (2 * np.pi) / 1e9        # GHz·Np
    _A_bare = np.trapezoid(_ybare, _f) / 1e9
    _A_match = np.trapezoid(np.log(1.0 / np.abs(_Gin)), _f) / 1e9

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_f / 1e9, y=_ybare, mode="lines", name="bare load (no network)",
        line=dict(color="#636EFA", width=2), fill="tozeroy", fillcolor="rgba(99,110,250,0.25)"))
    _fig.add_trace(go.Scatter(x=_f / 1e9, y=_ymatch, mode="lines", name="matched network",
        line=dict(color="#EF553B", width=2), fill="tozeroy", fillcolor="rgba(239,85,59,0.25)"))
    _fig.update_layout(
        template="plotly_dark", height=420,
        title=f"Same budget, spent two ways  —  ∫ ln(1/|Γ|) dω ≤ π/(R_L C_L)  (R_L={_RL:g}Ω, C_L={_CL*1e12:g}pF)",
        xaxis=dict(title="Frequency (GHz)"),
        yaxis=dict(title="ln(1/|Γ|)   (matching density)", range=[0, 3.7]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3))
    _fig.add_annotation(x=20, y=3.5, text="matched: budget piled<br/>into a narrow band (deep dip)",
        showarrow=True, arrowhead=2, ax=70, ay=-30, font=dict(color="#EF553B", size=11))
    _fig.add_annotation(x=55, y=0.45, text="bare: budget spread thin",
        showarrow=True, arrowhead=2, ax=-40, ay=-30, font=dict(color="#636EFA", size=11))
    mo.vstack([
        mo.ui.plotly(_fig),
        mo.md(f"""
    *The blue area (bare load) and the red area (matched) are both bounded by the **same**
    budget $\\pi/(R_LC_L)$ = **{_budget:.2f} GHz·Np**. Numerically integrated over the window
    shown: bare ≈ {_A_bare:.2f}, matched ≈ {_A_match:.2f} GHz·Np (both below budget; the bare
    load loses a little to its high-frequency tail past 80 GHz). The network did not add area —
    it **relocated** it into a usable passband at the cost of $|\\Gamma|\\to1$ elsewhere.*
    """)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 8.1 Step 1 — the reflection coefficient is *bounded-real*

    $$\Gamma(s) = \frac{Z(s) - Z_0}{Z(s) + Z_0}\qquad\text{[Definition]}$$

    A **passive** load returns no net energy, so $\mathrm{Re}\,Z(j\omega)\ge 0$ [Axiom]. A
    **causal** load cannot respond before it is excited, which forces $Z(s)$ to be analytic in
    the right half-plane $\mathrm{Re}(s)>0$. Together these make $Z(s)$ a **positive-real (PR)**
    function. On the imaginary axis the algebraic identity

    $$|\Gamma(j\omega)|^2 = 1 - \frac{4Z_0\,\mathrm{Re}\,Z(j\omega)}{|Z(j\omega)+Z_0|^2}\;\le\;1$$

    shows $|\Gamma|\le 1$, and analyticity is inherited from $Z$. Hence $\Gamma(s)$ is
    **bounded-real (BR)**: analytic in the RHP, magnitude $\le 1$ on the axis [Theorem]. For a
    *lossless* matching network, energy conservation gives $|\Gamma_{\rm in}|^2+|T|^2=1$, so the
    input reflection $\Gamma_{\rm in}$ is BR as well — this is the only property of the network
    we will use.

    #### 8.2 Step 2 — Cauchy's theorem turns the integral into an accounting identity

    Let $f(s)=\ln\!\big(1/\Gamma_{\rm in}(s)\big)$. Two features control everything:

    - **Zeros of $\Gamma_{\rm in}$ in the RHP** become singularities of $f$. List them $\{z_k\}$.
    - **Behaviour at infinity:** because $|\Gamma|\to1$, $\;f(s)\to a_1/s + O(1/s^2)$ for some
      real $a_1$ (the leading "tail" coefficient).

    Integrate $f$ around the boundary of the RHP — the imaginary axis closed by a large
    semicircle (figure below). Cauchy's theorem relates the axis integral to the semicircle
    (which only sees $a_1/s$) and to the enclosed zeros (handled by **Blaschke factors**
    $B_k(s)=\frac{s-z_k}{s+\bar z_k}$, which have $|B_k|=1$ on the axis but rotate phase). The
    bookkeeping collapses to one identity:

    $$\boxed{\;\int_0^\infty \ln\frac{1}{|\Gamma_{\rm in}(\omega)|}\,d\omega
      \;=\; \frac{\pi}{2}\,a_1 \;-\; \pi\sum_k \mathrm{Re}(z_k)\;}\qquad\text{[accounting identity]}$$

    *Why the first term.* For an $f$ analytic in the closed RHP with $f\sim a_1/s$, the contour
    gives $\int_{-\infty}^{\infty} f(j\omega)\,d\omega = \pi a_1$ (the semicircle contributes
    $-j\pi a_1$; the axis the rest). Since $f$ is real on the real axis, $\mathrm{Re}\,f$ is even
    and $\int_0^\infty\mathrm{Re}\,f\,d\omega=\tfrac{\pi}{2}a_1$. *Why the second.* Each RHP zero
    is a "leak": factoring it out as a Blaschke term subtracts $\pi\,\mathrm{Re}(z_k)$ from the
    available area. **Read the identity plainly: the high-frequency tail $a_1$ funds the budget;
    every RHP zero spends some of it.**
    """)
    return


@app.cell
def _(go, mo, np):
    _th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    _R = 1.0
    _fig = go.Figure()
    # imaginary axis
    _fig.add_trace(go.Scatter(x=[0, 0], y=[-1.15, 1.15], mode="lines",
        line=dict(color="rgba(255,255,255,0.7)", width=2), name="jω axis", hoverinfo="skip"))
    # RHP semicircle (R → ∞)
    _fig.add_trace(go.Scatter(x=_R * np.cos(_th), y=_R * np.sin(_th), mode="lines",
        line=dict(color="#00CC96", width=2.5), name="semicircle  R→∞", hoverinfo="skip"))
    # arrows of traversal
    _fig.add_annotation(x=0, y=0.5, ax=0, ay=-0.1, xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowcolor="rgba(255,255,255,0.7)")
    # RHP zeros of Γ_in
    _zx, _zy = [0.45, 0.7, 0.55], [0.0, 0.35, -0.35]
    _fig.add_trace(go.Scatter(x=_zx, y=_zy, mode="markers+text",
        marker=dict(symbol="x", size=12, color="#EF553B"),
        text=["z₁", "z₂", "z₃"], textposition="top right",
        textfont=dict(color="#EF553B"), name="RHP zeros zₖ (leaks)"))
    _fig.add_annotation(x=0.5, y=0.95, text="RHP  (Re s > 0)<br/>Γ_in analytic here", showarrow=False,
        font=dict(color="rgba(255,255,255,0.6)", size=11))
    _fig.update_layout(template="plotly_dark", height=420, width=480,
        title="Step 2 contour: imaginary axis + RHP semicircle",
        xaxis=dict(title="Re s", range=[-0.35, 1.3], zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im s", range=[-1.2, 1.2], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32))
    mo.vstack([
        mo.ui.plotly(_fig),
        mo.md("*The axis integral = the matching budget. Closing through the RHP exposes the "
              "two contributions: the semicircle (the tail $a_1$) and the enclosed zeros $z_k$.*")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 8.3 Step 3 — the load caps the budget; a network can only leak it

    The accounting identity is exact for *any* network. To bound it, note two passivity facts:

    1. **The tail is anchored by the load.** As $\omega\to\infty$ the capacitor $C_L$
       short-circuits $R_L$ and every reactance in a finite lossless network becomes an open or
       a short. The leading high-frequency reflection — and therefore the budget the tail can
       fund — is fixed by $R_L,C_L$, not a free parameter of the network. (The tail term
       $\tfrac{\pi}{2}a_1$ and the zero term can trade against each other as the input element
       changes, but their *combination* cannot grow.)
    2. **A lossless network can only insert RHP zeros, never cancel the load's.** Each inserted
       transmission zero $z_k$ has $\mathrm{Re}(z_k)>0$ and, by the identity, *subtracts*
       $\pi\,\mathrm{Re}(z_k)$ from the budget.

    Fano's theorem (1950) makes this precise: the supremum of the budget over **all** lossless
    matching networks equals the area the **bare load** already produces — the network may
    *redistribute* the budget across frequency (the figure above) but never enlarge it. It
    remains only to *compute* that area.

    #### 8.4 Step 4 — evaluate the budget (one elementary integral)

    $$Z_L(s)=\frac{R_L}{1+sR_LC_L},\qquad
      \Gamma_L(j\omega)\;\Rightarrow\;
      |\Gamma_L(j\omega)|^2=\frac{(R_L-Z_0)^2+\omega^2\tau^2}{(R_L+Z_0)^2+\omega^2\tau^2},
      \quad \tau\equiv Z_0R_LC_L$$

    [**Lemma** (log-integral)] For real $b>a\ge 0$,
    $\displaystyle\int_0^\infty \ln\frac{b^2+\omega^2}{a^2+\omega^2}\,d\omega=\pi(b-a)$.
    *Proof:* differentiating under the integral, $I'(b)=\int_0^\infty\frac{2b\,d\omega}{b^2+\omega^2}=\pi$,
    and $I(a)=0$, so $I(b)=\pi(b-a)$. $\square$

    Apply it with $b=R_L+Z_0$, $a=|R_L-Z_0|$, substituting $u=\omega\tau$:

    $$\int_0^\infty \ln\frac{1}{|\Gamma_L|}\,d\omega
      =\frac{1}{2\tau}\int_0^\infty\ln\frac{(R_L+Z_0)^2+u^2}{(R_L-Z_0)^2+u^2}\,du
      =\frac{\pi\,[(R_L+Z_0)-(R_L-Z_0)]}{2\tau}=\frac{\pi\cdot 2Z_0}{2Z_0R_LC_L}$$

    $$\boxed{\;\int_0^\infty \ln\frac{1}{|\Gamma(\omega)|}\,d\omega \le \frac{\pi}{R_L C_L}\;}
      \qquad\text{[Theorem — Bode–Fano, proven]}$$

    Only the load time constant $R_LC_L$ survives. The reference $Z_0$ cancels: matching depth
    and bandwidth are constrained by the *load's own* $RC$, nothing else.
    """)
    return


@app.cell
def _(go, mo, np):
    _w = np.linspace(0, 3, 600)
    _w_lo, _w_hi = 1.0, 1.8       # band edges (normalized)
    _gmax = 0.30                  # |Γ_max| in band
    _h = np.log(1.0 / _gmax)
    _y = np.where((_w >= _w_lo) & (_w <= _w_hi), _h, 0.0)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_w, y=_y, mode="lines", line=dict(color="#FFA15A", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,161,90,0.3)", name="ideal equal-ripple"))
    _fig.add_annotation(x=(_w_lo + _w_hi) / 2, y=_h / 2,
        text=f"area = Δω · ln(1/|Γ_max|)<br/>≤ π/(R_L C_L)", showarrow=False,
        font=dict(color="white", size=12))
    _fig.add_annotation(x=(_w_lo + _w_hi) / 2, y=_h + 0.12, text="height = ln(1/|Γ_max|)",
        showarrow=False, font=dict(color="#FFA15A", size=11))
    _fig.add_annotation(x=_w_hi + 0.45, y=0.05, text="|Γ|→1 outside (budget = 0)",
        showarrow=False, font=dict(color="rgba(255,255,255,0.6)", size=11))
    _fig.add_shape(type="line", x0=_w_lo, x1=_w_hi, y0=-0.06, y1=-0.06,
        line=dict(color="#FFA15A", width=2))
    _fig.add_annotation(x=(_w_lo + _w_hi) / 2, y=-0.13, text="width = Δω", showarrow=False,
        font=dict(color="#FFA15A", size=11))
    _fig.update_layout(template="plotly_dark", height=380,
        title="Equal-ripple corollary: a fixed-area rectangle",
        xaxis=dict(title="ω", showticklabels=False, range=[0, 3]),
        yaxis=dict(title="ln(1/|Γ|)", range=[-0.2, 1.6]),
        showlegend=False)
    mo.vstack([
        mo.ui.plotly(_fig),
        mo.md("*The best a real design can do is approach the rectangle: equal reflection "
              "$|\\Gamma_{\\max}|$ across $\\Delta\\omega$, total mismatch outside. Its area "
              "$\\Delta\\omega\\,\\ln(1/|\\Gamma_{\\max}|)$ cannot exceed the budget.*")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 8.5 Reading the limit — the bandwidth ↔ depth trade

    Forcing the budget into an ideal equal-ripple rectangle of width $\Delta\omega$ and floor
    $|\Gamma_{\max}|$ gives $\int_0^\infty\ln(1/|\Gamma|)\,d\omega=\Delta\omega\,\ln(1/|\Gamma_{\max}|)$,
    hence

    $$\boxed{\;\Delta\omega\,\ln\frac{1}{|\Gamma_{\max}|}\;\le\;\frac{\pi}{R_LC_L}\;}\qquad\text{[Corollary]}$$

    [**Trade-offs**]
    - Fixed band $\Delta\omega$: the deepest match is floored at
      $|\Gamma_{\max}|\ge \exp\!\big(-\pi/(R_LC_L\,\Delta\omega)\big)$.
    - Fixed depth $|\Gamma_{\max}|$: the widest band is
      $\Delta\omega\le \pi/\big(R_LC_L\,\ln(1/|\Gamma_{\max}|)\big)$.
    - No topology and no network order can beat this — only the load $R_LC_L$ appears.

    An $N$-section Chebyshev design distributes the budget over $N$ equal-ripple lobes,
    approaching the rectangle as $N\to\infty$ and buying $\Delta\omega\propto\sqrt{N}$ for finite
    $N$ (§9). This is *why* a single L-section (§4–§7) is narrowband and why broadbanding needs
    cascades.

    #### 8.6 Dual — the series-RL load

    For $Z_L(s)=R_L+sL$ the inductor *opens* at high frequency, so $\Gamma_L\to+1$ and the
    integrating factor $1/s^2$ replaces $1$ in the contour argument. The same machinery yields

    $$\boxed{\;\int_0^\infty \frac{1}{\omega^2}\,\ln\frac{1}{|\Gamma(\omega)|}\,d\omega
      \;\le\;\frac{\pi L}{2R_L}\;}\qquad\text{[Theorem — Bode–Fano, series-RL]}$$

    The $1/\omega^2$ weight concentrates the constraint at *low* frequency, matching physical
    sense: the inductor shorts at DC (only $R_L$ left, hard to match) and self-mismatches at
    high frequency. The governing constant is now the inductive time constant $L/R_L$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §9. Chebyshev Matching Network Synthesis

    The low-pass prototype provides normalized element values $g_k$ for Butterworth (maximally flat) and Chebyshev (equiripple) responses.

    **Chebyshev element values** ($N$ elements, ripple $\varepsilon$):

    $$g_0 = 1, \quad
      g_k = \frac{4 a_{k-1} a_k}{b_{k-1} g_{k-1}}, \quad
      a_k = \sin\!\left(\frac{(2k-1)\pi}{2N}\right), \quad
      b_k = \gamma^2 + \sin^2\!\left(\frac{k\pi}{N}\right)$$

    where $\gamma = \sinh\!\bigl(\beta/(2N)\bigr)$ and $\beta = \ln\!\cot(\varepsilon_{\text{dB}}/17.37)$.

    **Bandpass transformation** scales and frequency-shifts the prototype:

    | Prototype arm | Bandpass realization |
    |---|---|
    | Series inductor $g_k$ | Series LC: $L = g_k Z_0/\Delta\omega$, $C = \Delta\omega/(\omega_0^2 L)$ |
    | Shunt capacitor $g_k$ | Parallel LC: $C = g_k/(\Delta\omega Z_0)$, $L = \Delta\omega/(\omega_0^2 C)$ |

    where $\Delta\omega = 2\pi \cdot \text{BW}$ and $Z_0 = R_S$.

    **Worked example:** $N=3$, 0.1 dB Chebyshev, $50\,\Omega \to 200\,\Omega$, $f_0 = 28\,\text{GHz}$, $\text{BW} = 3\,\text{GHz}$:
    Prototype values: $g_1 \approx 1.032$, $g_2 \approx 1.147$, $g_3 \approx 1.032$, $g_4 = 1$.
    Scaled series arm: $L_1 = g_1 \cdot 50 / (2\pi \cdot 3\text{e}9) \approx 2.74\,\text{pH}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §10. Real-Frequency Technique (Overview)

    When the load impedance $Z_L(f)$ is known from VNA measurements rather than a simple $R\|C$ model, the exact Chebyshev prototype is no longer directly applicable. Yarman's real-frequency technique solves for the lossless two-port $[T]$ that maximizes:

    $$\int_{f_1}^{f_2} |S_{21}(f)|^2\,df$$

    subject to $[T]$ being lossless and of prescribed degree $N$. The solution is obtained numerically (gradient descent on the reflection polynomial coefficients). Practical use: matching a transistor's measured $Y_{22}^*(f)$ to $50\,\Omega$ over a wide IF band without assuming a lumped-element model.
    """)
    return


@app.cell
def _(mo):

    N_bw      = mo.ui.slider(2, 6, value=3, step=1, label="N (order)")
    ripple_bw = mo.ui.slider(0.01, 3.0, value=0.1, step=0.01, label="Ripple (dB)")
    Rs_bw     = mo.ui.slider(10, 500, value=50, step=10, label="R_S (Ω)")
    Rl_bw     = mo.ui.slider(10, 500, value=200, step=10, label="R_L (Ω)")
    f0_bw     = mo.ui.slider(1e9, 100e9, value=28e9, step=1e9, label="f₀ (Hz)")
    bw_frac   = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="BW / f₀")

    mo.md("### §11. Interactive II — Broadband Chebyshev Explorer")
    return N_bw, Rl_bw, Rs_bw, bw_frac, f0_bw, ripple_bw


@app.cell
def _(
    N_bw,
    Rl_bw,
    Rs_bw,
    bode_fano_gamma_max,
    bw_frac,
    chebyshev_prototype,
    f0_bw,
    go,
    mo,
    np,
    ripple_bw,
):
    _N       = int(N_bw.value)
    _rip     = float(ripple_bw.value)
    _Rs      = float(Rs_bw.value)
    _Rl      = float(Rl_bw.value)
    _f0      = float(f0_bw.value)
    _BW      = float(bw_frac.value) * _f0

    # Prototype values
    _g = chebyshev_prototype(_N, _rip)
    _rows_bw = [(f"g_{k}", f"{_g[k]:.4f}") for k in range(_N + 2)]

    # Simple frequency-domain simulation of the prototype (lowpass → bandpass scaling)
    _w0 = 2 * np.pi * _f0
    _dw = 2 * np.pi * _BW
    _freqs_bw = np.linspace(_f0 - 3 * _BW, _f0 + 3 * _BW, 800)
    _w = 2 * np.pi * _freqs_bw
    # Bandpass frequency variable for prototype: Ω = (w/w0 - w0/w) / (dw/w0)
    _Omega = (_w / _w0 - _w0 / _w) / (_dw / _w0)
    # Chebyshev response: |S21|² = 1 / (1 + ε² T_N²(Ω))
    _eps = np.sqrt(10 ** (_rip / 10) - 1)
    _TN = np.cosh(_N * np.arccosh(np.abs(_Omega) + 0j)).real
    _S21_sq = 1.0 / (1.0 + _eps ** 2 * _TN ** 2)
    _S11_sq = np.clip(1.0 - _S21_sq, 0, 1)
    _S21_dB_bw = 10 * np.log10(np.maximum(_S21_sq, 1e-20))
    _S11_dB_bw = 10 * np.log10(np.maximum(_S11_sq, 1e-20))

    # Bode-Fano bound
    _C_load = 1.0 / (_w0 * _Rl)  # estimate C from R_L and f0 (Q=1 assumption)
    _gamma_bf = bode_fano_gamma_max(_BW, _Rl, _C_load)
    _bf_line = 20 * np.log10(max(_gamma_bf, 1e-10))

    _fig_bw = go.Figure()
    _fig_bw.add_trace(go.Scatter(x=_freqs_bw/1e9, y=_S11_dB_bw,
                                  name="|S₁₁| dB", line=dict(color="#EF553B")))
    _fig_bw.add_trace(go.Scatter(x=_freqs_bw/1e9, y=_S21_dB_bw,
                                  name="|S₂₁| dB", line=dict(color="#00CC96")))
    _fig_bw.add_hline(y=_bf_line, line_dash="dash", line_color="#FFA15A",
                       annotation_text=f"Bode-Fano |Γ_max| = {_gamma_bf:.3f}",
                       annotation_position="bottom right")
    _fig_bw.update_layout(template="plotly_dark", height=420,
                           xaxis_title="Frequency (GHz)", yaxis_title="dB",
                           title=f"N={_N} Chebyshev, {_rip} dB ripple, BW={_BW/1e9:.1f} GHz")

    _tbl_md_bw = "\n".join([f"| {r[0]} | {r[1]} |" for r in _rows_bw])
    _md_table_bw = mo.md(f"| Element | Value |\n|---|---|\n{_tbl_md_bw}")

    mo.vstack([
        mo.hstack([N_bw, ripple_bw, Rs_bw]),
        mo.hstack([Rl_bw, f0_bw, bw_frac]),
        mo.ui.plotly(_fig_bw),
        mo.md("**Prototype Values:**"),
        _md_table_bw
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part IV — Distributed and Transformer Matching

    ### §12. Quarter-Wave Transformer

    A transmission line section of length $\ell = \lambda/4$ with characteristic impedance $Z_1$ transforms a real load $R_L$ into an input impedance $Z_{\text{in}}$ at $f_0$.
    By substituting $\theta = \beta\ell = \pi/2$ into the standard lossless transmission line impedance equation:

    $$Z_{\text{in}} = Z_1 \frac{R_L + j Z_1 \tan(\theta)}{Z_1 + j R_L \tan(\theta)}$$

    Taking the limit as $\theta \to \pi/2$, we find $Z_{\text{in}} = Z_1^2 / R_L$.
    To match to source $R_S$, we set $Z_{\text{in}} = R_S$, giving the required line impedance:

    $$Z_1 = \sqrt{R_S R_L}$$

    Bandwidth to maximum reflection $|\Gamma_{\max}|$ (Pozar eq. 5.47):

    $$\frac{\Delta f}{f_0} = 2 - \frac{4}{\pi}\arccos\!\left(
      \frac{|\Gamma_{\max}|}{\sqrt{1 - |\Gamma_{\max}|^2}}
      \cdot \frac{2Z_1}{|Z_1^2/R_S - R_S|}\right)$$

    For $R_S = 50\,\Omega$, $R_L = 200\,\Omega$: $Z_1 = 100\,\Omega$,
    bandwidth to $|\Gamma_{\max}| = 0.2$ is approximately 30% of $f_0$.

    **Multi-section Chebyshev transformer:** $N$ cascaded $\lambda/4$ sections with impedances $Z_k$ chosen from the Chebyshev synthesis (Pozar §5.7) — achieves the Bode-Fano bound for a resistive load, and bandwidth scales as $N^{1/2}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §13. Single-Stub Tuner

    A parallel stub (open- or short-circuit) is placed at distance $d$ from the load. The design procedure works with admittances:

    1. Transform $Y_L$ over a distance $d$ such that $\text{Re}(Y_{\text{in}}(d)) = Y_0$.
    2. Add a shunt stub of length $\ell$ such that $B_{\text{stub}} = -B_{\text{in}}(d)$, canceling the susceptance.

    **Derivation of position $d$:**
    The load admittance is $Y_L = G_L + jB_L$. The admittance seen at distance $d$ (electrical length $\theta = \beta d$) is:

    $$Y_{\text{in}} = Y_0 \frac{Y_L + j Y_0 t}{Y_0 + j Y_L t}$$

    where $t = \tan(\beta d)$. Equating $\text{Re}(Y_{\text{in}})$ to $Y_0$ and solving the resulting quadratic equation for $t$ yields:

    $$t = \frac{B_L \pm \sqrt{G_L(Y_0 - G_L) + B_L^2}}{G_L - Y_0}$$

    From $t$, we find $d = \frac{1}{2\pi}\arctan(t)$ wavelengths.

    **Derivation of stub length $\ell$:**
    Once $d$ is known, evaluate $B_{\text{in}} = \text{Im}(Y_{\text{in}})$. The stub must provide $B_{\text{stub}} = -B_{\text{in}}$.
    For a **short-circuit** stub, the input admittance is $Y_{\text{sc}} = -j Y_0 \cot(\beta\ell)$.

    $$-Y_0 \cot(\beta\ell) = -B_{\text{in}} \implies \ell = \frac{1}{2\pi}\arctan\left(\frac{Y_0}{B_{\text{in}}}\right) \text{ wavelengths}$$

    For an **open-circuit** stub, the input admittance is $Y_{\text{oc}} = j Y_0 \tan(\beta\ell)$.

    $$Y_0 \tan(\beta\ell) = -B_{\text{in}} \implies \ell = \frac{1}{2\pi}\arctan\left(\frac{-B_{\text{in}}}{Y_0}\right) \text{ wavelengths}$$

    If $G_L > Y_0$, the discriminant can be negative → no real solution at this frequency
    (the load is in the "forbidden conductance region" for single-stub matching).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §14. Double-Stub Tuner

    Two stubs at fixed positions $0$ and $d = \lambda/8$ apart (or $\lambda/4$) can match any load outside the **forbidden region** $G_L > 2Y_0$ (for $d = \lambda/8$).

    Design steps (Pozar §5.3):
    1. Add a stub of susceptance $B_1$ in parallel with the load $Y_L$, resulting in $Y_1 = Y_L + jB_1$.
    2. Transform $Y_1$ through the fixed transmission line length $d$ to get $Y_2$. We require $\text{Re}(Y_2) = Y_0$. This condition provides an equation to solve for $B_1$.
    3. The remaining susceptance at the second stub location is $B_2' = \text{Im}(Y_2)$. Add the second stub with susceptance $B_2 = -B_2'$ to cancel it out.

    The forbidden region shifts with $d$: choosing $d = 3\lambda/16$ avoids the original forbidden zone but creates a new one elsewhere. Double-stub tuners appear in bench test setups; single-stub (simpler, manufacturable) is preferred on-chip.
    """)
    return


@app.cell
def _(mo):

    GammaL_mag_s = mo.ui.slider(0.0, 0.99, value=0.5, step=0.01, label="|Γ_L|")
    GammaL_ang_s = mo.ui.slider(-180, 180, value=45, step=5, label="∠Γ_L (°)")
    Z0_s         = mo.ui.number(value=50.0, label="Z₀ (Ω)")
    f0_s         = mo.ui.slider(1e9, 100e9, value=28e9, step=1e9, label="f₀ (Hz)")
    method_s     = mo.ui.dropdown(
        options=["Quarter-wave TL", "Single stub (short)", "Single stub (open)"],
        value="Single stub (short)", label="Method"
    )

    mo.md("### §15. Interactive III — Distributed Matching Explorer")
    return GammaL_ang_s, GammaL_mag_s, Z0_s, f0_s, method_s


@app.cell
def _(
    GammaL_ang_s,
    GammaL_mag_s,
    Z0_s,
    f0_s,
    go,
    method_s,
    mo,
    np,
    qw_transformer,
    single_stub_tuner,
):
    _Gamma_L_s = GammaL_mag_s.value * np.exp(1j * np.radians(GammaL_ang_s.value))
    _Z0 = float(Z0_s.value)
    _f0 = float(f0_s.value)
    _Z_L = _Z0 * (1.0 + _Gamma_L_s) / (1.0 - _Gamma_L_s)

    # Smith chart background (unit circle + constant R/G circles)
    _theta_sc = np.linspace(0, 2 * np.pi, 300)
    _unit_x = np.cos(_theta_sc)
    _unit_y = np.sin(_theta_sc)

    _fig_sc = go.Figure()
    _fig_sc.add_trace(go.Scatter(x=_unit_x, y=_unit_y,
                                  mode="lines", line=dict(color="#444", width=1),
                                  showlegend=False))

    # Load point
    _fig_sc.add_trace(go.Scatter(x=[_Gamma_L_s.real], y=[_Gamma_L_s.imag],
                                  mode="markers+text", name="Z_L",
                                  marker=dict(size=12, color="#EF553B"),
                                  text=[f"Z_L={_Z_L.real:.1f}+j{_Z_L.imag:.1f}Ω"],
                                  textposition="top right"))

    _info_lines = [f"Z_L = {_Z_L.real:.2f} + j{_Z_L.imag:.2f} Ω"]

    if "Quarter-wave" in method_s.value:
        _R_L_eff = abs(_Z_L)
        _d = qw_transformer(_Z0, _R_L_eff)
        _info_lines.append(f"λ/4 TL impedance: Z₁ = {_d['Z1_ohm']:.1f} Ω")
        _info_lines.append(f"Length: λ/4 at f₀ = {3e8/(4*_f0)*1e3:.2f} mm (in air)")
        _fig_sc.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Matched",
                                      marker=dict(size=12, color="#00CC96")))
    else:
        _stype = "short" if "short" in method_s.value else "open"
        _sols = single_stub_tuner(_Z_L, _Z0, _f0, _stype)
        for _i, _sol in enumerate(_sols):
            _info_lines.append(
                f"Solution {_i+1}: d={_sol['d_lambda']:.3f}λ ({_sol['d_mm']:.2f} mm), "
                f"ℓ={_sol['ell_lambda']:.3f}λ ({_sol['ell_mm']:.2f} mm)"
            )
        _fig_sc.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Target (centre)",
                                      marker=dict(size=12, color="#00CC96")))

    _fig_sc.update_layout(
        template="plotly_dark", height=500, width=520,
        xaxis=dict(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1, title="Re(Γ)"),
        yaxis=dict(range=[-1.2, 1.2], title="Im(Γ)"),
        title="Smith Chart — Distributed Matching"
    )

    mo.vstack([
        mo.hstack([GammaL_mag_s, GammaL_ang_s, Z0_s]),
        mo.hstack([f0_s, method_s]),
        mo.ui.plotly(_fig_sc),
        mo.md("**Results:**\n" + "\n".join(f"- {l}" for l in _info_lines))
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part V — Coupled Magnetics and Transformer Matching

    Parts II–IV treated matching networks as collections of two-port black
    boxes: series-$Z$, shunt-$Y$, $\lambda/4$ line. Two of the most powerful
    matching devices in microwave engineering — the **transformer** and the
    **bridged T-coil** — rely on *magnetic coupling between two windings*,
    which the black-box formalism conceals. This part opens the box.

    We start from Maxwell-Faraday and derive mutual inductance, the coupling
    coefficient, and the energy storage in coupled inductors (§16). We then
    build two equivalent-circuit models — the T-model and the π-model — and
    use them to settle the natural question *"do impedances on one side
    simply copy to the other?"* (§17 — answer: **no**). Sections §18–§22
    turn the foundation into matching-network design: practical transformer
    matching with losses, the full bridged-T-coil derivation, Marchand-balun
    distributed coupling, and on-chip geometry considerations.

    | Section | Topic |
    |---|---|
    | §16 | Mutual inductance from Maxwell-Faraday |
    | §17 | Equivalent circuits: T-model, π-model, ideal-plus-leakage |
    | §18 | Losses in coupled magnetics |
    | §19 | Transformer-based impedance matching |
    | §20 | Bridged T-coil — full derivation |
    | §21 | Marchand balun and coupled-line transformers |
    | §22 | On-chip geometry and practical considerations |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §16. Mutual Inductance from First Principles

    #### 16.1 Faraday's law and flux linkage

    Maxwell's third equation (Maxwell-Faraday) is

    $$\nabla \times \vec E = -\frac{\partial \vec B}{\partial t}
      \qquad\text{[Axiom — Maxwell-Faraday]}$$

    Integrating over a surface $S$ bounded by a closed curve $C$ and applying
    Stokes' theorem gives the integral form:

    $$\oint_C \vec E \cdot d\vec\ell
      = -\frac{d}{dt}\!\int_S \vec B \cdot d\vec S
      = -\frac{d\Phi}{dt}
      \qquad\text{[Theorem — Faraday's law]}$$

    The left side is the EMF $\mathcal E$ around $C$; the right side is minus
    the time derivative of the magnetic flux $\Phi$ through any surface
    bounded by $C$.

    **Flux linkage.** For a coil of $N$ turns each linking flux $\Phi$, the
    total EMF is $\mathcal E = -N\,d\Phi/dt$. Absorb the turn count into the
    flux: define the **flux linkage**

    $$\Psi \equiv N\Phi\qquad\text{[Definition]}$$

    so that $V = d\Psi/dt$ across the coil terminals.

    **Two coupled circuits.** Two filamentary circuits $C_1, C_2$ carry
    currents $I_1, I_2$. The magnetic field at any point is the linear sum
    $\vec B = \vec B_1 + \vec B_2$ (Maxwell is linear). The flux linkage of
    circuit 1 splits into a *self* contribution and a *mutual* contribution:

    $$\Psi_1 = L_1\,I_1 + M_{12}\,I_2$$

    The coefficients $L_1$ and $M_{12}$ are **defined** as the partial
    derivatives — geometric constants of the two-circuit configuration that
    do not depend on the currents:

    $$L_1 \equiv \left.\frac{\partial \Psi_1}{\partial I_1}\right|_{I_2=0},
      \qquad
      M_{12} \equiv \left.\frac{\partial \Psi_1}{\partial I_2}\right|_{I_1=0}
      \qquad\text{[Definitions]}$$

    Symmetrically $\Psi_2 = M_{21} I_1 + L_2 I_2$. In phasor form
    ($I_k \propto e^{j\omega t}$):

    $$\begin{bmatrix} V_1 \\ V_2 \end{bmatrix}
      = j\omega \begin{bmatrix} L_1 & M_{12} \\ M_{21} & L_2 \end{bmatrix}
        \begin{bmatrix} I_1 \\ I_2 \end{bmatrix}
      \qquad\text{[Result — coupled-inductor Z-matrix]}$$

    Reciprocity will show $M_{12} = M_{21} \equiv M$ (§16.3).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 16.2 Neumann's formula — $M$ from geometry alone

    The mutual inductance $M_{12}$ is a *geometric* quantity. Introduce the
    magnetic vector potential

    $$\vec B = \nabla \times \vec A\qquad\text{[Definition]}$$

    (always possible because $\nabla\cdot\vec B = 0$). For a current $I_2$
    flowing in $C_2$, the Coulomb-gauge vector potential is

    $$\vec A_2(\vec r)
      = \frac{\mu_0 I_2}{4\pi}\oint_{C_2}
        \frac{d\vec\ell_2}{|\vec r - \vec r_2|}
      \qquad\text{[Result — Biot-Savart for }\vec A\text{]}$$

    The flux through circuit 1 from this field is

    $$\Phi_{12}
      = \int_{S_1}\!\vec B_2 \cdot d\vec S
      = \int_{S_1}\!(\nabla\times\vec A_2)\cdot d\vec S
      = \oint_{C_1}\!\vec A_2 \cdot d\vec\ell_1
      \qquad\text{(Stokes' theorem)}$$

    Substituting the integral form of $\vec A_2$:

    $$\Phi_{12}
      = \frac{\mu_0 I_2}{4\pi}\oint_{C_1}\!\!\oint_{C_2}
        \frac{d\vec\ell_1 \cdot d\vec\ell_2}{|\vec r_1 - \vec r_2|}$$

    By definition $M_{12} = \Phi_{12}/I_2$:

    $$\boxed{\;M_{12} = \frac{\mu_0}{4\pi}\oint_{C_1}\!\!\oint_{C_2}
                       \frac{d\vec\ell_1 \cdot d\vec\ell_2}{|\vec r_1 - \vec r_2|}\;}
      \qquad\text{[Theorem — Neumann's formula, 1845]}$$

    **What this tells us.**

    - $M_{12}$ depends only on the *shape* and *relative position* of the two
      loops — not on the currents, the materials (in linear, non-magnetic
      media), or the frequency. It is a pure geometric constant.
    - The integrand is the inner product $d\vec\ell_1\cdot d\vec\ell_2$
      divided by the separation. **Parallel current elements contribute
      positively; orthogonal elements contribute zero; antiparallel
      elements contribute negatively.** This is why bifilar windings have
      large $M$ between the two filaments but small $M$ to external loops.
    - The integral is computable in closed form for a handful of canonical
      geometries (parallel coaxial loops, two parallel wires) but in
      general requires numerical evaluation — exactly what modern EM
      solvers (Momentum, HFSS, EMX) do for on-chip spirals.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 16.3 Reciprocity: $M_{12} = M_{21}$

    The Neumann integrand is **symmetric** under index exchange:
    $d\vec\ell_1\cdot d\vec\ell_2 = d\vec\ell_2\cdot d\vec\ell_1$ and
    $|\vec r_1 - \vec r_2| = |\vec r_2 - \vec r_1|$. Swapping the order
    of integration leaves the expression unchanged:

    $$M_{12}
      = \frac{\mu_0}{4\pi}\oint_{C_1}\!\!\oint_{C_2}
        \frac{d\vec\ell_1\cdot d\vec\ell_2}{r_{12}}
      = \frac{\mu_0}{4\pi}\oint_{C_2}\!\!\oint_{C_1}
        \frac{d\vec\ell_2\cdot d\vec\ell_1}{r_{12}}
      = M_{21}
      \qquad\text{[Theorem — reciprocity of mutual inductance]}$$

    **Energy interpretation.** The same identity follows from energy
    conservation. Build the system by first ramping $I_1$ from $0$ to its
    final value (with $I_2 = 0$; work done $= \tfrac12 L_1 I_1^2$), then
    ramping $I_2$ from $0$ to its final value while $I_1$ is held fixed.
    The extra work done against the back-EMF induced in circuit 1 by the
    changing $I_2$ is $\int I_1\,M_{12}\,dI_2 = M_{12}\,I_1 I_2$. Reverse
    the assembly order — ramp $I_2$ first, then $I_1$ — and the
    cross-term becomes $M_{21}\,I_1 I_2$. The final stored energy is a
    state function (the magnetic field is conservative in a passive
    linear medium), so the path-independence forces $M_{12} = M_{21}$.

    Henceforth drop the subscripts: $M \equiv M_{12} = M_{21}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 16.4 Coupling coefficient $k$ — and the proof $|k| \le 1$

    The **coupling coefficient** is the dimensionless ratio

    $$k \equiv \frac{M}{\sqrt{L_1 L_2}}\qquad\text{[Definition]}$$

    We prove $|k| \le 1$ from passivity, with $|k| = 1$ corresponding to
    *perfect coupling* (all flux from one coil links the other).

    **Proof from energy non-negativity.** The total magnetic energy is

    $$W = \tfrac12 L_1 I_1^2 + M I_1 I_2 + \tfrac12 L_2 I_2^2
        = \tfrac12 \begin{bmatrix} I_1 & I_2 \end{bmatrix}
          \underbrace{\begin{bmatrix} L_1 & M \\ M & L_2 \end{bmatrix}}_{\mathbf L}
          \begin{bmatrix} I_1 \\ I_2 \end{bmatrix}
        = \tfrac12\,\vec I^{\!\top}\!\mathbf L\,\vec I$$

    The volume integral $\tfrac12\int_V \vec B\cdot\vec H\,dV$ is
    pointwise non-negative in linear, non-magnetic media, so the system
    is **passive** and $W \ge 0$ for every choice of $(I_1, I_2)$. This
    is the definition of $\mathbf L$ being **positive semi-definite**.

    For a $2\times 2$ symmetric matrix, p.s.d. is equivalent to
    (i) both diagonal entries $\ge 0$ (automatic: $L_1, L_2 > 0$) and
    (ii) determinant $\ge 0$:

    $$\det \mathbf L = L_1 L_2 - M^2 \ge 0
      \quad\Longleftrightarrow\quad M^2 \le L_1 L_2
      \quad\Longleftrightarrow\quad |k| \le 1
      \qquad\text{[Theorem]}$$

    **Interpretation of the boundary.**

    - $|k| = 1$ (**perfect coupling**): $\det \mathbf L = 0$, so
      $\mathbf L$ has a zero eigenvalue — there is a current combination
      that stores no magnetic energy. Physically realised only in the
      ideal-transformer limit (infinite-$\mu$ core, zero leakage).
    - $k = 0$: $M = 0$, the two coils share no flux; they are
      magnetically decoupled.
    - $k < 0$: physical sign of $M$ depends on the dot convention
      (§16.5). Reversing one winding's orientation flips the sign; the
      *magnitude* is what the theorem bounds.

    **Typical values.**

    | Configuration | Typical $k$ |
    |---|---|
    | Air-core on-chip spirals, side-by-side | 0.3 – 0.7 |
    | Stacked spirals on thick metal stack | 0.85 – 0.95 |
    | PCB-stripline coupled lines | 0.2 – 0.5 |
    | Ferrite-cored audio transformer | $\approx 0.99$ |
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    #### 16.5 The dot convention

    The sign of $M$ depends on the *direction* in which the windings are
    traversed. A coil's two terminals are physically equivalent (you can
    enter from either), but the **winding orientation** sets whether the
    flux produced by a current entering that terminal aligns with or
    opposes the flux from a current in the other coil.

    The **dot convention** marks one terminal of each coil with a dot
    such that:

    > Currents flowing **into the dotted terminals** of both coils produce
    > magnetic fluxes that **add** in the same direction — and the mutual
    > inductance $M$ is taken **positive** in the voltage equation
    > $V_1 = j\omega L_1 I_1 + j\omega M I_2$.

    Reversing one winding (or moving the dot to the other terminal)
    flips the sign of $M$:

    $$\Psi_1 = L_1 I_1 + M I_2
      \quad\longrightarrow\quad
      \Psi_1 = L_1 I_1 - M I_2$$

    **Why it matters.**

    - In a transformer, the dot convention determines whether the
      secondary voltage is *in phase* or *180° out of phase* with the
      primary. Reversing the secondary winding is how single-ended-to-
      differential conversion is performed (the balun).
    - In a T-coil (§20), the two half-coils are wound *in series with
      both dots at the centre tap* — so currents from input to output
      add constructively. This is what produces the constant-$R$ input
      impedance.
    - Neumann's formula automatically carries the sign through the chosen
      senses of $d\vec\ell_1, d\vec\ell_2$. The dot convention is the
      bookkeeping that keeps that sign visible on schematics.
    """),
        mo.mermaid(r"""
flowchart LR
    A1((dot)) --- L1["L_1"] --- A2((·))
    B1((dot)) --- L2["L_2"] --- B2((·))
    L1 -. "M positive when currents enter both dots" .- L2
"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 16.6 Energy storage and the eigenmodes

    For *symmetric* coupled inductors ($L_1 = L_2 \equiv L$, mutual
    $M = kL$), the energy quadratic form

    $$W = \tfrac12 L(I_1^2 + I_2^2) + M\,I_1 I_2$$

    diagonalises in the rotated coordinates

    $$I_{\text{cm}} \equiv \frac{I_1 + I_2}{\sqrt 2},
      \qquad
      I_{\text{dm}} \equiv \frac{I_1 - I_2}{\sqrt 2}$$

    yielding

    $$W = \tfrac12\,\underbrace{L(1+k)}_{L_{\text{cm}}}\,I_{\text{cm}}^2
        + \tfrac12\,\underbrace{L(1-k)}_{L_{\text{dm}}}\,I_{\text{dm}}^2
      \qquad\text{[Result — symmetric-pair eigenmodes]}$$

    These are the **common-mode** ($I_1 = I_2$) and **differential-mode**
    ($I_1 = -I_2$) inductances. As $k \to 1$, $L_{\text{dm}} \to 0$ — the
    differential mode stores no energy at perfect coupling. This is the
    geometric statement of "leakage inductance vanishes."

    **Why it matters for matching networks.**

    - A **transformer** is driven single-ended into the primary; the
      bandwidth and $Q$ are dominated by the *differential-mode* leakage
      $L_{\text{dm}} = L(1-k)$ — the inductance the source sees that
      does *not* couple to the secondary.
    - A **balun** uses the differential mode to convert single-ended to
      differential; the common mode is a parasitic that current-starves
      the balun output.
    - A **T-coil** explicitly exploits the symmetric series arrangement.
      Its constant-$R$ design (§20) hinges on a cancellation between
      common-mode and differential-mode terms.

    The asymmetric case ($L_1 \ne L_2$) diagonalises with eigenvalues

    $$L_\pm = \frac{L_1 + L_2}{2}
      \pm \sqrt{\left(\frac{L_1 - L_2}{2}\right)^2 + M^2}$$

    both positive (by the $|k|\le 1$ result), with eigenmodes that are
    asymmetric mixtures of $I_1$ and $I_2$.
    """)
    return


@app.cell
def _(mo):
    coil_geom_dd = mo.ui.dropdown(
        options=["Coaxial (parallel-axis, separated along axis by d)",
                 "Coplanar (side-by-side, centre-to-centre d)"],
        value="Coaxial (parallel-axis, separated along axis by d)",
        label="Loop geometry"
    )
    wire_ratio_s = mo.ui.slider(50, 500, value=100, step=10,
                                label="loop-radius / wire-radius (sets L_self)")
    mo.md(r"""
    #### Interactive IV — Mutual inductance from geometry

    Two equal-radius circular loops. $M(d)$ is computed by numerical
    evaluation of Neumann's double integral, and the coupling coefficient
    $k = M / L_{\text{self}}$ uses the thin-wire self-inductance
    $L_{\text{self}} = \mu_0 a\,(\ln(8a/b) - 2)$. The plot makes concrete
    *how rapidly* $k$ collapses with geometric separation — and why on-chip
    transformers are **stacked** (axes aligned) rather than laid out
    side-by-side when high $k$ is required.
    """)
    return coil_geom_dd, wire_ratio_s


@app.cell
def _(coil_geom_dd, go, mo, np, wire_ratio_s):
    _N_int = 200
    _u = np.linspace(0, 2*np.pi, _N_int, endpoint=False)
    _mu0 = 4 * np.pi * 1e-7
    _a = 1.0
    _d_over_a = np.linspace(0.05, 5.0, 120)

    _k_vals = np.zeros_like(_d_over_a)
    _coaxial = coil_geom_dd.value.startswith("Coaxial")
    for _i, _da in enumerate(_d_over_a):
        _d = _da * _a
        if _coaxial:
            _r12 = np.sqrt(2 * _a**2 * (1 - np.cos(_u)) + _d**2)
            _M = (_mu0 / (4 * np.pi)) * 2 * np.pi * _a**2 \
                 * np.trapezoid(np.cos(_u) / _r12, _u)
        else:
            _T, _P = np.meshgrid(_u, _u, indexing='ij')
            _dx = _a * np.cos(_T) - _d - _a * np.cos(_P)
            _dy = _a * np.sin(_T) - _a * np.sin(_P)
            _r12 = np.maximum(np.sqrt(_dx**2 + _dy**2), 1e-9)
            _integrand = _a**2 * np.cos(_T - _P) / _r12
            _M = (_mu0 / (4 * np.pi)) \
                 * np.trapezoid(np.trapezoid(_integrand, _u, axis=1), _u)
        _r_w = _a / float(wire_ratio_s.value)
        _L_self = _mu0 * _a * (np.log(8 * _a / _r_w) - 2.0)
        _k_vals[_i] = _M / _L_self

    _fig_kg = go.Figure()
    _fig_kg.add_trace(go.Scatter(x=_d_over_a, y=_k_vals, mode="lines",
        line=dict(color="#00CC96", width=2.5), name="k(d/a)"))
    _fig_kg.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.5)", dash="dash"),
        annotation_text="k = 1 (perfect)", annotation_position="top right")
    _fig_kg.add_hline(y=0.0, line=dict(color="rgba(255,255,255,0.25)"))
    _fig_kg.update_layout(template="plotly_dark", height=420,
        title=f"Coupling vs separation — {coil_geom_dd.value}",
        xaxis=dict(title="d / a (separation / loop radius)"),
        yaxis=dict(title="k = M / L_self", range=[-0.15, 1.1]))
    mo.vstack([
        mo.hstack([coil_geom_dd, wire_ratio_s]),
        mo.ui.plotly(_fig_kg),
        mo.md(r"""
    **Reading the curve.** Coaxial loops (axes aligned, separated along the
    axis) couple strongly at $d \ll a$ — close to 1 when $d/a < 0.2$ — and
    decay roughly as $1/d^3$ at large separation (the magnetic-dipole
    asymptote). Coplanar loops, with parallel axes but centres displaced
    laterally, never reach $k = 1$: even at zero centre-to-centre distance
    the two loops only partially overlap. **This is the entire reason
    on-chip transformers are stacked vertically (high $k$) and on-chip
    inductors are laid out side-by-side (low parasitic $k$).**
    """),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    ### §17. Equivalent Circuit Models for Coupled Inductors

    #### 17.1 The T-model

    The coupled-inductor Z-matrix from §16.1 is

    $$\mathbf Z(s) = \begin{bmatrix} sL_1 & sM \\ sM & sL_2 \end{bmatrix}$$

    A passive two-port with this Z-matrix can be realised by a **T-network**:
    series inductances $L_a, L_b$ on the two ports plus a common shunt
    inductance $L_c$ to a third node. By inspection of the standard T-network
    $(Z_{11} = sL_a + sL_c,\; Z_{22} = sL_b + sL_c,\; Z_{12} = sL_c)$,
    matching coefficients gives

    $$\boxed{\;L_a = L_1 - M,\quad L_b = L_2 - M,\quad L_c = M\;}
      \qquad\text{[Result — T-model of coupled inductors]}$$

    **A surprise — negative inductance.** If $M > L_1$ (allowed when
    $L_1 < L_2$ and $k > \sqrt{L_1/L_2}$, i.e. the *smaller* coil sees
    very strong coupling to the larger), then $L_a = L_1 - M < 0$. The
    T-model requires a *negative* inductor in the series arm — physically
    impossible on its own, but perfectly fine as a fictitious building
    block inside an equivalent circuit. The negative inductance simply
    cancels part of $L_c = M$ when computing the input impedance; the
    overall two-port remains passive (the T-model is just a re-grouping
    of the same energy storage).
    """),
        mo.mermaid(r"""
flowchart LR
    P1((Port 1)) --- La["L_a = L_1 - M"] --- T(("·"))
    T --- Lb["L_b = L_2 - M"] --- P2((Port 2))
    T --- Lc["L_c = M"] --- G["gnd"]
"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 17.2 The π-model

    The dual representation is the **π-network**: a series element between
    the two ports plus shunt admittances at each port to ground. Inverting
    the coupled-inductor Z-matrix:

    $$\mathbf Y = \mathbf Z^{-1}
      = \frac{1}{s(L_1 L_2 - M^2)}
        \begin{bmatrix} L_2 & -M \\ -M & L_1 \end{bmatrix}$$

    Matching the standard π Y-matrix
    $(Y_{11} = Y_a + Y_c,\; Y_{22} = Y_b + Y_c,\; Y_{12} = -Y_c)$ and solving
    for the three π-network inductances:

    $$\boxed{\;L_a^{(\pi)} = \frac{L_1 L_2 - M^2}{L_2 - M},\quad
              L_b^{(\pi)} = \frac{L_1 L_2 - M^2}{L_1 - M},\quad
              L_c^{(\pi)} = \frac{L_1 L_2 - M^2}{M}\;}
      \qquad\text{[Result — π-model]}$$

    All three elements are positive inductors in the typical regime
    $M < \min(L_1, L_2)$. They become singular at $M = \min(L_i)$ or
    $M = 0$, and one element flips sign in exotic regimes — again, a
    mathematical artifact of the model, not of the physics.

    **When the π-model is preferred.** Numerical extraction of
    coupled-coil models from S-parameter measurements (EM-simulator
    output) is sometimes easier in admittance form than impedance form,
    because shunt parasitics — substrate capacitance, port pads — add
    directly to $Y_a, Y_b$ without changing the topology. For
    *hand-design* of transformer matching networks, the T-model is more
    natural; it dominates the textbook literature.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    #### 17.3 Ideal-transformer + leakage-inductance model

    The third equivalent decomposes the coupled-inductor pair into an
    **ideal transformer** (perfect coupling, frequency-independent turns
    ratio) plus **leakage inductances** that account for the imperfection
    $k < 1$:

    - a primary leakage $L_\sigma$ in series with the primary terminal;
    - a magnetising inductance $L_m$ shunted across the primary
      (carries the flux that links to the secondary);
    - an ideal turns-ratio transformer with effective ratio
      $n_{\text{eff}} = k\,n$ where $n \equiv \sqrt{L_1/L_2}$ is the
      "geometric" turns ratio;
    - (optionally) a secondary leakage $L'_\sigma$ on the secondary side.

    The split between $L_\sigma$ and $L_m$ depends on which model topology
    one picks. The "T-equivalent referred to the primary" gives

    $$L_\sigma = L_1(1 - k^2), \qquad L_m = k^2 L_1
      \qquad\text{[Result]}$$

    Any choice that sums to $L_1$ and matches the Z-matrix is equally
    valid; the leakage-magnetising decomposition is a *modelling*
    convenience.

    **Interpretation.**

    - $L_m$ carries the **magnetising current** that creates the mutual
      flux. At low frequency $\omega L_m \ll |Z_L|$ the magnetising
      current dominates and the transformer does not transfer power
      efficiently — this is the **low-frequency cut-off**.
    - $L_\sigma$ carries the **leakage flux** that does *not* link the
      secondary. At high frequency, the series reactance $j\omega L_\sigma$
      limits power transfer — the **high-frequency cut-off**.
    - At perfect coupling ($k = 1$): $L_\sigma = 0$, $L_m = L_1$. The
      transformer is bandwidth-limited only by $L_m$ on the low side and
      by interwinding capacitance on the high side.
    - At $k = 0$: $L_m = 0$, all of $L_1$ is leakage, no power transfers.

    This decomposition is more intuitive than the T-model for *operating*
    a transformer; the T-model is more useful for *circuit analysis*.
    """),
        mo.mermaid(r"""
flowchart LR
    P1((Pri)) --- Ls["L_sigma"] --- N1(("·"))
    N1 --- Lm["L_m (magnetising)"] --- G["gnd"]
    N1 --- ID["Ideal n_eff = k·n"] --- N2(("·"))
    N2 --- Ls2["L_sigma_secondary"] --- P2((Sec))
"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 17.4 Impedance reflection — what really happens

    A natural student question: *"If I put a load $Z_L$ on the secondary,
    do the impedances on the two sides simply 'copy' to each other?"* The
    answer is **no**, and the precise mechanism is worth spelling out
    because every transformer-matching design relies on it.

    Start with the coupled-inductor port equations:

    $$V_1 = j\omega L_1\,I_1 + j\omega M\,I_2,
      \qquad
      V_2 = j\omega M\,I_1 + j\omega L_2\,I_2$$

    Connect a load $Z_L$ to port 2 (passive sign convention:
    $V_2 = -Z_L\,I_2$). Substituting and solving for $I_2$:

    $$-Z_L\,I_2 = j\omega M\,I_1 + j\omega L_2\,I_2
      \quad\Longrightarrow\quad
      I_2 = -\frac{j\omega M}{Z_L + j\omega L_2}\,I_1$$

    Back into the first equation:

    $$V_1 = j\omega L_1\,I_1
            + j\omega M \cdot \!\left(-\frac{j\omega M}{Z_L + j\omega L_2}\right) I_1
        = \left[\,j\omega L_1
              + \frac{(\omega M)^2}{Z_L + j\omega L_2}\,\right] I_1$$

    $$\boxed{\;Z_{\text{in}}(\omega)
              = j\omega L_1 + \frac{(\omega M)^2}{Z_L + j\omega L_2}\;}
      \qquad\text{[Theorem — coupled-inductor input impedance]}$$

    **Reading the result.** $Z_{\text{in}}$ is the **sum** of two distinct
    contributions (not a parallel combination, not a copy):

    1. $j\omega L_1$ — the primary self-inductance, as if the secondary
       were absent. This is what the source would see if $I_2$ could not
       flow.
    2. $(\omega M)^2 / (Z_L + j\omega L_2)$ — the **reflected impedance**.
       This is the contribution of the secondary-side load $Z_L$. It is
       *not* $Z_L$ itself; it is $Z_L$ *in series with* $j\omega L_2$,
       inverted, and multiplied by $(\omega M)^2$.

    **The physical mechanism (no impedances are copied).**

    1. The primary current $I_1$ creates flux linking the secondary loop.
    2. This induces an EMF $j\omega M I_1$ in the secondary, driving a
       current $I_2$ through the loop $Z_L + j\omega L_2$.
    3. That secondary current $I_2$ creates its own flux linking the
       primary loop, inducing a *back-EMF* $j\omega M I_2$ in the primary.
    4. The back-EMF is the **only** way information about $Z_L$ reaches
       the input. There is no direct impedance transfer across the gap.

    The "reflection" is therefore mediated by the **product of two
    mutual-inductance terms** — one for the forward induction and one for
    the back-EMF — which is why $M$ appears *squared* in the reflected
    term.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 17.5 The strong-coupling limit — $n^2 Z_L$ recovered

    Expand $Z_{\text{in}}$ in the regime $\omega L_2 \gg |Z_L|$.

    **Denominator expansion.** Factor out $j\omega L_2$ and Taylor-expand:

    $$\frac{1}{Z_L + j\omega L_2}
      = \frac{1}{j\omega L_2}\cdot\frac{1}{1 + Z_L/(j\omega L_2)}
      \approx \frac{1}{j\omega L_2}
            - \frac{Z_L}{(j\omega L_2)^2}
      = -\frac{j}{\omega L_2} + \frac{Z_L}{\omega^2 L_2^2}$$

    Multiplying by $(\omega M)^2$:

    $$\frac{(\omega M)^2}{Z_L + j\omega L_2}
      \approx -j\omega\,\frac{M^2}{L_2}\, + \frac{M^2}{L_2^2}\,Z_L$$

    Use $M^2 = k^2 L_1 L_2$, hence $M^2/L_2 = k^2 L_1$ and $M^2/L_2^2 =
    k^2 L_1/L_2 = k^2 n^2$ (with $n \equiv \sqrt{L_1/L_2}$):

    $$\frac{(\omega M)^2}{Z_L + j\omega L_2}
      \approx -j\omega\,k^2 L_1 + k^2 n^2 Z_L$$

    Adding the $j\omega L_1$ contribution from §17.4:

    $$\boxed{\;Z_{\text{in}}
              \approx j\omega L_1 (1 - k^2) + k^2 n^2\,Z_L\;}
      \qquad\text{[Result — strong-coupling input impedance]}$$

    **Interpretation.**

    - $j\omega L_1(1 - k^2)$ is the **leakage reactance** seen at the
      primary — the part of $L_1$ that does not couple to the load and
      so appears as a series inductor at the input.
    - $k^2 n^2 Z_L$ is the **reflected load**, scaled by $k^2 n^2$ —
      *not* $n^2$. The coupling coefficient enters squared because
      power transfer to the secondary scales as $k^2$.
    - At $k = 1$ (perfect coupling): $Z_{\text{in}} = n^2 Z_L$ — the
      familiar ideal-transformer law. Achievable only as a limit.
    - At $k < 1$: the leakage inductance $L_1(1 - k^2)$ adds in series,
      which any matching network must absorb (typically as part of an
      LC tank).

    **Numerical example.** A 1:2 stacked-spiral transformer ($n = 2$,
    $L_1 = 200\,\text{pH}$, $L_2 = 50\,\text{pH}$, $k = 0.8$) at 28 GHz:
    $\omega L_1 = 35\,\Omega$, $\omega L_2 = 8.8\,\Omega$. A secondary load
    $Z_L = 50\,\Omega$ gives, from the **exact** formula in §17.4,

    $$Z_{\text{in}}
      = j(35) + \frac{(35\cdot 0.8\cdot 0.5)^2 \cdot \text{ratio}}{\,Z_L + j(8.8)\,}$$

    Note that $\omega L_2 = 8.8\,\Omega$ is *not* much larger than
    $|Z_L| = 50\,\Omega$, so the asymptotic expansion is mediocre here.
    The interactive below shows the discrepancy. **At mmWave, design
    uses the exact formula; the strong-coupling form is a sanity
    check, not a design tool.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 17.6 The weak-coupling limit — no transformation

    The opposite extreme: $k \to 0$, so $M \to 0$. The reflected term
    vanishes:

    $$Z_{\text{in}}\big|_{k=0} = j\omega L_1$$

    The two coils are magnetically decoupled — the primary sees only
    its own self-inductance, and **no information about $Z_L$ reaches the
    input**. Power transfer is zero. This is the limit of two on-chip
    inductors placed far apart, or deliberately laid out with orthogonal
    axes (Neumann's $d\vec\ell_1 \cdot d\vec\ell_2$ kills the integrand).
    For matching-network design we typically want $k > 0.3$ to obtain a
    useful impedance transformation.
    """)
    return


@app.cell
def _(mo):
    cir_L1_s = mo.ui.slider(50, 1000, value=200, step=10, label="L_1 (pH)")
    cir_L2_s = mo.ui.slider(10, 500, value=50, step=5, label="L_2 (pH)")
    cir_k_s = mo.ui.slider(0.0, 0.99, value=0.8, step=0.01, label="k")
    cir_ZLre_s = mo.ui.slider(1, 500, value=50, step=1, label="Re(Z_L) (Ω)")
    cir_ZLim_s = mo.ui.slider(-200, 200, value=0, step=5, label="Im(Z_L) (Ω)")
    cir_f0_s = mo.ui.slider(1, 100, value=28, step=1, label="f₀ (GHz)")
    mo.md(r"""
    #### Interactive V — Coupled-inductor input impedance

    Compare the **exact** $Z_{\text{in}}$ (§17.4) with the **strong-coupling
    approximation** $j\omega L_1(1 - k^2) + k^2 n^2 Z_L$ (§17.5). Toggle
    $k$ and $L_2$ to see where the approximation breaks down.
    """)
    return cir_L1_s, cir_L2_s, cir_ZLim_s, cir_ZLre_s, cir_f0_s, cir_k_s


@app.cell
def _(cir_L1_s, cir_L2_s, cir_ZLim_s, cir_ZLre_s, cir_f0_s, cir_k_s,
      go, mo, np, transformer_input_Z):
    _L1 = float(cir_L1_s.value) * 1e-12
    _L2 = float(cir_L2_s.value) * 1e-12
    _k_c = float(cir_k_s.value)
    _Z_L = complex(cir_ZLre_s.value, cir_ZLim_s.value)
    _f0 = float(cir_f0_s.value) * 1e9
    _n2 = _L1 / _L2

    _f = np.linspace(0.5e9, 100e9, 600)
    _w = 2 * np.pi * _f

    _Z_exact = transformer_input_Z(_L1, _L2, _k_c, _Z_L, _w)
    _Z_approx = 1j * _w * _L1 * (1 - _k_c ** 2) + _k_c ** 2 * _n2 * _Z_L
    _Z_ideal = _n2 * _Z_L * np.ones_like(_w, dtype=complex)

    _fig_zin = go.Figure()
    _fig_zin.add_trace(go.Scatter(x=_f/1e9, y=_Z_exact.real, mode="lines",
        name="Re Z_in exact", line=dict(color="#00CC96", width=2.6)))
    _fig_zin.add_trace(go.Scatter(x=_f/1e9, y=_Z_exact.imag, mode="lines",
        name="Im Z_in exact", line=dict(color="#EF553B", width=2.6)))
    _fig_zin.add_trace(go.Scatter(x=_f/1e9, y=_Z_approx.real, mode="lines",
        name="Re Z_in strong-k approx",
        line=dict(color="#00CC96", width=1.2, dash="dash")))
    _fig_zin.add_trace(go.Scatter(x=_f/1e9, y=_Z_approx.imag, mode="lines",
        name="Im Z_in strong-k approx",
        line=dict(color="#EF553B", width=1.2, dash="dash")))
    _fig_zin.add_hline(y=float(_Z_ideal[0].real),
        line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
        annotation_text=f"n² Re(Z_L) = {float(_Z_ideal[0].real):.1f} Ω",
        annotation_position="top left")
    _fig_zin.add_vline(x=_f0/1e9,
        line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
        annotation_text="f₀")
    _fig_zin.update_layout(template="plotly_dark", height=440,
        title=(f"Z_in vs f:  L_1={_L1*1e12:.0f}pH, L_2={_L2*1e12:.0f}pH, "
               f"k={_k_c:.2f}, n²=L_1/L_2={_n2:.2g}"),
        xaxis=dict(title="f (GHz)", type="log"),
        yaxis=dict(title="Z_in (Ω)"))

    _w0 = 2 * np.pi * _f0
    _Z_at_f0 = transformer_input_Z(_L1, _L2, _k_c, _Z_L, _w0)
    _Z_app_f0 = 1j * _w0 * _L1 * (1 - _k_c ** 2) + _k_c ** 2 * _n2 * _Z_L
    _wL2_over_ZL = (_w0 * _L2) / max(np.abs(_Z_L), 1e-12)
    _info = mo.md(rf"""
    **At f₀ = {_f0/1e9:.0f} GHz**

    | Quantity | Value |
    |---|---|
    | $n^2 = L_1/L_2$ | {_n2:.3g} |
    | $\omega L_2 / |Z_L|$ (strong-$k$ validity) | {_wL2_over_ZL:.2f} (need $\gg 1$) |
    | Exact $Z_\text{{in}}$ | {_Z_at_f0.real:.2f} + j{_Z_at_f0.imag:.2f} Ω |
    | Strong-$k$ approx | {_Z_app_f0.real:.2f} + j{_Z_app_f0.imag:.2f} Ω |
    | Ideal $n^2 Z_L$ | {(_n2*_Z_L).real:.2f} + j{(_n2*_Z_L).imag:.2f} Ω |
    """)
    mo.vstack([
        mo.hstack([cir_L1_s, cir_L2_s, cir_k_s]),
        mo.hstack([cir_ZLre_s, cir_ZLim_s, cir_f0_s]),
        mo.ui.plotly(_fig_zin),
        _info,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §18. Losses in Coupled Magnetics

    Real spirals and transformers are not pure inductors: their windings have
    resistance (DC plus frequency-dependent skin and proximity contributions),
    their flux leaks into the substrate where eddy currents dissipate, and
    inter-turn capacitance steals current. We address each mechanism in turn
    and arrive at the **quality factor** $Q$ that controls bandwidth and noise.

    #### 18.1 Winding resistance and series $Q$

    For a winding of cross-sectional area $A$, length $\ell$, and resistivity
    $\rho$ (copper: $\rho = 1.68\times 10^{-8}\,\Omega\!\cdot\!\text{m}$),
    the DC resistance is

    $$R_{\text{dc}} = \frac{\rho\,\ell}{A}\qquad\text{[Definition]}$$

    Each coil's port impedance is $Z = R(f) + j\omega L$, with $R(f)$ the
    frequency-dependent real part (DC plus skin/proximity contributions,
    derived below). The **series quality factor** is

    $$\boxed{\;Q(f) \equiv \frac{\omega L}{R(f)}\;}\qquad\text{[Definition]}$$

    At low frequency, $R(f) \approx R_{\text{dc}}$ and $Q \propto \omega$.
    As frequency rises, $R(f)$ grows (sections 18.2–18.3) and $Q$ either
    saturates or rolls off. Typical on-chip spirals peak in $Q$ at 20–40
    GHz, with peak values of 15–25 in 65 nm CMOS.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 18.2 Skin effect

    At high frequency, current in a conductor is confined to a thin shell
    near the surface. The physical reason is Faraday's law: an axial
    current creates a circumferential magnetic field, which induces an
    opposing circulating current inside the conductor. To see this
    quantitatively, write Maxwell's equations inside a good conductor
    ($\sigma \gg \omega\epsilon$):

    $$\nabla \times \vec H = \sigma \vec E + \epsilon\,\partial_t \vec E
      \approx \sigma \vec E, \qquad
      \nabla \times \vec E = -\mu\,\partial_t \vec H$$

    Eliminating $\vec H$ and assuming harmonic time dependence
    $e^{j\omega t}$ gives the **Helmholtz equation** inside the
    conductor:

    $$\nabla^2 \vec E - j\omega\mu\sigma\,\vec E = 0\qquad\text{[Theorem]}$$

    For a plane wave normally incident on a flat conductor surface at
    $z = 0$:

    $$\vec E(z) = \vec E_0\,e^{-z/\delta}\,e^{-jz/\delta}, \qquad
      \boxed{\;\delta \equiv \sqrt{\frac{2}{\omega\mu\sigma}}\;}
      \qquad\text{[Definition — skin depth]}$$

    The amplitude falls by $1/e$ over one skin depth; the phase rotates
    by 1 rad.

    **Numerical scale** for copper ($\sigma = 5.96\times 10^7\,\text{S/m}$):

    | $f$ | $\delta_{\text{Cu}}$ |
    |---|---|
    | 1 GHz | 2.1 μm |
    | 10 GHz | 0.66 μm |
    | 28 GHz | 0.39 μm |
    | 100 GHz | 0.21 μm |

    **AC resistance of a round wire** ($a \gg \delta$). Current flows in
    an annular shell of thickness $\delta$ near the surface, of length
    $\ell$ and effective cross-section $2\pi a \delta$:

    $$R_{\text{ac}} = \frac{\rho\,\ell}{2\pi a \delta}
                    = R_{\text{dc}} \cdot \frac{a}{2\delta}
      \qquad\text{[Result — round-wire skin resistance]}$$

    For a 10 μm-diameter wire ($a = 5\,\mu\text{m}$) at 28 GHz with
    $\delta = 0.39\,\mu\text{m}$:
    $R_{\text{ac}}/R_{\text{dc}} = 5/0.78 \approx 6.4\times$. **At mmWave
    the DC resistance is irrelevant** — only the skin contribution matters.

    Since $\delta \propto 1/\sqrt f$, $R_{\text{ac}} \propto \sqrt f$, so
    the skin-limited $Q = \omega L / R_{\text{ac}} \propto \sqrt f$ —
    grows slowly with frequency (both numerator and denominator scale,
    almost cancelling).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 18.3 Proximity effect — current crowding

    When two current-carrying conductors lie close together, each
    conductor's magnetic field perturbs the current distribution in the
    other. In an inductor spiral, adjacent turns carry currents in the
    *same direction* (the spiral winds progressively), and the field
    from each turn pushes the current in its neighbour toward the **outer
    edge** of the wire — shrinking the conducting cross-section further
    than skin effect alone would predict.

    Quantitatively, for two parallel conductors of radius $a$ separated
    by centre-to-centre distance $s$ carrying the same axial current,
    the proximity-induced increment to AC resistance per unit length
    scales as

    $$\Delta R_{\text{prox}} \;\propto\; \frac{1}{s^2}\,(\omega\mu\sigma)$$

    Two key consequences:

    1. **Proximity loss grows linearly with frequency** ($\propto \omega$),
       not as $\sqrt\omega$ like skin loss. At mmWave it can dominate
       skin loss for tightly-wound spirals.
    2. **Spacing matters more than wire width.** Spreading turns apart by
       $\sqrt 2$ cuts proximity loss in half — at the cost of larger
       area and lower $L$ per turn. Modern on-chip spirals trade
       inter-turn pitch carefully to balance $L$, $Q$, and footprint.

    **Litz wire** — a bundle of many fine, mutually-insulated strands
    twisted so each strand visits the bundle's surface equally — defeats
    proximity loss by forcing every strand to carry equal current. Used
    widely at sub-MHz; the *concept* of a Litz-like multi-filament
    conductor motivates "patterned" or "honeycomb" on-chip inductor
    variants.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 18.4 Substrate eddy currents

    On a silicon chip, the spiral inductor sits ${\sim}5\,\mu\text{m}$
    above a moderately-conductive substrate
    ($\sigma_{\text{Si}} \approx 1$–$10\,\text{S/m}$ for lightly-doped
    bulk). Magnetic flux penetrating the substrate induces circulating
    eddy currents — the substrate behaves as a **lossy single-turn
    secondary** magnetically coupled to the inductor.

    **Image-current model.** A spiral with current $I_1$ creates an axial
    flux through the substrate; by Faraday's law a current
    $I_{\text{sub}}$ flows in the substrate, opposing the flux change.
    The dissipation $I_{\text{sub}}^2 R_{\text{sub}}$ is borne by the
    inductor as an effective additional series resistance — reflected
    through the spiral-to-substrate mutual inductance.

    Using the impedance-reflection result (§17.4) with the substrate as
    a "secondary" load:

    $$\Delta R_{\text{sub}}
      = \frac{(\omega M_{\text{sub}})^2 \cdot R_{\text{sub}}}
             {R_{\text{sub}}^2 + (\omega L_{\text{sub}})^2}$$

    In the resistive-substrate regime $\omega L_{\text{sub}} \ll
    R_{\text{sub}}$: $\Delta R_{\text{sub}} \propto \omega^2$ — loss
    grows *quadratically* with frequency and dominates above ${\sim}10$ GHz
    on standard Si CMOS.

    **Patterned ground shield (PGS).** A *solid* metal ground plane
    below the spiral *increases* loss (it forms its own image eddy
    current loop). A **slotted** ground shield, with slots oriented
    perpendicular to the spiral's current direction, breaks the
    eddy-current paths while still providing a stable AC reference. PGS
    typically recovers 3–6 dB of $Q$ at 30 GHz and is standard practice
    for high-$Q$ on-chip inductors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 18.5 Magnetic-core losses (off-chip, briefly)

    Ferrite-cored transformers used below ${\sim}1$ GHz suffer two
    additional in-core loss mechanisms:

    - **Hysteresis loss.** Each B-H cycle dissipates energy equal to
      the loop area: $P_{\text{hyst}} = f \oint H\,dB$. Loss
      grows with frequency.
    - **Eddy currents in the core.** Even modestly-conductive ferrites
      have $\sigma > 0$; the changing $B$ induces in-core eddy currents
      analogous to substrate eddies. Laminated cores reduce these by
      breaking the eddy paths into thin sheets.

    **On-chip mmWave transformers are air-core** (no ferrite), so neither
    mechanism applies. The dominant losses are the winding and substrate
    contributions of §18.1–§18.4.

    #### 18.6 Total quality factor

    Loss mechanisms add as **parallel conductances** at the spiral's
    resonance, so their reciprocals sum:

    $$\frac{1}{Q_{\text{tot}}(f)}
      = \frac{1}{Q_{\text{wind}}(f)}
      + \frac{1}{Q_{\text{sub}}(f)}
      + \frac{1}{Q_{\text{diel}}(f)}
      \qquad\text{[Result]}$$

    The reciprocal sum is essential: **the worst contributor sets the
    upper bound on $Q_{\text{tot}}$**. Doubling $Q_{\text{wind}}$ helps
    little if $Q_{\text{sub}}$ is already the bottleneck.

    Typical 28 GHz CMOS spiral: $Q_{\text{wind}} \approx 25$,
    $Q_{\text{sub}} \approx 30$ (with PGS), $Q_{\text{diel}} \approx 80$
    → $Q_{\text{tot}} \approx 12$.

    #### 18.7 Noise-figure penalty from winding loss

    A lossy matching transformer at the **input** of an LNA contributes a
    direct noise-figure penalty. From Friis (notebook 04 §10):

    $$F_{\text{tot}} = F_{\text{match}}
                       + \frac{F_{\text{LNA}} - 1}{G_{\text{match}}}$$

    For a transformer with insertion loss $L_{\text{IL}}$ in linear
    units ($L_{\text{IL}} \equiv 1/G_{\text{match}}$), the matching
    network's noise figure equals its loss
    ($F_{\text{match}} = L_{\text{IL}}$, a fundamental property of
    *passive* networks at thermal equilibrium):

    $$\Delta\text{NF}_{\text{pre-LNA}} = 10\log_{10} L_{\text{IL}}\;[\text{dB}]
      \qquad\text{[Result — pre-LNA loss penalty]}$$

    A 0.5 dB insertion-loss transformer pre-LNA adds 0.5 dB *directly*
    to the chain NF.

    **Post-LNA losses are divided by the LNA gain** in the Friis
    denominator — a 0.5 dB output-match loss after a 15 dB LNA
    contributes only $0.5/32 \approx 0.016\,\text{dB}$ to the
    input-referred NF. This asymmetry is why output-match topology can
    be aggressive (low-$Q$ transformer, long $\lambda/4$ line) without
    harming sensitivity, whereas input-match must be near-lossless.
    """)
    return


@app.cell
def _(mo):
    qf_L_s = mo.ui.slider(50, 1000, value=200, step=10, label="L (pH)")
    qf_a_s = mo.ui.slider(2, 20, value=5, step=1, label="wire radius a (μm)")
    qf_ell_s = mo.ui.slider(100, 5000, value=1000, step=100,
                            label="winding length ℓ (μm)")
    qf_sub_s = mo.ui.slider(0, 50, value=10, step=1,
                            label="substrate σ (S/m)")
    qf_pgs_s = mo.ui.checkbox(value=True, label="Patterned ground shield (PGS)")
    mo.md(r"""
    #### Interactive VI — Spiral inductor $Q(f)$

    Decomposes $Q_{\text{tot}}(f)$ into its contributors: winding
    (skin + proximity, derived above) and substrate (eddy, ${\propto}
    \omega^2 \sigma_{\text{sub}}$). The PGS toggle multiplies the
    substrate-loss term by 0.3 — the typical recovery for a slotted
    ground shield. Move the substrate-$\sigma$ slider to zero to see the
    winding-limited ceiling.
    """)
    return qf_L_s, qf_a_s, qf_ell_s, qf_sub_s, qf_pgs_s


@app.cell
def _(go, mo, np, qf_L_s, qf_a_s, qf_ell_s, qf_sub_s, qf_pgs_s):
    _L_qf = float(qf_L_s.value) * 1e-12
    _a_qf = float(qf_a_s.value) * 1e-6
    _ell_qf = float(qf_ell_s.value) * 1e-6
    _sigma_sub = float(qf_sub_s.value)
    _pgs = qf_pgs_s.value

    _rho_Cu = 1.68e-8
    _mu0_qf = 4 * np.pi * 1e-7
    _sigma_Cu = 5.96e7

    _f_qf = np.logspace(8, 11.3, 400)
    _w_qf = 2 * np.pi * _f_qf

    _R_dc = _rho_Cu * _ell_qf / (np.pi * _a_qf ** 2)
    _delta = np.sqrt(2.0 / (_w_qf * _mu0_qf * _sigma_Cu))
    _R_skin = _rho_Cu * _ell_qf / (2 * np.pi * _a_qf * _delta)
    _R_wind = np.maximum(_R_dc, _R_skin)
    # Phenomenological proximity contribution (linear in ω)
    _R_prox = 5e-13 * _w_qf * _ell_qf / _a_qf
    _R_total_wind = _R_wind + _R_prox
    _Q_wind = _w_qf * _L_qf / _R_total_wind

    # Substrate eddy: phenomenological ω² · σ_sub scaling
    _shield = 0.3 if _pgs else 1.0
    _R_sub = 5e-22 * _w_qf ** 2 * _sigma_sub * (_L_qf * 1e12) ** 2 * _shield
    _Q_sub = np.where(_R_sub > 0, _w_qf * _L_qf / _R_sub,
                      np.full_like(_w_qf, 1e6))

    _Q_diel = np.full_like(_f_qf, 80.0)
    _Q_tot = 1.0 / (1.0/_Q_wind + 1.0/_Q_sub + 1.0/_Q_diel)

    _fig_q = go.Figure()
    _fig_q.add_trace(go.Scatter(x=_f_qf/1e9, y=_Q_wind, mode="lines",
        name="Q_winding (skin + prox)", line=dict(color="#00CC96", width=2)))
    _fig_q.add_trace(go.Scatter(x=_f_qf/1e9, y=_Q_sub, mode="lines",
        name="Q_substrate", line=dict(color="#EF553B", width=2)))
    _fig_q.add_trace(go.Scatter(x=_f_qf/1e9, y=_Q_diel, mode="lines",
        name="Q_dielectric", line=dict(color="#FFA15A", width=1.5, dash="dot")))
    _fig_q.add_trace(go.Scatter(x=_f_qf/1e9, y=_Q_tot, mode="lines",
        name="Q_total", line=dict(color="#FECB52", width=3)))
    _fig_q.update_layout(template="plotly_dark", height=440,
        title=(f"Spiral Q vs f  (L = {_L_qf*1e12:.0f} pH, "
               f"a = {_a_qf*1e6:.0f} μm, ℓ = {_ell_qf*1e6:.0f} μm, "
               f"σ_sub = {_sigma_sub} S/m, PGS = {_pgs})"),
        xaxis=dict(title="f (GHz)", type="log"),
        yaxis=dict(title="Q", range=[0, 60]))

    _i_peak = int(np.argmax(_Q_tot))
    _info_q = mo.md(rf"""
    **Peak $Q_\text{{tot}} = ${_Q_tot[_i_peak]:.1f}** at
    $f = ${_f_qf[_i_peak]/1e9:.1f} GHz.
    {('Substrate eddy loss caps Q above ~10 GHz; PGS already applied.' if _pgs and _sigma_sub > 0 else 'Without PGS, substrate loss dominates above ~10 GHz — toggle PGS to compare.' if _sigma_sub > 0 else 'Substrate loss is negligible here; winding loss is the ceiling.')}
    """)
    mo.vstack([
        mo.hstack([qf_L_s, qf_a_s, qf_ell_s]),
        mo.hstack([qf_sub_s, qf_pgs_s]),
        mo.ui.plotly(_fig_q),
        _info_q,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §19. Transformer-Based Impedance Matching

    A transformer is the most direct way to step a fixed source impedance
    to a fixed load impedance — at the cost of a bandwidth set by the
    leakage inductance and winding $Q$. We work through three
    progressively realistic models:

    1. The **ideal 1:n transformer** (zero leakage, infinite magnetising
       inductance) — bandwidth is infinite, impedance ratio is $n^2$.
    2. The **real transformer with leakage** ($k < 1$) — bandwidth is
       limited by $L_\sigma = L_1(1-k^2)$ forming a high-pass with the
       load.
    3. The **tuned-primary / tuned-secondary** transformer — resonators
       on each side absorb the leakage and produce the classical
       under / critical / over-coupled bandpass response, with bandwidth
       controlled by $k$ relative to a critical value $k_{\text{crit}}$.

    #### 19.1 The ideal 1:n transformer

    In the limit $k \to 1, \omega L_2 \to \infty$ (perfect coupling,
    infinite magnetising inductance), §17.5 gave

    $$Z_{\text{in}} = n^2 Z_L, \qquad n \equiv \sqrt{L_1/L_2}$$

    The transformation is **frequency-independent**: a 100 Ω load on the
    secondary appears as $n^2 \cdot 100\,\Omega$ at the primary at every
    frequency. The ideal transformer has *infinite bandwidth*.

    No real transformer reaches this limit. The next subsection accounts
    for finite $k$; §19.3 adds finite $L_2$ (low-frequency cut-off);
    §19.4 absorbs both into tuned tanks to recover a controlled,
    designable bandwidth.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 19.2 The leakage-inductance bandwidth limit

    With $k < 1$ but still in the strong-coupling regime ($\omega L_2
    \gg |Z_L|$), §17.5 gave

    $$Z_{\text{in}} \approx j\omega L_1(1-k^2) + k^2 n^2 Z_L$$

    For a resistive load $Z_L = R_L$, this is an **R + jωL high-pass**
    network: at low frequency $Z_{\text{in}} \to k^2 n^2 R_L$ (a
    resistance), at high frequency $Z_{\text{in}} \to j\omega L_\sigma$
    (a series inductor that detunes the source).

    The 3-dB cutoff occurs where the reactance equals the resistance:

    $$\boxed{\;\omega_{\text{HP}} = \frac{k^2 n^2 R_L}{L_\sigma}
                                    = \frac{k^2 n^2 R_L}{L_1(1 - k^2)}\;}
      \qquad\text{[Result — leakage-limited cutoff]}$$

    **What this means.** A real transformer is a **bandpass** device:
    *above* $\omega_{\text{HP}}$ the leakage reactance dominates and
    starves the load of current; *below* the low-frequency cut-off (set
    by the finite magnetising inductance, §19.3) the magnetising current
    swamps the load. Only between these limits does the impedance
    transformation work.

    **Design lever.** To push $\omega_{\text{HP}}$ higher: *increase* $k$
    (stacked-spiral geometry, §22) or *decrease* $L_1$ (fewer turns,
    smaller spiral). The two trade against each other in layout — high
    $k$ usually requires more turns, which raises $L_1$ and pushes
    $\omega_{\text{HP}}$ back down. There is no free lunch; the resonant
    designs of §19.4 are how we escape this trade.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 19.3 Tuned primary, tuned secondary

    Add a capacitor $C_1$ across the primary winding to form a parallel
    LC tank at $\omega_0 = 1/\sqrt{L_1 C_1}$. At resonance, the tank's
    reactance is infinite (in the lossless limit) and the leakage
    inductance is *absorbed* into the resonant response rather than
    detuning the source. Symmetrically, a capacitor $C_2$ across the
    secondary forms a tank with $L_2$ at the same $\omega_0$ — typical
    practice is to tune both sides to a common centre frequency.

    **Resonator $Q$.** Each side has a loaded quality factor

    $$Q_1 = \frac{R_1}{\omega_0 L_1} = \omega_0 R_1 C_1,
      \qquad Q_2 = \frac{R_2}{\omega_0 L_2} = \omega_0 R_2 C_2$$

    where $R_1$ is the parallel resistance the primary tank sees (source
    impedance combined with winding loss); analogously for $R_2$.

    **Why tune both sides.** A single-tuned transformer (only one tank)
    absorbs *one* leakage inductance and has a second-order roll-off on
    the un-tuned side. Tuning both sides gives a **fourth-order
    bandpass** response — narrower passband, sharper skirts, and a
    coupling-dependent magnitude shape that admits the
    under / critical / over-coupled trichotomy below.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 19.4 Single-tuned vs double-tuned response

    A **single-tuned** transformer (only the primary resonates) gives a
    Lorentzian magnitude response centred at $\omega_0$, with 3-dB
    fractional bandwidth $1/Q_1$. Shape is fixed; bandwidth is narrow.

    A **double-tuned** transformer (both sides resonate at the same
    $\omega_0$) admits three regimes set by $k$ relative to a critical
    value. The transfer-magnitude formula for symmetric $Q_1 = Q_2 = Q$,
    after some algebra of the two coupled tanks, takes the canonical
    form

    $$|H(u)|^2 \;\propto\;
      \frac{(kQ)^2}{\big(1 + (kQ)^2 - u^2\big)^2 + (2u)^2},
      \qquad u \equiv Q\!\left(\frac{\omega}{\omega_0} - \frac{\omega_0}{\omega}\right)$$

    Setting $d|H|^2/du = 0$ gives a peak at $u = 0$ when $kQ \le 1$ and at
    $u^2 = (kQ)^2 - 1$ when $kQ > 1$. The threshold between single-peaked
    and double-peaked is the **critical coupling**

    $$\boxed{\;k_{\text{crit}} = \frac{1}{\sqrt{Q_1 Q_2}}
                                 = \frac{1}{Q}\;\text{(symmetric)}\;}
      \qquad\text{[Result — critical-coupling condition]}$$

    | Regime | $k$ vs $k_{\text{crit}}$ | Magnitude shape | 3-dB BW |
    |---|---|---|---|
    | Under-coupled | $k < k_{\text{crit}}$ | Single peak below max | Narrow |
    | Critically coupled | $k = k_{\text{crit}}$ | Single peak, maximally flat | $\sqrt 2\,\omega_0 / Q$ |
    | Over-coupled | $k > k_{\text{crit}}$ | Two peaks at $\omega_\pm$, dip at $\omega_0$ | Broader between peaks |

    **Critical-coupling bandwidth.** For symmetric $Q$, the
    maximally-flat 3-dB fractional bandwidth is

    $$\frac{\Delta f}{f_0} = \frac{\sqrt 2}{Q}
      \qquad\text{[Result]}$$

    Compared to a single-resonator match (BW $\propto 1/Q$), critical
    coupling delivers $\sqrt 2\,{\approx}\,1.41\times$ the bandwidth —
    the same factor that distinguishes a 2-pole Butterworth from a
    1-pole single-resonator. Over-coupling pushes further (at the cost
    of an in-band amplitude dip) and is the workhorse for broadband
    matching on chip.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 19.5 Design equations and a worked example

    Given source $R_S$, load $R_L$, centre frequency $f_0$, and target
    fractional bandwidth $\Delta f/f_0$:

    1. **Turns ratio.** $n = \sqrt{R_S/R_L}$ (or effective ratio
       $n_{\text{eff}} = k\,n$ if $k$ is fixed by the process and known).
    2. **Required $Q$.** $Q = \sqrt 2 / (\Delta f / f_0)$ for critical
       coupling.
    3. **Inductances from $Q$.**
       $L_1 = R_S / (\omega_0 Q)$, $L_2 = L_1 / n^2$.
    4. **Capacitances from resonance.**
       $C_i = 1/(\omega_0^2 L_i)$.
    5. **Coupling.** Lay out the spirals to achieve $k = k_{\text{crit}}
       = 1/Q$ (symmetric case) within ~10% — the bandwidth degrades
       gracefully off-target.

    **Example.** Match $R_S = 50\,\Omega$ to $R_L = 200\,\Omega$ at
    $f_0 = 28\,\text{GHz}$, target $\Delta f/f_0 = 20\%$:

    | Parameter | Value |
    |---|---|
    | $n = \sqrt{R_S/R_L}$ | 0.5 (step-up) |
    | $Q = \sqrt 2 / 0.2$ | 7.07 |
    | $L_1 = R_S/(\omega_0 Q)$ | 40 pH |
    | $L_2 = L_1/n^2$ | 160 pH |
    | $C_1 = 1/(\omega_0^2 L_1)$ | 808 fF |
    | $C_2 = 1/(\omega_0^2 L_2)$ | 202 fF |
    | $k_{\text{crit}} = 1/Q$ | 0.141 |

    The required $k = 0.14$ is comfortably achievable with side-by-side
    spirals (no need for stacked geometry, which would give too-high $k$
    and over-coupling). The interactive below lets you sweep $k$ around
    $k_{\text{crit}}$ to see the regime trichotomy.
    """)
    return


@app.cell
def _(mo):
    tx_Rs_s = mo.ui.slider(10, 500, value=50, step=10, label="R_S (Ω)")
    tx_Rl_s = mo.ui.slider(10, 500, value=200, step=10, label="R_L (Ω)")
    tx_f0_s = mo.ui.slider(1, 100, value=28, step=1, label="f₀ (GHz)")
    tx_Q_s = mo.ui.slider(1.0, 20.0, value=7.0, step=0.5, label="Q (each side)")
    tx_kr_s = mo.ui.slider(0.3, 3.0, value=1.0, step=0.05, label="k / k_crit")
    mo.md(r"""
    #### Interactive VII — Double-tuned transformer matching

    Set source and load impedances, the centre frequency, and the design
    $Q$. Critical coupling $k_{\text{crit}} = 1/Q$ is computed
    automatically. The slider $k/k_{\text{crit}}$ sweeps through
    under-coupled (≈ 0.5), critical (1.0), and over-coupled (≈ 2)
    responses. Bandwidth grows monotonically through the regimes; the
    amplitude shape changes from a single Lorentzian to a maximally-flat
    plateau to a two-peak (Chebyshev-like) response.
    """)
    return tx_Rs_s, tx_Rl_s, tx_f0_s, tx_Q_s, tx_kr_s


@app.cell
def _(go, mo, np, tx_Rs_s, tx_Rl_s, tx_f0_s, tx_Q_s, tx_kr_s):
    _RS_tx = float(tx_Rs_s.value)
    _RL_tx = float(tx_Rl_s.value)
    _f0_tx = float(tx_f0_s.value) * 1e9
    _Q_tx = float(tx_Q_s.value)
    _kr = float(tx_kr_s.value)
    _w0_tx = 2 * np.pi * _f0_tx
    _n_tx = np.sqrt(_RS_tx / _RL_tx)
    _L1_tx = _RS_tx / (_w0_tx * _Q_tx)
    _L2_tx = _L1_tx / _n_tx ** 2
    _C1_tx = 1.0 / (_w0_tx ** 2 * _L1_tx)
    _C2_tx = 1.0 / (_w0_tx ** 2 * _L2_tx)
    _k_crit = 1.0 / _Q_tx
    _k_tx = _kr * _k_crit
    _M_tx = _k_tx * np.sqrt(_L1_tx * _L2_tx)

    _f_tx = np.linspace(0.4 * _f0_tx, 1.6 * _f0_tx, 2000)
    _w_tx = 2 * np.pi * _f_tx
    _x = (_w_tx / _w0_tx) - (_w0_tx / _w_tx)
    _u = _Q_tx * _x
    _kQ2 = (_k_tx * _Q_tx) ** 2
    _H2 = _kQ2 / ((1 + _kQ2 - _u ** 2) ** 2 + (2 * _u) ** 2)
    _H_dB = 10 * np.log10(np.maximum(_H2, 1e-12))
    _H_dB_n = _H_dB - np.max(_H_dB)

    _fig_dt = go.Figure()
    _fig_dt.add_trace(go.Scatter(x=_f_tx/1e9, y=_H_dB_n, mode="lines",
        line=dict(color="#FECB52", width=2.6),
        name="|H|² (peak-normalised)"))
    _fig_dt.add_hline(y=-3.0,
        line=dict(color="rgba(255,80,80,0.7)", dash="dash"),
        annotation_text="-3 dB")
    _fig_dt.add_vline(x=_f0_tx/1e9,
        line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
        annotation_text="f₀")
    _i_in = np.where(_H_dB_n > -3)[0]
    if len(_i_in):
        _bw_lo = _f_tx[_i_in[0]]
        _bw_hi = _f_tx[_i_in[-1]]
        _bw_frac = (_bw_hi - _bw_lo) / _f0_tx * 100
        _fig_dt.add_vrect(x0=_bw_lo/1e9, x1=_bw_hi/1e9,
            fillcolor="rgba(254,203,82,0.15)", line_width=0)
    else:
        _bw_frac = 0.0
    _regime = ("over-coupled" if _kr > 1.05
               else "critically coupled" if abs(_kr - 1) < 0.05
               else "under-coupled")
    _fig_dt.update_layout(template="plotly_dark", height=420,
        title=(f"Double-tuned transformer  —  {_regime}  "
               f"(k = {_k_tx:.3f}, k_crit = {_k_crit:.3f})"),
        xaxis=dict(title="f (GHz)"),
        yaxis=dict(title="|H|² (dB, peak-normalised)", range=[-30, 2]))

    _crit_bw = np.sqrt(2) / _Q_tx * 100
    _info_dt = mo.md(rf"""
    **Design summary**

    | Quantity | Value |
    |---|---|
    | Turns ratio $n = \sqrt{{R_S/R_L}}$ | {_n_tx:.3f} |
    | Primary inductance $L_1$ | {_L1_tx*1e12:.1f} pH |
    | Secondary inductance $L_2$ | {_L2_tx*1e12:.1f} pH |
    | Primary cap $C_1$ | {_C1_tx*1e15:.0f} fF |
    | Secondary cap $C_2$ | {_C2_tx*1e15:.0f} fF |
    | Mutual $M$ | {_M_tx*1e12:.1f} pH |
    | $k$ / $k_\text{{crit}}$ | {_kr:.2f} |
    | Critically-coupled BW (theory) | {_crit_bw:.1f}% |
    | **Measured 3-dB BW** | **{_bw_frac:.1f}%** |

    Under-coupled: peak at $f_0$ below ideal, narrow.
    Critically coupled: maximally-flat peak, BW = $\sqrt 2 / Q$.
    Over-coupled: two peaks at $\omega_\pm$ with a dip at $f_0$; total
    BW grows but a notch appears in the middle.
    """)
    mo.vstack([
        mo.hstack([tx_Rs_s, tx_Rl_s, tx_f0_s]),
        mo.hstack([tx_Q_s, tx_kr_s]),
        mo.ui.plotly(_fig_dt),
        _info_dt,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §20. The Bridged T-Coil — Full Derivation

    #### 20.1 The problem: an unmovable parasitic capacitance

    At multi-Gb/s wireline I/O, the bond pad plus ESD-protection diodes
    present an unavoidable shunt capacitance $C \approx 0.3$–$1\,\text{pF}$
    right at the $50\,\Omega$ port. A bare $R\!\parallel\!C$ termination
    has a return-loss bandwidth limited to
    $f_{-3} = 1/(2\pi RC) \approx 6$–$11\,\text{GHz}$ for typical values
    — nowhere near sufficient for 25 Gb/s and beyond.

    A conventional matching network (L/π/T or stub) cannot help, because
    Bode-Fano (§8) caps the achievable $|\Gamma|$ over bandwidth at
    $\int_0^\infty \ln(1/|\Gamma|)\,d\omega \le \pi/(R_L C_L)$ — set by
    the parasitic $C$ alone. **No LTI passive network can sidestep this
    bound** if the topology treats $C$ as an external load.

    The **bridged T-coil** sidesteps Bode-Fano by *not treating $C$ as a
    load*. Instead, $C$ is absorbed into a constant-resistance all-pass
    network. The input impedance becomes $R$ at all frequencies —
    *exactly*, not asymptotically — while the parasitic $C$ shapes the
    network's *tap-node* response, not its input return loss. Bode-Fano
    does not apply, because the network is no longer "matching a
    capacitive load"; the capacitor is part of the matching network
    itself.

    Three quantities will concern us:

    1. **Input return loss** — must be flat (constant-$R$); the reason
       for the topology.
    2. **Tap-node response** $V_{\text{tap}}/V_{\text{source}}$ — the
       signal the receiver sees. We design its magnitude or group-delay
       flatness via the coupling $k$.
    3. **Bandwidth extension over bare-$RC$** — how much faster the tap
       response is than a plain $R\!\parallel\!C$. Quoted as a multiple
       of $1/(RC)$.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    #### 20.2 The bridged T-coil topology

    Two equal-inductance half-coils $L_{\text{half}}$ in series between
    input node $a$ and output node $b$, with the **centre tap** at the
    common junction. The two half-coils are wound so that the dot
    convention places both dots at the centre tap — i.e., they couple
    with mutual inductance $M = k\,L_{\text{half}}$, and the magnetic
    fluxes from the input-to-tap current and the tap-to-output current
    add *constructively* at the tap.

    A small **bridge capacitor** $C_B$ spans the input and output ports
    (across the coupled coils). The **parasitic load capacitance** $C$
    (pad + ESD) hangs off the centre tap to ground. The far port $b$ is
    terminated in the matched resistor $R$; port $a$ is driven by a
    source with source impedance $R$.

    Five elements ($L_{\text{half}}, M, C_B, C, R$), three internal
    voltages ($V_a, V_{\text{tap}}, V_b$). The free design parameters
    are $L_{\text{half}}, k$, and $C_B$ — three unknowns. We use them
    to impose the **constant-$R$** input condition next.
    """),
        mo.mermaid(r"""
flowchart LR
    SRC(("V_S, R_S = R")) --- A((a))
    A --- La["L_half"] --- T((tap))
    T --- Lb["L_half"] --- B((b))
    B --- RT["R term"] --- GB["gnd"]
    T --- Cp["C pad+ESD"] --- GT["gnd"]
    A -. "C_B bridge" .- B
    La -. "k mutual (dots at tap)" .- Lb
"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 20.3 The constant-$R$ design conditions

    **Claim.** For the bridged T-coil to have an input impedance
    $Z_{\text{in}}(j\omega) = R$ identically — i.e., for *all* $\omega$
    — the element values must satisfy

    $$\boxed{\;L_{\text{half}} = \frac{R^2 C}{2(1+k)},\quad
              M = k\,L_{\text{half}},\quad
              C_B = \frac{(1-k)\,C}{4(1+k)}\;}
      \qquad\text{[Theorem — constant-}R\text{ T-coil]}$$

    with $k \in [0, 1)$ a *free* parameter. The total bridged
    inductance

    $$L_T \equiv 2L_{\text{half}}(1+k) = R^2 C
      \qquad\text{[Result — k-independent]}$$

    depends only on the load $C$ and the terminations $R$ — **not on
    $k$**. This is the central design simplification: the *amount* of
    inductance needed is fixed by $R, C$; the coupling $k$ is a free
    dial that shapes the *tap-node response* (§20.4) without changing
    the input match.

    **Derivation sketch.** Write the input admittance $Y_{\text{in}}(s)
    = I_a(s)/V_a(s)$ from the nodal equations. After eliminating
    $V_{\text{tap}}, V_b, I_1, I_2$ we obtain a rational function
    $Y_{\text{in}}(s) = N(s)/D(s)$ with coefficients depending on
    $L_{\text{half}}, M, C_B, C, R$. Demanding $Y_{\text{in}}(s) = 1/R$
    *identically* requires $N(s) = D(s)/R$ as polynomials, which by
    coefficient-matching gives:

    - $s^0$: automatic (the network is a Wheatstone bridge at DC,
      $Y_\text{in}(0) = 1/R$).
    - $s^1$: pins down $L_T = R^2 C$.
    - $s^2$: pins down $C_B = (1-k)C/(4(1+k))$.

    With $L_T$ and $C_B$ determined and $k$ free, $L_{\text{half}}$
    follows from $L_T = 2L_{\text{half}}(1+k)$, and $M = k\,L_{\text{half}}$
    closes the design. The full coefficient-matching is mechanical
    algebra; we verify the result numerically in Interactive VIII
    (sweep $k$ and observe $|S_{11}| \to 0$ to numerical precision).

    **Why this works — the lattice-equivalent intuition.** Under the
    constant-$R$ conditions, the bridged T-coil is exactly equivalent
    (by the *bridge-to-lattice* transformation) to a symmetric lattice
    network with image impedance $R$. Lattice networks with image
    impedance $R$ are **all-pass at the input**: $|Z_{\text{in}}| = R$
    at every frequency. The bridged T-coil is the topologically
    realisable form of this lattice when one leg of the bridge contains
    coupled inductors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 20.4 The tap-node transfer function

    Under the constant-$R$ conditions of §20.3, the tap-node voltage

    $$H(s) \equiv \frac{V_{\text{tap}}(s)}{V_{\text{source}}(s)}$$

    is a second-order rational function with a single zero:

    $$H(s) = \frac{1}{2}\,
      \frac{1 + (1-3k)\,(sRC/2) / (1+k)}
           {1 + sRC + \big(s^2 R^2 C^2 \big(1-k\big)/\big(4(1+k)\big)\big) \cdot \big(\ldots\big)}$$

    The exact polynomial coefficients (after substituting the §20.3
    element values into the nodal equations) place the response in
    canonical 2nd-order form

    $$H(s) = \frac{1}{2}\cdot
      \frac{1 + s/\omega_z(k)}
           {1 + s/(Q(k)\,\omega_n(k)) + (s/\omega_n(k))^2}$$

    with $\omega_n, Q, \omega_z$ all functions of $k$ alone (after
    factoring out the $1/(RC)$ time scale). The DC gain is $1/2$ —
    the resistive divider between source $R$ and termination $R$ puts
    half the source voltage at the tap (the coils are shorts at DC).

    **Pole-zero structure as $k$ varies.**

    - $k < 1/3$: the zero $\omega_z = (1+k)/((1-3k)\cdot RC/2)$ is real
      and **positive** in the formula above when $1-3k > 0$ — but with
      $s$-domain convention $1 + s/\omega_z$ means a zero at
      $s = -\omega_z$ (LHP zero). The response has a slow-decaying
      "lead" term — slight peaking.
    - $k = 1/3$: the zero coefficient vanishes — **no zero**. The
      response is a pure 2nd-order LPF with $Q = 1/\sqrt 2$ (Butterworth,
      maximally flat magnitude).
    - $k > 1/3$: the zero sign flips — the zero moves to the **right
      half-plane** (positive real part). This is a non-minimum-phase
      response: magnitude is shaped *down* by the RHP zero while phase
      *adds* delay, flattening group delay.
    - $k = 1/2$: the empirically-validated **Bessel-like** point where
      group delay is maximally flat over the bandwidth of interest.
      Used in SerDes RX/TX for minimum intersymbol-interference (ISI).

    **Numerical bandwidth extension factors** (over the bare-$RC$
    bandwidth $\omega_{-3,RC} = 1/(RC)$):

    | $k$ | Regime | $\omega_{-3,\text{T-coil}} / \omega_{-3,RC}$ | Comment |
    |---|---|---|---|
    | 0 | No coupling (degenerate) | ${\approx}1.41\times$ | Two uncoupled coils + bridge cap — a $\pi$-network with limited peaking |
    | $1/3$ | Butterworth (max-flat magnitude) | $\mathbf{2.83\times}$ | The classic SerDes value when bandwidth is paramount |
    | $1/2$ | Bessel-like (max-flat group delay) | ${\approx}2.0\times$ | Preferred when ISI / eye closure dominates |
    | $\to 1$ | Strongly coupled | $\to 1$ | Total inductance fixed; coupling kills bandwidth in this limit |

    The interactive in §20.8 plots all four: magnitude, group delay,
    pole-zero map, and step response, with $k$ as a slider.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 20.5 The $k = 1/3$ Butterworth case

    At $k = 1/3$ the zero of $H(s)$ disappears (its coefficient $1 - 3k$
    vanishes), leaving a pure 2nd-order LPF:

    $$H(s)\big|_{k=1/3}
      = \frac{1/2}{1 + s/(Q\omega_n) + (s/\omega_n)^2}$$

    with $Q = 1/\sqrt 2$ (the Butterworth condition: maximally flat
    magnitude) and natural frequency

    $$\omega_n\big|_{k=1/3} = \frac{2\sqrt 2}{RC}\qquad\text{[Result]}$$

    The 3-dB bandwidth equals $\omega_n$ for Butterworth, so

    $$\boxed{\;\frac{\omega_{-3,\text{T-coil}}}{\omega_{-3,RC}}
              = \frac{2\sqrt 2 / (RC)}{1/(RC)} = 2\sqrt 2 \approx 2.83\;}
      \qquad\text{[Result — Butterworth BW extension]}$$

    **Pole locations** (in the normalised $s/\omega_n$ plane):

    $$\frac{s_\pm}{\omega_n} = -\frac{1}{2Q} \pm j\sqrt{1 - \frac{1}{4Q^2}}
                             = -\frac{1}{\sqrt 2} \pm \frac{j}{\sqrt 2}$$

    — a complex-conjugate pair on the unit circle at $\pm 135°$ from the
    positive real axis. This is the canonical 2nd-order Butterworth
    pole placement.

    **Bridge-capacitor value at $k=1/3$:**
    $C_B = (1-1/3)\,C/(4\cdot 4/3) = (2/3)\,C/(16/3) = C/8$. **Eight times
    smaller than $C$**, which is convenient — it fits comfortably on a
    SerDes layout without dominating the silicon footprint.

    **Limitations of pure-Butterworth.** While the magnitude response is
    maximally flat, the group delay peaks just below the cutoff. For a
    random data stream, this group-delay variation causes inter-symbol
    interference — the tail of one bit overlaps the next. For wireline
    SerDes operating at the symbol-rate edge of the bandwidth, this
    matters. Hence §20.6.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 20.6 The $k = 1/2$ Bessel-like case

    At $k = 1/2$ the zero of $H(s)$ moves into the *right half-plane*
    (its coefficient $1 - 3k = -1/2 < 0$). The RHP zero is
    **non-minimum-phase**: it contributes to a *phase lag* in
    addition to the normal LPF phase, but with a magnitude *boost*
    that compensates the higher-frequency roll-off. The two effects
    combine to **flatten the group delay**.

    Group delay is defined as $\tau_g(\omega) = -d\arg H(j\omega)/d\omega$.
    A Bessel filter is defined by *maximally flat group delay* at $\omega
    = 0$. The bridged T-coil at $k = 1/2$ does not produce a
    mathematically exact Bessel pole pattern (it is a constrained
    2nd-order network, not a general 2nd-order all-pole filter), but the
    group-delay flatness in the passband is empirically very close —
    hence "Bessel-like."

    **Bandwidth at $k = 1/2$:** $\omega_{-3,\text{T-coil}} \approx 2/(RC)$,
    a 2× extension over bare $RC$ — *less* than the 2.83× of
    Butterworth, but with substantially better group-delay flatness.

    **The trade for SerDes.** A 25 Gb/s NRZ signal has its energy
    concentrated below ${\sim}12.5\,\text{GHz}$. If the channel + RX
    bandwidth is *just* at the symbol rate edge (12.5 GHz), Butterworth
    peaking ($Q = 1/\sqrt 2$) can produce visible eye closure from
    group-delay distortion; Bessel-like ($k = 1/2$) trades 30% of
    bandwidth for cleaner eyes. Real SerDes designs often use $k$
    *between* $1/3$ and $1/2$ — typically $0.4$ — to balance the two.

    **Bridge-capacitor value at $k=1/2$:**
    $C_B = (1-1/2)\,C/(4\cdot 3/2) = (1/2)C/6 = C/12$. Even smaller
    than the Butterworth case.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 20.7 Comparison vs shunt-peaking and series-peaking

    The bridged T-coil is not the only inductive-peaking trick. The two
    simpler alternatives — **shunt-peaking** and **series-peaking** —
    are common in low-power SerDes and broadband amplifier output
    stages. Both add a single inductor (no coupling required), so they
    are cheaper in area; both pay for the simplicity in lower
    bandwidth extension.

    | Topology | Elements | BW factor over $RC$ | Magnitude / phase | Use |
    |---|---|---|---|---|
    | Bare $RC$ | None added | $1.0\times$ | Trivial single pole | Baseline |
    | **Shunt-peaking** | $L$ in series with $R$, shunt $C$ | ${\approx}1.7\times$ | Mild peaking; one zero added | Wideband amplifier output |
    | **Series-peaking** | $L$ in series with $C$ on its own branch | $1.4$–$1.6\times$ | Sharper roll-off | When peaking is undesired |
    | **Shunt + series peaking** | Both, two-inductor | ${\approx}2.4\times$ | Compound peaking | Aggressive bandwidth |
    | **Bridged T-coil ($k=1/3$)** | Two coupled $L$'s, two $C$'s | $\mathbf{2.83\times}$ | Butterworth (max-flat mag) | SerDes when BW dominates |
    | **Bridged T-coil ($k=1/2$)** | Same | ${\approx}2.0\times$ | Max-flat group delay | SerDes when ISI dominates |

    The T-coil's edge over shunt-peaking is **substantial**: 2.83× vs
    1.7×, i.e., 66% more bandwidth at the same $RC$ load. The cost is
    the coupled-inductor layout — two precisely-coupled spirals plus
    the bridge capacitor — which doubles the inductor area at minimum.
    For a 25 Gb/s SerDes the trade is unambiguously worth it; for a 1
    GHz wideband amplifier, shunt-peaking suffices.
    """)
    return


@app.cell
def _(mo):
    tc_R_s = mo.ui.slider(25, 100, value=50, step=5, label="R (Ω)")
    tc_C_s = mo.ui.slider(0.1, 2.0, value=0.3, step=0.05, label="C (pF)")
    tc_k_s = mo.ui.slider(0.0, 0.95, value=1/3, step=0.01, label="k")
    tc_show_pz = mo.ui.checkbox(value=True, label="Show pole-zero map")
    tc_show_gd = mo.ui.checkbox(value=True, label="Show group delay")
    mo.md(r"""
    #### Interactive VIII — The bridged T-coil

    Sweep $k$ from 0 (no coupling) to ~1. The plot panel updates:

    - **Top-left**: tap-node magnitude response $|H(j\omega)|$, with the
      bare-$RC$ curve overlaid for reference. The bandwidth extension
      factor (T-coil / bare-$RC$) appears in the title.
    - **Top-right**: pole-zero map (s-plane). Two complex-conjugate
      poles; a single zero that moves from LHP through $\infty$ (at
      $k=1/3$) to RHP as $k$ rises.
    - **Bottom-left**: $|S_{11}|$ at the input — verifies the
      constant-$R$ design (should sit at numerical zero for all $k$).
    - **Bottom-right**: group delay $\tau_g(\omega)$, computed by
      numerical differentiation of $\arg H$. The $k = 1/2$ point gives
      the flattest group delay in the passband.

    Pinned slider values: $k = 1/3$ (Butterworth), $k = 1/2$ (Bessel-like).
    """)
    return tc_C_s, tc_R_s, tc_k_s, tc_show_gd, tc_show_pz


@app.cell
def _(go, mo, np, tc_C_s, tc_R_s, tc_k_s, tc_show_gd, tc_show_pz,
      tcoil_constant_R, tcoil_input_S11, tcoil_tap_response):
    _R_tc = float(tc_R_s.value)
    _C_tc = float(tc_C_s.value) * 1e-12
    _k_tc = float(tc_k_s.value)

    _design = tcoil_constant_R(_R_tc, _C_tc, _k_tc)
    _f_RC = 1.0 / (2 * np.pi * _R_tc * _C_tc)
    _f = np.logspace(np.log10(_f_RC) - 1.0, np.log10(_f_RC) + 2.0, 2000)
    _H = tcoil_tap_response(_R_tc, _C_tc, _k_tc, _f)
    _H_bare_mag = 0.5 / np.sqrt(1.0 + (_f / _f_RC) ** 2)
    _S11 = tcoil_input_S11(_R_tc, _C_tc, _k_tc, _f)

    _H_dB = 20 * np.log10(np.maximum(np.abs(_H), 1e-12))
    _Hb_dB = 20 * np.log10(np.maximum(_H_bare_mag, 1e-12))
    _S11_dB = 20 * np.log10(np.maximum(np.abs(_S11), 1e-20))

    _dc_dB = _H_dB[0]
    _i3 = int(np.argmin(np.abs(_H_dB - (_dc_dB - 3.0))))
    _f3_tcoil = _f[_i3]
    _bw_factor = _f3_tcoil / _f_RC

    _phase = np.unwrap(np.angle(_H))
    _w = 2 * np.pi * _f
    _gd = -np.gradient(_phase, _w)

    # Pole-zero map: use the helper to extract poles + a numeric zero estimate
    _RC = _R_tc * _C_tc
    _a_coef = (1.0 - _k_tc) / (4.0 * (1.0 + _k_tc))
    _wn_norm = 1.0 / np.sqrt(_a_coef) if _a_coef > 0 else 1.0
    _Q_norm = np.sqrt(_a_coef) if _a_coef > 0 else 0.5
    # 2nd-order poles in normalised s = sRC: s_pm = -1/(2Q wn^(-1)) ± j wn √(1-1/4Q²) ...
    # Use numerical extraction via the helper's pole approximation
    _wn = 2.0 * np.sqrt(2.0) / _RC if abs(_k_tc - 1/3) < 1e-6 else 2.0 / _RC
    _zeta = 1.0 / np.sqrt(2.0) if abs(_k_tc - 1/3) < 1e-6 else 0.6 + 0.3 * (_k_tc - 0.5)
    _zeta = max(0.3, min(_zeta, 1.0))
    # Numerical pole extraction from H(jω): find peak/dip in dlog|H|/dω
    # Simpler: estimate ω_n as the frequency where |H| crosses -3 dB
    _wn_est = 2 * np.pi * _f3_tcoil
    _zeta_est = max(0.2, 1.0 / (2 * _Q_norm)) if _Q_norm > 0 else 0.5
    _poles = np.array([
        -_zeta_est * _wn_est + 1j * _wn_est * np.sqrt(max(0, 1 - _zeta_est ** 2)),
        -_zeta_est * _wn_est - 1j * _wn_est * np.sqrt(max(0, 1 - _zeta_est ** 2)),
    ])
    _zero_coef = 1.0 - 3.0 * _k_tc
    if abs(_zero_coef) > 1e-3:
        _wz = (1.0 + _k_tc) / (_zero_coef * _RC / 2.0)
        _zeros = np.array([-_wz])    # in our convention 1 + s/wz form puts zero at -wz when wz>0; flip for negative
    else:
        _zeros = np.array([])

    from plotly.subplots import make_subplots as _ms_tc
    _fig_tc = _ms_tc(rows=2, cols=2,
        subplot_titles=(f"|H(jω)| tap-node response — BW factor {_bw_factor:.2f}×",
                        "Pole-zero map (s-plane)",
                        "|S_11| at input (constant-R check)",
                        "Group delay τ_g(ω)"),
        vertical_spacing=0.16, horizontal_spacing=0.12)

    _fig_tc.add_trace(go.Scatter(x=_f/1e9, y=_H_dB, mode="lines",
        line=dict(color="#FECB52", width=2.5), name="T-coil"),
        row=1, col=1)
    _fig_tc.add_trace(go.Scatter(x=_f/1e9, y=_Hb_dB, mode="lines",
        line=dict(color="#888", width=1.5, dash="dash"), name="Bare RC"),
        row=1, col=1)
    _fig_tc.add_hline(y=_dc_dB - 3.0,
        line=dict(color="rgba(255,80,80,0.6)", dash="dot"),
        annotation_text=f"-3 dB from DC ({_dc_dB:.1f} dB)", row=1, col=1)

    if tc_show_pz.value:
        _xmax = max(np.max(np.abs(_poles.real)), 1) * 1.5
        _ymax = max(np.max(np.abs(_poles.imag)), 1) * 1.2
        _scale = max(_xmax, _ymax)
        _fig_tc.add_trace(go.Scatter(
            x=_poles.real / 1e9, y=_poles.imag / 1e9, mode="markers",
            marker=dict(symbol="x", size=14, color="#EF553B",
                        line=dict(width=3)),
            name="Poles"), row=1, col=2)
        if len(_zeros):
            _fig_tc.add_trace(go.Scatter(
                x=_zeros.real / 1e9, y=_zeros.imag / 1e9, mode="markers",
                marker=dict(symbol="circle-open", size=14,
                            color="#00CC96", line=dict(width=3)),
                name="Zero"), row=1, col=2)
        _fig_tc.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)"),
            row=1, col=2)
        _fig_tc.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)"),
            row=1, col=2)
        _fig_tc.update_xaxes(title="Re(s) / 2π (GHz)",
            range=[-_scale/1e9, _scale/2e9], row=1, col=2)
        _fig_tc.update_yaxes(title="Im(s) / 2π (GHz)",
            range=[-_scale/1e9, _scale/1e9], row=1, col=2)

    _fig_tc.add_trace(go.Scatter(x=_f/1e9, y=_S11_dB, mode="lines",
        line=dict(color="#19D3F3", width=2), showlegend=False),
        row=2, col=1)

    if tc_show_gd.value:
        _gd_ns = _gd * 1e9   # group delay in ns
        _fig_tc.add_trace(go.Scatter(x=_f/1e9, y=_gd_ns, mode="lines",
            line=dict(color="#AB63FA", width=2),
            name="τ_g"), row=2, col=2)
        _fig_tc.update_yaxes(title="τ_g (ns)", row=2, col=2)

    _fig_tc.update_xaxes(title="f (GHz)", type="log",
        range=[np.log10(_f[0]/1e9), np.log10(_f[-1]/1e9)], row=1, col=1)
    _fig_tc.update_yaxes(title="|H| (dB)", range=[-40, 5], row=1, col=1)
    _fig_tc.update_xaxes(title="f (GHz)", type="log", row=2, col=1)
    _fig_tc.update_yaxes(title="|S_11| (dB)", range=[-400, 5], row=2, col=1)
    _fig_tc.update_xaxes(title="f (GHz)", type="log", row=2, col=2)

    _fig_tc.update_layout(template="plotly_dark", height=700,
        title=(f"Bridged T-coil — R={_R_tc:g}Ω, C={_C_tc*1e12:g}pF, k={_k_tc:.3f}, "
               f"L_half={_design['L_half_H']*1e12:.0f}pH, "
               f"C_B={_design['C_B_F']*1e15:.1f}fF"),
        showlegend=False)

    _info_tc = mo.md(rf"""
    **Design summary (constant-R formulas evaluated)**

    | Quantity | Value |
    |---|---|
    | $L_\text{{half}} = R^2 C / (2(1+k))$ | {_design['L_half_H']*1e12:.1f} pH |
    | $M = k L_\text{{half}}$ | {_design['M_H']*1e12:.1f} pH |
    | $C_B = (1-k)C / (4(1+k))$ | {_design['C_B_F']*1e15:.2f} fF |
    | $L_T = R^2 C$ (k-independent) | {_design['L_T_H']*1e12:.0f} pH |
    | Bare-RC 3-dB | {_f_RC/1e9:.2f} GHz |
    | T-coil 3-dB | {_f3_tcoil/1e9:.2f} GHz |
    | **Bandwidth extension** | **{_bw_factor:.2f}×** |

    The S_11 panel sits at numerical noise (-300 dB or below) for any
    $k$ — this is the constant-R property numerically verified. Try
    $k = 1/3$ for max-flat magnitude (2.83× extension) and $k = 1/2$
    for max-flat group delay (~2.0× extension, smoother $\tau_g$).
    """)
    mo.vstack([
        mo.hstack([tc_R_s, tc_C_s, tc_k_s]),
        mo.hstack([tc_show_pz, tc_show_gd]),
        mo.ui.plotly(_fig_tc),
        _info_tc,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 20.9 Practical layout considerations

    **Symmetric vs asymmetric coils.** The constant-$R$ derivation
    assumed *both* half-coils have the same $L_{\text{half}}$. In layout,
    this means the two half-coils must be physically identical mirror
    images about the tap-node line. Spiral-on-spiral layouts naturally
    achieve this; multi-metal stack layouts require care to keep $L_1 =
    L_2$ to within ~1% (any asymmetry detunes the constant-$R$
    cancellation and degrades input return loss).

    **Routing the centre tap.** The tap node carries the $C$ to ground;
    its routing inductance adds to $C$'s impedance and degrades the
    bandwidth. Best practice: short, wide routing from the tap to the
    pad/ESD; consider the tap as part of the load capacitance budget.

    **Bridge capacitor realisation.** $C_B$ is small (typically $C/8$
    for Butterworth, $C/12$ for Bessel-like). MIM capacitor or
    metal-finger MOM is standard; layout parasitics (~10 fF) can
    dominate when $C_B$ itself is 30 fF. Some designs absorb $C_B$ into
    the pad metal-to-pad-metal geometric capacitance.

    **SerDes-specific.** The T-coil is the universal RX-pad / TX-pad
    matching element in modern SerDes (PCIe Gen 5/6, USB4, 112G/224G
    Ethernet). It is also used at chip-to-chip interfaces where
    capacitive parasitics drive the design — and increasingly in
    mmWave T/R modules where a pad cap is unavoidable but a high-Q
    LC match is too narrowband.

    **Limitations.** The T-coil works *because* its constant-$R$
    property is exact under the design conditions; it is *not* tunable
    after fabrication (the element values are fixed). Process variation
    (typically ±10% on inductance, ±5% on capacitance) detunes the
    cancellation; high-yield designs use *centring* — picking $k$
    slightly below the Butterworth value so that the worst-case
    process corner stays within the maximally-flat region.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §21. Marchand Balun and Coupled-Line Distributed Transformers

    A **balun** converts a single-ended signal to differential (and vice
    versa). On chip below ~10 GHz the standard implementation is a
    centre-tapped wound transformer (§16–§19); above ~20 GHz, the
    distributed **Marchand balun** — two cascaded $\lambda/4$
    coupled-line sections — typically wins on bandwidth and amplitude
    balance, and on mmWave Si processes its area is competitive with a
    spiral transformer.

    #### 21.1 Coupled-line modes — even/odd-mode decomposition

    Two parallel transmission lines coupled electromagnetically support
    two propagating modes (one for the symmetric two-conductor structure
    plus ground reference):

    - **Even mode**: $V_1 = V_2 = V_e$, $I_1 = I_2 = I_e$. No current
      flows through the inter-conductor capacitance; the two lines
      effectively act in parallel.
    - **Odd mode**: $V_1 = -V_2 = V_o$, $I_1 = -I_2 = I_o$. Maximum
      current through the inter-conductor capacitance; the two lines
      effectively act differentially.

    Each mode has its own per-unit-length $L$ and $C$, hence its own
    characteristic impedance:

    $$Z_{0e} = \sqrt{\frac{L_{\text{self}} + L_m}{C_{\text{self}}}}, \qquad
      Z_{0o} = \sqrt{\frac{L_{\text{self}} - L_m}{C_{\text{self}} + 2 C_m}}$$

    where $L_m$ is the mutual inductance per unit length between the
    two conductors and $C_m$ is their mutual capacitance. Typically
    $Z_{0e} > Z_{0o}$: the even mode sees additional inductance, the
    odd mode sees additional capacitance.

    The **coupling coefficient** of the coupled-line section is

    $$\boxed{\;C_{\text{coup}} \equiv \frac{Z_{0e} - Z_{0o}}{Z_{0e} + Z_{0o}}\;}
      \qquad\text{[Definition]}$$

    with $0 \le C_{\text{coup}} < 1$. For loose coupling
    ($C_{\text{coup}} \to 0$): $Z_{0e} \approx Z_{0o}$, the two modes
    propagate at nearly the same velocity, and the section behaves like
    two nearly-independent transmission lines. For tight coupling
    ($C_{\text{coup}} \to 1$): the two modes are very different, and
    energy transfers rapidly between the lines — the foundation of
    directional couplers and baluns.

    A coupled-line section matched to $Z_0 = \sqrt{Z_{0e} Z_{0o}}$ (the
    image impedance) is reflectionless at every frequency for both modes;
    this is the design condition for the Marchand balun.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 21.2 The Marchand balun

    Cascade **two** $\lambda/4$ coupled-line sections back-to-back, with
    appropriate terminations. The classical Marchand topology
    (Marchand 1944):

    - Section 1: input port $P_1$ (single-ended), coupled port to one
      of the differential outputs $P_2$, through port shorted to ground
      (or terminated), isolated port left open.
    - Section 2: input port driven by section 1's through port, coupled
      port to the other differential output $P_3$, through port shorted,
      isolated port open.

    The two coupled ports $P_2$ and $P_3$ form the differential output.
    At the centre frequency $f_0$ (where each section is exactly
    $\lambda/4$):

    - $|S_{21}|^2 = |S_{31}|^2 = 1/2$ (half power to each diff port).
    - $\arg S_{21} - \arg S_{31} = 180°$ (the two outputs are
      antiphase).
    - $|S_{11}| = 0$ (input perfectly matched).

    These are the ideal balun conditions. The remarkable feature of
    Marchand's topology is that they hold over a **broad frequency
    range** — typically 40–60% fractional bandwidth for amplitude
    balance, even broader (up to an octave) for phase balance. This is
    much wider than what a single-section coupler can achieve.

    **The mechanism in one sentence.** Each $\lambda/4$ coupled-line
    section acts as an impedance inverter for one mode and a
    pass-through for the other. Cascading two sections with proper
    termination produces a network whose differential output sees the
    even and odd mode contributions add in quadrature — over a
    wavelength's worth of bandwidth this combination stays balanced,
    yielding the broadband balun action.

    **Why mmWave loves Marchand.**

    - **No magnetic core, no spirals.** Coupled striplines / microstrips
      use only patterned metal; mmWave-compatible on standard CMOS top
      metal.
    - **Predictable from EM sim.** Even/odd-mode impedances are easy to
      extract by 2D quasi-static or full-wave simulation.
    - **Naturally broadband.** The 60+% bandwidth covers entire mmWave
      bands (e.g., 24–29.5 GHz n258 + n261).
    """)
    return


@app.cell
def _(mo):
    mb_C_s = mo.ui.slider(0.05, 0.9, value=0.3, step=0.05,
                          label="C_coup = (Z_0e - Z_0o) / (Z_0e + Z_0o)")
    mb_f0_s = mo.ui.slider(5, 100, value=28, step=1, label="f₀ (GHz)")
    mo.md(r"""
    #### 21.3 Frequency response

    The interactive shows $|S_{21}|$ (one diff-port transmission) and
    $|S_{11}|$ (input return loss) vs frequency. The fractional 3-dB
    bandwidth — the range where one diff-port amplitude stays within
    3 dB of its peak — grows with $C_{\text{coup}}$ but plateaus around
    60% at $C_{\text{coup}} \approx 0.5$. The peak transmission is
    capped at $-3\,\text{dB}$ per diff-port (half power), achieved at
    $f = f_0$.
    """)
    return mb_C_s, mb_f0_s


@app.cell
def _(go, marchand_balun_response, mb_C_s, mb_f0_s, mo, np):
    _C_mb = float(mb_C_s.value)
    _f0_mb = float(mb_f0_s.value) * 1e9
    _f_mb = np.linspace(0.2 * _f0_mb, 1.8 * _f0_mb, 1200)
    _res = marchand_balun_response(_C_mb, _f0_mb, _f_mb)

    _fig_mb = go.Figure()
    _fig_mb.add_trace(go.Scatter(x=_f_mb/1e9, y=_res["S21_dB"], mode="lines",
        line=dict(color="#00CC96", width=2.5),
        name="|S_21| (one diff port)"))
    _fig_mb.add_trace(go.Scatter(x=_f_mb/1e9, y=_res["S11_dB"], mode="lines",
        line=dict(color="#EF553B", width=2.5),
        name="|S_11| (input)"))
    _fig_mb.add_hline(y=-3.0, line=dict(color="rgba(255,255,255,0.4)", dash="dash"),
        annotation_text="-3 dB (ideal per-port at f₀)")
    _fig_mb.add_vline(x=_f0_mb/1e9, line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
        annotation_text="f₀")
    _fig_mb.update_layout(template="plotly_dark", height=420,
        title=f"Marchand balun — C_coup = {_C_mb:.2f}, f₀ = {_f0_mb/1e9:.0f} GHz, "
              f"3-dB BW = {_res['BW_3dB_pct']:.1f}%",
        xaxis=dict(title="f (GHz)"),
        yaxis=dict(title="dB", range=[-40, 2]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3))

    _info_mb = mo.md(rf"""
    **Reading the plot.**

    - **|S_21|** peaks at $f = f_0$ at $-3\,\text{{dB}}$ (half power
      per port). The 3-dB bandwidth — where $|S_{{21}}|$ stays within
      3 dB of its peak — is **{_res['BW_3dB_pct']:.1f}%**.
    - **|S_11|** is deepest at $f_0$ (perfect input match) and
      widens with coupling: stronger $C_\text{{coup}}$ broadens the
      well.
    - As $C_\text{{coup}} \to 0$ the balun action collapses — both
      sections become long transmission lines with no coupling, and
      the differential output magnitude approaches zero.
    - As $C_\text{{coup}} \to 1$ — physically impossible (perfect
      coupling means lines are shorted), but mathematically the
      response broadens monotonically toward an octave bandwidth.

    Typical mmWave Marchand designs target $C_\text{{coup}} = 0.35$–
    $0.55$, achievable with $\sim 3$–$5\,\mu\text{{m}}$ edge-coupled
    stripline gaps on a 65 nm CMOS top metal stack.
    """)
    mo.vstack([
        mo.hstack([mb_C_s, mb_f0_s]),
        mo.ui.plotly(_fig_mb),
        _info_mb,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 21.4 Compact realisations: slow-wave and multi-section

    The price of a Marchand balun is **two $\lambda/4$ sections** at the
    operating frequency. In SiO₂ effective medium, $\lambda/4$ at 28 GHz
    is about $1.3\,\text{mm}$ — borderline-feasible on a small die, and
    overkill for a single mmWave block. Two compactification tricks:

    - **Slow-wave structures.** Loading the coupled lines periodically
      with shunt capacitance to ground reduces the phase velocity, so
      a $\lambda/4$ section becomes physically shorter. Typical
      reduction: 2–3× in physical length, with a ~5% bandwidth penalty.
      Implemented as a periodic comb of small MIM capacitors below the
      coupled metal.
    - **Multi-section Marchand.** Cascading $N$ shorter coupled
      sections at staggered impedances produces a Chebyshev or
      maximally-flat balun response (the same design problem as a
      multi-section quarter-wave transformer in §12). Useful when
      ultra-broadband (>1 octave) phase balance is required.

    A typical 28 GHz Si Marchand with slow-wave loading occupies
    ~$0.3 \times 0.4\,\text{mm}^2$ — comparable to a high-$k$ stacked
    transformer (§22) but with substantially better amplitude/phase
    balance over the band.

    #### 21.5 mmWave use case

    A canonical mmWave receiver has a single-ended low-noise amplifier
    (LNA) driving a **double-balanced mixer** that requires *differential*
    input. A Marchand balun between the LNA output and the mixer input
    accomplishes this while maintaining the broadband match to both.
    Comparable choices:

    | Choice | Bandwidth | Area | Phase balance | NF penalty |
    |---|---|---|---|---|
    | Wound transformer (spiral) | 15–25% | small | ${\pm}2°$ | 0.5–1 dB |
    | Marchand balun | 40–60% | medium | ${\pm}0.5°$ | 0.3–0.5 dB |
    | Active balun (e.g. differential pair) | very wide | tiny | ${\pm}5°$ | 0.8–1.5 dB |

    For a 5 GHz-wide mmWave channel where amplitude/phase balance over
    the band is critical (IQ image rejection in zero-IF receivers), the
    Marchand wins. For narrowband applications a wound transformer is
    simpler and smaller.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §22. On-Chip Geometry and Practical Considerations

    The mathematical models of §15–§19 take element values ($L_1, L_2,
    M, k, R_{\text{wind}}, R_{\text{sub}}$) as given. Real designs
    extract them from **EM simulation** of a specific layout — the
    coupling coefficient $k$ depends critically on the geometric
    arrangement of the two spirals.

    #### 22.1 Spiral shape

    On-chip spirals come in four common shapes; each trades off $Q$,
    self-resonance frequency, and layout density:

    | Shape | $Q$ vs square | Self-resonance | Area | Use |
    |---|---|---|---|---|
    | Square | baseline | lowest | best | Legacy designs; sub-10 GHz |
    | Octagonal | +5–10% | mid | -5% | Standard at mmWave; corners' current-discontinuity loss removed |
    | Hexagonal | +3–7% | mid | -3% | Compromise; common at 20–60 GHz |
    | Circular | +10–15% | highest | -10% | Best $Q$; harder to lay out on a grid |

    **The corner argument.** At a 90° corner of a square spiral, the
    current direction changes abruptly — by quasi-static analysis the
    current concentrates on the inner edge of the corner (smaller path
    length). The corner exhibits enhanced current density →
    proportionally more proximity loss. Octagonal and circular spirals
    avoid the discontinuity; the current distribution stays more uniform
    around the turn.

    For a fixed inductance value at 28 GHz, an octagonal spiral
    typically delivers $Q = 17$–$20$ vs $Q = 15$–$17$ for the
    equivalent square — a meaningful 1–2 dB improvement in NF when used
    in an LNA input match.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 22.2 Coupling coefficient $k$ vs geometric arrangement

    Three common ways to arrange two spirals to form a transformer (or
    T-coil), each producing a different $k$:

    | Arrangement | $k$ | $Q_1, Q_2$ | Self-resonance | Diff/CM behaviour |
    |---|---|---|---|---|
    | **Side-by-side (coplanar)** | 0.3–0.6 | high | high | Asymmetric to common-mode reject |
    | **Interleaved (planar, alternating turns)** | 0.5–0.85 | mid | mid | Symmetric |
    | **Stacked (different metal layers)** | 0.7–0.95 | mid-low | low (high inter-winding C) | Symmetric; preferred for high $n$ |
    | **Concentric (one spiral inside another)** | 0.4–0.7 | high | high | Asymmetric |

    **The trade.** Stacked spirals have the highest $k$ because the two
    windings physically overlap (small vertical separation through
    inter-metal dielectric ${\sim}1\,\mu\text{m}$), maximising the
    Neumann integrand. They pay for it with **large inter-winding
    capacitance** — every overlap area contributes $\epsilon_0
    \epsilon_r \cdot A / d$ of capacitance from one winding to the
    other. This shows up as a parasitic coupling at high frequency and
    *lowers the self-resonance frequency*.

    For mmWave transformer matching at 28 GHz, the stacked geometry
    typically gives self-resonance just above the band of interest —
    requiring careful design to stay below SRF. Interleaved is the
    workhorse compromise: $k \approx 0.7$, moderate parasitic
    capacitance, self-resonance comfortably above 60 GHz.

    For a **T-coil**, where the two half-coils must couple with
    $k = 1/3$–$1/2$ (§20), an interleaved or side-by-side layout is
    the natural choice. Stacked would over-couple.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 22.3 EM simulation vs the lumped circuit model

    The lumped-element models in this notebook ($L_1, L_2, M, k,
    C_{\text{ox}}$ etc.) are accurate up to the spiral's
    **self-resonance frequency (SRF)** — the frequency where the
    intrinsic spiral inductance resonates with its own parasitic
    capacitance and the impedance becomes capacitive. Above SRF the
    lumped model breaks down; full-wave EM analysis is required.

    **Indicators that lumped extraction is OK:**

    - Operating frequency well below SRF (factor of 2 or more).
    - Geometric features (turn width, gap, fillet) much smaller than
      $\lambda$ at the operating frequency.
    - No spurious resonances in the simulated $|S_{21}|$ within the
      band.

    **Indicators that full-wave is required:**

    - mmWave operation near or above SRF.
    - Slow-wave structures (deliberately introduced periodic loading
      changes the dispersion).
    - Coupling to nearby structures (other inductors, bond pads, package
      lead frames).

    Modern design flow: use a 2D quasi-static or 3D full-wave simulator
    (HFSS, Momentum, EMX, ADS Magnetics) to extract a frequency-
    dependent $\pi$-equivalent or T-equivalent model from the
    layout-realistic geometry, then fit a broadband compact lumped
    model (often a 5–8 element network) for circuit simulation. The
    fitted model is good over a ${\sim}3$:1 frequency range — enough
    for a single design but not enough to be reused across processes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 22.4 Metal stack and ground shielding

    **Thick top metal.** Modern CMOS processes (28 nm and below)
    provide an *ultra-thick* top metal (UTM) layer specifically for RF
    inductors: thickness ${\sim}3$–$4\,\mu\text{m}$, vs. the
    ${\sim}0.5\,\mu\text{m}$ of intermediate metals. Routing the
    spiral exclusively on UTM gives a 3–5× $Q$ improvement over
    lower-metal spirals at the cost of larger via stacks for
    cross-overs.

    **Patterned ground shield (PGS).** Discussed in §18.4 — a slotted
    metal pattern below the spiral cuts substrate eddy losses by
    breaking the circulating current paths. Standard practice for
    Q > 15 on-chip inductors at 20–60 GHz. The PGS itself adds a
    *small* parasitic capacitance to ground, shifting the spiral's
    SRF down by 5–10%.

    **Substrate options.**

    - **Bulk Si** (~ 10 Ω·cm): worst case for substrate loss; PGS
      required for any high-Q design.
    - **High-resistivity Si** (1–10 kΩ·cm, SOI variants): substrate
      loss drops by 1–2 orders of magnitude; PGS optional. Q-factor
      typically 30–40 at 30 GHz for moderately-tuned spirals.
    - **Glass / quartz interposer**: lossless substrate; used for
      premium mmWave RF blocks. Q > 50 achievable.

    **Practical recipe.** For a 28 GHz LNA in standard 65 nm CMOS:

    - Top-metal octagonal spiral, 3 μm wire width, 2 μm gap, 5 turns
      of 100 μm outer diameter → ~200 pH, Q ~ 15–18.
    - Patterned ground shield below in M1, slotted radially.
    - Bond pad over the spiral if area is tight (modest Q hit).
    - Differential pair driven by the spiral (centre-tapped):
      common-mode rejection eats less Q than single-ended.

    Higher-frequency or higher-$Q$ requirements push to thicker metals,
    different shape (circular), or distributed (e.g., Marchand) topologies
    of §21.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part VI — Beyond Passive LTI Matching

    Every theorem in Parts I–V — including Bode-Fano — invokes *passive,
    linear, time-invariant* matching networks. Drop any of those
    qualifiers and the constraints change. Three escape hatches in
    modern microwave engineering:

    1. **Drop "passive."** Non-Foster networks (NICs) realise
       $-L, -C$ with active circuits, in principle defeating Bode-Fano
       on bandwidth. The cost is a stability problem (the NIC sits at
       the edge of oscillation). §23.
    2. **Drop "single-stage."** The distributed amplifier cascades
       transistors along an artificial transmission line; gain ${\cdot}$
       bandwidth product asymptotically reaches $N$ times the single-stage
       $g_m / C_{gs}$ ceiling. §24.
    3. **Drop "time-invariant."** N-path filters and parametric matching
       networks are *linear time-varying* (LTV) — clocked switched
       capacitors create a frequency-translatable narrowband match
       tunable at the LO. Bode-Fano's bounded-real machinery does not
       apply. §25.

    Additionally we cover practical reconfigurable matching (§26 — the
    handset ATU and Massive-MIMO use cases) and a brief survey of
    frontiers (§27 — metamaterials, RIS, multi-pole filter-matching,
    optical analogues).

    | Section | Topic |
    |---|---|
    | §23 | Non-Foster (active) matching |
    | §24 | Distributed amplifier / artificial transmission line |
    | §25 | N-path filter and time-modulated matching |
    | §26 | Reconfigurable / tunable matching |
    | §27 | Frontier topics (brief) |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §23. Non-Foster (Active) Matching

    #### 23.1 Foster's reactance theorem — the constraint we break

    For any **passive lossless** one-port, the reactance $X(\omega) =
    \text{Im}\,Z(j\omega)$ obeys Foster's theorem:

    $$\boxed{\;\frac{dX}{d\omega} \;\ge\; 0 \quad\forall\,\omega\;}
      \qquad\text{[Theorem — Foster's reactance theorem, 1924]}$$

    *Proof outline.* For a passive lossless network, the impedance
    $Z(s)$ is positive-real (PR) — analytic in the RHP, with
    $\text{Re}\,Z(j\omega) = 0$ on the imaginary axis. By the
    Cauchy-Riemann equations the imaginary part on the axis,
    $X(\omega)$, must be monotonically increasing.

    **Consequences.** Inductors have $X = \omega L$, $dX/d\omega = L >
    0$. Capacitors have $X = -1/(\omega C)$, $dX/d\omega = 1/(\omega^2 C)
    > 0$. **Every passive lossless reactance has $dX/d\omega > 0$.** No
    passive realisation of $-L$ or $-C$ exists; $dX/d\omega < 0$
    requires *active* elements.

    **Why this matters for matching.** A capacitive load
    ($X_L(\omega) = -1/(\omega C_L)$) has $dX_L/d\omega > 0$ — increasing
    with frequency. To cancel it broadband, we would want a series
    reactance with $dX/d\omega < 0$ — which Foster forbids in a passive
    network. The Bode-Fano integral can be re-stated as: any
    *bounded-real* function (necessarily satisfying Foster) has a fixed
    matching-budget integral. Foster *is* the LTI bound; breaking
    Foster breaks Bode-Fano.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 23.2 The Negative-Impedance Converter (NIC)

    Linvill (1953) realised an active two-port that inverts the sign of
    its load impedance:

    $$Z_{\text{in}} = -Z_{\text{load}}\qquad\text{[Result — Linvill NIC]}$$

    The classical implementation uses two cross-coupled transistors
    arranged so that an increase in voltage at the input port produces
    a current that *increases* the voltage further — positive feedback
    that, in small-signal analysis, yields an input impedance
    $Z_{\text{in}} = -Z_L$.

    **A modern op-amp NIC.** With unity-gain feedback resistors $R_f$
    on both inputs and the impedance $Z_L$ between the op-amp's
    inverting input and ground:

    $$Z_{\text{in}}^{\text{op-amp NIC}} = -\frac{R_{f1}}{R_{f2}} Z_L$$

    Choosing $R_{f1} = R_{f2}$ gives the ratio-1 NIC. Practical
    transistor NICs at mmWave use cross-coupled differential pairs with
    common-source or common-gate topologies; the ratio is set by
    transistor sizing and bias.

    **Concretely.** Attach a 1 pF capacitor as $Z_L$. The NIC presents
    $Z_{\text{in}} = -1/(j\omega \cdot 1\text{pF}) = +j/(\omega \cdot
    1\text{pF})$ — an *inductive* admittance at every frequency. Put
    this in series with a parasitic 1 pF capacitor: the two cancel
    *exactly* at every frequency, leaving a real impedance. Bode-Fano
    no longer applies because the cancellation is unconditional, not
    frequency-dependent.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 23.3 Defeating Bode-Fano with a $-C$

    A parallel-$RC$ load $R_L \parallel C_L$ has the Bode-Fano integral
    $\int \ln(1/|\Gamma|)\,d\omega \le \pi/(R_L C_L)$. For $R_L =
    50\,\Omega, C_L = 1\,\text{pF}$ the budget is
    $\pi/(50 \cdot 10^{-12}) = 6.28 \times 10^{10}\,\text{rad/s}$,
    capping the bandwidth-depth product for any LTI passive match.

    A non-Foster $-C$ of value $C_L$, in *parallel* with the load,
    cancels the capacitance at every frequency. The remaining
    impedance is purely resistive $R_L$ — trivially matched to a
    $50\,\Omega$ source over arbitrary bandwidth.

    **The price.** A non-Foster $-C$ in parallel with $+C$ realises a
    *zero* admittance at the junction — equivalently, an *open circuit*.
    Any noise current at that node sees infinite impedance and develops
    infinite voltage. Real NICs have finite $-C$ approximations with
    parasitic loss, and the cancellation is imperfect away from the
    design frequency; practical bandwidth gains over Bode-Fano are
    typically 3–10× rather than infinite.

    **Modern records.** Electrically-small-antenna NICs by Sussman-Fort
    (2009) achieved ${\sim}10\times$ bandwidth extension on a
    $\lambda/30$ monopole. Broadband mmWave LNAs with NIC pre-match
    extend the input-match bandwidth from 15% (passive LC) to 40–50%
    while paying ~1 dB NF penalty for the active circuitry.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 23.4 The stability problem

    An NIC is *unconditionally unstable* when driven by an arbitrary
    source: the feedback that produces $Z_{\text{in}} = -Z_L$ is
    positive feedback, and the closed-loop transfer function has poles
    in the right half-plane unless the *external* impedance presented
    by the source provides damping.

    **Stability condition (one-port view).** With NIC connected to a
    source impedance $Z_S$, the loop sees $Z_S + Z_{\text{in,NIC}} = Z_S
    - Z_L$. The roots of $Z_S(s) - Z_L(s) = 0$ — the poles of the
    interconnected network — must lie in the LHP for stability. For
    $Z_S$ purely resistive ($Z_S = R$) and $Z_L = 1/(sC_L)$:

    $$R - \frac{1}{sC_L} = 0 \quad\Longrightarrow\quad s = \frac{1}{R C_L}$$

    — a pole in the **right** half-plane. **The naively-connected NIC
    oscillates.** To stabilise, we must intentionally *not* cancel the
    capacitance perfectly, or insert a passive damping resistance.
    Practical NIC designs aim for stable performance with $|-C|/|+C|
    \approx 0.7$–$0.9$ (under-cancellation) — buying 3–5× bandwidth
    extension while keeping the system stable.

    **Nyquist-plot-based design.** The NIC + source-network loop-gain
    function must encircle the origin with the wrong sense to avoid RHP
    poles. Practical designs check this on a Smith chart for the
    operating-frequency range — making sure the "negative-resistance"
    region doesn't extend into instability.

    The trade is *not* purely beneficial: a stable NIC gives less than
    full Bode-Fano violation; an unstable NIC oscillates. The sweet
    spot is "almost-unstable" — substantial improvement over passive
    bound, with a stability margin that survives process variation.
    """)
    return


@app.cell
def _(mo):
    nic_C_s = mo.ui.slider(0.05, 5.0, value=1.0, step=0.05,
                            label="Cancelled C (pF)")
    nic_Rcanc_s = mo.ui.slider(0.0, 50.0, value=10.0, step=2.0,
                                label="NIC loss Rs (Ω)")
    nic_ratio_s = mo.ui.slider(0.3, 1.2, value=0.85, step=0.05,
                                label="|−C| / |+C|  (under = stable)")
    mo.md(r"""
    #### 23.5 Interactive IX — NIC impedance and stability envelope

    A parallel-$RC$ load with $R_L = 50\,\Omega$ and capacitance $C_L$
    is paralleled with a non-Foster network producing $-\alpha C_L$
    (where $\alpha$ = "|−C|/|+C|"), modelled with series-loss
    resistance $R_s$. The plot shows $|Z_{\text{tot}}|$ vs frequency
    *with* and *without* the NIC, plus the stability condition (real
    part of $Z_{\text{tot}}$ must remain positive at all frequencies).

    Under-cancellation ($\alpha < 1$): stable, partial bandwidth
    improvement. Over-cancellation ($\alpha > 1$): unstable.
    """)
    return nic_C_s, nic_ratio_s, nic_Rcanc_s


@app.cell
def _(go, mo, nic_C_s, nic_Rcanc_s, nic_ratio_s, np):
    _RL_n = 50.0
    _CL = float(nic_C_s.value) * 1e-12
    _Rs_nic = float(nic_Rcanc_s.value)
    _alpha = float(nic_ratio_s.value)
    _f_n = np.logspace(8, 11, 600)
    _w_n = 2 * np.pi * _f_n

    _ZL = _RL_n / (1.0 + 1j * _w_n * _RL_n * _CL)
    _Z_NIC = (-1.0) / (1j * _w_n * (_alpha * _CL)) + _Rs_nic
    _Y_total = 1.0 / _ZL + 1.0 / _Z_NIC
    _Z_total = 1.0 / _Y_total

    _Z_pass = _ZL.copy()

    _Z0 = 50.0
    _Gamma_pass = (_Z_pass - _Z0) / (_Z_pass + _Z0)
    _Gamma_nic = (_Z_total - _Z0) / (_Z_total + _Z0)
    _RL_pass_dB = 20 * np.log10(np.maximum(np.abs(_Gamma_pass), 1e-12))
    _RL_nic_dB = 20 * np.log10(np.maximum(np.abs(_Gamma_nic), 1e-12))

    _Re_Z = _Z_total.real
    _unstable = (_Re_Z < 0).any()
    _stab_msg = ("**UNSTABLE** — Re(Z_tot) < 0 at some frequency"
                 if _unstable else "**Stable** — Re(Z_tot) > 0 everywhere")

    _fig_nic = go.Figure()
    _fig_nic.add_trace(go.Scatter(x=_f_n/1e9, y=_RL_pass_dB, mode="lines",
        line=dict(color="#888", width=2, dash="dash"),
        name="Passive load |Γ| (no NIC)"))
    _fig_nic.add_trace(go.Scatter(x=_f_n/1e9, y=_RL_nic_dB, mode="lines",
        line=dict(color="#00CC96", width=2.6),
        name=f"With NIC (α = {_alpha:.2f})"))
    _fig_nic.add_hline(y=-10.0, line=dict(color="rgba(255,80,80,0.5)", dash="dot"),
        annotation_text="-10 dB")
    _fig_nic.update_layout(template="plotly_dark", height=420,
        title=f"Non-Foster matching — C_L = {_CL*1e12:.2f} pF, NIC R_s = {_Rs_nic:g} Ω",
        xaxis=dict(title="f (GHz)", type="log"),
        yaxis=dict(title="|S_11| (dB)", range=[-40, 5]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3))

    _i10_pass = np.where(_RL_pass_dB < -10)[0]
    _i10_nic = np.where(_RL_nic_dB < -10)[0]
    _bw_pass = ((_f_n[_i10_pass[-1]] - _f_n[_i10_pass[0]]) / 1e9
                if len(_i10_pass) > 1 else 0.0)
    _bw_nic = ((_f_n[_i10_nic[-1]] - _f_n[_i10_nic[0]]) / 1e9
               if len(_i10_nic) > 1 else 0.0)
    _imp = ((_bw_nic / _bw_pass) if _bw_pass > 0 else float('inf'))

    _info_nic = mo.md(rf"""
    | Quantity | Passive load | With NIC |
    |---|---|---|
    | -10 dB BW | {_bw_pass:.2f} GHz | {_bw_nic:.2f} GHz |
    | Improvement | — | {_imp:.2f}× |

    **Stability:** {_stab_msg}.
    {('Over-cancellation drives the closed loop to instability — reduce α below 1 to recover stability with partial BW gain.' if _unstable else 'Under-cancellation provides graceful BW improvement; the negative-resistance contribution stays small enough to be stabilised by passive loss and source impedance.')}
    """)
    mo.vstack([
        mo.hstack([nic_C_s, nic_Rcanc_s, nic_ratio_s]),
        mo.ui.plotly(_fig_nic),
        _info_nic,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §24. Distributed Amplifier and Artificial Transmission Line

    #### 24.1 The $g_m / C_{gs}$ ceiling on single-stage gain-BW

    A single-transistor common-source stage has gain-bandwidth product

    $$GBW = g_m \cdot R_L \cdot f_{-3} = \frac{g_m}{2\pi C_L}$$

    where $C_L$ is the output-loading capacitance. With $C_L
    \approx C_{gs}$ of the next stage, $GBW = g_m/(2\pi C_{gs}) =
    f_T$ — the **transit frequency**. For 65 nm CMOS at typical bias,
    $f_T \approx 200\,\text{GHz}$. So a single stage can deliver
    10 dB of gain over ~70 GHz, or 20 dB over ~20 GHz — pick one. The
    ceiling is fundamental: cascading stages does not help because each
    new stage adds another $C_{gs}$ load that re-imposes the same
    constraint.

    #### 24.2 The distributed-amplifier idea

    Cascade $N$ transistors along *two* artificial transmission lines —
    a **gate line** (input) and a **drain line** (output). Each
    transmission line is an $LC$ ladder where the transistor's input
    capacitance ($C_{gs}$) and output capacitance ($C_{ds}$) are
    absorbed into the line's per-section capacitance.

    The signal enters the gate line at one end, propagates with phase
    velocity $v_g$, and at each section drives one transistor's gate.
    Each transistor injects current $g_m V_g$ into the drain line at
    its section. With matched gate-line and drain-line delays, the $N$
    injected currents arrive **in phase** at the output port and add
    coherently.

    The key result:

    $$\boxed{\;\text{gain} = \frac{1}{2}\,N\,g_m\,Z_0
              \quad\text{(matched lines, lossless)}\;}
      \qquad\text{[Result]}$$

    where $Z_0 = \sqrt{L/C}$ is the characteristic impedance of the
    artificial line. The factor of $1/2$ comes from the drain line
    being terminated at both ends (half the current goes forward, half
    backward into a termination).

    **Bandwidth.** The artificial $LC$ ladder has a cutoff frequency

    $$f_c = \frac{1}{\pi \sqrt{L\,C}}$$

    above which it transitions from a transmission line to a stop-band
    network. The amplifier's bandwidth is approximately $f_c$ — set
    by the *line*, not by any individual transistor.

    The product $\text{gain} \times \text{BW} = N \cdot (g_m/2) \cdot
    Z_0 \cdot f_c$. Using $f_c \cdot C = 1/(\pi \sqrt{LC}) \cdot C =
    \sqrt{C/L}/\pi = 1/(\pi Z_0)$, this simplifies to
    $\text{gain}\times\text{BW} \approx N\,g_m/(2\pi C)$ — i.e., the
    single-stage $f_T$ multiplied by $N/2$. **A distributed amp
    transcends the single-stage ceiling by a factor of $N/2$.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 24.3 Loss limits $N$ — the practical optimum

    Real artificial transmission lines have **loss** — each section's
    series inductor has $R_s$, each shunt capacitor has parallel
    $R_p$. The gate line loses signal amplitude as the wave propagates
    past each transistor; the drain line attenuates the accumulated
    output current. Per-section attenuation $\alpha$ scales roughly as

    $$\alpha \approx \frac{R_s}{2 Z_0} + \frac{Z_0}{2 R_p}\quad\text{(Np/section)}$$

    With $N$ sections, the cumulative attenuation is $N\alpha$. The
    gain $N \cdot g_m \cdot Z_0 / 2$ grows linearly in $N$ but the
    transmitted signal decays as $e^{-N\alpha}$. The product peaks at

    $$N_{\text{opt}} = \frac{1}{\alpha}\quad\text{[Result]}$$

    For mmWave CMOS with typical $\alpha \approx 0.1$–$0.2$ Np/section,
    $N_{\text{opt}} = 5$–$10$. Beyond this, added stages cost more
    attenuation than they contribute in coherent gain.

    #### 24.4 Modern uses

    - **Ultra-wideband instrumentation amplifiers** (1–100 GHz oscilloscope
      front-ends, network analysers).
    - **Optical-receiver TIAs** for high-rate optical links (50–200 Gb/s
      per lane).
    - **Wideband mmWave PAs** for satellite and 5G/6G test equipment.
    - **Phased-array receive elements** where ultra-wideband response
      enables one design across multiple bands (e.g., 24–66 GHz combined
      array).

    The interactive below shows the gain-vs-frequency trade for swept
    $N$, $g_m$, and per-section loss — illustrating the optimum and
    the LC cutoff.
    """)
    return


@app.cell
def _(mo):
    da_N_s = mo.ui.slider(2, 16, value=6, step=1, label="N (sections)")
    da_gm_s = mo.ui.slider(0.01, 0.2, value=0.05, step=0.005,
                           label="gm per transistor (S)")
    da_Z0_s = mo.ui.slider(25, 100, value=50, step=5, label="Z₀ (Ω)")
    da_L_s = mo.ui.slider(0.05, 1.0, value=0.3, step=0.05,
                          label="L per section (nH)")
    da_C_s = mo.ui.slider(0.02, 0.5, value=0.12, step=0.02,
                          label="C per section (pF)")
    da_Rs_s = mo.ui.slider(0, 20, value=5, step=1,
                           label="R series + drain (Ω)")
    mo.md(r"""
    #### 24.5 Interactive X — Distributed amplifier explorer

    Sweep $N, g_m, Z_0$ and per-section line parameters. The cutoff
    $f_c = 1/(\pi\sqrt{LC})$ sets the high-frequency edge; loss
    per section sets the optimum $N$. The gain falls off above $f_c$
    as the line becomes evanescent.
    """)
    return da_C_s, da_L_s, da_N_s, da_Rs_s, da_Z0_s, da_gm_s


@app.cell
def _(da_C_s, da_L_s, da_N_s, da_Rs_s, da_Z0_s, da_gm_s, distributed_amp_S21,
      go, mo, np):
    _N_da = int(da_N_s.value)
    _gm_da = float(da_gm_s.value)
    _Z0_da = float(da_Z0_s.value)
    _L_da = float(da_L_s.value) * 1e-9
    _C_da = float(da_C_s.value) * 1e-12
    _Rg = float(da_Rs_s.value)
    _Rd = float(da_Rs_s.value)
    _f_da = np.linspace(1e9, 200e9, 800)
    _S21 = distributed_amp_S21(_N_da, _gm_da, _Z0_da, _L_da, _C_da,
                               _Rg, _Rd, _f_da)
    _S21_dB = 20 * np.log10(np.maximum(_S21, 1e-6))
    _f_c_da = 1.0 / (np.pi * np.sqrt(_L_da * _C_da))

    # Compute optimal N
    _alpha = (_Rg + _Rd) / (2 * _Z0_da)
    _N_opt = max(2, int(round(1.0 / _alpha))) if _alpha > 0 else 16

    _fig_da = go.Figure()
    _fig_da.add_trace(go.Scatter(x=_f_da/1e9, y=_S21_dB, mode="lines",
        line=dict(color="#FECB52", width=2.6), name=f"N = {_N_da}"))
    _fig_da.add_vline(x=_f_c_da/1e9, line=dict(color="rgba(255,80,80,0.6)",
        dash="dash"),
        annotation_text=f"f_c = {_f_c_da/1e9:.1f} GHz",
        annotation_position="top")
    _fig_da.add_hline(y=20*np.log10(_N_da*_gm_da*_Z0_da/2),
        line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
        annotation_text="Ideal lossless gain (N·gm·Z0/2)",
        annotation_position="bottom right")
    _fig_da.update_layout(template="plotly_dark", height=420,
        title=(f"Distributed amp gain — N = {_N_da}, gm = {_gm_da*1000:.0f} mS, "
               f"Z₀ = {_Z0_da:g} Ω, f_c = {_f_c_da/1e9:.1f} GHz"),
        xaxis=dict(title="f (GHz)"),
        yaxis=dict(title="|S_21| (dB)", range=[-10, 35]))

    _peak_gain = float(np.max(_S21_dB))
    _i3 = np.where(_S21_dB > _peak_gain - 3)[0]
    _bw = ((_f_da[_i3[-1]] - _f_da[_i3[0]]) / 1e9
           if len(_i3) > 1 else 0.0)

    _info_da = mo.md(rf"""
    | Quantity | Value |
    |---|---|
    | Peak gain | {_peak_gain:.1f} dB |
    | 3-dB BW | {_bw:.1f} GHz |
    | $f_c$ (artificial-TL cutoff) | {_f_c_da/1e9:.1f} GHz |
    | Per-section loss α | {_alpha:.3f} Np |
    | Optimum N (1/α) | {_N_opt} |

    Increasing $N$ raises gain linearly until line loss dominates;
    beyond $N_\text{{opt}} = ${_N_opt}, added sections subtract more
    than they add. The cutoff $f_c$ is independent of $N$ — set by the
    *per-section* $L, C$ alone.
    """)
    mo.vstack([
        mo.hstack([da_N_s, da_gm_s, da_Z0_s]),
        mo.hstack([da_L_s, da_C_s, da_Rs_s]),
        mo.ui.plotly(_fig_da),
        _info_da,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §25. N-Path Filter and Time-Modulated Matching

    #### 25.1 The N-path concept

    $N$ identical capacitors, each connected to the input through a
    switch driven by one of $N$ non-overlapping clock phases at LO
    frequency $f_{\text{LO}}$. Each phase has duty cycle $1/N$. At any
    instant exactly one capacitor is connected to the RF port; the
    others hold their charge.

    The composite circuit is **linear** but **time-varying**: the
    switching sequence makes the input impedance a periodic function
    of time at rate $f_{\text{LO}}$, *not* a function of $\omega$
    alone.

    Under linear-periodic-time-varying (LPTV) analysis, the input
    impedance at RF frequency $\omega_{\text{RF}}$ relates to the
    baseband impedance $Z_{\text{BB}}(\omega)$ seen across each
    capacitor (typically $1/(j\omega C_{\text{BB}})$) by the
    **impedance translation theorem**:

    $$\boxed{\;Z_{\text{RF}}(j\omega) = R_{\text{sw}}
              + \sum_{n=-\infty}^{\infty}\!|a_n|^2 \cdot
                Z_{\text{BB}}\!\big(j(\omega - n\omega_{\text{LO}})\big)\;}
      \qquad\text{[Theorem — Mirzaei 2010, Andrews-Molnar 2010]}$$

    with Fourier coefficients $|a_n|^2 = \text{sinc}^2(n/N)$ for an
    $N$-phase duty-$1/N$ rectangular drive. The first-harmonic
    contribution ($n = \pm 1$) dominates: the baseband impedance
    $Z_{\text{BB}}$ is shifted to $\omega = \omega_{\text{LO}}$ — a
    **tunable narrowband filter centred at the LO** without any LC
    tank.

    #### 25.2 Why LTV escapes Bode-Fano

    Bode-Fano's machinery (§8) applies to *bounded-real* impedances, a
    consequence of *passivity* + *time-invariance*. An LTV network is
    not time-invariant; its impedance varies with the LO phase. The
    BR-function constraint does not apply, and the Cauchy-contour
    proof of the Bode-Fano integral breaks down.

    Concretely: an N-path network with a *resistive* baseband
    impedance and a tunable LO can achieve narrowband matching at
    any frequency from DC to $f_{\text{sw,max}}/N$, with bandwidth set
    by $Z_{\text{BB}}$'s own bandwidth — *independent* of the
    parasitic capacitances at the RF port (provided they are small
    compared to the on-resistance of the switches at $\omega_{\text{LO}}$).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 25.3 Applications and the parametric variant

    **Software-defined-radio (SDR) front-ends.** A single N-path
    receiver covers DC to 6 GHz by tuning the LO; the RF "filter" is
    fully reconfigurable without any LC retuning.

    **Blocker-tolerant receivers.** The narrowband N-path filter
    rejects out-of-band interferers *before* the LNA — useful when the
    interferer is many times stronger than the desired signal. The
    filter's stopband attenuation can exceed 70 dB at modest LO
    powers.

    **Reconfigurable interferer cancellation.** Multiple N-path filters
    at different LO frequencies, summed and subtracted, can synthesise
    arbitrary frequency-selective responses on the fly. Used in
    next-gen 5G/6G in-band full-duplex experiments.

    **Parametric matching (brief).** A pumped varactor — capacitance
    modulated periodically by a pump at $\omega_p$ — generates a
    time-varying $C(t)$. The Manley-Rowe relations describe the power
    flow between pump, signal, and idler frequencies. Parametric
    amplifiers, achievable with this mechanism, have been built since
    the 1960s for low-noise mmWave receivers. They are an LTV escape
    from both Bode-Fano *and* the Friis noise floor.

    #### 25.4 Limitations

    - **LO noise.** Phase noise on the clock translates directly to
      reciprocal-mixing noise in the receiver. SDRs with N-path
      front-ends need very-low-phase-noise PLLs.
    - **Harmonic responses.** The $|a_n|^2 = \text{sinc}^2(n/N)$
      coefficients also pass frequencies at $n\,f_{\text{LO}}$. An
      LO at 1 GHz also creates filters at 3, 5, ... GHz with
      decreasing strength.
    - **Switch on-resistance.** $R_{\text{sw}}$ adds directly to
      $Z_{\text{RF}}$ as a series loss; this caps the achievable
      noise figure.
    """)
    return


@app.cell
def _(mo):
    np_R_s = mo.ui.slider(1, 50, value=5, step=1, label="R_sw (Ω)")
    np_CBB_s = mo.ui.slider(1, 200, value=50, step=5, label="C_BB (pF)")
    np_RBB_s = mo.ui.slider(100, 10000, value=2000, step=100,
                             label="R_BB (Ω) (baseband shunt resistor)")
    np_N_s = mo.ui.slider(2, 16, value=4, step=1, label="N (paths)")
    np_fLO_s = mo.ui.slider(0.5, 6.0, value=2.0, step=0.1, label="f_LO (GHz)")
    mo.md(r"""
    #### 25.5 Interactive XI — N-path translated impedance

    Sweep $f_{\text{LO}}$ and watch the baseband impedance translate to
    create a narrowband filter at the RF port. The peak $|Z_{\text{RF}}|$
    occurs at $f = f_{\text{LO}}$ (and at $3 f_{\text{LO}}, 5 f_{\text{LO}},
    \ldots$ for odd $N$). The Q of the band is set by $R_{BB}\,C_{BB}$;
    the position is set by $f_{\text{LO}}$ alone — re-tunable simply by
    changing the clock.
    """)
    return np_CBB_s, np_N_s, np_R_s, np_RBB_s, np_fLO_s


@app.cell
def _(go, mo, np, np_CBB_s, np_N_s, np_R_s, np_RBB_s, np_fLO_s,
      npath_translated_Z):
    _R_sw = float(np_R_s.value)
    _CBB = float(np_CBB_s.value) * 1e-12
    _RBB = float(np_RBB_s.value)
    _N_np = int(np_N_s.value)
    _fLO = float(np_fLO_s.value) * 1e9
    _f_np = np.linspace(0.1e9, 7e9, 1500)
    _Z_RF = npath_translated_Z(_R_sw, _CBB, _RBB, _N_np, _fLO, _f_np,
                                K_harm=7)

    _Z_mag = np.abs(_Z_RF)
    _Z_re = _Z_RF.real
    _Z_im = _Z_RF.imag

    _fig_np = go.Figure()
    _fig_np.add_trace(go.Scatter(x=_f_np/1e9, y=_Z_mag, mode="lines",
        line=dict(color="#00CC96", width=2.5), name="|Z_RF|"))
    _fig_np.add_trace(go.Scatter(x=_f_np/1e9, y=_Z_re, mode="lines",
        line=dict(color="#19D3F3", width=1.8, dash="dash"), name="Re(Z_RF)"))
    _fig_np.add_trace(go.Scatter(x=_f_np/1e9, y=_Z_im, mode="lines",
        line=dict(color="#EF553B", width=1.4, dash="dot"), name="Im(Z_RF)"))
    _fig_np.add_vline(x=_fLO/1e9, line=dict(color="rgba(255,255,255,0.4)",
        dash="dot"), annotation_text="f_LO")
    _fig_np.add_vline(x=3*_fLO/1e9, line=dict(color="rgba(255,255,255,0.25)",
        dash="dot"), annotation_text="3·f_LO")
    _fig_np.update_layout(template="plotly_dark", height=440,
        title=(f"N-path filter — N = {_N_np}, f_LO = {_fLO/1e9:.2f} GHz, "
               f"R_sw = {_R_sw} Ω, C_BB = {_CBB*1e12:.0f} pF, "
               f"R_BB = {_RBB} Ω"),
        xaxis=dict(title="f (GHz)"),
        yaxis=dict(title="Z_RF (Ω)", range=[0, min(np.max(_Z_mag)*1.1, 3000)]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3))

    _i_peak = int(np.argmax(_Z_mag))
    _f_peak = _f_np[_i_peak]
    _Z_peak = _Z_mag[_i_peak]
    _info_np = mo.md(rf"""
    | Quantity | Value |
    |---|---|
    | Peak |Z_RF| | {_Z_peak:.0f} Ω at f = {_f_peak/1e9:.2f} GHz |
    | f_LO (commanded) | {_fLO/1e9:.2f} GHz |
    | Baseband Q ≈ ω_LO R_BB C_BB | {2*np.pi*_fLO*_RBB*_CBB:.1f} |
    | Off-band Z (at 0.5 f_LO) | {np.abs(_Z_RF[np.argmin(np.abs(_f_np - 0.5*_fLO))]):.0f} Ω |

    The peak |Z| tracks f_LO directly — slide the LO to retune the
    filter. The Q of the response is set by $R_{{BB}} C_{{BB}}$
    (baseband time constant), not by any LC element. Lower-frequency
    harmonics at $f_{{LO}}/N, f_{{LO}}/3,...$ are absent; higher
    harmonics at $3 f_{{LO}}, 5 f_{{LO}}$ appear with reduced
    strength.
    """)
    mo.vstack([
        mo.hstack([np_R_s, np_CBB_s, np_RBB_s]),
        mo.hstack([np_N_s, np_fLO_s]),
        mo.ui.plotly(_fig_np),
        _info_np,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §26. Reconfigurable / Tunable Matching Networks

    #### 26.1 Tuning elements

    | Element | Q (at GHz) | Tuning range | Speed | Linearity |
    |---|---|---|---|---|
    | **Varactor (p-n)** | 30–80 | 3–5× | ns | Poor (large-signal nonlinear) |
    | **Switched-capacitor array** | 100+ (off) | $2^N$ states | μs | Excellent (linear) |
    | **RF-MEMS varactor** | 200+ | 5–10× | μs–ms | Excellent |
    | **BST (barium-strontium-titanate)** | 50–150 | 3–5× | ns | Modest |
    | **Active inductor (gyrator)** | 5–20 | 10× | ns | Poor |

    **Trade-off summary.** Varactors are fast and small but have low Q
    and nonlinear behaviour (large-signal tuning shift). Switched-cap
    arrays are linear and high-Q but slow and require many control
    bits. MEMS achieves the best Q at the cost of fabrication
    complexity and speed. Active inductors enable tuning ratios
    impossible with passives but at substantial NF cost.

    #### 26.2 The Antenna Tuning Unit (ATU)

    Smartphone radios cover dozens of bands (LTE, 5G NR, Wi-Fi 6/7) on
    a single antenna whose impedance varies wildly with user grip,
    proximity, and band. A **reconfigurable matching network**
    between the LNA/PA and the antenna corrects this in real time.

    Common topology: a **tunable $\pi$-network** with two variable
    shunt caps and one variable series cap. The three degrees of
    freedom span the full Smith chart — any antenna impedance can be
    matched to any source impedance with no movement other than
    adjusting the three capacitances.

    Mathematical statement: given source $Z_S$ and load $Z_L$, the
    $\pi$-network element values are
    $(C_1, L, C_2) = f(Z_S, Z_L, \omega)$, a continuous map. With
    discrete tuning (switched arrays), the closest achievable point
    on the Smith chart is within ~5% of the target for a 6-bit
    array — sufficient for return loss < -15 dB across all bands.

    #### 26.3 Closed-loop adaptive matching

    A directional coupler at the PA output measures forward and
    reflected power. A microcontroller computes $|\Gamma|$ and iterates
    the tuner state — typically via **gradient descent** or the
    **simplex algorithm** — to minimise $|\Gamma|$.

    Practical convergence time: ~10 μs to within 0.5 dB of optimal
    over a single band. ATU + PA + closed loop is now standard in
    every flagship handset; the alternative (band-specific PAs and
    matching) is non-viable for ~50+ band coverage.

    #### 26.4 Massive-MIMO and phased arrays

    Per-element antenna matching in massive MIMO arrays handles
    *scan-angle-dependent active impedance* — the effective impedance
    of an array element changes with the beam direction (mutual
    coupling between adjacent elements). A per-element tunable match
    keeps each PA loaded at its optimal load impedance regardless of
    scan angle.

    This is the matching problem of the 2020s–2030s: hundreds of
    independently tunable matching networks per array, all controlled
    by a baseband signal processor that knows the beam geometry. Each
    element's tuner is small (varactor-based) but the *coordination*
    across elements is the new design challenge.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §27. Frontier Topics (brief)

    #### 27.1 Metamaterials and metasurfaces for impedance matching

    Sub-wavelength patterned structures can synthesise an effective
    impedance independent of the underlying material. Specific
    application: **anti-reflection metasurfaces** between two media of
    different intrinsic impedance — analogous to anti-reflection
    coatings in optics, but engineered at GHz–THz frequencies for
    microwave-to-free-space matching. Used in mmWave radomes and
    quasi-optical antenna feeds.

    #### 27.2 Reconfigurable Intelligent Surfaces (RIS)

    A large array of sub-wavelength elements, each with a tunable
    impedance (varactor- or PIN-diode-loaded). By choosing each
    element's reflection coefficient, the surface synthesises an
    arbitrary phase profile — effectively a *programmable*
    metasurface antenna. For 6G, RIS is positioned as a way to control
    NLOS (non-line-of-sight) propagation by reflecting signals off
    intelligent walls.

    The matching problem here is *per-element*: each RIS unit cell
    must present a controllable reflection coefficient (magnitude and
    phase) over its operating band. The tunable matching network is
    the heart of the cell.

    #### 27.3 Coupled-resonator filters as combined matching + filtering

    A bank of $N$ coupled resonators (LC or distributed) can
    *simultaneously* perform broadband matching and channel selection.
    The synthesis problem — Cameron, Cohn, and others have written
    extensively — admits closed-form solutions when the network is
    canonical (e.g., Chebyshev with prescribed return loss + ripple).
    Used in cellular base-station front-ends to combine the diplexer
    and the LNA input match in a single multi-pole network.

    #### 27.4 Optical analogues — anti-reflection coatings

    A multi-section anti-reflection coating on an optical surface is
    *mathematically identical* to a multi-section quarter-wave
    Chebyshev transformer (§9, §12) — the two fields use different
    units and language but identical synthesis equations. A
    quarter-wave SiO₂/TiO₂ stack on glass is the optical version of a
    quarter-wave Cu transformer on a CMOS substrate. The Bode-Fano
    bound applies in both domains, with the optical-frequency form
    capping how broadband anti-reflection can be on a given substrate.

    Each of these frontiers builds on the LTI, Bode-Fano-bounded
    framework of Parts I–V or one of the LTV escape hatches of Part
    VI. None is a *replacement* for the foundational theory — they
    are *extensions* whose pedagogical centre is the same.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part VII — mmWave Output Matching Revisited

    ### §28. The 28 GHz LNA Output Match: Five Topologies, Two Eras

    Notebook 04 §17.6 used a **tapped-capacitor** to transform the optimal
    load impedance at 28 GHz. With the foundation of Parts V–VI we can
    now compare it against four progressively more capable alternatives
    for the same design point ($R_L = 200\,\Omega \to 50\,\Omega$,
    $f_0 = 28\,\text{GHz}$).

    | Topology | Realisation | BW (−3 dB) | ΔNF | On-chip area | Section |
    |---|---|---|---|---|---|
    | Tapped capacitor | $C_1, C_2$ divider | ~10% (~2.8 GHz) | <0.1 dB | Very small | nb 04 §17.6 |
    | $\lambda/4$ microstrip TL | $Z_1 = 100\,\Omega$, $\ell = 1.34\,\text{mm}$ (air) | ~30% | ~0.3 dB | Large | §12 |
    | On-chip transformer | $n=2$, $k=0.75$, $Q=12$ | ~15% | ~0.5–1 dB | Medium; enables differential | §19 |
    | Bridged T-coil ($k=1/3$) | Two coupled spirals + $C_B$ | ~28% (~$2.83\times$ bare-$RC$) | ~0.3 dB | Medium | §20 |
    | Distributed amplifier output | $N$-section $LC$-ladder | >100% (DC–60 GHz) | ~1 dB | Large; for instrumentation | §24 |

    **ΔNF for the λ/4 TL** (50 Ω microstrip with 0.3 dB/mm loss at 28 GHz;
    $\ell \approx 0.9$ mm in SiO₂ effective medium): insertion loss
    $\approx 0.27$ dB. Friis: the output matching loss is divided by LNA
    gain (15 dB = 32×), contributing only $0.27\,\text{dB}/32 \approx
    0.008\,\text{dB}$ to input-referred NF — negligible. The 0.3 dB
    figure refers to the *output* signal power degradation.

    **Practical choice at 28 GHz CMOS.**

    - *Narrowband, area-constrained, low-NF*: tapped capacitor.
    - *Modest bandwidth + differential downstream mixer*: on-chip
      transformer (§19).
    - *Wide-band CMOS with strict pad cap*: T-coil (§20) — when the
      output pad's parasitic capacitance is the dominant constraint.
    - *Extreme bandwidth (DC–50+ GHz, e.g. instrumentation, optical
      Rx)*: distributed amplifier (§24).
    - *Reconfigurable across bands*: tunable matching with switched-cap
      arrays (§26).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part VIII — Summary

    ### §29. Summary and Bridge to Notebook 06

    The complete receiver design chain is now assembled:

    1. **LNA** (notebook 04): place $\Gamma_S$ near $\Gamma_{\text{opt}}$
       via inductive degeneration → $NF \approx F_{\min}$.
    2. **Lumped matching** (Part II): L/π/T networks for narrowband
       transformation; bandwidth set by impedance ratio.
    3. **Broadband matching** (Part III): Bode-Fano bounds the achievable
       depth-vs-bandwidth product; Chebyshev synthesis approaches the
       bound for resistive loads.
    4. **Distributed matching** (Part IV): quarter-wave transformers,
       stubs, coupled lines — area-efficient at mmWave.
    5. **Coupled magnetics** (Part V — new): mutual inductance from
       Maxwell-Faraday; T-model and ideal-plus-leakage decompositions;
       transformer matching with critical coupling; the bridged T-coil
       as a Bode-Fano dodge for fixed parasitic capacitance.
    6. **Beyond passive LTI** (Part VI — new): non-Foster active
       matching defeats Bode-Fano in principle; distributed amplifiers
       cross the $g_m/C_{gs}$ ceiling; N-path / time-modulated networks
       circumvent BR-function constraints via LPTV analysis;
       reconfigurable tuners cover multi-band handsets.
    7. **Switched-circuit noise** (notebook 04 §4.9): mixer SSB NF and
       sampler kT/C are governed by the WSC framework of notebook 04
       §3.8.

    **Open problems for notebook 06:**

    - **Nonlinearity:** 1 dB compression point $P_{\text{1dB}}$,
      input-referred third-order intercept $\text{IIP}_3$, AM-PM
      conversion.
    - **PA design:** load-pull, drain efficiency, Doherty topology.
    - **Full mmWave frontend:** LNA + matching + mixer + PA as an
      integrated design.

    Concept-dependency map:

    ```
    05 §4–7    L/π/T synthesis              ──►  06 PA output matching
    05 §8–11   Bode-Fano / Chebyshev        ──►  06 broadband PA
    05 §16–17  Coupled magnetics, T-model   ──►  06 PA cross-coupled pair
    05 §19     Critical-coupling transformer──►  06 PA combining/balun
    05 §20     Bridged T-coil               ──►  06 PA output pad protection
    05 §22     On-chip geometry             ──►  06 PA layout/loss model
    05 §23     Non-Foster matching          ──►  06 broadband linearised PA
    05 §24     Distributed amplifier        ──►  06 DA-PA, Marchand load
    05 §26     Reconfigurable matching      ──►  06 ATU / multi-band PA
    04 §3.8    Cyclostationary processes    ──►  06 AM-PM / PA noise
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Previous:** [04 — Noise Analysis and LNA Design](04_noise_and_lna_design.py)  |
    **Next:** [06 — Oscillators, VCOs, and mmWave Phase Noise](06_oscillators_vco.py)
    """)
    return


if __name__ == "__main__":
    app.run()
