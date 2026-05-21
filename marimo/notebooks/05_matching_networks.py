# v1.0
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

    # v1.0
    """Pure-Python helpers for cyclostationary noise analysis.

    Used by notebook 05 (Matching Networks and Cyclostationary Noise).
    No marimo imports. Covers mixer conversion-matrix NF and kT/C sampler noise.
    """




    # ---------------------------------------------------------------------------
    # Mixer noise figure (conversion-matrix approximation)
    # ---------------------------------------------------------------------------

    def mixer_NF_DSB(L_c_dB: float, T0: float = 290.0) -> float:
        """DSB noise figure of an ideal resistive mixer (dB).

        For a purely resistive mixing element (switches between R_on and R_off),
        the DSB NF equals the conversion loss in dB.
        L_c_dB: conversion loss (dB), positive number.
        """
        assert L_c_dB >= 0
        return float(L_c_dB)


    def mixer_NF_SSB(L_c_dB: float, IRR_dB: float = 0.0, T0: float = 290.0) -> float:
        """SSB noise figure of a mixer (dB), accounting for image noise.

        IRR_dB: image rejection ratio (dB). 0 dB = no image rejection (single balanced).
        For ideal double-balanced mixer with infinite IRR: NF_SSB = NF_DSB + 3 dB.
        """
        L_c = 10 ** (L_c_dB / 10)
        IRR = 10 ** (IRR_dB / 10)
        F_DSB = L_c  # linear
        # Image noise contribution: power folds in with factor 1/IRR
        F_SSB = F_DSB + (1.0 / IRR) * (T0 / T0)  # simplified: image adds kT_0/IRR
        # For zero IRR (no rejection): F_SSB = 2*F_DSB → NF_SSB = NF_DSB + 3 dB
        return float(10 * np.log10(F_SSB))


    def friis_cascade(stages: list[dict]) -> dict:
        """Friis noise cascade for a list of stages.

        Each stage dict: {"F_dB": float, "G_dB": float}.
        Returns: {"F_total_dB": float, "contributions": list[float]}.
        The contribution list shows (F_i - 1)/prod(G_j, j<i) / (F_total - 1) as a fraction.
        """
        Fs = [10 ** (s["F_dB"] / 10) for s in stages]
        Gs = [10 ** (s["G_dB"] / 10) for s in stages]
        F_total = Fs[0]
        G_prod = Gs[0]
        for k in range(1, len(stages)):
            F_total += (Fs[k] - 1.0) / G_prod
            G_prod *= Gs[k]
        contributions = []
        G_prod = 1.0
        for k, (F, G) in enumerate(zip(Fs, Gs)):
            contrib = (F - 1.0) / G_prod / (F_total - 1.0) if F_total > 1.0 else 0.0
            contributions.append(float(contrib))
            G_prod *= G
        return {"F_total_dB": float(10 * np.log10(F_total)),
                "F_total_lin": float(F_total),
                "contributions": contributions}


    # ---------------------------------------------------------------------------
    # kT/C sampler noise
    # ---------------------------------------------------------------------------

    def ktc_noise_voltage(C_F: float, T: float = 290.0) -> float:
        """RMS noise voltage on a hold capacitor (V).

        V_n = sqrt(kT / C). Independent of switch resistance.
        """
        k = 1.380649e-23
        return float(np.sqrt(k * T / C_F))


    def noise_folding_penalty_dB(N_aliases: int) -> float:
        """Noise figure penalty (dB) from N-fold aliasing in an under-sampled receiver.

        Assumes the noise is white over the full pre-sample bandwidth.
        """
        assert N_aliases >= 1
        return float(10 * np.log10(N_aliases))


    def sampler_settling_requirement(f_s: float, N_bits: int) -> float:
        """Maximum RC settling time constant (s) for N-bit accuracy at sample rate f_s.

        Condition: exp(-T_s / (2*tau)) < 2^(-N) → tau < T_s / (2 * N * ln2).
        """
        T_s = 1.0 / f_s
        return float(T_s / (2.0 * N_bits * np.log(2.0)))


    # ---------------------------------------------------------------------------
    # Cyclostationary PSD helper
    # ---------------------------------------------------------------------------

    def cyclo_time_avg_psd(S_n: list[float], f_LO: float,
                           freqs: np.ndarray) -> np.ndarray:
        """Time-averaged PSD of a cyclostationary process.

        Given cyclic spectral components S_n (one-sided, n=0,1,...) at
        IF frequencies near f_IF = freqs, the time-averaged PSD is:

            S_avg(f) = Σ_n S_n * δ-like contribution at f ± n*f_LO

        Here we return a broadened representation: each S_n contributes
        a Gaussian of width σ=f_LO/50 centred at f_IF + n*f_LO (n=0 term only
        is at the IF; higher n appear at image-band offsets).

        This is for illustration only — the delta functions are convolved with
        a narrow Gaussian to be plottable.
        """
        psd = np.zeros_like(freqs, dtype=float)
        sigma = f_LO / 50.0
        for n, Sn in enumerate(S_n):
            f_center = n * f_LO
            psd += Sn * np.exp(-0.5 * ((freqs - f_center) / sigma) ** 2)
        return psd

    return (
        bode_fano_gamma_max,
        chebyshev_prototype,
        friis_cascade,
        l_network,
        l_network_S,
        mixer_NF_DSB,
        mixer_NF_SSB,
        noise_folding_penalty_dB,
        pi_network,
        pi_network_S,
        qw_transformer,
        single_stub_tuner,
        t_network,
        t_network_S,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 05 — Matching Networks, Broadband Design, and Cyclostationary Noise

    Builds on notebooks 01–04. Lumped and distributed impedance matching,
    Bode-Fano bandwidth limits, transformer coupling, and mixer/sampler noise.
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
    parallel-$RC$ load. The bridged T-coil (§6.2) is the special-case escape hatch when
    the load is a *pure parasitic capacitance* rather than a resistance to transform.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    ### §6.2 T-Coils for Wireline (Capacitive-Load) Matching

    **The problem.** At multi-Gb/s wireline I/O, the bond pad plus ESD protection presents
    an unavoidable shunt capacitance $C$ ($\sim 0.3$–$1\,$pF) right at the $50\,\Omega$ port.
    A bare $R\!\parallel\!C$ termination has return-loss bandwidth limited to $\sim 1/(2\pi R C)$,
    and a conventional LC match cannot help — Bode-Fano applies because the load is a *fixed
    capacitance*, not a resistance to transform.

    **The bridged T-coil.** A centre-tapped inductor (two coupled half-windings, coupling
    coefficient $k$) bridges the pad to the on-chip termination $R$. The parasitic $C$ hangs
    off the **centre tap**, and a small bridge capacitor $C_B$ spans the two ends. Solving the
    nodal equations and demanding a frequency-independent input impedance gives the
    constant-resistance design (derived, not quoted):

    $$Z_{\text{in}}(j\omega) = R \quad \forall\,\omega \quad\text{[theorem, holds for any } k]$$

    $$L_{\text{half}} = \frac{R^2 C}{2(1+k)}, \qquad M = k\,L_{\text{half}}, \qquad
      C_B = \frac{(1-k)\,C}{4(1+k)}, \qquad L_T \equiv 2L_{\text{half}}(1+k) = R^2 C$$

    The total bridged inductance $L_T = R^2 C$ is **independent of $k$** [result]; $k$ is the
    free knob that shapes the *tap-node* (received-signal) response:

    - $k = 1/3$ → **maximally flat magnitude** (Butterworth) at the tap; the $-3\,$dB
      bandwidth is $\;2\sqrt2/(RC) \approx 2.83\times\;$ the bare-$RC$ value, with $C_B = C/8$.
    - $k = 1/2$ → **maximally flat group delay** (Bessel); preferred for wireline because it
      minimises ISI / eye closure.

    **Why it wins.** The input match (return loss) is broadband *for any* $k$ — the parasitic
    $C$ is fully absorbed into a constant-$R$ all-pass network, sidestepping Bode-Fano. The
    $2.83\times$ figure applies to the through response and beats shunt-peaking ($1.7\times$)
    and series-peaking ($1.4$–$1.6\times$), which is why T-coils are ubiquitous in SerDes
    TX/RX front-ends.
    """),
        mo.mermaid(r"""
flowchart LR
    PAD(("pad / line<br/>Z_0 = R")) --- X(("&middot;"))
    X --- La["L_half coil"] --- Z(("tap"))
    Z --- Lb["L_half coil"] --- Y(("&middot;"))
    Y --- Rt["R term"] --- G1["gnd"]
    Z --- Cp["C  pad+ESD"] --- G2["gnd"]
    X -. "C_B bridge" .- Y
    La -. "k coupling" .- Lb
"""),
    ])
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
    mo.md(r"""
    ### §15. On-Chip Transformer Coupling

    A coupled-inductor pair with mutual inductance $M = k\sqrt{L_1 L_2}$ and turns ratio $n = \sqrt{L_1/L_2}$ transforms impedances.
    The impedance matrix of the two-port is:

    $$\begin{bmatrix}V_1\\V_2\end{bmatrix} = \begin{bmatrix}j\omega L_1 & j\omega M \\ j\omega M & j\omega L_2\end{bmatrix} \begin{bmatrix}I_1\\I_2\end{bmatrix}$$

    Connecting a load $Z_L$ to port 2 ($V_2 = -I_2 Z_L$) gives $I_2 = \frac{-j\omega M}{Z_L + j\omega L_2} I_1$.
    The input impedance is $Z_{\text{in}} = V_1/I_1$:

    $$Z_{\text{in}} = j\omega L_1 + \frac{(\omega M)^2}{Z_L + j\omega L_2} = j\omega L_1(1 - k^2) + \frac{k^2 L_1/L_2}{1/(j\omega L_2) + 1/Z_L}$$

    In the limit of strong coupling ($k \to 1$) and large inductance ($\omega L_2 \gg |Z_L|$):

    $$Z_{\text{in}} \approx n^2 Z_L$$

    **Q penalty.** Winding resistance $r_1 = \omega L_1/Q_1$ adds series noise.
    The noise figure penalty (Friis referred back through the network):

    $$\Delta NF \approx 10\log_{10}\!\left(1 + \frac{r_1}{R_S}\right) \quad [\text{dB}]$$

    At 28 GHz in 65 nm CMOS: $Q_{\text{spiral}} \approx 10$–$15$, $k \approx 0.7$–$0.8$.
    For $Q = 12$, $L_1 = 100\,\text{pH}$, $R_S = 50\,\Omega$:
    $r_1 = 2\pi \cdot 28\text{e}9 \cdot 100\text{e-12}/12 \approx 1.5\,\Omega$,
    $\Delta NF \approx 0.13\,\text{dB}$ (referred to input — acceptable).

    **Balun application.** A 1:2 turns-ratio transformer converts single-ended to differential, enabling a differential mixer drive from a single-ended LNA. $\Delta NF$ rises to 0.5–1 dB including imbalance and layout parasitics.
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

    mo.md("### §16. Interactive III — Distributed Matching Explorer")
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
    ## Part V — mmWave Output Matching Revisited

    ### §17. The 28 GHz LNA Output Match: Three Topologies

    Notebook 04 §17.6 used a **tapped-capacitor** to transform the optimal load impedance at 28 GHz. Here we compare it against two distributed alternatives for the same design point ($R_L = 200\,\Omega \to 50\,\Omega$, $f_0 = 28\,\text{GHz}$).

    | Topology | Realization | BW (−3 dB) | ΔNF | On-chip area |
    |---|---|---|---|---|
    | Tapped capacitor | $C_1$, $C_2$ divider | ~10% (~2.8 GHz) | <0.1 dB | Very small |
    | λ/4 microstrip TL | $Z_1 = 100\,\Omega$, $\ell = 1.34\,\text{mm}$ (air) | ~30% | ~0.3 dB | Large |
    | On-chip transformer | $n=2$, $k=0.75$, $Q=12$ | ~15% | ~0.5–1 dB | Medium; enables differential |

    **ΔNF calculation for the λ/4 TL** (50 Ω microstrip with 0.3 dB/mm loss at 28 GHz; $\ell \approx 0.9$ mm in SiO₂ effective medium): insertion loss $\approx 0.27$ dB.
    Friis: the output matching loss is divided by LNA gain (15 dB = 32×), contributing only $0.27\,\text{dB}/32 \approx 0.008\,\text{dB}$ to input-referred NF — negligible. The 0.3 dB figure refers to the degradation of the output signal power delivered to the next stage.

    **Practical choice at 28 GHz CMOS:** tapped-capacitor for narrowband; transformer coupling when a differential IF mixer is required downstream.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part VI — Cyclostationary Noise in Mixers and Samplers

    ### §18. Cyclostationary Processes

    A process $x(t)$ is **wide-sense cyclostationary** with period $T = 1/f_{\text{LO}}$ if its mean and autocorrelation function are periodic:

    $$R_x(t, \tau) \triangleq E[x(t)\,x(t+\tau)] = R_x(t+T, \tau)$$

    Fourier expanding in $t$:

    $$R_x(t, \tau) = \sum_{n=-\infty}^{\infty} R_n(\tau)\,e^{jn\omega_{\text{LO}} t}, \qquad
      R_n(\tau) = \frac{1}{T}\int_0^T R_x(t,\tau)\,e^{-jn\omega_{\text{LO}} t}\,dt$$

    The **cyclic spectral densities** $S_n(f) = \mathcal{F}\{R_n(\tau)\}$ describe spectral correlation between frequency components separated by $nf_{\text{LO}}$. The $n=0$ term is the ordinary time-averaged power spectral density (PSD).

    **Why stationary analysis fails in a mixer.** Thermal noise at the RF input at frequency $f_{\text{RF}}$ and image frequency $f_{\text{RF}} - 2f_{\text{IF}}$ both mix to the same IF frequency $f_{\text{IF}}$. The **conversion matrix** $\mathbf{C}$ maps input noise spectral vector $\mathbf{a}$ (components at $f \pm nf_{\text{LO}}$) to output vector $\mathbf{b} = \mathbf{C}\mathbf{a}$, with $C_{mn} = g_{m-n}$ (Fourier coefficient of the time-varying conductance).
    Output noise matrix: $\mathbf{S}_{\text{out}} = \mathbf{C}\,\mathbf{S}_{\text{in}}\,\mathbf{C}^\dagger$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §19. Mixer Noise Theory

    Model the MOSFET switch as a time-varying conductance $g(t)$ driven by a square-wave LO: $g(t) = g_{\text{on}}$ for $0 < t < T/2$, $g_{\text{off}}$ otherwise.
    The Fourier coefficients are given by:

    $$g_n = \frac{1}{T}\int_0^T g(t)\,e^{-jn\omega_{\text{LO}} t}\,dt = \frac{g_{\text{on}} - g_{\text{off}}}{j\pi n},\quad n \neq 0$$

    Channel thermal noise current PSD is directly proportional to the instantaneous conductance: $S_i(t,f) = 4kT\,g(t)$ — a fundamental cyclostationary process.

    **DSB noise figure.** The IF output collects noise from both the signal band ($f_{\text{LO}} + f_{\text{IF}}$) and the image band ($f_{\text{LO}} - f_{\text{IF}}$) with equal weight. For a resistive switch, it can be proven that $NF_{\text{DSB}} = L_c$ (conversion loss in dB).

    **SSB noise figure.** Only one sideband carries the signal; the image contributes noise without signal. For an ideal balanced mixer with **no** image rejection:

    $$NF_{\text{SSB}} = NF_{\text{DSB}} + 10\log_{10}(2) \approx NF_{\text{DSB}} + 3\,\text{dB}$$

    With image rejection ratio $\text{IRR}$ (dB), the image noise contribution is suppressed by $10^{-\text{IRR}/10}$, recovering $NF_{\text{SSB}} \to NF_{\text{DSB}}$ as $\text{IRR} \to \infty$.

    **Friis with mixer** (extending notebook 04 §11):

    $$F_{\text{sys}} = F_{\text{LNA}} + \frac{F_{\text{mixer}} - 1}{G_{A,\text{LNA}}}$$

    For $G_{A,\text{LNA}} = 15\,\text{dB}$ (32×) and $F_{\text{mixer}} = 10$ (10 dB SSB):
    $(10 - 1)/32 = 0.28$ linear → negligible vs. $F_{\text{LNA}} - 1 \approx 0.78$ for 3.5 dB NF.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §20. Switched-Capacitor (Sampler) Noise

    A track-and-hold circuit samples an input voltage with switch resistance $R_{\text{sw}}$ and hold capacitor $C_h$. During track phase, it forms an RC low-pass filter with noise bandwidth $BW_n = 1/(4R_{\text{sw}} C_h)$.

    The total integrated noise charge on the capacitor is the integral of the filtered Johnson noise PSD:

    $$\langle v_n^2 \rangle = \int_0^\infty \frac{4kT R_{\text{sw}}}{1 + (\omega R_{\text{sw}} C_h)^2}\,\frac{d\omega}{2\pi} = \frac{kT}{C_h}$$

    The $R_{\text{sw}}$ cancels entirely — the noise power on the capacitor is fundamentally independent of the switch resistance. This is the **kT/C noise** limit.

    **Noise folding from aliasing.** If the input noise BW is $B$ and the sample rate is $f_s$, then $N = \lceil B/f_s \rceil$ noise images fold into the baseband $[0, f_s/2]$.
    The effective noise PSD rises by $10\log_{10}(N)\,\text{dB}$:

    $$\text{Noise folding penalty} = 10\log_{10}(N)\,\text{dB}$$

    **Settling requirement** for $N_b$-bit resolution:
    The track settling error must be less than 1 LSB, so $e^{-T_s/(2\tau)} < 2^{-N_b}$. Thus:
    $\tau = R_{\text{sw}} C_h \ll \frac{T_s}{2 N_b \ln 2}$.

    At 28 GHz direct RF sampling ($f_s = 56\,\text{GHz}$, $N_b = 8$):
    $\tau \ll 1.3\,\text{ps}$ → $R_{\text{sw}} C_h \ll 1.3\,\text{ps}$,
    requiring $R_{\text{sw}} < 10\,\Omega$ for $C_h = 100\,\text{fF}$.
    Sub-10 Ω switch on-resistance at 28 GHz is feasible but challenging.
    This motivates heterodyne architecture (down-convert first to a lower IF, then sample).
    """)
    return


@app.cell
def _(mo):

    Lc_mx      = mo.ui.slider(3.0, 15.0, value=7.0, step=0.5, label="Conv. Loss (dB)")
    IRR_mx     = mo.ui.slider(0.0, 60.0, value=20.0, step=1.0, label="Image Reject. (dB)")
    fLO_mx     = mo.ui.slider(1e9, 100e9, value=28e9, step=1e9, label="f_LO (Hz)")
    GA_LNA_mx  = mo.ui.slider(0.0, 30.0, value=15.0, step=0.5, label="LNA Gain (dB)")
    NF_LNA_mx  = mo.ui.slider(0.5, 10.0, value=3.5, step=0.1, label="LNA NF (dB)")
    N_aliases  = mo.ui.slider(1, 16, value=1, step=1, label="Alias folds N")

    mo.md("### §21. Interactive IV — Mixer Noise Explorer")
    return GA_LNA_mx, IRR_mx, Lc_mx, NF_LNA_mx, N_aliases, fLO_mx


@app.cell
def _(
    GA_LNA_mx,
    IRR_mx,
    Lc_mx,
    NF_LNA_mx,
    N_aliases,
    fLO_mx,
    friis_cascade,
    go,
    mixer_NF_DSB,
    mixer_NF_SSB,
    mo,
    noise_folding_penalty_dB,
):
    _Lc    = float(Lc_mx.value)
    _IRR   = float(IRR_mx.value)
    _GA    = float(GA_LNA_mx.value)
    _NF_L  = float(NF_LNA_mx.value)
    _N_al  = int(N_aliases.value)

    _NF_DSB_mx = mixer_NF_DSB(_Lc)
    _NF_SSB_mx = mixer_NF_SSB(_Lc, _IRR)
    _fold_dB   = noise_folding_penalty_dB(_N_al)

    _stages = [
        {"F_dB": _NF_L, "G_dB": _GA},
        {"F_dB": _NF_SSB_mx, "G_dB": -_Lc},
    ]
    _res = friis_cascade(_stages)
    _F_total_dB = _res["F_total_dB"]
    _contrib = _res["contributions"]

    _fig_mx = go.Figure()
    _fig_mx.add_trace(go.Bar(
        x=["LNA", "Mixer (SSB)"],
        y=[_contrib[0] * 100, _contrib[1] * 100],
        marker_color=["#636EFA", "#EF553B"],
        text=[f"{c*100:.1f}%" for c in _contrib],
        textposition="outside"
    ))
    _fig_mx.update_layout(
        template="plotly_dark", height=400,
        yaxis_title="Contribution to F_total − 1 (%)",
        title=f"System NF = {_F_total_dB:.2f} dB | DSB NF = {_NF_DSB_mx:.1f} dB | SSB NF = {_NF_SSB_mx:.1f} dB"
    )

    _summary = mo.md(f"""
    | Quantity | Value |
    |---|---|
    | Mixer DSB NF | {_NF_DSB_mx:.2f} dB |
    | Mixer SSB NF | {_NF_SSB_mx:.2f} dB |
    | System NF (LNA + Mixer) | {_F_total_dB:.2f} dB |
    | Noise folding penalty ({_N_al}×) | {_fold_dB:.2f} dB |
    """)

    mo.vstack([
        mo.hstack([Lc_mx, IRR_mx, fLO_mx]),
        mo.hstack([GA_LNA_mx, NF_LNA_mx, N_aliases]),
        mo.ui.plotly(_fig_mx),
        _summary
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part VII — Summary

    ### §22. Summary and Bridge to Notebook 06

    The complete receiver design chain is now assembled:

    1. **LNA** (notebook 04): place $\Gamma_S$ near $\Gamma_{\text{opt}}$ via inductive
       degeneration → $NF \approx F_{\min}$.
    2. **Input/output matching** (this notebook): Bode-Fano bounds the achievable
       bandwidth; L/π/T or stub/transformer networks realise the target impedance.
    3. **Mixer** (this notebook): SSB $NF_{\text{mixer}} = NF_{\text{DSB}} + 3\,\text{dB}$
       (absent image rejection); Friis shows it is suppressed by LNA gain.
    4. **Sampler** (this notebook): kT/C sets the noise floor; noise folding penalises
       direct-RF architectures.

    **Open problems for notebook 06:**

    - **Nonlinearity:** 1 dB compression point $P_{\text{1dB}}$, input-referred
      third-order intercept $\text{IIP}_3$, AM-PM conversion.
    - **PA design:** load-pull, drain efficiency, Doherty topology.
    - **Full mmWave frontend:** LNA + matching + mixer + PA as an integrated design.

    Concept-dependency map:

    ```
    05 §4–6  L/π/T synthesis  ──►  06 PA output matching
    05 §8–10 Bode-Fano / Chebyshev  ──►  06 broadband PA
    05 §18–20 Cyclostationary  ──►  06 AM-PM / PA nonlinear noise
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
