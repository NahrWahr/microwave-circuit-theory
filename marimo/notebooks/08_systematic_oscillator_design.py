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

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import math

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, math, mo, np


# ---------------------------------------------------------------------------
# Hybrid-pi MOSFET small-signal Y-parameter helpers
# ---------------------------------------------------------------------------


@app.cell
def _(np):
    # v1.0
    """Hybrid-pi small-signal Y-parameters and activity-condition helpers
    for notebook 08 (Systematic mmWave Oscillator Design via Activity Condition).

    Conventions:
    - Frequencies in Hz, angular frequencies in rad/s.
    - Capacitances in F, transconductances in S, resistances in Ω.
    - Y matrices use port 1 = gate, port 2 = drain, common = source.
    - Activity / U / f_max derivations follow Mason 1954 and the (A, φ)
      power-flow analysis of Momeni and Afshari, IEEE JSSC 46(3), 2011.
    """

    def y_intrinsic(omega, gm, cgs, cgd, cds, ro):
        """Intrinsic hybrid-pi Y at the (intrinsic gate, drain) ports.

        omega : angular frequency (rad/s), scalar or array.
        Returns (Y11, Y12, Y21, Y22) with the same shape as omega.
        """
        omega = np.asarray(omega, dtype=float)
        Y11 = 1j * omega * (cgs + cgd)
        Y12 = -1j * omega * cgd
        Y21 = gm - 1j * omega * cgd
        Y22 = (1.0 / ro) + 1j * omega * (cds + cgd)
        return Y11, Y12, Y21, Y22

    def y_external(omega, gm, cgs, cgd, cds, ro, rg):
        """External hybrid-pi Y with series gate resistance R_g at port 1."""
        Y11p, Y12p, Y21p, Y22p = y_intrinsic(omega, gm, cgs, cgd, cds, ro)
        denom = 1.0 + Y11p * rg
        Y11 = Y11p / denom
        Y12 = Y12p / denom
        Y21 = Y21p / denom
        Y22 = Y22p - Y21p * rg * Y12p / denom
        return Y11, Y12, Y21, Y22

    def mason_U(Y):
        """Mason invariant U = |Y21 - Y12|² / (4(G11 G22 - G12 G21))."""
        Y11, Y12, Y21, Y22 = Y
        G11 = Y11.real
        G22 = Y22.real
        G12 = Y12.real
        G21 = Y21.real
        num = np.abs(Y21 - Y12) ** 2
        den = 4.0 * (G11 * G22 - G12 * G21)
        # Numerical safeguard at the activity boundary
        den_safe = np.where(np.abs(den) < 1e-30, np.sign(den) * 1e-30 + 1e-30, den)
        return num / den_safe

    def activity_margin(Y):
        """|Y12 + Y21*|² - 4 G11 G22.  > 0 ⇒ active (some embedding oscillates)."""
        Y11, Y12, Y21, Y22 = Y
        G11 = Y11.real
        G22 = Y22.real
        return np.abs(Y12 + np.conj(Y21)) ** 2 - 4.0 * G11 * G22

    def power_flow_AP(A, phi, Y):
        """P_R / (|V1||V2|) for forced V2/V1 = A e^{jφ}.

        A   : amplitude ratio |V2|/|V1|, scalar or array.
        phi : phase ∠(V2/V1) in radians, scalar or array.
        Y   : (Y11, Y12, Y21, Y22) tuple, scalar or array of identical shape.
        """
        Y11, Y12, Y21, Y22 = Y
        G11 = Y11.real
        G22 = Y22.real
        Y_sum = Y12 + np.conj(Y21)
        return -(G11 / A + A * G22) - np.abs(Y_sum) * np.cos(np.angle(Y_sum) + phi)

    def optimum_AP(Y):
        """Unconstrained (A_opt, phi_opt) and max P_R / (|V1||V2|).

        Returns
        -------
        A_opt   : sqrt(G11/G22)         (optimum amplitude ratio)
        phi_opt : (2k+1)π - ∠(Y12+Y21*) (k chosen to wrap into (-π, π])
        max_PR  : -2 sqrt(G11 G22) + |Y12+Y21*|
        """
        Y11, Y12, Y21, Y22 = Y
        G11 = Y11.real
        G22 = Y22.real
        Y_sum = Y12 + np.conj(Y21)
        with np.errstate(invalid="ignore"):
            A_opt = np.where((G11 > 0) & (G22 > 0), np.sqrt(G11 / np.maximum(G22, 1e-30)), np.nan)
        phi_opt = np.pi - np.angle(Y_sum)
        phi_opt = ((phi_opt + np.pi) % (2.0 * np.pi)) - np.pi
        max_PR = -2.0 * np.sqrt(np.maximum(G11 * G22, 0.0)) + np.abs(Y_sum)
        return A_opt, phi_opt, max_PR

    def G_m_section(A_p, phi_p, Y, G_d=0.0):
        """G_m = P_R/(|V1||V2|) for one section with forced (A', phi').

        The shunt loss G_d (e.g. parallel inductor conductance) augments G22:
            G_m = -(A'^-1 G11 + A' (G22 + G_d)) - |Y12+Y21*| cos(∠(Y12+Y21*) + φ').

        G_m > 0 ⇒ section is active; G_m = 0 is the boundary of oscillation.
        """
        Y11, Y12, Y21, Y22 = Y
        G11 = Y11.real
        G22_eff = Y22.real + G_d
        Y_sum = Y12 + np.conj(Y21)
        return -(G11 / A_p + A_p * G22_eff) - np.abs(Y_sum) * np.cos(np.angle(Y_sum) + phi_p)

    def find_zero_crossing(f_grid, values):
        """Linear interpolation of the largest f at which values changes from + to -.

        Returns NaN if no positive-to-negative crossing is found in f_grid.
        """
        signs = np.sign(values)
        last = float("nan")
        for i in range(1, len(f_grid)):
            if signs[i-1] > 0 and signs[i] <= 0:
                a, b = values[i-1], values[i]
                if a == b:
                    last = f_grid[i]
                else:
                    last = f_grid[i-1] + (f_grid[i] - f_grid[i-1]) * a / (a - b)
        return last

    def f_max(f_grid, gm, cgs, cgd, cds, ro, rg):
        """f_max from activity-margin zero crossing on f_grid (Hz)."""
        omega_grid = 2.0 * np.pi * f_grid
        Y = y_external(omega_grid, gm, cgs, cgd, cds, ro, rg)
        return find_zero_crossing(f_grid, activity_margin(Y))

    return (
        y_intrinsic,
        y_external,
        mason_U,
        activity_margin,
        power_flow_AP,
        optimum_AP,
        G_m_section,
        find_zero_crossing,
        f_max,
    )


# ---------------------------------------------------------------------------
# Top-level title and intro
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    # 08 — Systematic mmWave Oscillator Design via the Activity Condition

    Notebook 06 asked **how clean** a millimeter-wave oscillator can be (phase noise). This notebook asks the orthogonal question — **how high in frequency, and how much output power**, can a given technology actually deliver, and how does the choice of topology determine the answer.

    Mason's invariant power gain $U$ and the maximum frequency of oscillation $f_{\max}$ (notebook 02 §2) describe the **upper bound** of what a transistor can do under any lossless reciprocal embedding, but they do not by themselves tell you whether a *specific* topology — cross-coupled, three-stage ring, Colpitts, triple-push — can actually reach that bound. The Momeni–Afshari paper (IEEE JSSC 46(3), 2011) closes this gap by extending the activity condition with an explicit $(A, \varphi)$ parameterization: the amplitude ratio and phase shift between the device's two ports. Every topology imposes a particular $(A', \varphi')$ on the device. The maximum oscillation frequency $f_m$ of that topology is the highest frequency at which the constrained $(A', \varphi')$ still satisfies the activity condition.

    The result is a clean separation of three quantities:

    | Quantity   | Definition                                                | Scope                 |
    |-----------|-----------------------------------------------------------|-----------------------|
    | $f_{\max}$ | Frequency where $U = 1$                                   | Property of device    |
    | $f_m$      | Highest $f$ where forced $(A', \varphi')$ keeps $G_m > 0$ | Property of topology  |
    | $f_\text{osc}$ | Operating frequency, set by the resonator and bias   | Property of design    |

    Three-stage rings happen to land near the unconstrained $(A_\text{opt}, \varphi_\text{opt})$ for typical CMOS, reaching $f_m \approx f_{\max}$. Cross-coupled topologies do not. Above $f_{\max}$, no fundamental oscillator exists by any topology, but a sub-$f_{\max}$ ring's third harmonic can be summed coherently — the **triple-push** technique of paper Section V — pushing useful output power up to the 256 GHz / 482 GHz range in 0.13 µm and 65 nm CMOS respectively.

    **Layout.** Part I extends the notebook-02 activity condition with the $(A, \varphi)$ optimum. Part II derives $f_m$ for ring-oscillator topologies and contrasts cross-coupled with three-stage rings. Part III adds large-signal swing dynamics. Part IV gives the seven-step systematic methodology and walks the 121 / 104 GHz fundamental example. Part V covers the triple-push extension above $f_{\max}$. Part VI summarises and connects back to the phase-noise framework of notebook 06.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part I — Activity, $(A, \varphi)$, and the topology-constrained optimum
    """)
    return


# ---------------------------------------------------------------------------
# §1 — Recap from notebook 02
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Recap — Mason's $U$, the activity condition, and $f_{\max}$

    Notebook 02 §2 proved (via Sylvester's criterion on the Hermitian part of $\mathbf{Y}$) that a lossy passive two-port satisfies

    $$
    G_{11} \ge 0, \quad G_{22} \ge 0, \quad G_{11} G_{22} - \tfrac{1}{4}|Y_{12} + Y_{21}^*|^2 \ge 0,
    $$

    so the device is **active** (capable of generating power) when at least one inequality is reversed. For a CMOS transistor at normal bias, $G_{11}$ and $G_{22}$ are both strictly positive, so activity reduces to the **transfer-activity** condition

    $$
    4\, G_{11} G_{22} < |Y_{12} + Y_{21}^*|^2.
    $$

    Mason's invariant unilateral power gain is

    $$
    U = \frac{|Y_{21} - Y_{12}|^2}{4 (G_{11} G_{22} - G_{12} G_{21})},
    $$

    and the equivalence $U > 1 \iff 4(G_{11} G_{22} - G_{12} G_{21}) < |Y_{21} - Y_{12}|^2$ follows by direct algebra together with the identity

    $$
    |Y_{12} + Y_{21}^*|^2 = |Y_{21} - Y_{12}|^2 + 4\, G_{12} G_{21}.
    $$

    Both forms describe the **same** boundary in the $(G_{11}, G_{22}, Y_{12}, Y_{21})$ space; we will use whichever is more convenient. The frequency at which the boundary is crossed,

    $$
    \boxed{\, U(f_{\max}) = 1 \,},
    $$

    is the **maximum frequency of oscillation**: above $f_{\max}$ no lossless reciprocal embedding can render the device active, so a fundamental oscillator at $f > f_{\max}$ is impossible regardless of topology.

    What notebook 02 did not address is whether a *given* topology — cross-coupled, three-stage ring, Colpitts — can reach $f_{\max}$. That is the question this notebook answers, and the answer turns on the constrained $(A, \varphi)$ analysis of the next section.
    """)
    return


# ---------------------------------------------------------------------------
# §2 — Power flow with constrained (A, φ)
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Power flow with constrained $(A, \varphi)$

    **Problem.** A two-port device sits inside an oscillator. The surrounding circuit forces a particular relationship between the port voltages,

    $$
    \frac{V_2}{V_1} \;=\; A\, e^{j\varphi}, \qquad A = \frac{|V_2|}{|V_1|}, \quad \varphi = \angle(V_2/V_1).
    $$

    Here $A$ and $\varphi$ are not free parameters — they are pinned by Kirchhoff's laws applied to the topology. We want the time-averaged real power flowing out of the device into the lossless-passive embedding, since that is what must equal the embedding's loss in steady state.

    **Derivation.** The total complex power flowing **into** the two-port is

    $$
    P_\text{in} \;=\; V_1^* I_1 \,+\, V_2^* I_2.
    $$

    Substituting $I_i = Y_{ij} V_j$ and using $V_1^* V_2 = |V_1||V_2|\, e^{j\varphi}$, $V_2^* V_1 = |V_1||V_2|\, e^{-j\varphi}$:

    $$
    P_\text{in} \;=\; Y_{11}|V_1|^2 + Y_{22}|V_2|^2 + Y_{12} V_1^* V_2 + Y_{21} V_2^* V_1.
    $$

    Dividing by $|V_1||V_2|$ and using $|V_1|/|V_2| = A^{-1}$, $|V_2|/|V_1| = A$:

    $$
    \frac{P_\text{in}}{|V_1||V_2|} \;=\; A^{-1} Y_{11} + A\, Y_{22} + Y_{12}\, e^{j\varphi} + Y_{21}\, e^{-j\varphi}.
    $$

    Taking the real part (the only part that contributes to time-averaged real power) and applying the trigonometric identity

    $$
    \mathrm{Re}\!\left[ Y_{12} e^{j\varphi} + Y_{21} e^{-j\varphi} \right] \;=\; \mathrm{Re}\!\left[ (Y_{12} + Y_{21}^*) e^{j\varphi} \right] \;=\; |Y_{12} + Y_{21}^*|\, \cos\!\bigl( \angle(Y_{12} + Y_{21}^*) + \varphi \bigr),
    $$

    we obtain

    $$
    \frac{P_\text{in}}{|V_1||V_2|} \;=\; A^{-1} G_{11} + A\, G_{22} + |Y_{12} + Y_{21}^*|\, \cos\!\bigl( \angle(Y_{12} + Y_{21}^*) + \varphi \bigr).
    $$

    **Power flowing out of the device** is the negative,

    $$
    \boxed{\;
    \dfrac{P_R}{|V_1||V_2|} \;=\; -\bigl(A^{-1} G_{11} + A\, G_{22}\bigr) \;-\; |Y_{12} + Y_{21}^*|\, \cos\!\bigl( \angle(Y_{12} + Y_{21}^*) + \varphi \bigr).
    \;}
    $$

    This is the Momeni–Afshari power-flow expression (paper eq. 6). Two observations:

    - The **first term** $-(A^{-1} G_{11} + A G_{22})$ is the loss inside the device. For positive $G_{11}, G_{22}$, this term is always negative — the device's port conductances dissipate power regardless of topology.
    - The **second term** has the only freedom of sign: when $\cos(\angle(Y_{12}+Y_{21}^*) + \varphi) = -1$, the cross-coupling between ports *injects* power equal to $|Y_{12}+Y_{21}^*|$. Whether the device is net-active depends on whether this injection beats the dissipation.

    Activity ($P_R > 0$) is therefore a competition between $A$ (which controls the dissipation balance) and $\varphi$ (which controls the magnitude of the injection). The optimum is derived next.
    """)
    return


# ---------------------------------------------------------------------------
# §3 — Optimum (A, φ) and recovery of the activity condition
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Optimum amplitude and phase

    **Phase optimum.** The cosine in the second term is bounded by $\pm 1$. The most positive $P_R$ occurs at $\cos(\angle(Y_{12}+Y_{21}^*) + \varphi) = -1$, i.e.

    $$
    \boxed{\;\varphi_\text{opt} \;=\; (2k+1)\pi - \angle(Y_{12} + Y_{21}^*)\;}, \qquad k \in \mathbb{Z}.
    $$

    At this phase the second term contributes $+|Y_{12} + Y_{21}^*|$ to the right-hand side.

    **Amplitude optimum.** The first term $A^{-1} G_{11} + A G_{22}$ is minimised over $A > 0$ by AM-GM:

    $$
    A^{-1} G_{11} + A\, G_{22} \;\ge\; 2 \sqrt{G_{11} G_{22}}, \qquad \text{equality at}\ \boxed{\;A_\text{opt} \;=\; \sqrt{G_{11}/G_{22}}\;.}
    $$

    **Combined.** Substituting the two optima back into $P_R$:

    $$
    \boxed{\;
    \max_{A, \varphi} \frac{P_R}{|V_1||V_2|}
    \;=\;
    -2\sqrt{G_{11} G_{22}} \;+\; |Y_{12} + Y_{21}^*|.
    \;}
    $$

    **Recovering the activity condition.** A device can deliver positive real power to *some* lossless embedding if and only if this maximum is positive,

    $$
    -2\sqrt{G_{11} G_{22}} + |Y_{12} + Y_{21}^*| \;>\; 0
    \quad\Longleftrightarrow\quad
    4\, G_{11} G_{22} \;<\; |Y_{12} + Y_{21}^*|^2,
    $$

    which is exactly the transfer-activity condition $\neg C$ from notebook 02 §2 and is equivalent to $U > 1$.

    **Interpretation.** Two derivations of activity now coexist:

    - **Sylvester / unilateralisation argument** (notebook 02): activity $\iff$ the unilateralised Hermitian part is indefinite $\iff$ $U > 1$. The argument is silent about how to actually achieve the boundary.
    - **$(A, \varphi)$ argument** (this notebook): activity $\iff$ a specific $(A_\text{opt}, \varphi_\text{opt})$ exists at which power can be extracted. The argument identifies *the* optimum excitation, which is the design target for real topologies.

    The two views are equivalent for the boundary, but they disagree about what is *easy*. Sylvester's argument allows you to embed the device freely; the $(A, \varphi)$ argument tells you that the embedding must produce a particular amplitude ratio and phase shift between the two ports. A real topology — cross-coupled, ring, Colpitts — pins $(A', \varphi')$ to a discrete set of values determined by the circuit graph. If those values happen to lie near $(A_\text{opt}, \varphi_\text{opt})$, the topology reaches close to $f_{\max}$. If not, the topology stops at a strictly lower frequency.
    """)
    return


# ---------------------------------------------------------------------------
# §4 — Negative-conductance vs transfer activity
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Negative-conductance vs. transfer activity

    The activity condition $\neg(A \cap B \cap C)$ is satisfied by failing **any** of:

    - **$\neg A$** : $G_{11} < 0$. The input port presents negative conductance — the device pushes current out of port 1 even with no excitation at port 2.
    - **$\neg B$** : $G_{22} < 0$. The output port presents negative conductance.
    - **$\neg C$** : $4 G_{11} G_{22} < |Y_{12} + Y_{21}^*|^2$. **Transfer activity** — neither port is locally negative, but the cross-coupling between them is strong enough that a coordinated $(A, \varphi)$ excitation extracts power.

    Negative-conductance activity ($\neg A$ or $\neg B$) is the regime of tunnel diodes, IMPATTs, and Gunn diodes. It is rare in MOSFETs at normal bias: well below $f_T$ the input conductance is dominated by gate-to-channel displacement currents (purely susceptive plus a small positive ohmic part from $R_g$), and the output conductance is dominated by $1/r_o$. The CMOS oscillator regime is therefore **transfer activity**, and every derivation in this notebook will assume $G_{11}, G_{22} > 0$.

    The remaining frequency dependence is straightforward:

    - As $f$ rises, $G_{11}$ grows (mainly from the series gate resistance $R_g$ coupling through the gate-to-channel capacitance), and $G_{22}$ grows (from feedback through $C_{gd}$ and $R_g$).
    - $|Y_{12} + Y_{21}^*|$ falls (the transconductance term $g_m$ stays roughly constant up to $f_T$, but the displacement currents through $C_{gd}$ partially cancel it in the $Y_{12} + Y_{21}^*$ combination).
    - At $f_{\max}$ the two sides meet: $4 G_{11} G_{22} = |Y_{12} + Y_{21}^*|^2$.
    """)
    return


# ---------------------------------------------------------------------------
# Interactive I — Activity power-flow explorer
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive I — Activity power-flow explorer

    For a hybrid-π MOSFET with gate resistance $R_g$, the heatmap below shows the constrained power-flow $P_R / (|V_1||V_2|)$ over the $(A, \varphi)$ plane at the chosen frequency. The optimum $(A_\text{opt}, \varphi_\text{opt})$ is marked. The right panel shows how $A_\text{opt}(f)$ and $\varphi_\text{opt}(f)$ evolve over the device's useful frequency range.

    Defaults target $f_{\max} \approx 150\text{–}170$ GHz, representative of an aggressive 0.13 µm-class transistor.
    """)
    return


@app.cell
def _(mo):
    gm_I = mo.ui.slider(10.0, 100.0, step=5.0, value=50.0, label="g_m (mS)", show_value=True)
    cgs_I = mo.ui.slider(20.0, 150.0, step=5.0, value=70.0, label="C_gs (fF)", show_value=True)
    cgd_I = mo.ui.slider(2.0, 30.0, step=1.0, value=15.0, label="C_gd (fF)", show_value=True)
    cds_I = mo.ui.slider(2.0, 30.0, step=1.0, value=10.0, label="C_ds (fF)", show_value=True)
    rg_I = mo.ui.slider(2.0, 40.0, step=1.0, value=12.0, label="R_g (Ω)", show_value=True)
    ro_I = mo.ui.slider(0.3, 5.0, step=0.1, value=1.5, label="r_o (kΩ)", show_value=True)
    f_I = mo.ui.slider(20.0, 250.0, step=2.0, value=100.0, label="f (GHz)", show_value=True)
    mo.vstack([
        mo.hstack([gm_I, cgs_I, cgd_I, cds_I], gap="2rem"),
        mo.hstack([rg_I, ro_I, f_I], gap="2rem"),
    ])
    return cds_I, cgd_I, cgs_I, f_I, gm_I, rg_I, ro_I


@app.cell
def _(
    activity_margin,
    cds_I,
    cgd_I,
    cgs_I,
    f_I,
    find_zero_crossing,
    gm_I,
    go,
    make_subplots,
    mo,
    np,
    optimum_AP,
    power_flow_AP,
    rg_I,
    ro_I,
    y_external,
):
    _gm = gm_I.value * 1e-3
    _cgs = cgs_I.value * 1e-15
    _cgd = cgd_I.value * 1e-15
    _cds = cds_I.value * 1e-15
    _rg = rg_I.value
    _ro = ro_I.value * 1e3

    _f = f_I.value * 1e9
    _w = 2.0 * np.pi * _f
    _Y_at_f = y_external(_w, _gm, _cgs, _cgd, _cds, _ro, _rg)

    _A = np.linspace(0.1, 4.0, 80)
    _phi = np.linspace(-np.pi, np.pi, 120)
    _AA, _PHI = np.meshgrid(_A, _phi, indexing="xy")
    _PR = power_flow_AP(_AA, _PHI, _Y_at_f)

    _A_opt, _phi_opt, _max_PR = optimum_AP(_Y_at_f)

    _f_grid = np.linspace(5e9, 300e9, 600)
    _Y_grid = y_external(2.0 * np.pi * _f_grid, _gm, _cgs, _cgd, _cds, _ro, _rg)
    _A_opt_curve, _phi_opt_curve, _max_PR_curve = optimum_AP(_Y_grid)
    _fmax = find_zero_crossing(_f_grid, activity_margin(_Y_grid))

    # Clip the heatmap z-range so the active band is visible. Using ±1.5×|max P_R|
    # focuses contrast near the activity boundary; deep-negative regions (A→0)
    # would otherwise saturate the scale.
    _z_cap = max(abs(_max_PR) * 1.5, 1e-3)

    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            rf"$P_R/(|V_1||V_2|)\text{{ at }}f={f_I.value:.0f}\text{{ GHz}}\;\;|\;\;\max={_max_PR*1e3:.2f}\text{{ mS}}$",
            rf"$(A_\mathrm{{opt}},\varphi_\mathrm{{opt}})\text{{ vs. }}f\;\;|\;\;f_\mathrm{{max}}\approx {_fmax/1e9:.1f}\text{{ GHz}}$",
        ),
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.18,
        specs=[[{"type": "heatmap"}, {"secondary_y": True}]],
    )

    _fig.add_trace(
        go.Heatmap(
            x=_A, y=_phi*180.0/np.pi, z=_PR*1e3,
            colorscale="RdBu", zmid=0.0,
            zmin=-_z_cap*1e3, zmax=_z_cap*1e3,
            colorbar=dict(
                title=dict(text=r"$P_R/(|V_1||V_2|)\text{ (mS)}$", side="right"),
                len=0.85, thickness=12, x=0.40,
            ),
            hovertemplate="A=%{x:.2f}<br>φ=%{y:.0f}°<br>P_R=%{z:.2f} mS<extra></extra>",
        ),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Contour(
            x=_A, y=_phi*180.0/np.pi, z=_PR*1e3,
            showscale=False, contours=dict(start=0, end=0, size=1, coloring="lines"),
            line=dict(color="#FFD700", width=2),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=[_A_opt], y=[_phi_opt*180.0/np.pi],
            mode="markers", name="(A_opt, φ_opt)",
            marker=dict(size=14, color="#FFD700", symbol="star",
                        line=dict(width=1.5, color="black")),
            showlegend=False,
            hovertemplate=f"A_opt={_A_opt:.2f}<br>φ_opt={_phi_opt*180/np.pi:.0f}°<extra></extra>",
        ),
        row=1, col=1,
    )

    _fig.add_trace(
        go.Scatter(
            x=_f_grid/1e9, y=_A_opt_curve, mode="lines",
            name=r"$A_\mathrm{opt}$", line=dict(color="#00CC96", width=2),
        ),
        row=1, col=2, secondary_y=False,
    )
    _fig.add_trace(
        go.Scatter(
            x=_f_grid/1e9, y=_phi_opt_curve*180.0/np.pi, mode="lines",
            name=r"$\varphi_\mathrm{opt}\,(\deg)$", line=dict(color="#EF553B", width=2),
        ),
        row=1, col=2, secondary_y=True,
    )
    if not np.isnan(_fmax):
        _fig.add_vline(x=_fmax/1e9, line=dict(color="#AB63FA", dash="dash"), row=1, col=2)

    _fig.update_xaxes(title_text=r"$A=|V_2|/|V_1|$", row=1, col=1)
    _fig.update_yaxes(title_text=r"$\varphi\text{ (deg)}$", row=1, col=1, range=[-180, 180])
    _fig.update_xaxes(title_text=r"$f\text{ (GHz)}$", row=1, col=2)
    _fig.update_yaxes(title_text=r"$A_\mathrm{opt}$", row=1, col=2, secondary_y=False)
    _fig.update_yaxes(title_text=r"$\varphi_\mathrm{opt}\text{ (deg)}$", row=1, col=2, secondary_y=True)

    _fig.update_layout(
        template="plotly_dark", height=520, showlegend=True,
        legend=dict(orientation="h", y=-0.20, x=0.55, xanchor="center"),
        margin=dict(l=60, r=20, t=60, b=80),
    )

    fig_act_I = mo.ui.plotly(_fig)
    fig_act_I
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What to look for.**

    - The black contour on the heatmap is the activity boundary $P_R = 0$ in the $(A, \varphi)$ plane. Inside the contour the device is active for that excitation; outside, the embedding cannot extract net power.
    - The gold star marks the optimum $(A_\text{opt}, \varphi_\text{opt})$ — the single $(A, \varphi)$ that maximises $P_R$ at the chosen frequency.
    - As $f$ approaches $f_{\max}$, the active region shrinks and the optimum becomes a narrow point of admissibility. Above $f_{\max}$ no $(A, \varphi)$ keeps $P_R > 0$.
    - In the right panel, $\varphi_\text{opt}(f)$ for typical CMOS sits near $135^\circ$ at low frequency and rises toward $180^\circ$ as $f \to f_{\max}$. $A_\text{opt}(f)$ stays close to unity. The fact that $\varphi_\text{opt}$ lingers near $120^\circ$ in the mmWave range is exactly why the **three-stage ring** (which forces $\varphi' = 120^\circ$) is the natural high-frequency topology.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part II — Topology-constrained $f_m$
    """)
    return


# ---------------------------------------------------------------------------
# §5 — The constraint a topology imposes
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The constraint a topology imposes

    A *connected* oscillator topology — one in which the device's two ports are linked through the surrounding passive network into a closed loop — fixes $A'$ and $\varphi'$ to discrete values determined by Kirchhoff's laws.

    **Worked illustration.** A common-source CMOS transistor with its drain wired through a passive feedback network back to the gate. Let the network be a transformer of voltage ratio $n$ and lossless phase shift $\theta$. Then by construction $V_1 = (1/n) e^{-j\theta} V_2$, i.e. $V_2/V_1 = n\, e^{j\theta}$. The transistor sees $A' = n$ and $\varphi' = \theta$, *not* $A_\text{opt}$ and $\varphi_\text{opt}$. The transformer ratio and phase shift are properties of the embedding network, not free parameters of the device.

    The figure of merit for a topology is therefore the *constrained* power flow

    $$
    G_m \;=\; \frac{P_R(A', \varphi')}{|V_1'||V_2'|} \;=\; -\bigl(A'^{-1} G_{11} + A' G_{22,\text{eff}}\bigr) \;-\; |Y_{12} + Y_{21}^*|\, \cos\!\bigl( \angle(Y_{12}+Y_{21}^*) + \varphi' \bigr),
    $$

    where $G_{22,\text{eff}}$ absorbs any shunt loss in the embedding (e.g. the parallel conductance $G_d$ of a finite-Q tank inductor). Three regimes:

    - $G_m > 0$ : the section is active. Oscillation grows.
    - $G_m = 0$ : steady state. The section's power generation exactly balances its loss.
    - $G_m < 0$ : the section is passive. Oscillation cannot start at this frequency in this topology.

    **The maximum oscillation frequency of a topology**, $f_m$, is the highest $f$ at which $G_m(f) > 0$. Strictly $f_m \le f_{\max}$, with equality if and only if the topology happens to land at $(A', \varphi') = (A_\text{opt}, \varphi_\text{opt})$.
    """)
    return


# ---------------------------------------------------------------------------
# §6 — N-stage ring with inductive loading
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. N-stage ring oscillator with inductive loading

    Consider $N$ identical common-source CMOS sections wired head-to-tail in a closed ring (Fig. 3 of the paper). Each drain drives the next gate; the ring closes back to the first gate after $N$ hops. For steady-state oscillation, the loop phase must satisfy

    $$
    \sum_{i=1}^{N} \varphi_i \;=\; 2\pi k, \qquad k \in \mathbb{Z}.
    $$

    With identical sections, each section contributes $\varphi' = k\,(2\pi/N)$. Identical sections likewise enforce identical amplitude ratios, $A' = 1$. Discrete modes index by $k = 1, 2, \dots, N-1$ (the $k = 0$ mode is DC and produces no power).

    **The output is loaded by a tank inductor** with parallel conductance $G_d = \omega L / Q$ (skin-effect resistance referred to a parallel form). The tank's job is to set the resonant frequency and provide the imaginary-axis poles required for sustained oscillation. The inductor adds $G_d$ to $G_{22}$ at port 2.

    Substituting $A' = 1$, $\varphi' = k\,(2\pi/N)$, $G_{22,\text{eff}} = G_{22} + G_d$ into the constrained power-flow expression:

    $$
    \boxed{\;
    G_m(f) \;=\; -\bigl(G_{11} + G_{22} + G_d\bigr) - |Y_{12} + Y_{21}^*|\, \cos\!\bigl( \angle(Y_{12}+Y_{21}^*) + k\tfrac{2\pi}{N} \bigr).
    \;}
    $$

    This is paper eq. 13. Notice that the inductor's *inductance* $L$ does not appear directly — only its parallel-conductance loss $G_d$. The inductor sets $f_\text{osc}$ (through $L C$ resonance with the device's input-port capacitance) but the question of *whether oscillation is possible at $f_\text{osc}$* depends only on $G_d$ and the device's $Y$-parameters. Lower-Q inductors (larger $G_d$) reduce the achievable $f_m$.

    **Operational interpretation.** $G_m$ is the largest *additional* parallel conductance the section can absorb beyond its existing $G_d$ while remaining active. Equivalently, if $G_m > 0$ at $f_\text{osc}$, the design has *margin*; the actual oscillation amplitude grows until large-signal compression brings $G_m$ down to zero (Part III).
    """)
    return


# ---------------------------------------------------------------------------
# §7 — Cross-coupled and three-stage ring
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Cross-coupled (N=2) and three-stage ring (N=3)

    Two cases dominate practical mmWave CMOS oscillators. We evaluate both with the hybrid-π default of §1 and compare to $f_{\max}$.

    ### Cross-coupled differential pair (N = 2, k = 1, $\varphi' = \pi$)

    The standard differential CMOS oscillator: two cross-coupled NMOS transistors, each transistor's drain driving the other's gate. Following the loop: $\varphi_1 + \varphi_2 = 2\pi$, with two identical sections $\Rightarrow \varphi' = \pi$ per section. The transistors operate $180^\circ$ out of phase.

    Substituting $\varphi' = \pi$:

    $$
    G_{m,\text{XC}}(f) \;=\; -(G_{11} + G_{22} + G_d) + |Y_{12} + Y_{21}^*|\, \cos\!\bigl( \angle(Y_{12}+Y_{21}^*) \bigr).
    $$

    For the hybrid-π, $Y_{12} + Y_{21}^* = -j\omega C_{gd} + g_m + j\omega C_{gd} = g_m$ in the *intrinsic* model (no $R_g$). Including $R_g$ adds small imaginary corrections; $\angle(Y_{12} + Y_{21}^*)$ stays close to zero through the mmWave range. Thus the cosine sits near $+1$, giving the maximum-possible regenerative term $\approx +g_m$.

    The cross-coupled arrangement is therefore the **easy-to-start** topology — at low frequencies it has very large $G_m$. But $\varphi' = \pi$ is far from $\varphi_\text{opt} \approx 130^\circ$ at mmWave; the $\cos(\angle + \pi) = -\cos(\angle)$ value is *not* the most negative the cosine can be over $\varphi$ at high frequency. As frequency rises, $G_{11} + G_{22}$ grows rapidly while $|Y_{12} + Y_{21}^*|$ falls, and $G_{m,\text{XC}}$ crosses zero at $f_{m,2}$, well below $f_{\max}$ for typical CMOS.

    ### Three-stage ring (N = 3, k = 1, $\varphi' = 2\pi/3 = 120^\circ$)

    Three identical common-source sections in a ring, each tank-loaded. Each section operates $120^\circ$ phase-shifted from its predecessor.

    $$
    G_{m,3}(f) \;=\; -(G_{11} + G_{22} + G_d) - |Y_{12} + Y_{21}^*|\, \cos\!\bigl( \angle(Y_{12}+Y_{21}^*) + 120^\circ \bigr).
    $$

    Since $\angle(Y_{12}+Y_{21}^*)$ is small, $\cos(\angle + 120^\circ) \approx \cos(120^\circ) = -1/2$, so the regenerative term is $|Y_{12}+Y_{21}^*|/2$ — only *half* of what cross-coupled gets at low frequency. The three-stage ring is harder to start.

    But the trade-off reverses at high frequency. As $f \to f_{\max}$, $\varphi_\text{opt}(f)$ drifts toward $180^\circ$; for typical CMOS at the relevant mmWave range, $\varphi_\text{opt}$ stays *between* $120^\circ$ and $180^\circ$ over a wide band. Cross-coupled overshoots toward $180^\circ$; three-stage undershoots toward $120^\circ$. Whichever is closer to $\varphi_\text{opt}$ at the operating frequency wins. **For typical CMOS at frequencies approaching $f_{\max}$, three-stage wins.**

    ### Numerical comparison

    With the default hybrid-π values ($g_m = 50$ mS, $C_{gs} = 70$ fF, $C_{gd} = 15$ fF, $C_{ds} = 10$ fF, $R_g = 12$ Ω, $r_o = 1.5$ kΩ), targeting $f_{\max} \approx 155$ GHz, the calculator below shows:

    - $f_{m,2}$ (cross-coupled) lies in the 90–115 GHz range with $G_d = 0$.
    - $f_{m,3}$ (three-stage ring) lies in the 140–160 GHz range, within 10 GHz of $f_{\max}$.
    - The gap $f_{m,3} - f_{m,2}$ is the technology penalty for forcing $\varphi' = \pi$ when $\varphi_\text{opt}$ is near $130^\circ$.
    """)
    return


# ---------------------------------------------------------------------------
# §8 — f_m vs forced phase shift
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. $f_m$ as a function of the forced phase shift $\varphi'$

    Plotting $f_m(\varphi')$ as $\varphi'$ ranges continuously over $[-180^\circ, 180^\circ]$ — the calculation imagines arbitrary phase shifters at each section, beyond the discrete $k\,(2\pi/N)$ values of N-stage rings — exposes the topology landscape clearly.

    For each $\varphi'$, $f_m(\varphi')$ is the highest $f$ at which

    $$
    G_m(f, \varphi') \;=\; -(G_{11}(f) + G_{22}(f)) \,-\, |Y_{12}(f) + Y_{21}^*(f)|\, \cos\!\bigl( \angle(Y_{12}+Y_{21}^*) + \varphi' \bigr) \;>\; 0,
    $$

    with $G_d = 0$ for the lossless-passives upper bound.

    The curve has three features (paper Fig. 7):

    1. **A broad plateau near $\varphi' = \varphi_\text{opt}$.** For typical CMOS, $\varphi_\text{opt}$ is in the $120^\circ$–$140^\circ$ range over the useful mmWave band, so $f_m(\varphi')$ peaks at $f_{\max}$ across $\varphi' \approx 110^\circ$–$170^\circ$.
    2. **A discrete N-stage ladder.** Three-stage rings ($\varphi' = 120^\circ$) sit on the high-$f_m$ plateau. Cross-coupled ($\varphi' = 180^\circ$) sit at the high-$\varphi'$ edge of the plateau, somewhat lower in $f_m$. Five-stage rings ($\varphi' = 144^\circ$ for $k=2$) sit deep on the plateau — even better than three-stage, but the section count grows rapidly.
    3. **A collapse region near $\varphi' = 0$** where $\cos(\angle + 0) \approx 1$, the regenerative term reverses sign, and the section becomes purely passive at all frequencies — $f_m = 0$. This is why a non-inverting common-drain feedback loop *cannot* oscillate (in our small-signal model).

    The next interactive plots both views: $G_m(f)$ for selectable $N$, and the continuous $f_m(\varphi')$ sweep.
    """)
    return


# ---------------------------------------------------------------------------
# Interactive II — Ring-oscillator f_m explorer
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive II — Ring-oscillator $f_m$ explorer

    Sliders set the hybrid-π parameters and the tank-inductor parallel conductance $G_d$. The left panel overlays $G_m(f)$ for several mode counts $N$. The right panel sweeps $f_m(\varphi')$ continuously and annotates the discrete N-stage rings.
    """)
    return


@app.cell
def _(mo):
    gm_II = mo.ui.slider(10.0, 100.0, step=5.0, value=50.0, label="g_m (mS)", show_value=True)
    cgs_II = mo.ui.slider(20.0, 150.0, step=5.0, value=70.0, label="C_gs (fF)", show_value=True)
    cgd_II = mo.ui.slider(2.0, 30.0, step=1.0, value=15.0, label="C_gd (fF)", show_value=True)
    cds_II = mo.ui.slider(2.0, 30.0, step=1.0, value=10.0, label="C_ds (fF)", show_value=True)
    rg_II = mo.ui.slider(2.0, 40.0, step=1.0, value=12.0, label="R_g (Ω)", show_value=True)
    ro_II = mo.ui.slider(0.3, 5.0, step=0.1, value=1.5, label="r_o (kΩ)", show_value=True)
    Gd_II = mo.ui.slider(0.0, 20.0, step=0.5, value=0.0, label="G_d (mS)", show_value=True)
    mo.vstack([
        mo.hstack([gm_II, cgs_II, cgd_II, cds_II], gap="2rem"),
        mo.hstack([rg_II, ro_II, Gd_II], gap="2rem"),
    ])
    return Gd_II, cds_II, cgd_II, cgs_II, gm_II, rg_II, ro_II


@app.cell
def _(
    G_m_section,
    Gd_II,
    activity_margin,
    cds_II,
    cgd_II,
    cgs_II,
    find_zero_crossing,
    gm_II,
    go,
    make_subplots,
    mo,
    np,
    rg_II,
    ro_II,
    y_external,
):
    _gm = gm_II.value * 1e-3
    _cgs = cgs_II.value * 1e-15
    _cgd = cgd_II.value * 1e-15
    _cds = cds_II.value * 1e-15
    _rg = rg_II.value
    _ro = ro_II.value * 1e3
    _Gd = Gd_II.value * 1e-3

    _f_grid = np.linspace(5e9, 350e9, 700)
    _w_grid = 2.0 * np.pi * _f_grid
    _Y_grid = y_external(_w_grid, _gm, _cgs, _cgd, _cds, _ro, _rg)
    _fmax = find_zero_crossing(_f_grid, activity_margin(_Y_grid))

    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            rf"$G_m(f)\text{{ for }}N\text{{-stage ring}}\;\;|\;\;G_d={Gd_II.value:.1f}\text{{ mS}}$",
            rf"$f_m(\varphi')\text{{ sweep}}\;\;|\;\;f_\mathrm{{max}}\approx {_fmax/1e9:.0f}\text{{ GHz}}$",
        ),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.13,
    )

    _colors = {2: "#EF553B", 3: "#00CC96", 4: "#636EFA", 5: "#FFA15A"}
    _f_m_ring = {}
    for _N in [2, 3, 4, 5]:
        _Gm_curve = G_m_section(1.0, 2.0*np.pi/_N, _Y_grid, G_d=_Gd) * 1e3  # mS
        _fig.add_trace(
            go.Scatter(
                x=_f_grid/1e9, y=_Gm_curve, mode="lines",
                name=rf"$N={_N}\,(\varphi'={int(360/_N)}°)$",
                line=dict(color=_colors[_N], width=2),
            ),
            row=1, col=1,
        )
        _Gm_W = _Gm_curve * 1e-3
        _f_m_ring[_N] = find_zero_crossing(_f_grid, _Gm_W)

    _fig.add_hline(y=0, line=dict(color="white", dash="dash", width=1), row=1, col=1)

    if not np.isnan(_fmax):
        _fig.add_vline(x=_fmax/1e9, line=dict(color="#AB63FA", dash="dash"), row=1, col=1)

    # f_m(φ') sweep
    _phi_p_grid = np.linspace(-np.pi, np.pi, 121)
    _f_m_curve = np.zeros_like(_phi_p_grid)
    for _i, _phi_p in enumerate(_phi_p_grid):
        _Gm_curve_phi = G_m_section(1.0, _phi_p, _Y_grid, G_d=_Gd)
        _val = find_zero_crossing(_f_grid, _Gm_curve_phi)
        _f_m_curve[_i] = _val if not np.isnan(_val) else 0.0

    _fig.add_trace(
        go.Scatter(
            x=_phi_p_grid*180.0/np.pi, y=_f_m_curve/1e9, mode="lines",
            name=r"$f_m(\varphi')$",
            line=dict(color="#FFFFFF", width=2),
            showlegend=False,
        ),
        row=1, col=2,
    )

    if not np.isnan(_fmax):
        _fig.add_hline(y=_fmax/1e9, line=dict(color="#AB63FA", dash="dash"), row=1, col=2)

    for _N in [2, 3, 4, 5]:
        _phi_p = 2.0 * np.pi / _N
        _val = _f_m_ring[_N]
        if not np.isnan(_val):
            _fig.add_trace(
                go.Scatter(
                    x=[_phi_p*180.0/np.pi], y=[_val/1e9],
                    mode="markers+text", marker=dict(size=12, color=_colors[_N], symbol="circle",
                                                     line=dict(width=1, color="white")),
                    text=[f"N={_N}"], textposition="top center",
                    showlegend=False,
                ),
                row=1, col=2,
            )

    _fig.update_xaxes(title_text=r"$f\text{ (GHz)}$", row=1, col=1)
    _fig.update_yaxes(title_text=r"$G_m\text{ (mS)}$", row=1, col=1)
    _fig.update_xaxes(title_text=r"$\varphi'\text{ (deg)}$", row=1, col=2, range=[-180, 180])
    _fig.update_yaxes(title_text=r"$f_m\text{ (GHz)}$", row=1, col=2,
                      range=[0, max(1.05*_fmax/1e9 if not np.isnan(_fmax) else 200, 200)])

    _fig.update_layout(template="plotly_dark", height=520, showlegend=True,
                       legend=dict(orientation="h", y=-0.18),
                       margin=dict(l=70, r=20, t=60, b=80))
    fig_fm_II = mo.ui.plotly(_fig)

    _summary_lines = [f"**Computed**:  f_max ≈ {_fmax/1e9:.1f} GHz  (G_d = 0 limit)"]
    for _N in [2, 3, 4, 5]:
        _val = _f_m_ring[_N]
        _txt = f"{_val/1e9:.1f} GHz" if not np.isnan(_val) else "—"
        _delta = (_val - _fmax) / 1e9 if (not np.isnan(_val) and not np.isnan(_fmax)) else float("nan")
        _delta_txt = f"({_delta:+.1f} GHz vs f_max)" if not np.isnan(_delta) else ""
        _summary_lines.append(f"- f_{{m,{_N}}} = {_txt}  {_delta_txt}")

    mo.vstack([fig_fm_II, mo.md("\n".join(_summary_lines))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **At low $G_d$**, the three-stage ring's $G_m(f)$ stays positive nearly up to $f_{\max}$, while the cross-coupled curve ($N=2$) crosses zero $\sim 30$–$50$ GHz lower. The five-stage ring ($\varphi' = 144^\circ$) is even better, but the four-stage ring ($\varphi' = 90^\circ$) is *worse* because $\cos(0 + 90^\circ) = 0$ kills the regenerative term entirely — the four-stage ring is locked out of activity at low frequency.
    - **As $G_d$ increases**, all curves shift down. The lossy-tank floor $G_d$ adds directly to the loss term and reduces every $f_{m,N}$. Eventually $G_d$ exceeds the maximum value of the regenerative term and oscillation becomes impossible at any frequency.
    - The right panel shows the continuous $f_m(\varphi')$ trace. The plateau at the top is where $\varphi'$ is close to $\varphi_\text{opt}$; the collapse to zero at $\varphi' \approx 0$ is the non-oscillating common-drain region. The N-stage markers tell you which discrete topologies sit on the plateau.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part III — Large-signal swing dynamics
    """)
    return


# ---------------------------------------------------------------------------
# §9 — From small-signal start-up to steady state
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. From small-signal start-up to steady state

    The analysis so far is *small-signal*: $\mathbf{Y}(f)$ is evaluated at the operating point with infinitesimal port voltages. $G_m(f, A', \varphi') > 0$ is the **start-up** condition — when the bias is first applied, even an infinitesimal voltage seed grows exponentially.

    But the device is not a small-signal element. As the oscillation amplitude $|V|$ grows, the transistor's effective transconductance compresses. The drain current saturates, then turns off below threshold. Quasi-statically, the small-signal $g_m$ is replaced by an effective $g_m^\text{eff}(|V|)$ that decreases monotonically with $|V|$. The steady-state oscillation amplitude $|V|_\text{ss}$ is determined by

    $$
    G_m\!\bigl( f_\text{osc},\, A',\, \varphi';\, g_m^\text{eff}(|V|_\text{ss}) \bigr) \;=\; 0.
    $$

    In words: the regenerative power exactly balances the dissipation (already reduced by the $G_d$ shunt). At $|V| < |V|_\text{ss}$, the section is super-active and grows; at $|V| > |V|_\text{ss}$, the section is sub-active and decays. The fixed point is unique under the standard assumption that $g_m^\text{eff}(|V|)$ is monotonically decreasing.

    **Output power.** Once $|V|_\text{ss}$ is known, the time-averaged power dissipated by the inductor's parallel conductance $G_d$ is

    $$
    P_\text{out} \;\approx\; \tfrac{1}{2}\, |V|_\text{ss}^2\, G_d.
    $$

    Higher $G_d$ extracts more power per unit swing, but lowers $f_m$ (Part II) — the central design tension between high-Q (better $f_m$) and lossy (better $P_\text{out}$) tanks.

    The figure of merit *for the active device* is the **swing-vs-frequency** curve: how much $|V|_\text{ss}$ you can achieve at a given $f_\text{osc}$ with a given $G_d$. Below we plot a family of $G_m(f)$ curves parameterized by $|V|$ and read off $|V|_\text{ss}$ as the swing at which $G_m(f_\text{osc})$ crosses the requisite $G_d$ floor.
    """)
    return


# ---------------------------------------------------------------------------
# §10 — Large-signal Y from the hybrid-π
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Large-signal $Y$ from the hybrid-π

    The simplest large-signal extension that captures swing-induced compression is to treat $g_m$ as the only swing-dependent parameter and let it follow a saturating tanh:

    $$
    g_m^\text{eff}(|V|) \;=\; g_m^{(0)}\, \frac{\tanh(\alpha\, |V|)}{\alpha\, |V|},
    $$

    with $g_m^\text{eff}(0) = g_m^{(0)}$ (small-signal value) and $g_m^\text{eff}(|V|) \to 0$ for $|V| \gg 1/\alpha$. The parameter $\alpha$ (units of 1/V) sets the swing at which compression becomes severe; for short-channel CMOS biased near peak $f_T$, $\alpha \approx 1$–$2$ V$^{-1}$ corresponds to the typical 2–3 dB compression behaviour. Capacitances $C_{gs}, C_{gd}, C_{ds}$ are held constant — bias-dependent capacitances would add small corrections (and AM–PM, treated in notebook 07) but do not change the qualitative picture.

    With this large-signal model, the family $G_m(f, |V|)$ is computed by reusing the small-signal expression of §6 with $g_m \to g_m^\text{eff}(|V|)$ for each member of the family. The result reproduces paper Fig. 8: a stack of $G_m(f)$ curves indexed by $|V|$, monotonically lower with increasing $|V|$.

    **Reading the family.** For a given $G_d$, draw the horizontal line $G_m = G_d$. The intersection of that line with each curve gives the frequency $f$ at which a section operating with swing $|V|$ would just oscillate. Conversely, at a fixed $f_\text{osc}$, the curve that intersects $G_m = G_d$ at $f_\text{osc}$ tells you $|V|_\text{ss}$ — the steady-state swing.

    The interactive below lets you sweep both $G_d$ and $\alpha$ and reads off $|V|_\text{ss}$ at a chosen $f_\text{osc}$.
    """)
    return


# ---------------------------------------------------------------------------
# Interactive III — Swing-dependent Gm(f, |V|)
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive III — Swing-dependent $G_m(f, |V|)$
    """)
    return


@app.cell
def _(mo):
    gm_III = mo.ui.slider(10.0, 100.0, step=5.0, value=50.0, label="g_m^(0) (mS)", show_value=True)
    cgs_III = mo.ui.slider(20.0, 150.0, step=5.0, value=70.0, label="C_gs (fF)", show_value=True)
    cgd_III = mo.ui.slider(2.0, 30.0, step=1.0, value=15.0, label="C_gd (fF)", show_value=True)
    cds_III = mo.ui.slider(2.0, 30.0, step=1.0, value=10.0, label="C_ds (fF)", show_value=True)
    rg_III = mo.ui.slider(2.0, 40.0, step=1.0, value=12.0, label="R_g (Ω)", show_value=True)
    ro_III = mo.ui.slider(0.3, 5.0, step=0.1, value=1.5, label="r_o (kΩ)", show_value=True)
    Gd_III = mo.ui.slider(0.0, 10.0, step=0.5, value=2.0, label="G_d (mS)", show_value=True)
    alpha_III = mo.ui.slider(0.3, 3.0, step=0.1, value=1.0, label="α (1/V)", show_value=True)
    fosc_III = mo.ui.slider(20.0, 200.0, step=2.0, value=120.0, label="f_osc (GHz)", show_value=True)
    N_III = mo.ui.dropdown(["2", "3", "4", "5"], value="3", label="N (ring stages)")
    mo.vstack([
        mo.hstack([gm_III, cgs_III, cgd_III, cds_III], gap="2rem"),
        mo.hstack([rg_III, ro_III, Gd_III, alpha_III], gap="2rem"),
        mo.hstack([fosc_III, N_III], gap="2rem"),
    ])
    return Gd_III, N_III, alpha_III, cds_III, cgd_III, cgs_III, fosc_III, gm_III, rg_III, ro_III


@app.cell
def _(
    G_m_section,
    Gd_III,
    N_III,
    alpha_III,
    cds_III,
    cgd_III,
    cgs_III,
    fosc_III,
    gm_III,
    go,
    mo,
    np,
    rg_III,
    ro_III,
    y_external,
):
    _gm0 = gm_III.value * 1e-3
    _cgs = cgs_III.value * 1e-15
    _cgd = cgd_III.value * 1e-15
    _cds = cds_III.value * 1e-15
    _rg = rg_III.value
    _ro = ro_III.value * 1e3
    _Gd = Gd_III.value * 1e-3
    _alpha = alpha_III.value
    _N = int(N_III.value)
    _phi_p = 2.0 * np.pi / _N
    _f_osc = fosc_III.value * 1e9

    def gm_eff(V):
        # Swing-dependent transconductance: small-signal limit gm0 at V=0.
        if V <= 0.0:
            return _gm0
        return _gm0 * np.tanh(_alpha * V) / (_alpha * V)

    _f_grid = np.linspace(20e9, 250e9, 600)
    _w_grid = 2.0 * np.pi * _f_grid

    _V_levels = [0.01, 0.5, 1.0, 1.2, 1.5, 2.0]
    _colors = ["#FFFFFF", "#FFD700", "#00CC96", "#636EFA", "#EF553B", "#AB63FA"]

    _fig = go.Figure()
    _Gm_at_fosc = []
    for _V, _c in zip(_V_levels, _colors):
        _gmV = gm_eff(_V)
        _Y = y_external(_w_grid, _gmV, _cgs, _cgd, _cds, _ro, _rg)
        _Gm = G_m_section(1.0, _phi_p, _Y, G_d=_Gd) * 1e3
        _label = (rf"$|V|={_V:.2f}\text{{ V}}$" if _V > 0.05
                  else r"$\text{small-signal}$")
        _fig.add_trace(go.Scatter(
            x=_f_grid/1e9, y=_Gm, mode="lines", name=_label,
            line=dict(color=_c, width=2),
        ))
        _idx = int(np.argmin(np.abs(_f_grid - _f_osc)))
        _Gm_at_fosc.append((_V, _Gm[_idx]))

    _fig.add_hline(y=0, line=dict(color="white", dash="dash", width=1))
    _fig.add_vline(x=fosc_III.value, line=dict(color="#FF6692", dash="dot", width=1.5))

    _fig.update_layout(
        template="plotly_dark",
        title=rf"$G_m(f,|V|)\text{{ for }}N={_N}\text{{ ring section}}\;\;|\;\;G_d={Gd_III.value:.1f}\text{{ mS}}\;\;|\;\;f_\mathrm{{osc}}={fosc_III.value:.0f}\text{{ GHz}}$",
        xaxis_title=r"$f\text{ (GHz)}$",
        yaxis_title=r"$G_m\text{ (mS)}$",
        height=480,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=70, r=20, t=70, b=80),
    )
    fig_swing_III = mo.ui.plotly(_fig)

    # Solve for |V|_ss: |V| where G_m(f_osc, |V|) crosses zero.
    _Vmax = 3.0
    _V_fine = np.linspace(0.01, _Vmax, 300)
    _Gm_fine = np.zeros_like(_V_fine)
    for _i, _V in enumerate(_V_fine):
        _gmV = gm_eff(_V)
        _Y = y_external(2.0*np.pi*_f_osc, _gmV, _cgs, _cgd, _cds, _ro, _rg)
        _Gm_fine[_i] = G_m_section(1.0, _phi_p, _Y, G_d=_Gd)

    _Vss = float("nan")
    for _i in range(1, len(_V_fine)):
        if _Gm_fine[_i-1] > 0 and _Gm_fine[_i] <= 0:
            _a, _b = _Gm_fine[_i-1], _Gm_fine[_i]
            _Vss = _V_fine[_i-1] + (_V_fine[_i] - _V_fine[_i-1]) * _a / (_a - _b)
            break

    if np.isnan(_Vss):
        _readout = (
            f"**Steady-state at f_osc = {fosc_III.value:.0f} GHz:**\n\n"
            f"Oscillation cannot start: even at |V| = 0 (small-signal), "
            f"G_m({fosc_III.value:.0f} GHz, N={_N}) ≤ 0. Increase g_m, lower G_d, "
            f"lower f_osc, or change topology."
        )
    else:
        _Pout_mW = 0.5 * _Vss**2 * _Gd * 1e3
        _Pout_dBm = 10.0 * np.log10(max(_Pout_mW, 1e-9))
        _readout = (
            f"**Steady-state at f_osc = {fosc_III.value:.0f} GHz:**\n\n"
            f"| Quantity   | Value |\n"
            f"|------------|-------|\n"
            f"| Mode       | N = {_N}, φ' = {int(360/_N)}° |\n"
            f"| g_m^(0)    | {gm_III.value:.0f} mS |\n"
            f"| α          | {alpha_III.value:.2f} 1/V |\n"
            f"| G_d        | {Gd_III.value:.1f} mS |\n"
            f"| **|V|_ss** | **{_Vss:.3f} V** |\n"
            f"| **P_out (≈ ½|V|² G_d)** | **{_Pout_mW:.2f} mW = {_Pout_dBm:.1f} dBm** |\n"
        )

    mo.vstack([fig_swing_III, mo.md(_readout)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What to look for.**

    - At very low $|V|$ (top curve, small-signal), $G_m(f)$ is largest. The intersection with $G_m = 0$ defines $f_m$ for the chosen $N$.
    - As $|V|$ grows, the transconductance compresses and the curves drop. The steady-state swing $|V|_\text{ss}$ is the $|V|$ at which the curve passes through zero at the chosen $f_\text{osc}$.
    - **Smaller $G_d$** (higher-Q tank) raises the family upward: bigger $|V|_\text{ss}$, but lower $P_\text{out} = \tfrac{1}{2}|V|_\text{ss}^2 G_d$. **Larger $G_d$** lowers the family: smaller $|V|_\text{ss}$, but higher $P_\text{out}$ until the family stops crossing zero — at which point oscillation collapses.
    - **Approaching $f_m$**, $|V|_\text{ss} \to 0$. Output power vanishes near $f_m$ even though oscillation is technically possible. This is why practical designs sit at $f_\text{osc} \lesssim 0.85\, f_m$ to retain useful swing.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part IV — Systematic 7-step methodology + 121 / 104 GHz example
    """)
    return


# ---------------------------------------------------------------------------
# §11 — The 7-step methodology
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. The seven-step methodology

    The Momeni–Afshari paper (Sec. III-C, Fig. 10) packages the analysis above into a **systematic flow** that takes a process technology and a target frequency and returns a topology, transistor sizing, and inductor specification — or proves that the target is infeasible. Each step has a defined input, output, and *failure mode* that returns control to a specific earlier step.

    ### Step 1 — Determine $f_{\max}$ of the process

    Measure or simulate the device's $Y$-parameters across frequency, compute $U(f)$, find $U(f_{\max}) = 1$. Extrapolation from low-frequency data is optimistic; in-process measurement is more reliable. **Outputs:** $f_{\max}$, the device $Y$-data over the design band.

    ### Step 2 — Choose $f_\text{osc} < f_{\max}$

    Initial pick. Common choice: $f_\text{osc} \approx 0.5$–$0.8\, f_{\max}$. Sets the design target. **Failure mode (later):** if the passives can't meet the constraint at this $f_\text{osc}$, return here and lower it.

    ### Step 3 — Compute $(A_\text{opt}, \varphi_\text{opt})$ at $f_\text{osc}$

    Direct formulas

    $$
    A_\text{opt}(f_\text{osc}) = \sqrt{G_{11}(f_\text{osc})/G_{22}(f_\text{osc})}, \qquad
    \varphi_\text{opt}(f_\text{osc}) = (2k+1)\pi - \angle(Y_{12}(f_\text{osc}) + Y_{21}^*(f_\text{osc})).
    $$

    These are the topology targets.

    ### Step 4 — Pick a topology with $(A', \varphi')$ near $(A_\text{opt}, \varphi_\text{opt})$

    For an N-stage ring, $\varphi' = k\,(2\pi/N)$ is discrete; choose $N$ to minimise $|\varphi' - \varphi_\text{opt}|$. For Colpitts, the capacitor divider sets the amplitude transformation. Cross-coupled forces $(1, \pi)$ regardless of $\varphi_\text{opt}$ and is usually a poor match. **Output:** topology choice.

    ### Step 5 — Size transistor and pick passive component values

    Choose width, finger count, and bias for the transconductance / parasitic balance. With ideal passives, choose $L_d, C_d$ to resonate at $f_\text{osc}$. Iteratively set the transistor size based on the desired $G_m$ margin and output swing target.

    ### Step 6 — Compute $G_d^{\max}(f_\text{osc}, A', \varphi')$

    The largest tank conductance the section can absorb while still oscillating at $f_\text{osc}$. From §6,

    $$
    G_d^{\max} \;=\; -(G_{11} + G_{22}) - |Y_{12} + Y_{21}^*|\, \cos(\angle(Y_{12} + Y_{21}^*) + \varphi'),
    $$

    evaluated at $f_\text{osc}$ with the chosen transistor's $Y$-data. **Output:** the inductor-Q budget.

    ### Step 7 — Realise the passives meeting $G_d^{\max}$

    Inductor Q, varactor Q, parasitic shunts. The inductor's parallel conductance at $f_\text{osc}$ is

    $$
    G_d^{\text{realised}} \;=\; \frac{\omega L}{Q_L^2 + 1} \cdot \frac{1}{(\omega L)^2} \;\approx\; \frac{1}{Q_L\, \omega L} \quad \text{for large } Q_L.
    $$

    If $G_d^{\text{realised}} > G_d^{\max}$, return to **Step 5** (reduce transistor size to lower the device's $G_{11} + G_{22}$ and increase $G_m$ margin) or **Step 2** (lower $f_\text{osc}$). If the inductor Q cannot be improved enough, the design is infeasible at this $(f_\text{osc}, \text{process})$.

    ### Loop structure

    ```
    Step 1  →  Step 2  →  Step 3  →  Step 4  →  Step 5  →  Step 6  →  Step 7
                  ↑                                            ↓ fail
                  └────────────── lower f_osc  ────────────────┘
                                  ↑                            ↓ fail
                                  └────── reduce W  ───────────┘
    ```

    The next interactive runs the seven steps with user inputs and reports pass/fail at each step.
    """)
    return


# ---------------------------------------------------------------------------
# §12 — Worked example: 121 / 104 GHz three-stage ring
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Worked example — 121 / 104 GHz three-stage ring

    The original paper builds two CMOS oscillators in 0.13 µm: a 121 GHz fundamental ($-3.5$ dBm output) and a 104 GHz fundamental ($-2.7$ dBm). Both use the **three-stage ring** topology because $\varphi' = 120^\circ$ is nearly optimal at this frequency in 0.13 µm.

    With our hybrid-π default values targeting paper-comparable behaviour, the methodology runs as follows.

    ### Step 1 — $f_{\max}$ from the model

    With $g_m = 50$ mS, $C_{gs} = 70$ fF, $C_{gd} = 15$ fF, $R_g = 12$ Ω, $r_o = 1.5$ kΩ, the activity-margin zero crossing is in the 150–170 GHz range — close to the paper's reported $f_{\max} \approx 174$ GHz for that process.

    ### Step 2 — Target $f_\text{osc} = 121$ GHz

    Below $f_{\max}$, in the regime where three-stage rings excel.

    ### Step 3 — $A_\text{opt}, \varphi_\text{opt}$ at 121 GHz

    From the optimum curves (Interactive I): $A_\text{opt} \approx 1.0$, $\varphi_\text{opt} \approx 130^\circ$. Both close to the three-stage ring's $(1, 120^\circ)$.

    ### Step 4 — Choose three-stage ring

    The mismatch $|\varphi' - \varphi_\text{opt}| \approx 10^\circ$ produces only a small penalty in $G_m$ relative to the optimum.

    ### Step 5 — Transistor sizing

    Paper uses $W = 10$ µm × 10 fingers (effective $W = 100$ µm), bias $V_{ds} = V_{gs} = 1.5$ V at 22 mW per oscillator (per transistor: $\sim 7$ mW). With our hybrid-π defaults representing roughly that geometry, $L_d \approx 70$ pH resonates with the device's input capacitance at 121 GHz.

    ### Step 6 — $G_d^{\max}$ at 121 GHz

    From Interactive II at $G_d = 0$, the three-stage curve has $G_m(121\,\text{GHz}) \approx 1.5$–$2.5$ mS depending on the exact hybrid-π values. This is the budget for inductor and parasitic loss.

    ### Step 7 — Inductor Q requirement

    For $L_d = 70$ pH at 121 GHz, $\omega L = 53$ Ω, so $G_d^{\text{realised}} \approx 1/(Q_L \cdot 53)$ S. A Q of 8–10 gives $G_d \approx 1.9$–$2.4$ mS, marginally meeting the $G_d^{\max}$ target. The paper achieves $Q \approx 30$ at 121 GHz with a shielded coplanar transmission-line inductor — large margin. (See nb 06 §10 for the mmWave inductor-Q mechanisms: skin effect, substrate eddy currents, shield design.)

    ### Result

    Steady-state swing $\sim 1.0$–$1.5$ V (limited by supply and device clipping), output power in the $-3$ to $-5$ dBm range — within a factor of two of the paper's measured $-3.5$ dBm at 121 GHz and $-2.7$ dBm at 104 GHz. The exact match depends on calibrating the hybrid-π parameters; the *structure* of the design — three-stage ring, $\sim 100$ µm transistor, $\sim 70$ pH inductors at $Q > 20$ — is robust.

    ### What 104 GHz costs

    Repeating with $f_\text{osc} = 104$ GHz: $\varphi_\text{opt}(104\,\text{GHz})$ drifts a bit lower (further from $120^\circ$); $L_d$ grows to $\sim 90$ pH; $G_d^{\max}$ is slightly larger because the transistor is further from $f_{\max}$. Lower frequency $\Rightarrow$ more margin, lower power requirement, similar $|V|_\text{ss}$. Output power slightly higher.
    """)
    return


# ---------------------------------------------------------------------------
# §13 — Inductor Q at mmWave
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Inductor Q at mmWave — pointer to notebook 06

    The methodology above takes the inductor Q as a *constraint* and verifies feasibility against $G_d^{\max}$. The Q itself is set by mechanisms covered in detail in **notebook 06 §10**:

    - **Skin-effect series resistance** $R_s \propto \sqrt{f}$, giving $Q_L = \omega L / R_s \propto \sqrt{f}$ in the skin-effect regime.
    - **Substrate eddy currents** in CMOS: induced currents in the lossy silicon substrate add an image inductance and additional resistance. Patterned ground shields (PGS) partially mitigate.
    - **Self-resonant frequency (SRF)** caused by parasitic shunt capacitance. Useful range $f < \text{SRF}/2$ to $2/3$.
    - **Varactor Q** $1/(\omega C R_\text{var})$ degrades rapidly above 30 GHz; for tunable designs, varactor-Q is often the binding constraint.

    Notebook 06 contains the full $Q_L(f)$ breakdown interactive (Interactive III in that notebook). Here we treat $G_d$ as a single design number.
    """)
    return


# ---------------------------------------------------------------------------
# Interactive IV — Design-flow calculator
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive IV — Design-flow calculator

    Choose target $f_\text{osc}$, transistor parameters, and inductor Q. The calculator runs the seven-step methodology and reports pass/fail at each gate.
    """)
    return


@app.cell
def _(mo):
    fosc_IV = mo.ui.slider(20.0, 250.0, step=2.0, value=121.0, label="f_osc target (GHz)", show_value=True)
    gm_IV = mo.ui.slider(10.0, 100.0, step=5.0, value=50.0, label="g_m (mS)", show_value=True)
    cgs_IV = mo.ui.slider(20.0, 150.0, step=5.0, value=70.0, label="C_gs (fF)", show_value=True)
    cgd_IV = mo.ui.slider(2.0, 30.0, step=1.0, value=15.0, label="C_gd (fF)", show_value=True)
    cds_IV = mo.ui.slider(2.0, 30.0, step=1.0, value=10.0, label="C_ds (fF)", show_value=True)
    rg_IV = mo.ui.slider(2.0, 40.0, step=1.0, value=12.0, label="R_g (Ω)", show_value=True)
    ro_IV = mo.ui.slider(0.3, 5.0, step=0.1, value=1.5, label="r_o (kΩ)", show_value=True)
    Q_IV = mo.ui.slider(3.0, 60.0, step=1.0, value=20.0, label="Inductor Q at f_osc", show_value=True)
    Ld_IV = mo.ui.slider(10.0, 300.0, step=5.0, value=70.0, label="L_d (pH)", show_value=True)
    mo.vstack([
        mo.hstack([fosc_IV, gm_IV, cgs_IV, cgd_IV], gap="2rem"),
        mo.hstack([cds_IV, rg_IV, ro_IV, Q_IV, Ld_IV], gap="2rem"),
    ])
    return Ld_IV, Q_IV, cds_IV, cgd_IV, cgs_IV, fosc_IV, gm_IV, rg_IV, ro_IV


@app.cell
def _(
    G_m_section,
    Ld_IV,
    Q_IV,
    activity_margin,
    cds_IV,
    cgd_IV,
    cgs_IV,
    find_zero_crossing,
    fosc_IV,
    gm_IV,
    mo,
    np,
    optimum_AP,
    rg_IV,
    ro_IV,
    y_external,
):
    _gm = gm_IV.value * 1e-3
    _cgs = cgs_IV.value * 1e-15
    _cgd = cgd_IV.value * 1e-15
    _cds = cds_IV.value * 1e-15
    _rg = rg_IV.value
    _ro = ro_IV.value * 1e3
    _Q = Q_IV.value
    _Ld = Ld_IV.value * 1e-12
    _f_osc = fosc_IV.value * 1e9
    _w_osc = 2.0 * np.pi * _f_osc

    # Step 1 — f_max
    _f_grid = np.linspace(5e9, 350e9, 700)
    _Y_grid = y_external(2.0*np.pi*_f_grid, _gm, _cgs, _cgd, _cds, _ro, _rg)
    _fmax = find_zero_crossing(_f_grid, activity_margin(_Y_grid))
    _step1_pass = not np.isnan(_fmax)

    # Step 2 — f_osc < f_max?
    _step2_pass = _step1_pass and (_f_osc < _fmax)

    # Step 3 — A_opt, phi_opt at f_osc
    _Y_at_fosc = y_external(_w_osc, _gm, _cgs, _cgd, _cds, _ro, _rg)
    _A_opt, _phi_opt, _max_PR = optimum_AP(_Y_at_fosc)
    _step3_pass = _step2_pass and not np.isnan(_A_opt)

    # Step 4 — best N-stage ring (closest phi' to phi_opt)
    _candidates = []
    for _N in [2, 3, 4, 5]:
        for _k in range(1, _N):
            _phi_p = _k * 2.0 * np.pi / _N
            # match to phi_opt within (-pi, pi]
            _phi_p_w = ((_phi_p + np.pi) % (2.0*np.pi)) - np.pi
            _phi_opt_w = ((_phi_opt + np.pi) % (2.0*np.pi)) - np.pi
            _delta = abs(_phi_p_w - _phi_opt_w)
            _delta = min(_delta, 2.0*np.pi - _delta)
            _candidates.append((_N, _k, _phi_p, _delta))
    _candidates.sort(key=lambda t: t[3])
    _N_best, _k_best, _phi_p_best, _delta_best = _candidates[0]
    _step4_pass = _step3_pass

    # Step 5 — transistor sizing assumed input
    _step5_pass = _step4_pass

    # Step 6 — G_d^max
    _Gd_max = G_m_section(1.0, _phi_p_best, _Y_at_fosc, G_d=0.0)
    _step6_pass = _step5_pass and (_Gd_max > 0)

    # Step 7 — realised G_d from Q and L_d
    _Gd_real = 1.0 / (_Q * _w_osc * _Ld)
    _step7_pass = _step6_pass and (_Gd_real < _Gd_max)
    _margin = (_Gd_max - _Gd_real) * 1e3 if _step6_pass else float("nan")

    # P_out estimate at the realised G_d
    if _step7_pass:
        _Vss_est = min(1.5, np.sqrt(2.0 * (_Gd_max - _Gd_real) / max(_gm * 0.5, 1e-3)))
        _Pout_est = 0.5 * _Vss_est**2 * _Gd_real
        _Pout_dBm = 10.0 * np.log10(max(_Pout_est * 1e3, 1e-6))
    else:
        _Vss_est = float("nan")
        _Pout_est = float("nan")
        _Pout_dBm = float("nan")

    def _ok(b):
        return "✅ PASS" if b else "❌ FAIL"

    _table = f"""
| Step | Action | Result | Status |
|------|--------|--------|--------|
| 1 | Determine f_max | {_fmax/1e9:.1f} GHz | {_ok(_step1_pass)} |
| 2 | f_osc < f_max?  | f_osc = {fosc_IV.value:.0f} GHz vs f_max = {_fmax/1e9:.0f} GHz | {_ok(_step2_pass)} |
| 3 | (A_opt, φ_opt) @ f_osc | A_opt = {_A_opt:.2f}, φ_opt = {_phi_opt*180/np.pi:.0f}° | {_ok(_step3_pass)} |
| 4 | Topology | N = {_N_best}, k = {_k_best}, φ' = {_phi_p_best*180/np.pi:.0f}° (Δ = {_delta_best*180/np.pi:.0f}°) | {_ok(_step4_pass)} |
| 5 | Transistor sizing | g_m = {gm_IV.value:.0f} mS, C_gs = {cgs_IV.value:.0f} fF | {_ok(_step5_pass)} |
| 6 | G_d^max | {_Gd_max*1e3:.2f} mS | {_ok(_step6_pass)} |
| 7 | Inductor realisation: G_d^realised = 1/(Q · ω L_d) | {_Gd_real*1e3:.2f} mS (Q = {_Q:.0f}, L_d = {Ld_IV.value:.0f} pH) | {_ok(_step7_pass)} |
"""

    if _step7_pass:
        _summary = f"""
**Design verdict: ✅ FEASIBLE**

- Topology: **{_N_best}-stage ring**, mode k = {_k_best} (φ' = {_phi_p_best*180/np.pi:.0f}°).
- Inductor margin: G_d^max − G_d^realised = **{_margin:.2f} mS** ({100.0*_margin/(_Gd_max*1e3):.0f}% of budget unused).
- Estimated steady-state swing |V|_ss ≈ **{_Vss_est:.2f} V**.
- Estimated output power ≈ **{_Pout_est*1e3:.2f} mW = {_Pout_dBm:.1f} dBm**.
"""
    else:
        if not _step1_pass:
            _msg = "Step 1 failed: f_max not found in 5–350 GHz grid. Increase g_m, lower R_g, or reduce parasitic capacitances."
        elif not _step2_pass:
            _msg = f"Step 2 failed: f_osc = {fosc_IV.value:.0f} GHz exceeds f_max = {_fmax/1e9:.0f} GHz. Lower f_osc, or use harmonic generation (Part V)."
        elif not _step6_pass:
            _msg = f"Step 6 failed: G_d^max = {_Gd_max*1e3:.2f} mS is not positive — chosen topology cannot oscillate at f_osc = {fosc_IV.value:.0f} GHz. Try a different topology or lower f_osc."
        else:
            _msg = (f"Step 7 failed: G_d^realised = {_Gd_real*1e3:.2f} mS exceeds G_d^max = {_Gd_max*1e3:.2f} mS. "
                    f"Increase inductor Q (currently {_Q:.0f}), lower f_osc, or reduce W (proxy: increase g_m proportionally with capacitances).")
        _summary = f"\n**Design verdict: ❌ INFEASIBLE**\n\n{_msg}"

    mo.md(_table + _summary)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part V — Beyond $f_{\max}$: triple-push for harmonic generation
    """)
    return


# ---------------------------------------------------------------------------
# §14 — Why fundamental oscillation above f_max is impossible
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Why fundamental oscillation above $f_{\max}$ is impossible

    $U(f) < 1$ for $f > f_{\max}$ ⇒ **no** lossless reciprocal embedding can render the device active at $f$. By the (A, φ) argument of §3, $\max_{(A, \varphi)} P_R(f) < 0$ above $f_{\max}$ — every excitation has the device dissipating, and no amount of clever circuit design can reverse that.

    Pushing useful output to $f > f_{\max}$ therefore requires a different mechanism: **harmonic generation**. Run a fundamental oscillator at $f_\text{fund} < f_{\max}$, and extract the $n$-th harmonic at $f_\text{out} = n\, f_\text{fund}$. The device is acting active at $f_\text{fund}$ (where $U > 1$), and the harmonic is generated by the device's nonlinearity (large-signal compression), not by linear power gain at $n f_\text{fund}$.

    Two harmonic-extraction architectures dominate at mmWave / sub-mmWave:

    - **Push-push** ($n = 2$). Two transistors driven $180^\circ$ apart; their second harmonics add in phase, their fundamentals cancel. Common but limited in output power because the second harmonic is only modestly enhanced by typical Class-B compression.
    - **Triple-push** ($n = 3$). Three sections of a ring driven $120^\circ$ apart; their *third* harmonics add in phase, their fundamentals and second harmonics cancel.

    Triple-push has two advantages over push-push for high output power:

    1. **Three transistors generate three times the harmonic current** instead of two.
    2. The three-stage ring is already the optimal *fundamental* topology near $f_{\max}$ (Part II), so the fundamental swing is large and clean, producing strong third-harmonic content under Class-AB or Class-B-ish compression.

    The remainder of Part V develops the triple-push design.
    """)
    return


# ---------------------------------------------------------------------------
# §15 — Triple-push topology
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 15. Triple-push topology — third-harmonic phase coherence

    Take a three-stage ring (Part II): three identical common-source CMOS sections, drains tank-loaded, ring closure providing $\varphi' = 120^\circ$ per section. Section $i \in \{1, 2, 3\}$ has drain voltage

    $$
    v_{D,i}(t) \;=\; V_\text{DC} + V_1 \cos\!\bigl(\omega_\text{fund} t - i\, \tfrac{2\pi}{3}\bigr) + V_2 \cos\!\bigl(2\omega_\text{fund} t - i\, \tfrac{4\pi}{3}\bigr) + V_3 \cos\!\bigl(3\omega_\text{fund} t - i\, \tfrac{6\pi}{3}\bigr) + \dots
    $$

    The harmonic phases are linear in $n$:

    | Harmonic | Phase shift between sections | Phasor sum at common node |
    |----------|------------------------------|---------------------------|
    | $n = 1$ (fundamental) | $-2\pi/3$ | $1 + e^{-j 2\pi/3} + e^{-j 4\pi/3} = 0$ |
    | $n = 2$               | $-4\pi/3$ | $1 + e^{-j 4\pi/3} + e^{-j 8\pi/3} = 0$ |
    | $n = 3$ (target)      | $-2\pi$   | $1 + e^{-j 2\pi} + e^{-j 4\pi} = 3$ |
    | $n = 4$               | $-8\pi/3$ | $0$ |
    | $n = 6, 9, \dots$ (multiples of 3) | $-2\pi k$ | $3$ |

    **The third harmonic adds coherently across the three sections; the fundamental and second harmonic cancel by symmetric phasor sum.** A common-node summing point — typically the bias-tee at the supply rail or a dedicated combining inductor — extracts $3 V_3 \cos(3\omega_\text{fund} t)$ at three times the voltage of a single section's third harmonic, i.e. **nine times the power**.

    In practice the cancellation of $n = 1, 2, 4, 5$ harmonics depends on the symmetry of the three sections. Layout asymmetry produces residual fundamental at $-30$ to $-50$ dBc relative to the third harmonic. The 60th-percentile state of practice in 0.13 µm and 65 nm CMOS publishes triple-push spectra with the third harmonic dominant by 20–25 dB.

    **Phase-noise penalty.** Phase modulation of the fundamental converts to *amplified* phase modulation of the third harmonic: $\delta\varphi_3 = 3\, \delta\varphi_1$, so $L_3(\Delta\omega) = L_1(\Delta\omega) + 20 \log 3 \approx L_1(\Delta\omega) + 9.5$ dB. Triple-push trades 9.5 dB of phase noise for the extension above $f_{\max}$. Notebook 06 develops the phase-noise framework that makes this trade-off concrete.
    """)
    return


# ---------------------------------------------------------------------------
# §16 — Gate-inductor L_g optimization
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 16. Gate-inductor $L_g$ optimisation

    A bare three-stage ring has each section sit at $(A' = 1, \varphi' = 120^\circ)$. At the high-frequency limit relevant to triple-push ($f_\text{fund} \approx 80$–$160$ GHz, well above the $f_\text{osc}$ of a fundamental design), the unconstrained optimum drifts somewhat: typical CMOS at $f_\text{fund} = 85$ GHz has $A_\text{opt} \approx 0.84$ and $\varphi_\text{opt} \approx 144^\circ$. The bare three-stage ring is suboptimal at this frequency.

    **Adding a gate inductor $L_g$** between the input pad and the intrinsic gate of each section accomplishes two things simultaneously:

    ### Goal 1 — Push $(A', \varphi')$ toward $(A_\text{opt}, \varphi_\text{opt})$ at $f_\text{fund}$

    The series gate inductor adds phase delay between the external gate node $V_1'$ and the intrinsic gate node $V_1$. Effective $\varphi'$ rises from $120^\circ$ toward $\varphi_\text{opt}$. Simultaneously, the gate-inductor partially resonates with the input capacitance $C_{gs} + C_{gd}$, increasing the gate voltage swing and reducing the effective $A' = |V_2'|/|V_1'|$ — pushing $A'$ from 1 down toward $A_\text{opt} < 1$.

    The result is a larger $G_m$ at $f_\text{fund}$ — the section is more active, the swing is larger, and the third-harmonic content is richer. Reproduces paper Fig. 21: $G_m$ at $f_\text{fund}$ rises monotonically with $L_g$ over a useful range, then collapses as $L_g$ exceeds $1/(\omega^2 C_{gg})$ and the gate node becomes inductive.

    ### Goal 2 — Match drain to gate at $3\, f_\text{fund}$

    Each transistor generates third-harmonic current at its drain. To extract that current efficiently into the load, the impedance looking into the drain at $3\,f_\text{fund}$ should match the impedance presented by the rest of the circuit at $3\,f_\text{fund}$. The gate inductor $L_g$ — together with the transistor's gate-to-drain feedforward $C_{gd}$ and the drain inductor $L_d$ — controls the drain reflection coefficient $\Gamma_D(3 f_\text{fund})$. Reproduces paper Fig. 22: $|\Gamma_D|$ at $3\,f_\text{fund}$ has a deep minimum near a specific $L_g$ (typically tens of pH for 65 nm at 150 GHz fundamental).

    ### Trade-off

    The optimum $L_g$ for fundamental swing (Goal 1) and the optimum $L_g$ for harmonic match (Goal 2) do not coincide. The design choice biases toward Goal 2 because output power at $3\,f_\text{fund}$ is the figure of merit; Goal 1 ensures sufficient harmonic generation, but the harmonic that is generated still has to be transferred to the load.

    A useful rule of thumb: pick $L_g$ at the harmonic-match optimum, then verify that $G_m$ at $f_\text{fund}$ is positive with margin. If the swing is too low (output power below target), back off $L_g$ slightly toward Goal 1 territory.

    Interactive V plots both $G_m(f, L_g)$ and $|\Gamma_D(f, L_g)|$ on a shared $L_g$ axis so the trade-off is visible.
    """)
    return


# ---------------------------------------------------------------------------
# §17 — Worked numbers: 256 GHz / 482 GHz
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 17. Worked numbers — 256 GHz / 482 GHz

    The paper builds two triple-push oscillators:

    ### 256 GHz in 0.13 µm CMOS

    - $f_\text{fund} \approx 85$ GHz (third harmonic at 255 GHz; layout-induced asymmetry shifts the actual centre frequency to 256 GHz).
    - Three sections, $W = 20$ µm × 20 fingers per transistor.
    - Drain inductor $L_d \approx 70$ pH realised as shielded coplanar transmission line (Q ≈ 30).
    - Gate inductor $L_g \approx 30$ pH (well below the bare-ring $G_m$ peak at $\sim 55$ pH; chosen for harmonic match at 255 GHz).
    - Simulated output power: $-3$ dBm. Measured: $-17$ dBm (degradation primarily from inaccurate device models above $f_{\max}$ and additional loss in interconnects).

    ### 482 GHz in 65 nm CMOS

    - $f_\text{fund} \approx 150$ GHz (paper $f_{\max}$ in 65 nm process is around 270 GHz).
    - Three sections, $W = 20$ µm × 20 fingers.
    - $L_d \approx 26$ pH, $L_g \approx 17$ pH.
    - Simulated: $-3$ dBm. Measured: $-7.9$ dBm. The 65 nm process's higher $f_{\max}$ means $f_\text{fund} = 150$ GHz is *farther* below $f_{\max}$ in relative terms than the 0.13 µm 256 GHz design — the device operates more comfortably, the model-to-measurement gap shrinks.

    **Phase-noise comparison.** The 256 GHz oscillator's measured phase noise is $-76$ dBc/Hz at 1 MHz offset; the equivalent fundamental at 85 GHz (measured separately on a buffered output) is $-97$ dBc/Hz. The 21 dB difference exceeds the predicted $20 \log 3 = 9.5$ dB; the additional $\sim 10$ dB comes from the buffer's contribution and harmonic-extraction loss.

    **General observations from Part V.**

    - Triple-push extends useful output to $\sim 3\,f_{\max}/2$ in 0.13 µm CMOS (255 GHz in a 174 GHz $f_{\max}$ process), and similarly in 65 nm.
    - Output power is set by $f_\text{fund}$ swing and harmonic-match efficiency. Both deteriorate as $f_\text{fund}$ approaches $f_{\max}$ — there's a sweet spot at $f_\text{fund} \approx 0.5$ to $0.7\, f_{\max}$.
    - Phase noise pays $\geq 20 \log 3$ dB. In phase-noise-critical applications, this rules out triple-push.
    """)
    return


# ---------------------------------------------------------------------------
# Interactive V — Triple-push gate-inductor optimization
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive V — Triple-push gate-inductor optimisation

    Sweep $L_g$ and observe the $G_m$ at $f_\text{fund}$ (Goal 1, paper Fig. 21) and the drain reflection coefficient at $3\,f_\text{fund}$ (Goal 2, paper Fig. 22) on a common axis.
    """)
    return


@app.cell
def _(mo):
    gm_V = mo.ui.slider(10.0, 150.0, step=5.0, value=80.0, label="g_m (mS)", show_value=True)
    cgs_V = mo.ui.slider(20.0, 200.0, step=5.0, value=100.0, label="C_gs (fF)", show_value=True)
    cgd_V = mo.ui.slider(2.0, 50.0, step=1.0, value=20.0, label="C_gd (fF)", show_value=True)
    cds_V = mo.ui.slider(2.0, 30.0, step=1.0, value=12.0, label="C_ds (fF)", show_value=True)
    rg_V = mo.ui.slider(2.0, 40.0, step=1.0, value=10.0, label="R_g (Ω)", show_value=True)
    ro_V = mo.ui.slider(0.3, 5.0, step=0.1, value=1.2, label="r_o (kΩ)", show_value=True)
    Ld_V = mo.ui.slider(20.0, 200.0, step=5.0, value=70.0, label="L_d (pH)", show_value=True)
    ffund_V = mo.ui.slider(50.0, 200.0, step=2.0, value=85.0, label="f_fund (GHz)", show_value=True)
    Gd_V = mo.ui.slider(0.0, 15.0, step=0.5, value=2.0, label="G_d (mS)", show_value=True)
    mo.vstack([
        mo.hstack([gm_V, cgs_V, cgd_V, cds_V], gap="2rem"),
        mo.hstack([rg_V, ro_V, Ld_V, Gd_V], gap="2rem"),
        mo.hstack([ffund_V], gap="2rem"),
    ])
    return Gd_V, Ld_V, cds_V, cgd_V, cgs_V, ffund_V, gm_V, rg_V, ro_V


@app.cell
def _(
    G_m_section,
    Gd_V,
    Ld_V,
    cds_V,
    cgd_V,
    cgs_V,
    ffund_V,
    gm_V,
    go,
    make_subplots,
    mo,
    np,
    rg_V,
    ro_V,
    y_external,
):
    _gm = gm_V.value * 1e-3
    _cgs = cgs_V.value * 1e-15
    _cgd = cgd_V.value * 1e-15
    _cds = cds_V.value * 1e-15
    _rg = rg_V.value
    _ro = ro_V.value * 1e3
    _Ld = Ld_V.value * 1e-12
    _f_fund = ffund_V.value * 1e9
    _f_3 = 3.0 * _f_fund
    _Gd = Gd_V.value * 1e-3

    def _y_with_lg(omega, lg_val):
        # Add series L_g to the gate (port 1) by extending the existing R_g + L_g series.
        # This is equivalent to replacing R_g with R_g + jωL_g in the port reduction.
        Y11p = 1j * omega * (_cgs + _cgd)
        Y12p = -1j * omega * _cgd
        Y21p = _gm - 1j * omega * _cgd
        Y22p = (1.0/_ro) + 1j * omega * (_cds + _cgd)
        Z_g_total = _rg + 1j * omega * lg_val
        denom = 1.0 + Y11p * Z_g_total
        Y11 = Y11p / denom
        Y12 = Y12p / denom
        Y21 = Y21p / denom
        Y22 = Y22p - Y21p * Z_g_total * Y12p / denom
        return Y11, Y12, Y21, Y22

    _Lg_grid = np.linspace(0.0, 100e-12, 80)

    # Panel (a): G_m at f_fund as a function of L_g
    _Gm_at_fund = np.zeros_like(_Lg_grid)
    for _i, _Lg in enumerate(_Lg_grid):
        _Y_fund = _y_with_lg(2.0*np.pi*_f_fund, _Lg)
        _Gm_at_fund[_i] = G_m_section(1.0, 2.0*np.pi/3.0, _Y_fund, G_d=_Gd) * 1e3  # mS

    # Panel (b): drain reflection coefficient at 3·f_fund vs L_g
    # Z_drain looking into drain at f_3 = 1/Y22(f_3) (with the gate biased through Z_g, source grounded).
    # Reference impedance: the parallel L_d || (rest), modeled as ω L_d at 3 f_fund.
    _Z_ref_3 = 1j * 2.0 * np.pi * _f_3 * _Ld
    _Gamma_at_3 = np.zeros_like(_Lg_grid, dtype=complex)
    for _i, _Lg in enumerate(_Lg_grid):
        _Y3 = _y_with_lg(2.0*np.pi*_f_3, _Lg)
        _Z_drain = 1.0 / (_Y3[3] + 1j*1e-12)
        _Gamma_at_3[_i] = (_Z_drain - np.conj(_Z_ref_3)) / (_Z_drain + _Z_ref_3)

    _Gamma_dB = 20.0 * np.log10(np.maximum(np.abs(_Gamma_at_3), 1e-6))

    # Panel (c): time-domain V_G(t), V_out(t) at the chosen Lg = optimum-of-(b)
    _idx_min = int(np.argmin(np.abs(_Gamma_at_3)))
    _Lg_opt = _Lg_grid[_idx_min]
    _T = 1.0 / _f_fund
    _t = np.linspace(0, 3*_T, 600)
    _Vss_est = 1.2  # rough swing estimate
    _V1 = _Vss_est
    _V3 = 0.4 * _Vss_est * (1.0 - np.abs(_Gamma_at_3[_idx_min]))
    _VG_t = _V1 * np.cos(2.0*np.pi*_f_fund*_t)
    _Vout_t = _V3 * np.cos(2.0*np.pi*_f_3*_t)

    _fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            rf"$\text{{(a) }}G_m\text{{ at }}f_\mathrm{{fund}}={ffund_V.value:.0f}\text{{ GHz}}$",
            rf"$\text{{(b) }}|\Gamma_D|\text{{ at }}3f_\mathrm{{fund}}={3*ffund_V.value:.0f}\text{{ GHz}}$",
            rf"$\text{{(c) Time domain at }}L_g={_Lg_opt*1e12:.0f}\text{{ pH}}$",
        ),
        horizontal_spacing=0.10,
    )
    _fig.add_trace(
        go.Scatter(x=_Lg_grid*1e12, y=_Gm_at_fund, mode="lines",
                   line=dict(color="#00CC96", width=2), name=r"$G_m$"),
        row=1, col=1,
    )
    _fig.add_hline(y=0, line=dict(color="white", dash="dash", width=1), row=1, col=1)
    _fig.add_vline(x=_Lg_opt*1e12, line=dict(color="#FFD700", dash="dot"), row=1, col=1)

    _fig.add_trace(
        go.Scatter(x=_Lg_grid*1e12, y=_Gamma_dB, mode="lines",
                   line=dict(color="#EF553B", width=2), name=r"$|\Gamma_D|$"),
        row=1, col=2,
    )
    _fig.add_vline(x=_Lg_opt*1e12, line=dict(color="#FFD700", dash="dot"), row=1, col=2)

    _fig.add_trace(
        go.Scatter(x=_t*1e12, y=_VG_t, mode="lines",
                   line=dict(color="#636EFA", width=2), name=r"$V_G(t)$"),
        row=1, col=3,
    )
    _fig.add_trace(
        go.Scatter(x=_t*1e12, y=_Vout_t, mode="lines",
                   line=dict(color="#FFA15A", width=2), name=r"$V_\mathrm{out}(t)\propto 3\mathrm{rd\,harm}$"),
        row=1, col=3,
    )

    _fig.update_xaxes(title_text=r"$L_g\text{ (pH)}$", row=1, col=1)
    _fig.update_yaxes(title_text=r"$G_m\text{ (mS)}$", row=1, col=1)
    _fig.update_xaxes(title_text=r"$L_g\text{ (pH)}$", row=1, col=2)
    _fig.update_yaxes(title_text=r"$|\Gamma_D|\text{ (dB)}$", row=1, col=2)
    _fig.update_xaxes(title_text=r"$t\text{ (ps)}$", row=1, col=3)
    _fig.update_yaxes(title_text=r"$V\text{ (V)}$", row=1, col=3)
    _fig.update_layout(template="plotly_dark", height=480, showlegend=True,
                       legend=dict(orientation="h", y=-0.20),
                       margin=dict(l=70, r=20, t=70, b=80))

    fig_tp_V = mo.ui.plotly(_fig)

    _Gm_at_opt = _Gm_at_fund[_idx_min]
    _Gamma_min = _Gamma_dB[_idx_min]
    _Pout_3 = 0.5 * _V3**2 * _Gd
    _Pout_3_dBm = 10.0 * np.log10(max(_Pout_3 * 1e3, 1e-9))

    _readout = f"""**Triple-push readout @ L_g = {_Lg_opt*1e12:.0f} pH (matches |Γ_D| minimum):**

- G_m at f_fund = **{_Gm_at_opt:.2f} mS**  (must be > 0 for fundamental oscillation)
- |Γ_D| at 3·f_fund = **{_Gamma_min:.1f} dB**  (deeper is better — more harmonic transfer)
- Estimated output power at 3·f_fund ≈ **{_Pout_3*1e3:.2f} mW = {_Pout_3_dBm:.1f} dBm**  (≈ ½ × (3·V_3)² G_d, three sections summed)
"""

    mo.vstack([fig_tp_V, mo.md(_readout)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **Panel (a)**: $G_m$ at $f_\text{fund}$ rises with $L_g$ over the useful range, peaks where $L_g$ resonates with the gate-side capacitance, then collapses. The peak is Goal 1 of §16.
    - **Panel (b)**: $|\Gamma_D|$ at $3\,f_\text{fund}$ has a deep minimum at a specific $L_g$ — Goal 2. Whether the Goal-1 peak and the Goal-2 minimum coincide depends on the geometry.
    - **Panel (c)**: time-domain $V_G(t)$ at the fundamental and $V_\text{out}(t) \propto V_3$ at the third harmonic, drawn at the Goal-2 optimum.
    - The output power readout uses the simplified estimate $\tfrac{1}{2} (3 V_3)^2 G_d$ for three sections summing in phase. Real designs pay layout and combiner losses on top.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part VI — Wrap-up
    """)
    return


# ---------------------------------------------------------------------------
# §18 — Wrap-up
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 18. Recap and connections

    ### The chain of reasoning

    1. **Mason's $U$ and $f_{\max}$ (notebook 02).** Activity is a property of the device; $U$ is invariant under any lossless reciprocal embedding; $U(f_{\max}) = 1$ is the absolute upper bound.

    2. **The $(A, \varphi)$ extension (this notebook §2–3).** Given a constrained excitation $V_2/V_1 = A e^{j\varphi}$, real power flowing out of the device is

       $$
       P_R/(|V_1||V_2|) = -(A^{-1} G_{11} + A G_{22}) - |Y_{12} + Y_{21}^*|\cos(\angle(Y_{12}+Y_{21}^*) + \varphi),
       $$

       maximised at $A_\text{opt} = \sqrt{G_{11}/G_{22}}$, $\varphi_\text{opt} = (2k+1)\pi - \angle(Y_{12}+Y_{21}^*)$. The maximum is $-2\sqrt{G_{11} G_{22}} + |Y_{12}+Y_{21}^*|$, and its sign reproduces the activity condition.

    3. **Topology constraint (§5–6).** A connected oscillator topology pins $(A', \varphi')$ to discrete values. The gap $|(A', \varphi') - (A_\text{opt}, \varphi_\text{opt})|$ determines $f_m \le f_{\max}$.

    4. **Three-stage rings win (§7–8).** $\varphi' = 120^\circ$ lands near the typical CMOS $\varphi_\text{opt} \approx 130^\circ$ over the useful mmWave band; cross-coupled at $\varphi' = 180^\circ$ overshoots and stops far short of $f_{\max}$.

    5. **Large-signal swing (§9–10).** The steady-state amplitude $|V|_\text{ss}$ is the swing where the compressed-$g_m$ section's $G_m$ exactly balances the inductor's $G_d$. Output power follows $\tfrac{1}{2} |V|_\text{ss}^2 G_d$.

    6. **Systematic methodology (§11–13).** The seven-step flow turns the analysis into a repeatable design procedure: $f_{\max} \to f_\text{osc} \to (A_\text{opt}, \varphi_\text{opt}) \to \text{topology} \to \text{transistor sizing} \to G_d^{\max} \to \text{passive realisation}$, with explicit failure-return paths.

    7. **Beyond $f_{\max}$ (§14–17).** Fundamental oscillation above $f_{\max}$ is impossible. Triple-push extracts the third harmonic of a sub-$f_{\max}$ three-stage ring: phase-coherent summing gives 9× power at $3\,f_\text{fund}$, with $20 \log 3 \approx 9.5$ dB phase-noise penalty. Gate inductor $L_g$ tunes the section for both fundamental optimum and harmonic match.

    ### Connections to other notebooks

    | Where used | What it provided |
    |------------|------------------|
    | Notebook 02 §2–3 | Mason's $U$, activity condition, $f_{\max}$ definition; invariance proof |
    | Notebook 03 | MAG / MSG ordering for context |
    | Notebook 06 §10 | mmWave inductor Q mechanisms (skin effect, substrate, SRF) |
    | Notebook 06 §11 | Phase-noise FOM (used here to interpret the triple-push penalty) |
    | Notebook 07 | Large-signal nonlinearity (extends §10's tanh model with full Volterra series; AM–PM) |

    ### What is *not* in this notebook

    - **Phase-noise theory**: notebook 06 derives Leeson and the ISF framework. This notebook is purely deterministic.
    - **Push-push ($n=2$)**: simpler, but lower output power and limited gain in the $f > f_{\max}$ regime relative to triple-push. Mostly omitted.
    - **EM-simulated passives**: real inductor $S$-parameters require Sonnet / HFSS / ADS Momentum simulation. Here we treat $G_d$ as a single number.
    - **Layout extraction, parasitic-aware iteration**: real CMOS designs revisit Steps 5–7 with parasitic estimates from layout; the methodology accommodates this naturally.
    - **Bias-dependent capacitances**: only $g_m$ is swing-dependent in §10's model. Bias-dependent $C_{gs}$, $C_{gd}$ produce AM–PM, treated in nb 07.

    ### Bridge

    Notebooks 04–08 now form a complete mmWave block-level design toolkit:

    - **04** — LNA noise design (front end of receive chain).
    - **05** — Matching networks and Bode–Fano (interfaces between blocks; mixers).
    - **06** — Oscillator phase noise (LO cleanliness).
    - **07** — PA design and large-signal nonlinearity (transmit chain).
    - **08** — Oscillator frequency and power (LO availability at mmWave).

    The remaining frontier — full transceiver integration, antenna interface, beamforming arrays, system noise/linearity budgeting end-to-end — is the natural continuation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Previous:** [07 — Power Amplifiers, Linearity, and Volterra Series](07_power_amplifiers_linearity.py)  |
    **Next:** *to be determined*
    """)
    return


if __name__ == "__main__":
    app.run()
