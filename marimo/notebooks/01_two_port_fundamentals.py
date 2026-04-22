import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # 01 — Two-Port Fundamentals

    Microwave circuit theory relies on the rigorous characterization of linear 
    networks. While a single-port network is described by a single impedance $Z$, 
    a multi-port network requires a matrix representation to capture the interactions 
    between its terminals.

    ---

    ## 1. The Port Condition

    A **port** is a pair of terminals where the current entering one terminal is exactly 
    equal to the current exiting the other. This ensures that the network can be 
    treated as a black box with well-defined terminal voltages and currents.

    ## 2. Z-Parameters (Impedance)

    Choose $(I_1, I_2)$ as independent variables:

    $$\begin{pmatrix} V_1 \\ V_2 \end{pmatrix} 
      = \begin{pmatrix} Z_{11} & Z_{12} \\ Z_{21} & Z_{22} \end{pmatrix} 
        \begin{pmatrix} I_1 \\ I_2 \end{pmatrix}$$

    - $Z_{11} = V_1/I_1 |_{I_2=0}$: Input impedance with output open-circuited.
    - $Z_{21} = V_2/I_1 |_{I_2=0}$: Forward transfer impedance.

    <div style="text-align: center; margin: 20px 0;">
    <svg width="450" height="220" viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>
      <g font-family="sans-serif" font-size="16" fill="currentColor">
        <!-- Port 1 -->
        <circle cx="50" cy="60" r="4" fill="currentColor"/>
        <circle cx="50" cy="160" r="4" fill="currentColor"/>
        <text x="30" y="115" text-anchor="middle">V<tspan dy="5" font-size="12">1</tspan></text>
        <line x1="10" y1="70" x2="10" y2="150" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        
        <!-- I1 arrow -->
        <line x1="60" y1="60" x2="100" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="80" y="50" text-anchor="middle">I<tspan dy="5" font-size="12">1</tspan></text>

        <!-- Network box -->
        <rect x="150" y="40" width="150" height="140" fill="none" stroke="currentColor" stroke-width="3" rx="8"/>
        <text x="225" y="115" text-anchor="middle" font-weight="bold">Linear Network</text>

        <!-- Port 2 -->
        <circle cx="400" cy="60" r="4" fill="currentColor"/>
        <circle cx="400" cy="160" r="4" fill="currentColor"/>
        <text x="420" y="115" text-anchor="middle">V<tspan dy="5" font-size="12">2</tspan></text>
        <line x1="440" y1="70" x2="440" y2="150" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        
        <!-- I2 arrow -->
        <line x1="390" y1="60" x2="350" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="370" y="50" text-anchor="middle">I<tspan dy="5" font-size="12">2</tspan></text>
      </g>
    </svg>
    </div>

    ---

    ## 3. Y-Parameters (Admittance)

    Choose $(V_1, V_2)$ as independent variables:

    $$\begin{pmatrix} I_1 \\ I_2 \end{pmatrix} 
      = \begin{pmatrix} Y_{11} & Y_{12} \\ Y_{21} & Y_{22} \end{pmatrix} 
        \begin{pmatrix} V_1 \\ V_2 \end{pmatrix}$$

    - $Y_{11} = I_1/V_1 |_{V_2=0}$: Input admittance with output short-circuited.
    - $Y_{21} = I_2/V_1 |_{V_2=0}$: Forward transfer admittance.

    <div style="text-align: center; margin: 20px 0;">
    <svg width="450" height="220" viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>
      <g font-family="sans-serif" font-size="16" fill="currentColor">
        <circle cx="150" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="65" text-anchor="middle">V<tspan dy="5" font-size="12">1</tspan></text>
        <circle cx="150" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="165" text-anchor="middle">I<tspan dy="5" font-size="12">1</tspan></text>
        <circle cx="300" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="65" text-anchor="middle">I<tspan dy="5" font-size="12">2</tspan></text>
        <circle cx="300" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="165" text-anchor="middle">V<tspan dy="5" font-size="12">2</tspan></text>
        <line x1="168" y1="60" x2="282" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="50" text-anchor="middle">Y<tspan dy="5" font-size="12">21</tspan></text>
        <line x1="282" y1="160" x2="168" y2="160" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="150" text-anchor="middle">Y<tspan dy="5" font-size="12">12</tspan></text>
        <line x1="150" y1="78" x2="150" y2="142" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="140" y="115" text-anchor="end">Y<tspan dy="5" font-size="12">11</tspan></text>
        <line x1="300" y1="142" x2="300" y2="78" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="310" y="115" text-anchor="start">Y<tspan dy="5" font-size="12">22</tspan></text>
      </g>
    </svg>
    </div>

    ---

    ## 4. H-Parameters (Hybrid)

    Choose $(I_1, V_2)$ as independent variables:

    $$\begin{pmatrix} V_1 \\ I_2 \end{pmatrix}
      = \begin{pmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{pmatrix}
        \begin{pmatrix} I_1 \\ V_2 \end{pmatrix}$$

    These are natural for BJTs (input ≈ current source, output ≈ voltage source)
    and are still used in transistor datasheets.

    <div style="text-align: center; margin: 20px 0;">
    <svg width="450" height="220" viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>
      <g font-family="sans-serif" font-size="16" fill="currentColor">
        <circle cx="150" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="65" text-anchor="middle">I<tspan dy="5" font-size="12">1</tspan></text>
        <circle cx="150" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="165" text-anchor="middle">V<tspan dy="5" font-size="12">1</tspan></text>
        <circle cx="300" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="65" text-anchor="middle">I<tspan dy="5" font-size="12">2</tspan></text>
        <circle cx="300" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="165" text-anchor="middle">V<tspan dy="5" font-size="12">2</tspan></text>
        <line x1="168" y1="60" x2="282" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="50" text-anchor="middle">h<tspan dy="5" font-size="12">21</tspan></text>
        <line x1="282" y1="160" x2="168" y2="160" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="150" text-anchor="middle">h<tspan dy="5" font-size="12">12</tspan></text>
        <line x1="150" y1="78" x2="150" y2="142" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="140" y="115" text-anchor="end">h<tspan dy="5" font-size="12">11</tspan></text>
        <line x1="300" y1="142" x2="300" y2="78" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="310" y="115" text-anchor="start">h<tspan dy="5" font-size="12">22</tspan></text>
      </g>
    </svg>
    </div>

    ---

    ## 5. ABCD (Transmission / Chain) Parameters

    Choose $(V_2, -I_2)$ as independent variables (note the sign on $I_2$ —
    this is the *output* current flowing **out** of port 2, which is conventional
    for cascade analysis):

    $$\begin{pmatrix} V_1 \\ I_1 \end{pmatrix}
      = \begin{pmatrix} A & B \\ C & D \end{pmatrix}
        \begin{pmatrix} V_2 \\ -I_2 \end{pmatrix}$$

    **Cascade property:** For two two-ports in cascade,
    $\mathbf{T}_{\text{total}} = \mathbf{T}_1 \cdot \mathbf{T}_2$.
    This makes ABCD parameters the natural choice for transmission-line and
    filter analysis.

    For a reciprocal network: $AD - BC = 1$.

    <div style="text-align: center; margin: 20px 0;">
    <svg width="450" height="220" viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>
      <g font-family="sans-serif" font-size="16" fill="currentColor">
        <circle cx="150" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="65" text-anchor="middle">V<tspan dy="5" font-size="12">1</tspan></text>
        <circle cx="150" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="165" text-anchor="middle">I<tspan dy="5" font-size="12">1</tspan></text>
        <circle cx="300" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="65" text-anchor="middle">V<tspan dy="5" font-size="12">2</tspan></text>
        <circle cx="300" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="165" text-anchor="middle">−I<tspan dy="5" font-size="12">2</tspan></text>
        
        <line x1="282" y1="60" x2="168" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="50" text-anchor="middle">A</text>
        
        <line x1="282" y1="160" x2="168" y2="160" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="150" text-anchor="middle">D</text>
        
        <line x1="285" y1="70" x2="165" y2="150" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="200" y="130" text-anchor="end">C</text>
        
        <line x1="285" y1="150" x2="165" y2="70" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="250" y="130" text-anchor="start">B</text>
      </g>
    </svg>
    </div>

    ---

    ## 6. Conversion Relations

    All parameter sets are equivalent representations of the same linear network.
    The conversions below are derived by substituting one set of equations into another.

    | From \\ To | **Z** | **Y** | **ABCD** |
    |---|---|---|---|
    | **Z** | — | $Y = Z^{-1}$ | $A=Z_{11}/Z_{21}$, $B=\Delta_Z/Z_{21}$, $C=1/Z_{21}$, $D=Z_{22}/Z_{21}$ |
    | **Y** | $Z = Y^{-1}$ | — | $A=-Y_{22}/Y_{21}$, $B=-1/Y_{21}$, $C=-\Delta_Y/Y_{21}$, $D=-Y_{11}/Y_{21}$ |
    | **ABCD** | $Z_{11}=A/C$, $Z_{12}=\Delta_T/C$, $Z_{21}=1/C$, $Z_{22}=D/C$ | $Y_{11}=D/B$, $Y_{12}=-\Delta_T/B$, $Y_{21}=-1/B$, $Y_{22}=A/B$ | — |

    where $\Delta_Z = Z_{11}Z_{22}-Z_{12}Z_{21}$, $\Delta_Y = Y_{11}Y_{22}-Y_{12}Y_{21}$,
    $\Delta_T = AD-BC$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Interactive Two-Port Explorer

    Tune component values and observe how the impedance parameters evolve across frequency. The matrices are evaluated at the chosen frequency; the plots show broadband magnitude and phase.
    """)
    return


@app.cell
def _(mo):
    topology_selector = mo.ui.dropdown(
        options={
            "Series impedance (Z_s)": "series",
            "Shunt admittance (Y_p)": "shunt",
            "T-network (Z1, Z2, Z3)": "tee",
            "π-network (Z1, Y2, Z3)": "pi",
            "L-section (series Z, shunt Y)": "lsec",
        },
        value="Series impedance (Z_s)",
        label="Topology",
    )
    return (topology_selector,)


@app.cell
def _(mo, topology_selector):
    t = topology_selector.value

    if t == "series":
        r_slider = mo.ui.slider(1, 1000, step=1, value=50, label="R (Ω)", show_value=True)
        l_slider = mo.ui.slider(0, 10, step=0.1, value=0.0, label="L (nH)", show_value=True)
        c_slider = mo.ui.slider(0, 100, step=1, value=0, label="C (fF) — 0 = open", show_value=True)
        topo_label = mo.md("**Series element:** $Z_s = R + j\\omega L \\;\\|\\; (1/j\\omega C)$ if C>0")
    elif t == "shunt":
        r_slider = mo.ui.slider(1, 10000, step=10, value=1000, label="R (Ω)", show_value=True)
        l_slider = mo.ui.slider(0, 10, step=0.1, value=0.0, label="L (nH)", show_value=True)
        c_slider = mo.ui.slider(0, 1000, step=10, value=100, label="C (fF)", show_value=True)
        topo_label = mo.md("**Shunt element:** $Y_p = 1/R + j\\omega C + 1/(j\\omega L)$ if L>0")
    elif t == "tee":
        r_slider = mo.ui.slider(0, 500, step=5, value=25, label="Z1: R (Ω)", show_value=True)
        l_slider = mo.ui.slider(0, 5, step=0.1, value=1.0, label="Z1: L (nH)", show_value=True)
        c_slider = mo.ui.slider(1, 1000, step=10, value=200, label="Z3 (shunt): R (Ω)", show_value=True)
        topo_label = mo.md("**T-network:** Z1 = series arm 1, Z3 = shunt arm, Z2 = Z1 (symmetric)")
    elif t == "pi":
        r_slider = mo.ui.slider(10, 2000, step=10, value=500, label="Y1 shunt (Ω)", show_value=True)
        l_slider = mo.ui.slider(1, 500, step=5, value=50, label="Z_series (Ω)", show_value=True)
        c_slider = mo.ui.slider(10, 2000, step=10, value=500, label="Y2 shunt (Ω)", show_value=True)
        topo_label = mo.md("**π-network:** Y1 = input shunt, Z_s = series arm, Y2 = output shunt")
    else:  # lsec
        r_slider = mo.ui.slider(1, 500, step=5, value=50, label="Series R (Ω)", show_value=True)
        l_slider = mo.ui.slider(0, 10, step=0.1, value=1.0, label="Series L (nH)", show_value=True)
        c_slider = mo.ui.slider(1, 1000, step=10, value=100, label="Shunt C (fF)", show_value=True)
        topo_label = mo.md("**L-section:** series $Z = R + j\\omega L$, shunt $Y = j\\omega C$")

    freq_slider = mo.ui.slider(
        steps=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
        value=10, label="Frequency (GHz)", show_value=True,
    )
    mo.vstack([
        mo.hstack([topology_selector, topo_label], justify="start", gap=1),
        mo.hstack([r_slider, l_slider, c_slider, freq_slider]),
    ])
    return c_slider, freq_slider, l_slider, r_slider, t


@app.cell
def _(c_slider, freq_slider, l_slider, mo, np, r_slider, t, go, make_subplots):
    # --- read controls ---
    R_val = float(r_slider.value)
    L_val = float(l_slider.value) * 1e-9
    C_val = float(c_slider.value) * 1e-15
    f0_Hz = freq_slider.value * 1e9
    w0 = 2 * np.pi * f0_Hz

    # --- helper: build Z-matrix at angular frequency w ---
    def _build_Z(w_):
        if t == "series":
            zs = R_val + 1j * w_ * L_val
            if C_val > 0:
                zs = 1.0 / (1.0 / zs + 1j * w_ * C_val)
            return np.array([[zs, zs], [zs, zs]])
        elif t == "shunt":
            yp = 1.0 / R_val
            if C_val > 0:
                yp += 1j * w_ * C_val
            if L_val > 0:
                yp += 1.0 / (1j * w_ * L_val)
            ym = np.array([[yp, -yp], [-yp, yp]])
            return np.linalg.inv(ym) if abs(np.linalg.det(ym)) > 1e-30 else np.full((2, 2), np.nan + 1j * np.nan)
        elif t == "tee":
            z1 = R_val + 1j * w_ * L_val
            z3 = float(c_slider.value)
            return np.array([[z1 + z3, z3], [z3, z1 + z3]])
        elif t == "pi":
            y1 = 1.0 / R_val
            zs_pi = float(l_slider.value) if abs(l_slider.value) > 1e-15 else 1e-9
            y2 = 1.0 / float(c_slider.value) if abs(c_slider.value) > 1e-15 else 0.0
            ys = 1.0 / zs_pi
            ym = np.array([[y1 + ys, -ys], [-ys, y2 + ys]])
            return np.linalg.inv(ym) if abs(np.linalg.det(ym)) > 1e-30 else np.full((2, 2), np.nan + 1j * np.nan)
        else:  # lsec
            zser = R_val + 1j * w_ * L_val
            ysh = 1j * w_ * C_val
            zsh = 1.0 / ysh if abs(ysh) > 1e-30 else 1e12
            return np.array([[zser + zsh, zsh], [zsh, zsh]])

    # --- single-frequency matrix evaluation ---
    Z = _build_Z(w0)
    detZ = Z[0, 0] * Z[1, 1] - Z[0, 1] * Z[1, 0]
    z11, z12, z21, z22 = Z[0, 0], Z[0, 1], Z[1, 0], Z[1, 1]

    if abs(detZ) > 1e-30:
        Y_mat = np.linalg.inv(Z)
        y_valid = True
    else:
        Y_mat = np.full((2, 2), np.nan)
        y_valid = False

    if abs(z21) > 1e-30:
        ABCD_mat = np.array([
            [z11 / z21, detZ / z21],
            [1.0 / z21, z22 / z21],
        ])
        abcd_valid = True
    else:
        ABCD_mat = np.full((2, 2), np.nan)
        abcd_valid = False

    def _fmt_z(z):
        r, i = float(np.real(z)), float(np.imag(z))
        if abs(i) < 1e-5:
            return f"{r:.2f}"
        if abs(r) < 1e-5:
            return f"{i:.2f}j"
        sign = "+" if i > 0 else "-"
        return f"{r:.2f} {sign} {abs(i):.2f}j"

    def _tex_mat(mat, unit=""):
        if np.any(np.isinf(mat)) or np.any(np.isnan(mat)):
            return r"\text{Undefined}"
        s = r"\begin{bmatrix} "
        s += _fmt_z(mat[0, 0]) + " & " + _fmt_z(mat[0, 1])
        s += r" \\ "
        s += _fmt_z(mat[1, 0]) + " & " + _fmt_z(mat[1, 1])
        s += r" \end{bmatrix}"
        if unit:
            s += r" \; " + unit
        return s

    z_tex = _tex_mat(Z, "Ω")
    y_tex = _tex_mat(Y_mat, r"\text{S}") if y_valid else r"\text{Singular}"
    abcd_tex = _tex_mat(ABCD_mat) if abcd_valid else r"\text{Undefined}"
    f_label = str(freq_slider.value)

    matrices_md = (
        r"**Z:** $\mathbf{Z} = " + z_tex + r"$"
        + r" &emsp; **Y:** $\mathbf{Y} = " + y_tex + r"$"
        + r" &emsp; **ABCD:** $\mathbf{T} = " + abcd_tex + r"$"
        + " &emsp; *(at " + f_label + " GHz)*"
    )

    # --- broadband sweep ---
    freqs = np.logspace(8, 11, 400)
    ws = 2 * np.pi * freqs
    z11_f = np.empty(len(freqs), dtype=complex)
    z21_f = np.empty(len(freqs), dtype=complex)
    for idx, _w in enumerate(ws):
        _Zf = _build_Z(_w)
        z11_f[idx] = _Zf[0, 0]
        z21_f[idx] = _Zf[1, 0]

    fGHz = freqs / 1e9
    mag11, mag21 = np.abs(z11_f), np.abs(z21_f)
    ph11, ph21 = np.degrees(np.angle(z11_f)), np.degrees(np.angle(z21_f))

    # --- dual-panel figure ---
    col_z11, col_z21 = "#636EFA", "#00CC96"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Magnitude (Ω)", "Phase (deg)"],
        horizontal_spacing=0.08,
    )
    # magnitude
    fig.add_trace(go.Scatter(
        x=fGHz, y=mag11, name="Z₁₁", legendgroup="z11",
        line=dict(color=col_z11, width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fGHz, y=mag21, name="Z₂₁", legendgroup="z21",
        line=dict(color=col_z21, width=2.5),
    ), row=1, col=1)
    # phase
    fig.add_trace(go.Scatter(
        x=fGHz, y=ph11, name="∠Z₁₁", legendgroup="z11", showlegend=False,
        line=dict(color=col_z11, width=2, dash="dot"),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=fGHz, y=ph21, name="∠Z₂₁", legendgroup="z21", showlegend=False,
        line=dict(color=col_z21, width=2, dash="dot"),
    ), row=1, col=2)

    # frequency marker line
    fig.add_vline(x=freq_slider.value, line_width=1, line_dash="dash",
                  line_color="rgba(255,255,255,0.35)")

    fig.update_xaxes(type="log", title_text="Frequency (GHz)")
    fig.update_yaxes(type="log", title_text="|Z| (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)", range=[-95, 95], row=1, col=2)
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=50, r=20, t=35, b=45),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )

    mo.vstack([mo.md(matrices_md), mo.ui.plotly(fig)])


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. ABCD Parameters and Cascaded Two-Port Stages

    ### 9.1 Why ABCD?

    The ABCD (or **transmission**) matrix shines when two-ports are connected **in cascade** (output of stage 1 drives input of stage 2).  The overall matrix is simply the product:

    $$\mathbf{T}_\text{total} = \mathbf{T}_1 \cdot \mathbf{T}_2 \cdot \mathbf{T}_3 \cdots$$

    This is because the ABCD convention propagates *port voltages and currents* from output to input:

    $$\begin{pmatrix}V_1 \\ I_1\end{pmatrix} = \mathbf{T} \begin{pmatrix}V_2 \\ -I_2\end{pmatrix}$$

    At the junction between stage 1 and stage 2, $V_2^{(1)} = V_1^{(2)}$ and $-I_2^{(1)} = I_1^{(2)}$, so the matrices multiply naturally.

    ### 9.2 Canonical element matrices

    | Element | ABCD matrix |
    |---|---|
    | Series impedance $Z$ | $\begin{pmatrix}1 & Z \\ 0 & 1\end{pmatrix}$ |
    | Shunt admittance $Y$ | $\begin{pmatrix}1 & 0 \\ Y & 1\end{pmatrix}$ |
    | Transmission line $Z_0$, $\theta=\beta\ell$ | $\begin{pmatrix}\cos\theta & jZ_0\sin\theta \\ j\sin\theta/Z_0 & \cos\theta\end{pmatrix}$ |

    An **L-section** (series $Z$, then shunt $Y$) is the product of the first two:

    $$\mathbf{T}_L = \begin{pmatrix}1 & Z \\ 0 & 1\end{pmatrix}\begin{pmatrix}1 & 0 \\ Y & 1\end{pmatrix} = \begin{pmatrix}1+ZY & Z \\ Y & 1\end{pmatrix}$$

    ### 9.3 Input impedance of a cascade

    With a load $Z_L$ at the output of an $N$-stage cascade ($\mathbf{T}_N = \mathbf{T}^N$):

    $$Z_\text{in} = \frac{A_N\, Z_L + B_N}{C_N\, Z_L + D_N}$$

    This **bilinear (Möbius) transformation** maps circles to circles in the complex plane — it is the mathematical foundation of the Smith chart.

    ### 9.4 Insertion loss of a cascade

    For a cascade driven by a source with internal impedance $Z_S$ and terminated in $Z_L$, the voltage transfer ratio is:

    $$H = \frac{V_2}{V_S} = \frac{1}{A_N + B_N/Z_L + C_N Z_S + D_N Z_S/Z_L}$$

    The **insertion loss** in dB is $-20\log_{10}|H|$.  The interactive plot below sweeps frequency and shows how $|H|$ evolves as you stack more stages.
    """)
    return


@app.cell
def _(mo):
    cascade_n = mo.ui.slider(1, 8, step=1, value=3, label="Number of stages N", show_value=True)
    cascade_r = mo.ui.slider(1, 200, step=1, value=50, label="Series R per stage (Ω)", show_value=True)
    cascade_l = mo.ui.slider(0, 5, step=0.1, value=0.5, label="Series L per stage (nH)", show_value=True)
    cascade_c = mo.ui.slider(1, 500, step=1, value=100, label="Shunt C per stage (fF)", show_value=True)
    cascade_zs = mo.ui.slider(1, 200, step=1, value=50, label="Source impedance Zs (Ω)", show_value=True)
    cascade_zl = mo.ui.slider(1, 200, step=1, value=50, label="Load impedance ZL (Ω)", show_value=True)
    mo.vstack([
        mo.md("### Interactive: N cascaded L-sections (series R+L, shunt C)"),
        mo.hstack([cascade_n, cascade_r, cascade_l]),
        mo.hstack([cascade_c, cascade_zs, cascade_zl]),
    ])
    return cascade_c, cascade_l, cascade_n, cascade_r, cascade_zl, cascade_zs


@app.cell
def _(cascade_c, cascade_l, cascade_n, cascade_r, cascade_zl, cascade_zs, go, make_subplots, mo, np):
    _freqs = np.logspace(8, 11, 600)
    _ws    = 2 * np.pi * _freqs
    _fGHz  = _freqs / 1e9

    _R  = float(cascade_r.value)
    _L  = float(cascade_l.value) * 1e-9
    _C  = float(cascade_c.value) * 1e-15
    _Zs = float(cascade_zs.value)
    _ZL = float(cascade_zl.value)
    _N  = int(cascade_n.value)

    _H_db  = np.zeros((len(_ws), _N))
    _Zin_r = np.zeros((len(_ws), _N))
    _Zin_i = np.zeros((len(_ws), _N))

    for _ni in range(1, _N + 1):
        for _i, _w in enumerate(_ws):
            _Zser = _R + 1j * _w * _L
            _Ysh  = 1j * _w * _C
            _T1 = np.array([[1 + _Zser * _Ysh, _Zser], [_Ysh, 1.0]])
            _Tn = np.linalg.matrix_power(_T1, _ni)
            _An, _Bn, _Cn, _Dn = _Tn[0,0], _Tn[0,1], _Tn[1,0], _Tn[1,1]
            _denom = _An + _Bn / _ZL + _Cn * _Zs + _Dn * _Zs / _ZL
            _H = 1.0 / _denom if abs(_denom) > 1e-30 else np.nan
            _H_db[_i, _ni-1] = 20 * np.log10(abs(_H)) if abs(_H) > 0 else np.nan
            _zin = (_An * _ZL + _Bn) / (_Cn * _ZL + _Dn) if abs(_Cn * _ZL + _Dn) > 1e-30 else np.nan
            _Zin_r[_i, _ni-1] = np.real(_zin) if not np.isnan(_zin) else np.nan
            _Zin_i[_i, _ni-1] = np.imag(_zin) if not np.isnan(_zin) else np.nan

    _palette = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880"]
    _fig_cas = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Insertion loss |H| (dB)", "Input impedance Re(Zin), Im(Zin) (Ω)"],
        horizontal_spacing=0.12,
    )
    for _ni in range(_N):
        _col = _palette[_ni % len(_palette)]
        _fig_cas.add_trace(
            go.Scatter(x=_fGHz, y=_H_db[:, _ni], name=f"N={_ni+1}",
                       line=dict(color=_col), legendgroup=f"N{_ni+1}"),
            row=1, col=1)
        _fig_cas.add_trace(
            go.Scatter(x=_fGHz, y=_Zin_r[:, _ni], name=f"Re Zin N={_ni+1}",
                       line=dict(color=_col), legendgroup=f"N{_ni+1}", showlegend=False),
            row=1, col=2)
        _fig_cas.add_trace(
            go.Scatter(x=_fGHz, y=_Zin_i[:, _ni], name=f"Im Zin N={_ni+1}",
                       line=dict(color=_col, dash="dot"), legendgroup=f"N{_ni+1}", showlegend=False),
            row=1, col=2)

    _fig_cas.update_xaxes(type="log", title_text="Frequency (GHz)")
    _fig_cas.update_yaxes(title_text="Insertion loss (dB)", row=1, col=1)
    _fig_cas.update_yaxes(title_text="Impedance (Ω)", row=1, col=2)
    _fig_cas.update_layout(
        template="plotly_dark",
        height=440,
        title=f"Cascade of up to {_N} L-sections  (R={_R:.0f}Ω, L={_L*1e9:.1f}nH, C={_C*1e15:.0f}fF)",
        legend=dict(orientation="h", y=-0.18),
    )
    mo.ui.plotly(_fig_cas)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. $N$-Port Networks and the Matrix Formulation of Power

    While we have extensively discussed two-ports, the concepts easily generalize to $N$-ports.
    An $N$-port network has $N$ voltage/current pairs. These are collected into column vectors:
    $$\mathbf{V} = [V_1, V_2, \dots, V_N]^T, \quad \mathbf{I} = [I_1, I_2, \dots, I_N]^T$$

    The impedance matrix $\mathbf{Z}$ and admittance matrix $\mathbf{Y}$ generalize to $N \times N$ matrices:
    $$\mathbf{V} = \mathbf{Z}\mathbf{I}, \quad \mathbf{I} = \mathbf{Y}\mathbf{V}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Power, Waves, and Network Representations

    This notebook establishes the vocabulary used by every subsequent notebook:
    complex power, available power and mismatch, power-wave variables, the
    scattering matrix, conversions between network representations, signal flow
    graphs, and Mason's non-touching loop rule. No gain definitions or stability
    criteria appear here — those build on this foundation in notebook 03.

    ## 1. Complex Power from First Principles

    ### 1.1 Time-domain instantaneous power

    For a linear circuit driven at a single angular frequency $\omega$, every voltage
    and current is a sinusoid.  Write the terminal voltage and current as:

    $$v(t) = V_m \cos(\omega t + \phi_v), \qquad
      i(t) = I_m \cos(\omega t + \phi_i)$$

    The **instantaneous power** entering a one-port is:

    $$p(t) = v(t)\,i(t)
           = V_m I_m \cos(\omega t+\phi_v)\cos(\omega t+\phi_i)$$

    Using the product-to-sum identity $\cos A \cos B = \tfrac{1}{2}[\cos(A-B)+\cos(A+B)]$:

    $$p(t) = \underbrace{\frac{V_m I_m}{2}\cos(\phi_v - \phi_i)}_{\text{average power }P} + \underbrace{\frac{V_m I_m}{2}\cos(2\omega t + \phi_v + \phi_i)}_{\text{oscillates at }2\omega,\;\text{zero mean}}$$

    The second term oscillates at $2\omega$ and averages to zero over one cycle.
    Therefore the **average power** is:

    $$\boxed{P = \frac{V_m I_m}{2}\cos\theta}$$

    where $\theta = \phi_v - \phi_i$ is the phase difference between voltage and current.

    ### 1.2 Phasor convention

    We adopt the **peak-value phasor** convention: $\tilde{V} = V_m e^{j\phi_v}$,
    $\tilde{I} = I_m e^{j\phi_i}$.  Then:

    $$\frac{1}{2}\tilde{V}\tilde{I}^* = \frac{1}{2}V_m I_m e^{j(\phi_v - \phi_i)}
      = \frac{V_m I_m}{2}(\cos\theta + j\sin\theta)$$

    This is the **complex power**:

    $$\boxed{S \;\equiv\; P + jQ \;=\; \frac{1}{2}\tilde{V}\tilde{I}^*}$$

    | Quantity | Symbol | Unit | Meaning |
    |---|---|---|---|
    | Real part | $P = \operatorname{Re}(S)$ | W | Average (active) power dissipated |
    | Imaginary part | $Q = \operatorname{Im}(S)$ | var | Reactive power (energy oscillating back and forth) |
    | Magnitude | $|S|$ | VA | Apparent power |

    ### 1.3 Complex power in terms of $Z_L$

    If the load impedance is $Z_L = R_L + jX_L$ then $\tilde{I} = \tilde{V}/Z_L$, so:

    $$S = \frac{1}{2}\tilde{V}\tilde{I}^* = \frac{1}{2}\tilde{V}\frac{\tilde{V}^*}{Z_L^*}
        = \frac{|\tilde{V}|^2}{2 Z_L^*}$$

    Separating real and imaginary parts:

    $$P = \frac{|\tilde{V}|^2}{2}\cdot\frac{R_L}{|Z_L|^2}, \qquad
      Q = \frac{|\tilde{V}|^2}{2}\cdot\frac{-X_L}{|Z_L|^2}$$

    This confirms that only the **resistive** part of the load dissipates real power.
    A purely reactive load ($R_L = 0$) delivers $P = 0$ regardless of $|\tilde{V}|$.

    Equivalently, using the current phasor $\tilde{I}$:

    $$S = \frac{1}{2}Z_L|\tilde{I}|^2 \implies P = \frac{1}{2}R_L|\tilde{I}|^2$$

    ### 1.4 Algebraic Formulation of Power for $N$-Ports

    Using matrices, the total complex power entering an $N$-port is the sum of the power at each port:

    $$S = \frac{1}{2} \sum_{i=1}^{N} \tilde{V}_i \tilde{I}_i^* = \frac{1}{2} \mathbf{I}^H \mathbf{V} = \frac{1}{2} \mathbf{V}^T \mathbf{I}^*$$

    where $\mathbf{I}^H = (\mathbf{I}^*)^T$ is the Hermitian transpose. The real active power is:

    $$P = \operatorname{Re}(S) = \frac{1}{2} \operatorname{Re}(\mathbf{I}^H \mathbf{V}) = \frac{1}{4} (\mathbf{I}^H \mathbf{V} + \mathbf{V}^H \mathbf{I})$$

    Substituting $\mathbf{V} = \mathbf{Z}\mathbf{I}$:

    $$P = \frac{1}{4} (\mathbf{I}^H \mathbf{Z} \mathbf{I} + \mathbf{I}^H \mathbf{Z}^H \mathbf{I}) = \frac{1}{2} \mathbf{I}^H \left( \frac{\mathbf{Z} + \mathbf{Z}^H}{2} \right) \mathbf{I}$$

    The matrix $\frac{1}{2}(\mathbf{Z} + \mathbf{Z}^H)$ is the **Hermitian part** of $\mathbf{Z}$. Thus, power dissipation is governed strictly by the Hermitian part of the impedance (or admittance) matrix.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Available Power of a Thévenin Source

    ### 2.1 The circuit

    A real source is modelled as an ideal voltage phasor $\tilde{V}_s$ in series with a
    **source impedance** $Z_s = R_s + jX_s$ (with $R_s > 0$).  The source drives a
    load $Z_L = R_L + jX_L$.

    The current drawn is:

    $$\tilde{I} = \frac{\tilde{V}_s}{Z_s + Z_L}$$

    The power delivered to $Z_L$:

    $$P_L = \frac{1}{2}R_L|\tilde{I}|^2
          = \frac{|\tilde{V}_s|^2}{2}\cdot\frac{R_L}{|Z_s + Z_L|^2}$$

    Expanding $|Z_s + Z_L|^2 = (R_s+R_L)^2 + (X_s+X_L)^2$:

    $$P_L = \frac{|\tilde{V}_s|^2}{2}\cdot\frac{R_L}{(R_s+R_L)^2 + (X_s+X_L)^2}$$

    ### 2.2 Maximising $P_L$ — conjugate match

    **Step 1 — optimise the reactive part.**
    For fixed $R_L$, $P_L$ is maximised by minimising the denominator with respect to
    $X_L$.  The denominator is $(R_s+R_L)^2 + (X_s+X_L)^2$, which is minimised when:

    $$X_L = -X_s$$

    This **cancels** the source reactance.

    **Step 2 — optimise the resistive part.**
    With $X_L = -X_s$, $P_L = |\tilde{V}_s|^2 R_L / [2(R_s+R_L)^2]$.
    Differentiating with respect to $R_L$ and setting to zero:

    $$\frac{dP_L}{dR_L} = 0 \;\implies\; \frac{(R_s+R_L)^2 - 2R_L(R_s+R_L)}{(R_s+R_L)^4}
      = 0 \;\implies\; R_L = R_s$$

    **Conjugate match condition:** $Z_L = Z_s^* = R_s - jX_s$.

    ### 2.3 Available power formula

    Substituting $Z_L = Z_s^*$ (i.e. $R_L = R_s$, $X_L = -X_s$):

    $$\boxed{P_{\text{avs}} = \frac{|\tilde{V}_s|^2}{8 R_s}}$$

    This is the **available power** of the source — the maximum power that can
    *ever* be extracted, independent of what load is connected.

    **Alternative form using Norton equivalent:**
    A Norton source has $\tilde{I}_s = \tilde{V}_s / Z_s$ and $Z_s$ in parallel.
    Then $P_{\text{avs}} = \frac{1}{2}|{\tilde{I}_s}|^2 R_s$.
    Both forms are equivalent by Thévenin–Norton duality.

    ### 2.4 Mismatch loss

    When the load is not conjugate-matched, the **mismatch factor** $M$ quantifies
    how much power is lost:

    $$M = \frac{P_L}{P_{\text{avs}}}
        = \frac{4 R_s R_L}{|Z_s + Z_L|^2}
        = 1 - |\Gamma|^2$$

    where $\Gamma = (Z_L - Z_s^*)/(Z_L + Z_s)$ is the **reflection coefficient**
    referenced to the source impedance conjugate.

    For a normalisation to a real characteristic impedance $Z_0$, the port reflection coefficients are defined as $\Gamma_S = (Z_s - Z_0)/(Z_s + Z_0)$ and $\Gamma_L = (Z_L - Z_0)/(Z_L + Z_0)$. We can invert these to express the impedances in terms of $\Gamma$:

    $$Z_s = Z_0 \frac{1 + \Gamma_S}{1 - \Gamma_S}, \qquad Z_L = Z_0 \frac{1 + \Gamma_L}{1 - \Gamma_L}$$

    The real parts of the impedances (resistances) are derived by taking $R = (Z + Z^*)/2$. For the source resistance $R_s$:

    $$R_s = \frac{Z_0}{2} \left[ \frac{1 + \Gamma_S}{1 - \Gamma_S} + \frac{1 + \Gamma_S^*}{1 - \Gamma_S^*} \right] = \frac{Z_0}{2} \frac{(1+\Gamma_S)(1-\Gamma_S^*) + (1+\Gamma_S^*)(1-\Gamma_S)}{|1-\Gamma_S|^2} = Z_0 \frac{1 - |\Gamma_S|^2}{|1-\Gamma_S|^2}$$

    Similarly, the load resistance is $R_L = Z_0 (1 - |\Gamma_L|^2) / |1-\Gamma_L|^2$.

    The denominator of the mismatch factor, $|Z_s + Z_L|^2$, can be evaluated by summing the impedances:

    $$Z_s + Z_L = Z_0 \left( \frac{1 + \Gamma_S}{1 - \Gamma_S} + \frac{1 + \Gamma_L}{1 - \Gamma_L} \right) = Z_0 \frac{(1+\Gamma_S)(1-\Gamma_L) + (1+\Gamma_L)(1-\Gamma_S)}{(1-\Gamma_S)(1-\Gamma_L)} = 2Z_0 \frac{1 - \Gamma_S\Gamma_L}{(1-\Gamma_S)(1-\Gamma_L)}$$

    Squaring this yields $|Z_s + Z_L|^2 = 4Z_0^2 |1 - \Gamma_S\Gamma_L|^2 / (|1-\Gamma_S|^2 |1-\Gamma_L|^2)$.

    Substituting $R_s$, $R_L$, and $|Z_s + Z_L|^2$ back into $M = 4 R_s R_L / |Z_s + Z_L|^2$, the $4Z_0^2$ scalars and the $|1-\Gamma|^2$ denominator fractions all cancel out beautifully, yielding:

    $$\boxed{M = \frac{(1-|\Gamma_S|^2)(1-|\Gamma_L|^2)}{|1-\Gamma_S\Gamma_L|^2}}$$

    This is the exact mismatch formula for a direct connection between a source with
    reflection $\Gamma_S$ and a load $\Gamma_L$.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Interactive: Mismatch Loss vs Source and Load Impedance
    """)
    return


@app.cell
def _(mo):
    mm_Rs  = mo.ui.slider(1,   200,  step=1,   value=50,  label="Rs (Ω)",  show_value=True)
    mm_Xs  = mo.ui.slider(-200, 200, step=5,   value=0,   label="Xs (Ω)",  show_value=True)
    mm_RL  = mo.ui.slider(1,   200,  step=1,   value=50,  label="RL (Ω)",  show_value=True)
    mm_XL  = mo.ui.slider(-200, 200, step=5,   value=0,   label="XL (Ω)",  show_value=True)
    mm_Vs  = mo.ui.slider(0.01, 2.0, step=0.01, value=1.0, label="|Vs| (V peak)", show_value=True)
    mo.hstack([
        mo.vstack([mo.md("**Source**"), mm_Rs, mm_Xs]),
        mo.vstack([mo.md("**Load**"),   mm_RL, mm_XL]),
        mo.vstack([mo.md("**Source voltage**"), mm_Vs]),
    ], gap="2rem")
    return mm_RL, mm_Rs, mm_Vs, mm_XL, mm_Xs


@app.cell
def _(mm_RL, mm_Rs, mm_Vs, mm_XL, mm_Xs, mo, np):
    _Rs = float(mm_Rs.value)
    _Xs = float(mm_Xs.value)
    _RL = float(mm_RL.value)
    _XL = float(mm_XL.value)
    _Vs = float(mm_Vs.value)

    _Zs = complex(_Rs, _Xs)
    _ZL = complex(_RL, _XL)

    # Power delivered to load
    _I = _Vs / (_Zs + _ZL)
    _PL = 0.5 * _RL * abs(_I)**2

    # Available power (conjugate match)
    _Pavs = _Vs**2 / (8 * _Rs)

    # Mismatch factor
    _M = _PL / _Pavs if _Pavs > 1e-30 else 0.0
    _M_db = 10 * np.log10(max(_M, 1e-30))

    # Reflection coefficients (Z0 = 50 Ω)
    _Z0 = 50.0
    _GS = (_Zs - _Z0) / (_Zs + _Z0)
    _GL = (_ZL - _Z0) / (_ZL + _Z0)
    _M2 = (1 - abs(_GS)**2) * (1 - abs(_GL)**2) / abs(1 - _GS * _GL)**2

    # Reactive power
    _Q_source = 0.5 * _Xs * abs(_I)**2
    _Q_load   = 0.5 * _XL * abs(_I)**2
    _S_total  = 0.5 * _Vs * abs(_I)

    mo.md(f"""
    ### Power Budget

    | Quantity | Value |
    |---|---|
    | Source impedance $Z_s$ | {_Rs:.1f} + j{_Xs:.1f} Ω |
    | Load impedance $Z_L$ | {_RL:.1f} + j{_XL:.1f} Ω |
    | Current $\\tilde{{I}}$ | {abs(_I)*1e3:.3f} mA ∠ {np.degrees(np.angle(_I)):.1f}° |
    | Power to load $P_L$ | {_PL*1e3:.4f} mW |
    | Available power $P_{{\\text{{avs}}}}$ | {_Pavs*1e3:.4f} mW |
    | **Mismatch factor $M$** | **{_M:.4f}** ({_M_db:.2f} dB) |
    | $\\Gamma_S$ (re 50 Ω) | {abs(_GS):.4f} ∠ {np.degrees(np.angle(_GS)):.1f}° |
    | $\\Gamma_L$ (re 50 Ω) | {abs(_GL):.4f} ∠ {np.degrees(np.angle(_GL)):.1f}° |
    | Mismatch (Γ formula) | {_M2:.4f} (cross-check) |
    | Reactive power $Q_L$ | {_Q_load*1e3:.4f} mvar |
    | Conjugate match condition | $Z_L = Z_s^*$ = {_Rs:.1f} − j{_Xs:.1f} Ω |

    {"✓ Load is conjugate-matched — maximum power transfer" if abs(_ZL - np.conj(_Zs)) < 0.5 else f"Mismatch loss = {-_M_db:.2f} dB below maximum"}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Power Waves

    ### 3.1 Why wave variables?

    At microwave frequencies, voltages and currents become **position-dependent**
    along a transmission line: the voltage you measure depends on *where* you probe.
    This makes them inconvenient as primary network variables.

    Instead, we decompose the voltage and current at each port into forward- and
    backward-travelling **power waves** that are uniquely defined at any reference
    plane.

    ### 3.2 Definitions

    Let $V_i$ and $I_i$ be the total voltage and current at port $i$, and let
    $Z_0$ be a purely real reference impedance (usually 50 Ω). Define the power waves:

    $$a_i = \frac{V_i + Z_0 I_i}{2\sqrt{Z_0}}, \qquad
      b_i = \frac{V_i - Z_0 I_i}{2\sqrt{Z_0}}$$

    **Physical interpretation of the power carried by each wave:**

    $$ \begin{aligned}
    \frac{1}{2}|a_i|^2 &= \text{time-averaged power flowing \textbf{into} port } i \\
    \frac{1}{2}|b_i|^2 &= \text{time-averaged power flowing \textbf{out of} port } i
    \end{aligned} $$

    *Proof.* By definition, the net time-averaged power entering port $i$ is $P_i = \frac{1}{2}\operatorname{Re}(V_i I_i^*)$.
    We want to prove that this equals $\frac{1}{2}|a_i|^2 - \frac{1}{2}|b_i|^2$.

    Let's substitute the wave definitions into the right-hand side and evaluate $|a_i|^2 - |b_i|^2$.
    Because $|x|^2 = x x^*$, we can expand the numerators using the identity $|A+B|^2 - |A-B|^2 = 4\operatorname{Re}(A B^*)$:

    $$ \begin{aligned}
    |a_i|^2 - |b_i|^2
      &= \left| \frac{V_i + Z_0 I_i}{2\sqrt{Z_0}} \right|^2 - \left| \frac{V_i - Z_0 I_i}{2\sqrt{Z_0}} \right|^2 \\
      &= \frac{1}{4Z_0}\Bigl[ |V_i + Z_0 I_i|^2 - |V_i - Z_0 I_i|^2 \Bigr] \\
      &= \frac{1}{4Z_0} \cdot 4\operatorname{Re}(V_i (Z_0 I_i)^*) \\
      &= \frac{1}{Z_0}\operatorname{Re}(V_i I_i^* Z_0^*)
    \end{aligned} $$

    Since our reference impedance $Z_0$ is purely real, its complex conjugate is simply $Z_0^* = Z_0$. A real scalar can be factored out of the real-part operator, allowing it to cancel with the $1/Z_0$ term:

    $$ \begin{aligned}
    |a_i|^2 - |b_i|^2
      &= \frac{1}{Z_0} \cdot Z_0 \operatorname{Re}(V_i I_i^*) \\
      &= \operatorname{Re}(V_i I_i^*)
    \end{aligned} $$

    Dividing both sides by 2, we recover the original net power equation:

    $$ \frac{1}{2}|a_i|^2 - \frac{1}{2}|b_i|^2 = \frac{1}{2}\operatorname{Re}(V_i I_i^*) = P_i \quad \blacksquare $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The Scattering Matrix

    The **scattering matrix** relates reflected/transmitted waves to incident waves:

    $$\begin{pmatrix} b_1 \\ b_2 \end{pmatrix}
      = \begin{pmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix}
        \begin{pmatrix} a_1 \\ a_2 \end{pmatrix}$$

    Each S-parameter is measured with only one port excited and the other terminated in $Z_0$:

    | Parameter | Condition | Meaning |
    |---|---|---|
    | $S_{11} = b_1/a_1$ | $a_2 = 0$ (port 2 matched) | Input reflection coefficient |
    | $S_{21} = b_2/a_1$ | $a_2 = 0$ | Forward transmission (gain if $\|S_{21}\|>1$) |
    | $S_{12} = b_1/a_2$ | $a_1 = 0$ (port 1 matched) | Reverse transmission (feedback) |
    | $S_{22} = b_2/a_2$ | $a_1 = 0$ | Output reflection coefficient |

    **Properties of the S-matrix:**

    - **Reciprocal network** ($Y_{12}=Y_{21}$): $S_{12} = S_{21}$
    - **Lossless network**: $\mathbf{S}^H \mathbf{S} = \mathbf{I}$ (unitary), which implies
      $|S_{11}|^2 + |S_{21}|^2 = 1$ and $|S_{22}|^2 + |S_{12}|^2 = 1$
    - **Matched network** (e.g. ideal amplifier): $S_{11} = S_{22} = 0$

    The source and load terminations are described by their own reflection coefficients
    (defined in §2.4):

    $$\Gamma_S = \frac{Z_s - Z_0}{Z_s + Z_0}, \qquad \Gamma_L = \frac{Z_L - Z_0}{Z_L + Z_0}$$

    For a source with Thévenin voltage $\tilde{V}_s$ behind $Z_s$, the wave launched into
    a matched $Z_0$ load — denoted $b_S$ — relates to the available power through:

    $$P_{\text{avs}} = \frac{|\tilde{V}_s|^2}{8 R_s} = \frac{1}{2} \frac{|b_S|^2}{1 - |\Gamma_S|^2}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Conversion Between Network Representations

    ### 5.1 S ↔ Z Conversion

    To convert between S-parameters and Z-parameters, we express the network's voltages and currents in terms of the incident ($\mathbf{a}$) and reflected ($\mathbf{b}$) wave vectors.

    From our definitions of the power waves, we can add and subtract $\mathbf{a}$ and $\mathbf{b}$ to isolate voltage and current. Assuming a real reference impedance $Z_0$:

    $$ \begin{aligned}
    \mathbf{a} + \mathbf{b} &= \frac{\mathbf{V}}{\sqrt{Z_0}} \implies \mathbf{V} = \sqrt{Z_0}(\mathbf{a} + \mathbf{b}) \\
    \mathbf{a} - \mathbf{b} &= \frac{Z_0 \mathbf{I}}{\sqrt{Z_0}} \implies \mathbf{I} = \frac{1}{\sqrt{Z_0}}(\mathbf{a} - \mathbf{b})
    \end{aligned} $$

    The Z-parameter matrix relates total voltage to total current via $\mathbf{V} = \mathbf{Z}\mathbf{I}$. Substituting our wave expressions into this relationship:

    $$\sqrt{Z_0}(\mathbf{a}+\mathbf{b}) = \mathbf{Z} \left[ \frac{1}{\sqrt{Z_0}}(\mathbf{a}-\mathbf{b}) \right]$$

    Multiplying both sides by $\sqrt{Z_0}$ to clear the fraction:

    $$Z_0(\mathbf{a}+\mathbf{b}) = \mathbf{Z}(\mathbf{a}-\mathbf{b})$$

    By definition, the S-matrix relates reflected waves to incident waves via $\mathbf{b} = \mathbf{S}\mathbf{a}$. Substituting $\mathbf{b}$:

    $$Z_0(\mathbf{I}+\mathbf{S})\mathbf{a} = \mathbf{Z}(\mathbf{I}-\mathbf{S})\mathbf{a}$$

    Since this identity must hold for any arbitrary incident wave vector $\mathbf{a}$, we can drop $\mathbf{a}$ and solve for $\mathbf{Z}$:

    $$\boxed{\mathbf{Z} = Z_0(\mathbf{I}+\mathbf{S})(\mathbf{I}-\mathbf{S})^{-1}}$$

    To find $\mathbf{S}$ in terms of $\mathbf{Z}$, we simply rearrange the equation to solve for $\mathbf{S}$:

    $$\boxed{\mathbf{S} = (\mathbf{Z} - Z_0\mathbf{I})(\mathbf{Z} + Z_0\mathbf{I})^{-1}}$$

    *Sanity check:* If a network is perfectly matched to the reference impedance, its impedance matrix is $\mathbf{Z} = Z_0 \mathbf{I}$. Substituting this into the equation yields $\mathbf{S} = (Z_0\mathbf{I} - Z_0\mathbf{I})(2Z_0\mathbf{I})^{-1} = \mathbf{0}$, which correctly confirms that a matched network has zero reflections.

    ### 5.2 S ↔ Y

    Since $\mathbf{Y} = \mathbf{Z}^{-1}$, we can invert the Z-formula.  Starting from
    $\mathbf{Z} = Z_0(\mathbf{I}+\mathbf{S})(\mathbf{I}-\mathbf{S})^{-1}$:

    $$\mathbf{Y} = \frac{1}{Z_0}(\mathbf{I}-\mathbf{S})(\mathbf{I}+\mathbf{S})^{-1}$$

    $$\mathbf{S} = (\mathbf{I} - Z_0\mathbf{Y})(\mathbf{I} + Z_0\mathbf{Y})^{-1}$$

    ### 5.3 S → ABCD

    For a two-port, the ABCD (transmission) parameters relate port 1 to port 2:
    $$\begin{pmatrix} V_1 \\ I_1 \end{pmatrix}
      = \begin{pmatrix} A & B \\ C & D \end{pmatrix}
        \begin{pmatrix} V_2 \\ -I_2 \end{pmatrix}$$

    Converting from S-parameters with reference impedance $Z_0$:

    $$A = \frac{(1+S_{11})(1-S_{22})+S_{12}S_{21}}{2S_{21}}, \qquad
      B = Z_0\frac{(1+S_{11})(1+S_{22})-S_{12}S_{21}}{2S_{21}}$$

    $$C = \frac{1}{Z_0}\frac{(1-S_{11})(1-S_{22})-S_{12}S_{21}}{2S_{21}}, \qquad
      D = \frac{(1-S_{11})(1+S_{22})+S_{12}S_{21}}{2S_{21}}$$

    *Check:* For the identity network ($S_{11}=S_{22}=0$, $S_{21}=S_{12}=1$):
    $A = (1)(1)+1)/(2) = 1$, $B=0$, $C=0$, $D=1$. ✓
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 16. Summary

    We have established the fundamentals of network parameters (Z, Y, H, ABCD) and generalized them to $N$-ports and Scattering parameters (S-matrix). Furthermore, the Hermitian matrix formulation elegantly captures power dissipation across an $N$-port.

    ---

    **Next:** [02 — Power Gains, Signal Flow Graphs, and Mason's Gain](02_power_gain_definitions.py)
    """)
    return


