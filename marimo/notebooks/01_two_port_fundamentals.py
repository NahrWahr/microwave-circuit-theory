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
    ## 7. Interactive Calculator

    Choose a canonical two-port topology, set its element values, and inspect all
    four parameter sets.  The computations below follow the definitions exactly.
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
        controls = mo.vstack([
            mo.md("**Series element:** $Z_s = R + j\\omega L \\;\\|\\; (1/j\\omega C)$ if C>0"),
            mo.hstack([r_slider, l_slider, c_slider]),
        ])
    elif t == "shunt":
        r_slider = mo.ui.slider(1, 10000, step=10, value=1000, label="R (Ω)", show_value=True)
        l_slider = mo.ui.slider(0, 10, step=0.1, value=0.0, label="L (nH)", show_value=True)
        c_slider = mo.ui.slider(0, 1000, step=10, value=100, label="C (fF)", show_value=True)
        controls = mo.vstack([
            mo.md("**Shunt element:** $Y_p = 1/R + j\\omega C + 1/(j\\omega L)$ if L>0"),
            mo.hstack([r_slider, l_slider, c_slider]),
        ])
    elif t == "tee":
        r_slider  = mo.ui.slider(0, 500, step=5, value=25,  label="Z1: R (Ω)", show_value=True)
        l_slider  = mo.ui.slider(0, 5,   step=0.1, value=1.0, label="Z1: L (nH)", show_value=True)
        c_slider  = mo.ui.slider(1, 1000, step=10, value=200, label="Z3 (shunt): R (Ω)", show_value=True)
        controls = mo.vstack([
            mo.md("**T-network:** Z1 = series arm 1, Z3 = shunt arm, Z2 = series arm 2 (= Z1 for symmetry here)"),
            mo.hstack([r_slider, l_slider, c_slider]),
        ])
    elif t == "pi":
        r_slider  = mo.ui.slider(10, 2000, step=10, value=500, label="Y1 shunt (Ω)", show_value=True)
        l_slider  = mo.ui.slider(1, 500,  step=5,  value=50,  label="Z_series (Ω)", show_value=True)
        c_slider  = mo.ui.slider(10, 2000, step=10, value=500, label="Y2 shunt (Ω)", show_value=True)
        controls = mo.vstack([
            mo.md("**π-network:** Y1 = input shunt resistor, Z_s = series arm, Y2 = output shunt resistor"),
            mo.hstack([r_slider, l_slider, c_slider]),
        ])
    else:  # lsec
        r_slider = mo.ui.slider(1, 500,  step=5,  value=50,  label="Series R (Ω)", show_value=True)
        l_slider = mo.ui.slider(0, 10,   step=0.1, value=1.0, label="Series L (nH)", show_value=True)
        c_slider = mo.ui.slider(1, 1000, step=10, value=100, label="Shunt C (fF)", show_value=True)
        controls = mo.vstack([
            mo.md("**L-section:** series $Z = R + j\\omega L$, shunt $Y = j\\omega C$"),
            mo.hstack([r_slider, l_slider, c_slider]),
        ])

    freq_slider = mo.ui.slider(
        steps=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
        value=10,
        label="Frequency (GHz)",
        show_value=True,
    )
    controls_full = mo.vstack([controls, freq_slider])
    controls_full
    return c_slider, controls, controls_full, freq_slider, l_slider, r_slider, t


@app.cell
def _(c_slider, freq_slider, l_slider, mo, np, r_slider, t):
    f_Hz = freq_slider.value * 1e9
    w = 2 * np.pi * f_Hz

    R_val = float(r_slider.value)
    L_val = float(l_slider.value) * 1e-9
    C_val = float(c_slider.value) * 1e-15

    # Build the 2x2 Z-matrix for each topology
    if t == "series":
        Zs = R_val + 1j * w * L_val
        if C_val > 0:
            Zs = 1.0 / (1.0 / Zs + 1j * w * C_val)
        Z = np.array([[Zs, Zs], [Zs, Zs]])

    elif t == "shunt":
        Yp = 1.0 / R_val
        if C_val > 0:
            Yp += 1j * w * C_val
        if L_val > 0:
            Yp += 1.0 / (1j * w * L_val)
        # Shunt two-port: Y = [[Yp, -Yp],[-Yp, Yp]]
        Y_mat = np.array([[Yp, -Yp], [-Yp, Yp]])
        # Z = Y^{-1} if invertible
        try:
            Z = np.linalg.inv(Y_mat)
        except np.linalg.LinAlgError:
            Z = np.full((2, 2), np.inf)

    elif t == "tee":
        Z1 = R_val + 1j * w * L_val
        Z2 = Z1  # symmetric T
        Z3 = float(c_slider.value)  # reuse c_slider as shunt resistor
        Z = np.array([[Z1 + Z3, Z3], [Z3, Z2 + Z3]])

    elif t == "pi":
        Y1 = 1.0 / R_val
        Zs = float(l_slider.value)  # series arm in Ω
        Y2 = 1.0 / float(c_slider.value)
        # π: Y11=Y1+1/Zs, Y12=Y21=-1/Zs, Y22=Y2+1/Zs
        Ys = 1.0 / Zs if Zs != 0 else 1e12
        Y_mat = np.array([[Y1 + Ys, -Ys], [-Ys, Y2 + Ys]])
        try:
            Z = np.linalg.inv(Y_mat)
        except np.linalg.LinAlgError:
            Z = np.full((2, 2), np.inf)

    else:  # lsec
        Zser = R_val + 1j * w * L_val
        Ysh  = 1j * w * C_val
        # L-section: series arm then shunt arm
        # Z11 = Zser + 1/Ysh, Z12=Z21=1/Ysh, Z22=1/Ysh
        Zsh = 1.0 / Ysh if abs(Ysh) > 1e-30 else 1e12
        Z = np.array([[Zser + Zsh, Zsh], [Zsh, Zsh]])

    # Derive Y, ABCD from Z
    detZ = Z[0, 0] * Z[1, 1] - Z[0, 1] * Z[1, 0]

    def fmt_complex(z, unit=""):
        mag = abs(z)
        ang = np.degrees(np.angle(z))
        re  = z.real
        im  = z.imag
        return f"{re:+.4f}{im:+.4f}j {unit}  →  |{mag:.4f}|∠{ang:.1f}°"

    Z11, Z12, Z21, Z22 = Z[0,0], Z[0,1], Z[1,0], Z[1,1]

    if abs(detZ) > 1e-30:
        Y_mat = np.linalg.inv(Z)
        Y11, Y12, Y21, Y22 = Y_mat[0,0], Y_mat[0,1], Y_mat[1,0], Y_mat[1,1]
        y_valid = True
    else:
        y_valid = False

    if abs(Z21) > 1e-30:
        A = Z11 / Z21
        B = detZ / Z21
        C_abcd = 1.0 / Z21
        D = Z22 / Z21
        abcd_valid = True
    else:
        abcd_valid = False

    reciprocal = abs(Z12 - Z21) < 1e-6 * max(abs(Z12), abs(Z21), 1e-30)

    lines = [
        f"### Results at f = {freq_slider.value} GHz\n",
        f"**Topology:** {t}  |  "
        f"**Reciprocal:** {'✓  ($Z_{{12}}=Z_{{21}}$)' if reciprocal else '✗ (active/non-reciprocal)'}",
        "",
        "#### Z-parameters (Ω)",
        f"- $Z_{{11}}$ = {fmt_complex(Z11, 'Ω')}",
        f"- $Z_{{12}}$ = {fmt_complex(Z12, 'Ω')}",
        f"- $Z_{{21}}$ = {fmt_complex(Z21, 'Ω')}",
        f"- $Z_{{22}}$ = {fmt_complex(Z22, 'Ω')}",
        f"- $\\Delta_Z = Z_{{11}}Z_{{22}}-Z_{{12}}Z_{{21}}$ = {fmt_complex(detZ, 'Ω²')}",
        "",
    ]
    if y_valid:
        lines += [
            "#### Y-parameters (S = Ω⁻¹)",
            f"- $Y_{{11}}$ = {fmt_complex(Y11, 'S')}",
            f"- $Y_{{12}}$ = {fmt_complex(Y12, 'S')}",
            f"- $Y_{{21}}$ = {fmt_complex(Y21, 'S')}",
            f"- $Y_{{22}}$ = {fmt_complex(Y22, 'S')}",
            "",
        ]
    else:
        lines.append("#### Y-parameters: **singular** (det Z = 0); Y = Z⁻¹ does not exist for this topology.\n")

    if abcd_valid:
        lines += [
            "#### ABCD-parameters",
            f"- $A$ = {fmt_complex(A, '')}",
            f"- $B$ = {fmt_complex(B, 'Ω')}",
            f"- $C$ = {fmt_complex(C_abcd, 'S')}",
            f"- $D$ = {fmt_complex(D, '')}",
            f"- $AD-BC$ = {fmt_complex(A*D - B*C_abcd, '')}  (= 1 iff reciprocal)",
        ]
    else:
        lines.append("#### ABCD-parameters: $Z_{21}=0$; cascade matrix undefined (no transmission).")

    mo.md("\n".join(lines))
    return (
        A,
        B,
        C_abcd,
        D,
        L_val,
        R_val,
        Y11,
        Y12,
        Y21,
        Y22,
        Z,
        Z11,
        Z12,
        Z21,
        Z22,
        abcd_valid,
        detZ,
        f_Hz,
        fmt_complex,
        reciprocal,
        w,
        y_valid,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Frequency Sweep — Parameter Magnitudes

    The plot below sweeps frequency from 100 MHz to 100 GHz and shows the Z-parameter
    magnitudes and phases for the selected topology and element values.
    """)
    return


@app.cell
def _(C_abcd, C_val, L_val, R_val, go, make_subplots, mo, np, t):
    freqs = np.logspace(8, 11, 500)  # 100 MHz to 100 GHz
    ws = 2 * np.pi * freqs

    Z11_f = np.zeros(len(ws), dtype=complex)
    Z12_f = np.zeros(len(ws), dtype=complex)
    Z21_f = np.zeros(len(ws), dtype=complex)
    Z22_f = np.zeros(len(ws), dtype=complex)

    for _i, _w in enumerate(ws):
        if t == "series":
            _Zs = R_val + 1j * _w * L_val
            if C_val > 0:
                _Zs = 1.0 / (1.0/_Zs + 1j*_w*C_val)
            _Z = np.array([[_Zs, _Zs], [_Zs, _Zs]])
        elif t == "shunt":
            _Yp = 1.0/R_val
            if C_val > 0:
                _Yp += 1j*_w*C_val
            if L_val > 0:
                _Yp += 1.0/(1j*_w*L_val)
            _Ym = np.array([[_Yp, -_Yp],[-_Yp, _Yp]])
            try:
                _Z = np.linalg.inv(_Ym)
            except Exception:
                _Z = np.full((2,2), np.nan+0j)
        elif t == "tee":
            _Z1 = R_val + 1j*_w*L_val
            _Z3 = float(C_abcd) # Reuse C_abcd if needed or logic
            _Z = np.array([[_Z1+_Z3, _Z3],[_Z3, _Z1+_Z3]])
        elif t == "pi":
            _Y1 = 1.0/R_val
            _Zs_pi = float(L_val) if abs(L_val) > 1e-15 else 1e-9
            _Y2 = 1.0/float(C_abcd) if abs(C_abcd) > 1e-15 else 0.0
            _Ys = 1.0/_Zs_pi
            _Ym = np.array([[_Y1+_Ys, -_Ys],[-_Ys, _Y2+_Ys]])
            try:
                _Z = np.linalg.inv(_Ym)
            except Exception:
                _Z = np.full((2,2), np.nan+0j)
        else:  # lsec
            _Zser = R_val + 1j*_w*L_val
            _Ysh  = 1j*_w*C_val
            _Zsh  = 1.0/_Ysh if abs(_Ysh) > 1e-30 else 1e12
            _Z = np.array([[_Zser+_Zsh, _Zsh],[_Zsh, _Zsh]])

        Z11_f[_i] = _Z[0,0]
        Z12_f[_i] = _Z[0,1]
        Z21_f[_i] = _Z[1,0]
        Z22_f[_i] = _Z[1,1]

    fGHz = freqs / 1e9

    def _safe_db(arr):
        mag = np.abs(arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 20 * np.log10(np.where(mag > 0, mag, np.nan))
        return db

    fig_sweep = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "|Z-parameters| (dB·Ω)", "∠Z-parameters (°)"
        ],
        horizontal_spacing=0.10,
    )

    _colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    _zlabels = ["|Z₁₁|", "|Z₁₂|", "|Z₂₁|", "|Z₂₂|"]
    _zarrs   = [Z11_f, Z12_f, Z21_f, Z22_f]
    for _lbl, _arr, _col in zip(_zlabels, _zarrs, _colors):
        fig_sweep.add_trace(
            go.Scatter(x=fGHz, y=_safe_db(_arr), name=_lbl,
                       line=dict(color=_col), legendgroup=_lbl),
            row=1, col=1)
        fig_sweep.add_trace(
            go.Scatter(x=fGHz, y=np.degrees(np.angle(_arr)), name=_lbl+" ∠",
                       line=dict(color=_col, dash="dot"), legendgroup=_lbl, showlegend=False),
            row=1, col=2)

    fig_sweep.update_xaxes(type="log", title_text="Frequency (GHz)")
    fig_sweep.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig_sweep.update_yaxes(title_text="Phase (°)", row=1, col=2)
    fig_sweep.update_layout(
        template="plotly_dark",
        height=400,
        title="Two-Port Parameters vs Frequency",
        legend=dict(orientation="h", y=-0.2),
    )
    mo.ui.plotly(fig_sweep)
    return Z11_f, Z12_f, Z21_f, Z22_f, fGHz, fig_sweep, freqs, ws


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. ABCD Parameters and Cascaded Two-Port Stages

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
    ---

    **Next:** [02 — Power, Waves, and Network Representations](02_power_gain_definitions.py)
    """)
    return


if __name__ == "__main__":
    app.run()
