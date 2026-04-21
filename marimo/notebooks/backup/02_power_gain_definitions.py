import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    return go, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # 02 — Power, Waves, and Network Representations

    This notebook establishes the foundational vocabulary for the project: 
    complex power, mismatch loss, power waves, the scattering matrix, 
    signal flow graphs, and Mason's non-touching loop rule.

    ---

    ## 1. Complex Power from First Principles

    ### 1.1 Phasor representation
    In the frequency domain, we represent a time-harmonic voltage as:
    $v(t) = \text{Re}\{V e^{j\omega t}\}$
    where $V = |V|e^{j\phi}$ is the complex phasor.

    ### 1.2 Average Power
    The time-average power delivered to a load is:
    $$P = \frac{1}{2} \text{Re}\{V I^*\}$$
    Substituting $I = V/Z_L$, we get $P = \frac{1}{2} |V|^2 \frac{\text{Re}\{Z_L\}}{|Z_L|^2}$.

    ### 1.3 Maximum Power Transfer (Conjugate Match)
    For a source with impedance $Z_S = R_S + jX_S$, the maximum power available 
    from the source ($P_{avs}$) is delivered when $Z_L = Z_S^*$.
    $$P_{avs} = \frac{|V_S|^2}{8 R_S}$$

    ### 1.4 Mismatch Loss
    If $Z_L \neq Z_S^*$, the power delivered ($P_L$) is less than $P_{avs}$.
    The ratio $P_L / P_{avs}$ is the **mismatch loss**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Interactive: Load Match Explorer

    Adjust the source and load impedances to see how $P_L / P_{avs}$ changes.
    """)
    return


@app.cell
def _(mo):
    rs_slider = mo.ui.slider(1, 200, value=50, label="Rs (Ω)")
    xs_slider = mo.ui.slider(-100, 100, value=0, label="Xs (Ω)")
    rl_slider = mo.ui.slider(1, 200, value=50, label="Rl (Ω)")
    xl_slider = mo.ui.slider(-100, 100, value=0, label="Xl (Ω)")

    mo.hstack([
        mo.vstack([mo.md("**Source Zs**"), rs_slider, xs_slider]),
        mo.vstack([mo.md("**Load ZL**"), rl_slider, xl_slider]),
    ])
    return rl_slider, rs_slider, xl_slider, xs_slider


@app.cell
def _(mo, rl_slider, rs_slider, xl_slider, xs_slider):
    import numpy as np
    rs = rs_slider.value; xs = xs_slider.value
    rl = rl_slider.value; xl = xl_slider.value

    zs = rs + 1j*xs
    zl = rl + 1j*xl

    pavs = 1.0 / (8 * rs) # Normalised to |Vs|=1
    pl = 0.5 * (rl / abs(zs + zl)**2)

    ratio = pl / pavs
    mismatch_db = 10 * (0 if ratio <= 0 else -np.log10(ratio))

    mo.md(f"""
    **Mismatch Analysis:**
    - Ratio $P_L / P_{{avs}}$: **{ratio:.4f}**
    - Mismatch Loss: **{mismatch_db:.2f} dB**
    - Status: {'Perfect match' if ratio > 0.999 else 'Mismatched'}
    """)
    return mismatch_db, pavs, pl, ratio, rl, rs, xl, xs, zl, zs


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Power Waves (a and b)

    In microwave circuits, we prefer variables that represent propagating power. 
    Following Kurokawa (1965), we define incident ($a$) and reflected ($b$) power 
    waves at a port with reference impedance $Z_0$:

    $$a = \frac{V + Z_0 I}{2\sqrt{\text{Re}\{Z_0\}}}, \quad b = \frac{V - Z_0^* I}{2\sqrt{\text{Re}\{Z_0\}}}$$

    When $Z_0$ is real ($Z_0 = R_0$):
    $$a = \frac{V + R_0 I}{2\sqrt{R_0}}, \quad b = \frac{V - R_0 I}{2\sqrt{R_0}}$$

    Key property: The net power entering the port is $P = |a|^2 - |b|^2$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. The Scattering Matrix (S-Matrix)

    The S-parameters relate the incident and reflected waves across a multi-port network:
    $$\mathbf{b} = \mathbf{S} \mathbf{a}$$

    For a two-port:
    $$\begin{pmatrix} b_1 \\ b_2 \end{pmatrix} 
      = \begin{pmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix} 
        \begin{pmatrix} a_1 \\ a_2 \end{pmatrix}$$

    - $S_{11}$: Input reflection coefficient (with $a_2=0$, matched load).
    - $S_{21}$: Forward transmission coefficient (gain/loss).
    - $S_{12}$: Reverse transmission (isolation).
    - $S_{22}$: Output reflection coefficient.

    A network is **lossless** if $\mathbf{S}$ is unitary ($\mathbf{S}^\dagger \mathbf{S} = \mathbf{I}$).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Parameter Transformations (Z ↔ S)

    ### 5.1 Scattering to Impedance
    Given a reference impedance $Z_0$ (usually 50 Ω):
    $$\mathbf{Z} = Z_0 (\mathbf{I} + \mathbf{S})(\mathbf{I} - \mathbf{S})^{-1}$$

    ### 5.2 Impedance to Scattering
    $$\mathbf{S} = (\mathbf{Z} - Z_0\mathbf{I})(\mathbf{Z} + Z_0\mathbf{I})^{-1}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Signal Flow Graphs (SFG)

    Signal flow graphs provide a topological way to solve wave interactions. 
    Nodes represent waves ($a_i, b_i$) and directed branches represent 
    S-parameters.

    <div style="text-align: center; margin: 20px 0;">
    <svg width="450" height="200" viewBox="0 0 450 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>
      <g font-family="sans-serif" font-size="14" fill="currentColor">
        <circle cx="100" cy="150" r="15" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="100" y="155" text-anchor="middle">a<tspan dy="5" font-size="10">1</tspan></text>
        <circle cx="100" cy="50" r="15" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="100" y="55" text-anchor="middle">b<tspan dy="5" font-size="10">1</tspan></text>
        <circle cx="300" cy="150" r="15" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="155" text-anchor="middle">b<tspan dy="5" font-size="10">2</tspan></text>
        <circle cx="300" cy="50" r="15" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="55" text-anchor="middle">a<tspan dy="5" font-size="10">2</tspan></text>

        <!-- S11 -->
        <path d="M 100 135 L 100 65" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="90" y="100" text-anchor="end">S<tspan dy="5" font-size="10">11</tspan></text>
        
        <!-- S21 -->
        <path d="M 115 150 L 285 150" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="200" y="165" text-anchor="middle">S<tspan dy="5" font-size="10">21</tspan></text>

        <!-- S12 -->
        <path d="M 285 50 L 115 50" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="200" y="40" text-anchor="middle">S<tspan dy="5" font-size="10">12</tspan></text>
        
        <!-- S22 -->
        <path d="M 300 65 L 300 135" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="310" y="100" text-anchor="start">S<tspan dy="5" font-size="10">22</tspan></text>

        <!-- b_s and Γs -->
        <circle cx="20" cy="100" r="15" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="20" y="105" text-anchor="middle">b<tspan dy="5" font-size="10">S</tspan></text>
        <line x1="35" y1="100" x2="85" y2="150" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        
        <path d="M 100 65 Q 50 100 100 135" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="70" y="100" text-anchor="end">Γ<tspan dy="5" font-size="10">S</tspan></text>

        <!-- ΓL -->
        <path d="M 300 135 Q 350 100 300 65" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="330" y="100" text-anchor="start">Γ<tspan dy="5" font-size="10">L</tspan></text>
      </g>
    </svg>
    </div>

    The branches encode the equations:
    - $b_1 = S_{11}a_1 + S_{12}a_2$ (S-matrix, row 1)
    - $b_2 = S_{21}a_1 + S_{22}a_2$ (S-matrix, row 2)
    - $a_1 = b_S + \Gamma_S b_1$ (source reflection: incident wave = source + reflected)
    - $a_2 = \Gamma_L b_2$ (load reflection: wave returning from load)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Mason's non-touching loop rule

    Solving signal flow graphs algebraically for the transfer function between two 
    nodes can become tedious. Samuel Mason derived a general formula:

    $$T = \frac{\sum_k P_k \Delta_k}{\Delta}$$

    Where:
    - $\Delta = 1 - \sum L_i + \sum L_i L_j - \dots$ (the graph determinant)
    - $P_k$ = Path gain of the $k$-th forward path.
    - $\Delta_k$ = The cofactor of path $k$ (the determinant of the part of the graph 
      not touching path $k$).

    ### 7.1 Deriving $\Gamma_{in}$
    To find $\Gamma_{in} = b_1 / a_1$ (ratio at the input port interface), we 
    set $b_S = 0$ (calculating reflection looking into the device) and find the transfer 
    from $a_1$ to $b_1$.
    - **Paths**: $P_1 = S_{11}$ (direct), $P_2 = S_{21}\Gamma_L S_{12}$ (via load).
    - **Loops**: $L_1 = S_{22}\Gamma_L$ (load reflection loop).
    - **Determinant**: $\Delta = 1 - S_{22}\Gamma_L$.
    - **Cofactors**: $\Delta_1 = 1 - S_{22}\Gamma_L$, $\Delta_2 = 1$ (path 2 touches the loop).

    Result:
    $$\Gamma_{in} = \frac{S_{11}(1 - S_{22}\Gamma_L) + S_{21}\Gamma_L S_{12}}{1 - S_{22}\Gamma_L} 
      = S_{11} + \frac{S_{12} S_{21} \Gamma_L}{1 - S_{22} \Gamma_L}$$

    ### 7.2 Deriving $\Gamma_{out}$
    Looking back from the load ($b_S=0$, stimulus into port 2):
    $$\Gamma_{out} = S_{22} + \frac{S_{12} S_{21} \Gamma_S}{1 - S_{11} \Gamma_S}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Interactive Signal Flow Graph Solver

    Adjust the S-parameters and terminations to see how the input reflection 
    changes. This uses the Mason's Rule result derived above.
    """)
    return


@app.cell
def _(mo):
    # S-parameter sliders
    s11_slider = mo.ui.slider(0, 0.99, value=0.5, label="|S11|")
    s21_slider = mo.ui.slider(0.1, 5, value=2.0, label="|S21|")
    s12_slider = mo.ui.slider(0, 0.5, value=0.1, label="|S12|")
    s22_slider = mo.ui.slider(0, 0.99, value=0.5, label="|S22|")

    # Termination sliders
    gl_re = mo.ui.slider(-1.0, 1.0, value=0.3, label="Re(ΓL)")
    gl_im = mo.ui.slider(-1.0, 1.0, value=0.2, label="Im(ΓL)")
    gs_re = mo.ui.slider(-1.0, 1.0, value=0.0, label="Re(ΓS)")
    gs_im = mo.ui.slider(-1.0, 1.0, value=0.0, label="Im(ΓS)")

    mo.vstack([
        mo.hstack([s11_slider, s21_slider, s12_slider, s22_slider]),
        mo.hstack([gl_re, gl_im, gs_re, gs_im])
    ])
    return gl_im, gl_re, gs_im, gs_re, s11_slider, s12_slider, s21_slider, s22_slider


@app.cell
def _(gl_im, gl_re, gs_im, gs_re, mo, np, s11_slider, s12_slider, s21_slider, s22_slider):
    s11 = s11_slider.value; s21 = s21_slider.value
    s12 = s12_slider.value; s22 = s22_slider.value
    
    # Assume zero phase for device S for simplicity
    S11 = s11 + 0j; S21 = s21 + 0j; S12 = s12 + 0j; S22 = s22 + 0j
    
    gammaL = gl_re.value + 1j*gl_im.value
    gammaS = gs_re.value + 1j*gs_im.value

    # Mason's results
    gin = S11 + (S12 * S21 * gammaL) / (1 - S22 * gammaL)
    gout = S22 + (S12 * S21 * gammaS) / (1 - S11 * gammaS)

    # Unit disk check
    in_stable = abs(gin) < 1.0
    out_stable = abs(gout) < 1.0

    mo.md(f"""
    **Calculated Results:**
    - Input Reflection $\\Gamma_{{in}}$: **{abs(gin):.3f}** ∠{np.degrees(np.angle(gin)):.1f}°
    - Output Reflection $\\Gamma_{{out}}$: **{abs(gout):.3f}** ∠{np.degrees(np.angle(gout)):.1f}°
    
    **Stability Check:**
    - Port 1: {'✓' if in_stable else '✗ (Potentially Unstable)'}
    - Port 2: {'✓' if out_stable else '✗ (Potentially Unstable)'}
    """)
    return gammaL, gammaS, gin, gout, in_stable, out_stable, s11, s12, s21, s22


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Next:** [03 — Power Gains and Stability](03_s_parameters_stability.py)
    """)
    return


if __name__ == "__main__":
    app.run()
