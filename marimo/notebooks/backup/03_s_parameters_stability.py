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
    # 03 — S-Parameters, Stability, and Gain Circles

    This notebook explores the boundaries of two-port performance: the canonical power 
    gains, the transition from conditional to unconditional stability, and 
    the mapping of gain/stability circles onto the Smith chart.

    ---

    ## 1. The Three Canonical Power Gains

    In a two-port system, we distinguish between three gain definitions based on 
    which power we reference.

    ### 1.1 Operating Power Gain ($G$)
    Ratio of power delivered to the load ($P_L$) to power entering the network ($P_{in}$).
    $$G = \frac{P_L}{P_{in}}$$
    Independent of source impedance $Z_S$.

    ### 1.2 Available Power Gain ($G_A$)
    Ratio of power available from the network ($P_{avn}$) to power available from 
    the source ($P_{avs}$).
    $$G_A = \frac{P_{avn}}{P_{avs}}$$
    Independent of load impedance $Z_L$.

    ### 1.3 Transducer Power Gain ($G_T$)
    Ratio of power delivered to the load ($P_L$) to power available from 
    the source ($P_{avs}$).
    $$G_T = \frac{P_L}{P_{avs}}$$
    Depends on both $Z_S$ and $Z_L$. This is the standard definition of "gain" 
    in a system.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Transducer Gain in S-Parameters

    Using the signal flow graph results from notebook 02, we can derive $G_T$:
    $$G_T = \frac{1 - |\Gamma_S|^2}{|1 - \Gamma_{in} \Gamma_S|^2} |S_{21}|^2 \frac{1 - |\Gamma_L|^2}{|1 - S_{22} \Gamma_L|^2}$$

    Equivalently, referenced to $\Gamma_{out}$:
    $$G_T = \frac{1 - |\Gamma_S|^2}{|1 - S_{11} \Gamma_S|^2} |S_{21}|^2 \frac{1 - |\Gamma_L|^2}{|1 - \Gamma_{out} \Gamma_L|^2}$$

    ### 2.1 Unilateral Case ($S_{12}=0$)
    If the device has infinite reverse isolation ($S_{12}=0$), then $\Gamma_{in}=S_{11}$ 
    and the gain factorizes:
    $$G_{TU} = \frac{1-|\Gamma_S|^2}{|1-S_{11}\Gamma_S|^2} \cdot |S_{21}|^2 \cdot \frac{1-|\Gamma_L|^2}{|1-S_{22}\Gamma_L|^2} = G_S \cdot G_0 \cdot G_L$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Unilateral Factorization and Error

    We define:
    - $G_0 = |S_{21}|^2$ (intrinsic gain)
    - $G_S = \frac{1-|\Gamma_S|^2}{|1-S_{11}\Gamma_S|^2}$ (source matching gain)
    - $G_L = \frac{1-|\Gamma_L|^2}{|1-S_{22}\Gamma_L|^2}$ (load matching gain)

    Note: Mason's $U$ (Notebook 04) is a different concept. The **Unilateral Figure of Merit** $U_m$ 
    defines the error bound when assuming $S_{12}=0$:
    $$\frac{1}{(1+U_m)^2} < \frac{G_T}{G_{TU}} < \frac{1}{(1-U_m)^2}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Stability — The Problem

    A two-port is **unconditionally stable** if the input/output impedances have 
    positive real parts ($|\Gamma_{in}| < 1, |\Gamma_{out}| < 1$) for **all possible** 
    passive source and load terminations ($|\Gamma_S| < 1, |\Gamma_L| < 1$).

    If there exist some passive terminations that make $|\Gamma_{in}| > 1$, the 
    device is **conditionally stable** and may oscillate.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Rollett's Stability Factor ($K$)

    A network is unconditionally stable if and only if:
    1. $K = \frac{1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2}{2 |S_{12} S_{21}|} > 1$
    2. $|\Delta| = |S_{11}S_{22} - S_{12}S_{21}| < 1$

    $K > 1$ ensures that the stability circles on the Smith chart do not intersect 
    the unit disk.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. The $\mu$-Test (Edwards–Sinsky)

    While the $K$-test requires two conditions ($K$ and $\Delta$), the $\mu$-test 
    is a single-parameter test:
    $$\mu = \frac{1 - |S_{11}|^2}{|S_{22} - \Delta S_{11}^*| + |S_{12}S_{21}|} > 1$$
    The device is unconditionally stable if and only if $\mu > 1$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Stability Circles

    When $K < 1$, we must visualize the forbidden regions on the Smith chart.

    ### 7.1 Output Stability Circle ($\Gamma_L$ plane)
    The locus of $\Gamma_L$ values that make $|\Gamma_{in}| = 1$.
    - Center $C_L = \frac{(S_{22} - \Delta S_{11}^*)^*}{ |S_{22}|^2 - |\Delta|^2 }$
    - Radius $R_L = \frac{|S_{12} S_{21}|}{ |S_{22}|^2 - |\Delta|^2 }$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. MAG and MSG

    ### 8.1 Maximum Stable Gain (MSG)
    At the boundary of stability ($K=1$), the maximum possible gain is:
    $$\text{MSG} = \left| \frac{S_{21}}{S_{12}} \right|$$

    ### 8.2 Maximum Available Gain (MAG)
    If the device is unconditionally stable ($K > 1$), we can achieve MAG by 
    simultaneous conjugate matching:
    $$\text{MAG} = \text{MSG} \cdot (K - \sqrt{K^2 - 1})$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Interactive Gain Explorer (Unilateral)

    Adjust the S-parameters and terminations to see the gain components.
    """)
    return


@app.cell
def _(mo):
    s11_slider = mo.ui.slider(0, 0.99, value=0.6, label="|S11|")
    s21_slider = mo.ui.slider(0.1, 10, value=3.5, label="|S21|")
    s22_slider = mo.ui.slider(0, 0.99, value=0.45, label="|S22|")
    gs_slider = mo.ui.slider(0, 0.99, value=0.5, label="|ΓS|")
    gl_slider = mo.ui.slider(0, 0.99, value=0.3, label="|ΓL|")
    
    mo.hstack([
        mo.vstack([mo.md("**Device S**"), s11_slider, s21_slider, s22_slider]),
        mo.vstack([mo.md("**Terminations**"), gs_slider, gl_slider]),
    ])
    return gl_slider, gs_slider, s11_slider, s21_slider, s22_slider


@app.cell
def _(gl_slider, gs_slider, mo, np, s11_slider, s21_slider, s22_slider):
    S11 = s11_slider.value; S21 = s21_slider.value; S22 = s22_slider.value
    GS = gs_slider.value; GL = gl_slider.value
    
    G0 = S21**2
    Gs = (1 - GS**2) / abs(1 - S11 * GS)**2
    Gl = (1 - GL**2) / abs(1 - S22 * GL)**2
    
    Gtu_db = 10 * np.log10(Gs * G0 * Gl)
    
    mo.md(f"""
    **Unilateral Transducer Gain:**
    - $G_0$: {10*np.log10(G0):.2f} dB
    - $G_S$: {10*np.log10(Gs):.2f} dB
    - $G_L$: {10*np.log10(Gl):.2f} dB
    - **Total $G_{{TU}}$**: **{Gtu_db:.2f} dB**
    """)
    return G0, GL, GS, Gl, Gs, Gtu_db, S11, S21, S22


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Visualizing Gain and Stability Circles
    """)
    return


@app.cell
def _(mo):
    show_src = mo.ui.dropdown(options=["Load circles", "Source circles", "Both"], value="Load circles", label="View")
    gL_param = mo.ui.slider(0.1, 0.95, value=0.8, label="gL (normalised)", show_value=True)
    gS_param = mo.ui.slider(0.1, 0.95, value=0.8, label="gS (normalised)", show_value=True)
    mo.hstack([show_src, gL_param, gS_param])
    return gL_param, gS_param, show_src


@app.cell
def _(S11, S22, gL_param, gS_param, go, mo, np, show_src):
    _theta = np.linspace(0, 2*np.pi, 300)

    def _smith_circle(c, r):
        t = np.linspace(0, 2*np.pi, 300)
        z = c + r * np.exp(1j*t)
        return z.real, z.imag

    fig_smith = go.Figure()

    # Smith chart grid
    for _r_v in [0, 0.2, 0.5, 1, 2, 5]:
        _cv = _r_v/(1+_r_v); _rv2 = 1/(1+_r_v)
        _zv = _cv + _rv2*np.exp(1j*_theta)
        _mv = _zv.real**2 + _zv.imag**2 <= 1.001
        fig_smith.add_trace(go.Scatter(x=_zv.real[_mv], y=_zv.imag[_mv], mode="lines",
            line=dict(color="rgba(150,150,150,0.25)", width=0.8), showlegend=False, hoverinfo="skip"))
    for _xv in [0.3, 0.5, 1, 2, -0.3, -0.5, -1, -2]:
        _cv2 = complex(1, 1/_xv); _rv2 = abs(1/_xv)
        _zv = _cv2 + _rv2*np.exp(1j*_theta)
        _mv = _zv.real**2 + _zv.imag**2 <= 1.001
        fig_smith.add_trace(go.Scatter(x=_zv.real[_mv], y=_zv.imag[_mv], mode="lines",
            line=dict(color="rgba(150,150,150,0.25)", width=0.8), showlegend=False, hoverinfo="skip"))
    fig_smith.add_trace(go.Scatter(x=np.cos(_theta), y=np.sin(_theta), mode="lines",
        line=dict(color="rgba(220,220,220,0.6)", width=1.5), showlegend=False))

    _show = show_src.value
    _colors_gL = [(1.0,"#00CC96"),(0.8,"#636EFA"),(0.5,"#EF553B"),(0.25,"#AB63FA")]
    _colors_gS = [(1.0,"#FFD700"),(0.8,"#FF6692"),(0.5,"#B6E880"),(0.25,"#FF97FF")]

    if _show in ["Load circles", "Both"]:
        for _gL_v, _col in _colors_gL:
            _d = 1-(1-_gL_v)*abs(S22)**2
            _cL = _gL_v*np.conj(S22)/_d
            _rL = np.sqrt(max(0,1-_gL_v))*(1-abs(S22)**2)/_d if _d > 1e-10 else 0
            _cx, _cy = _smith_circle(_cL, _rL)
            fig_smith.add_trace(go.Scatter(x=_cx, y=_cy, mode="lines", name=f"gL={_gL_v:.2f}",
                line=dict(color=_col, width=1.8)))
        # Highlighted selected circle
        _gLs = gL_param.value
        _ds = 1-(1-_gLs)*abs(S22)**2
        _cLs = _gLs*np.conj(S22)/_ds if _ds > 1e-10 else 0
        _rLs = np.sqrt(max(0,1-_gLs))*(1-abs(S22)**2)/_ds if _ds > 1e-10 else 0
        _cx, _cy = _smith_circle(_cLs, _rLs)
        fig_smith.add_trace(go.Scatter(x=_cx, y=_cy, mode="lines", name=f"gL={_gLs:.2f} (sel)",
            line=dict(color="white", width=2.5, dash="dash")))
        fig_smith.add_trace(go.Scatter(x=[np.conj(S22).real], y=[np.conj(S22).imag],
            mode="markers", name="S₂₂* (opt ΓL)",
            marker=dict(color="yellow", size=11, symbol="star")))

    if _show in ["Source circles", "Both"]:
        for _gS_v, _col in _colors_gS:
            _d = 1-(1-_gS_v)*abs(S11)**2
            _cS = _gS_v*np.conj(S11)/_d
            _rS = np.sqrt(max(0,1-_gS_v))*(1-abs(S11)**2)/_d if _d > 1e-10 else 0
            _cx, _cy = _smith_circle(_cS, _rS)
            fig_smith.add_trace(go.Scatter(x=_cx, y=_cy, mode="lines", name=f"gS={_gS_v:.2f}",
                line=dict(color=_col, width=1.8, dash="dot")))
        _gSs = gS_param.value
        _ds2 = 1-(1-_gSs)*abs(S11)**2
        _cSs = _gSs*np.conj(S11)/_ds2 if _ds2 > 1e-10 else 0
        _rSs = np.sqrt(max(0,1-_gSs))*(1-abs(S11)**2)/_ds2 if _ds2 > 1e-10 else 0
        _cx, _cy = _smith_circle(_cSs, _rSs)
        fig_smith.add_trace(go.Scatter(x=_cx, y=_cy, mode="lines", name=f"gS={_gSs:.2f} (sel)",
            line=dict(color="orange", width=2.5, dash="dash")))
        fig_smith.add_trace(go.Scatter(x=[np.conj(S11).real], y=[np.conj(S11).imag],
            mode="markers", name="S₁₁* (opt ΓS)",
            marker=dict(color="cyan", size=11, symbol="star")))

    fig_smith.update_layout(
        template="plotly_dark",
        title="Unilateral Gain Circles on Smith Chart",
        xaxis=dict(title="Re(Γ)", range=[-1.1,1.1], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(Γ)", range=[-1.1,1.1]),
        height=560,
        legend=dict(x=1.02, y=1),
    )
    mo.ui.plotly(fig_smith)
    return (fig_smith,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Interactive: Stability Circles
    """)
    return


@app.cell
def _(mo):
    s11m = mo.ui.slider(0.0, 0.99, step=0.01, value=0.60, label="|S₁₁|", show_value=True)
    s11p = mo.ui.slider(-180, 180, step=5, value=-70, label="∠S₁₁ (°)", show_value=True)
    s21m = mo.ui.slider(0.5, 10.0, step=0.1, value=3.5, label="|S₂₁|", show_value=True)
    s21p = mo.ui.slider(-180, 180, step=5, value=110, label="∠S₂₁ (°)", show_value=True)
    s12m = mo.ui.slider(0.0, 0.5,  step=0.01, value=0.10, label="|S₁₂|", show_value=True)
    s12p = mo.ui.slider(-180, 180, step=5, value=60,  label="∠S₁₂ (°)", show_value=True)
    s22m = mo.ui.slider(0.0, 0.99, step=0.01, value=0.45, label="|S₂₂|", show_value=True)
    s22p = mo.ui.slider(-180, 180, step=5, value=-55, label="∠S₂₂ (°)", show_value=True)

    mo.hstack([
        mo.vstack([mo.md("**S₁₁**"), s11m, s11p]),
        mo.vstack([mo.md("**S₂₁**"), s21m, s21p]),
        mo.vstack([mo.md("**S₁₂**"), s12m, s12p]),
        mo.vstack([mo.md("**S₂₂**"), s22m, s22p]),
    ], gap="1.5rem")
    return s11m, s11p, s12m, s12p, s21m, s21p, s22m, s22p


@app.cell
def _(mo, np, s11m, s11p, s12m, s12p, s21m, s21p, s22m, s22p):
    S11_c = s11m.value * np.exp(1j * np.radians(s11p.value))
    S21_c = s21m.value * np.exp(1j * np.radians(s21p.value))
    S12_c = s12m.value * np.exp(1j * np.radians(s12p.value))
    S22_c = s22m.value * np.exp(1j * np.radians(s22p.value))

    Delta_c = S11_c * S22_c - S12_c * S21_c

    # Rollett K
    K_num_s = 1 - abs(S11_c)**2 - abs(S22_c)**2 + abs(Delta_c)**2
    K_den_s = 2 * abs(S12_c * S21_c)
    K_c = K_num_s / K_den_s if K_den_s > 1e-30 else np.inf
    Delta_mag = abs(Delta_c)

    # Edwards–Sinsky μ
    mu_denom = abs(S22_c - Delta_c * np.conj(S11_c)) + abs(S12_c * S21_c)
    mu = (1 - abs(S11_c)**2) / mu_denom if mu_denom > 1e-30 else np.inf

    stable = K_c > 1 and Delta_mag < 1
    msg = abs(S21_c / S12_c) if abs(S12_c) > 1e-30 else np.inf
    mag = msg * (K_c - np.sqrt(K_c**2 - 1)) if stable else None

    lines = [
        f"### Stability Analysis",
        f"| Quantity | Value |",
        f"| Rollett $K$ | **{K_c:.4f}** |",
        f"| $\\mu$ (output) | **{mu:.4f}** |",
        f"| MSG | {msg:.3f} |",
    ]
    if mag is not None:
        lines.append(f"| MAG | {mag:.3f} |")
    mo.md("\n".join(lines))
    return Delta_c, Delta_mag, K_c, S11_c, S12_c, S21_c, S22_c, mag, msg, mu, stable


@app.cell
def _(Delta_c, S11_c, S12_c, S21_c, S22_c, go, mo, np):
    # Stability circles
    denom_L = abs(S22_c)**2 - abs(Delta_c)**2
    CL = np.conj(S22_c - Delta_c * np.conj(S11_c)) / denom_L if abs(denom_L) > 1e-10 else 0.0
    RL = abs(S12_c * S21_c / denom_L) if abs(denom_L) > 1e-10 else 0.0

    fig_stab = go.Figure()
    _theta = np.linspace(0, 2*np.pi, 400)
    fig_stab.add_trace(go.Scatter(x=np.cos(_theta), y=np.sin(_theta), mode="lines", line=dict(color="white")))
    
    # Draw CL circle
    _zL = CL + RL * np.exp(1j*_theta)
    fig_stab.add_trace(go.Scatter(x=_zL.real, y=_zL.imag, mode="lines", name="Output Stab. Circle"))

    fig_stab.update_layout(template="plotly_dark", height=500, width=500, xaxis=dict(scaleanchor="y"))
    mo.ui.plotly(fig_stab)
    return CL, RL, denom_L, fig_stab


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Next:** [04 — Mason's Unilateral Power Gain U](04_unilateral_power_gain.py)
    """)
    return


if __name__ == "__main__":
    app.run()
