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
    # 05 — MOSFET Small-Signal Analysis

    This notebook analyzes the high-frequency performance of a MOSFET using its 
    intrinsic Y-parameters, and derives the key figures of merit: $f_T$ and $f_{\max}$.

    ---

    ## 1. The Intrinsic Model

    A MOSFET at high frequencies is modeled by its transconductance ($g_m$), 
    output conductance ($g_{ds}$), and parasitic capacitances ($C_{gs}, C_{gd}, C_{ds}$).

    ### 1.1 Y-Matrix of the Intrinsic MOSFET
    Assuming a common-source configuration and neglecting $C_{ds}$ for simplicity:
    $$\mathbf{Y} = \begin{pmatrix} j\omega(C_{gs} + C_{gd}) & -j\omega C_{gd} \\ g_m - j\omega C_{gd} & g_{ds} + j\omega C_{gd} \end{pmatrix}$$

    ---

    ## 2. Unity Current-Gain Frequency ($f_T$)

    The frequency at which the short-circuit current gain $|h_{21}|$ falls to 1.
    $$h_{21} = \frac{I_2}{I_1} = \frac{Y_{21}}{Y_{11}} \approx \frac{g_m}{j\omega C_{gs}}$$
    $$f_T = \frac{g_m}{2\pi C_{gs}}$$

    $f_T$ is a measure of the device's speed in a current-amplifying configuration 
    and is largely determined by the channel length $L$.

    ---

    ## 3. Power Gain and $f_{\max}$

    While $f_T$ relates to current, $f_{\max}$ relates to power. $f_{\max}$ is the 
    frequency where Mason's $U$ falls to 1. 

    If we include the gate resistance $R_g$ (extrinsic parasitic):
    $$f_{\max} \approx \sqrt{\frac{f_T}{8\pi R_g C_{gd}}}$$

    ---

    ## 4. Technology Context: GF 22FDX

    In advanced nodes like GlobalFoundries 22FDX (22nm FDSOI), $f_T$ and $f_{\max}$ 
    can exceed 300 GHz. Performance is highly dependent on:
    - **Bias**: $g_m$ peaks at specific $V_{GS}$ and $I_D$ densities.
    - **Parasitics**: Layout (number of fingers, wiring) determines $R_g$ and $C_{gd}$.
    """)
    return


@app.cell
def _(mo):
    # Interactive: fT/fmax calculator
    gm_slider = mo.ui.slider(1, 100, value=20, label="gm (mS)")
    cgs_slider = mo.ui.slider(1, 100, value=15, label="Cgs (fF)")
    cgd_slider = mo.ui.slider(1, 50, value=3, label="Cgd (fF)")
    rg_slider = mo.ui.slider(0, 100, value=10, label="Rg (Ω)")
    
    mo.hstack([gm_slider, cgs_slider, cgd_slider, rg_slider])
    return cgd_slider, cgs_slider, gm_slider, rg_slider


@app.cell
def _(cgd_slider, cgs_slider, gm_slider, mo, np, rg_slider):
    gm = gm_slider.value * 1e-3
    cgs = cgs_slider.value * 1e-15
    cgd = cgd_slider.value * 1e-15
    rg = rg_slider.value
    
    ft = gm / (2 * np.pi * cgs)
    fmax = np.sqrt(ft / (8 * np.pi * rg * cgd + 1e-30))
    
    mo.md(f"""
    **Performance Estimates:**
    - $f_T$: **{ft/1e9:.1f} GHz**
    - $f_{{\max}}$: **{fmax/1e9:.1f} GHz**
    """)
    return cgd, cgs, fmax, ft, gm, rg


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Next:** [06 — Load-Pull and Source-Pull Analysis](06_loadpull_stability.py)
    """)
    return


if __name__ == "__main__":
    app.run()
