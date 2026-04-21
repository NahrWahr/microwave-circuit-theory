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
    # 04 — Mason's Unilateral Power Gain (U)

    This notebook covers Mason's invariant unilateral power gain ($U$), its properties, 
    and its role in defining the maximum frequency of oscillation ($f_{\max}$).

    ---

    ## 1. Motivation: Why $U$?

    While transducer gain ($G_T$) and MAG depend on the matching network and can be 
    easily modified, Samuel Mason (1954) sought a gain metric that is **intrinsic** 
    to the active device.

    He defined $U$ as the maximum power gain achievable when the device is made 
    **unilateral** (reverse feedback cancelled) using a lossless, reciprocal 
    external network.

    ---

    ## 2. Properties of $U$

    1. **Invariance**: $U$ is invariant under any lossless, reciprocal three-terminal 
       transformation (series/shunt feedback, cascading with matching networks).
    2. **Activity Criterion**:
       - If $U > 1$: The device is **active** (can oscillate or provide gain).
       - If $0 \leq U \leq 1$: The device is **passive**.
       - If $U < 0$: The device is **unconditionally stable** but potentially active.
    3. **Roll-off**: For most transistors, $U$ rolls off at $-20$ dB/decade (or $-6$ dB/octave) 
       at high frequencies.

    ---

    ## 3. Mathematical Definition

    In terms of Y-parameters:
    $$U = \frac{|Y_{21} - Y_{12}|^2}{4(\text{Re}\{Y_{11}\}\text{Re}\{Y_{22}\} - \text{Re}\{Y_{12}\}\text{Re}\{Y_{21}\})}$$

    In terms of S-parameters:
    $$U = \frac{\left| \frac{S_{21}}{S_{12}} - 1 \right|^2}{ 2 \left( K \left| \frac{S_{21}}{S_{12}} \right| - \text{Re} \left\{ \frac{S_{21}}{S_{12}} \right\} \right) }$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. The Maximum Frequency of Oscillation ($f_{\max}$)

    $f_{\max}$ is defined as the frequency where $U = 1$ (or 0 dB). 
    Beyond $f_{\max}$, the device is passive and cannot provide power gain to 
    a load, regardless of the matching network used.

    For most technologies:
    $$U(f) \approx \left( \frac{f_{\max}}{f} \right)^2 \implies U_{\text{dB}} = 20 \log_{10} \left( \frac{f_{\max}}{f} \right)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Interactive: U-Parameter Explorer

    Adjust the S-parameters and frequency to see how $U$ and the activity 
    of the device evolve.
    """)
    return


@app.cell
def _(mo):
    s11_m = mo.ui.slider(0, 0.99, value=0.6, label="|S11|")
    s22_m = mo.ui.slider(0, 0.99, value=0.5, label="|S22|")
    s21_m = mo.ui.slider(0.5, 10, value=4.0, label="|S21|")
    s12_m = mo.ui.slider(0.01, 0.5, value=0.08, label="|S12|")
    
    mo.hstack([s11_m, s21_m, s12_m, s22_m])
    return s11_m, s12_m, s21_m, s22_m


@app.cell
def _(mo, np, s11_m, s12_m, s21_m, s22_m):
    s11 = s11_m.value; s21 = s21_m.value; s12 = s12_m.value; s22 = s22_m.value
    
    # Assume fixed phase for simplicity in the explorer
    S11 = s11 * np.exp(1j*0); S22 = s22 * np.exp(1j*0); S21 = s21 * np.exp(1j*0); S12 = s12 * np.exp(1j*0)
    
    delta = S11*S22 - S12*S21
    K = (1 - abs(S11)**2 - abs(S22)**2 + abs(delta)**2) / (2 * abs(S12*S21))
    
    # Mason's formula
    ratio = s21/s12
    U = (abs(ratio - 1)**2) / (2 * (K * abs(ratio) - ratio.real))
    
    mo.md(f"""
    **Activity Analysis:**
    - Rollett K: {K:.3f}
    - Mason's U: **{U:.2f} ({10*np.log10(U):.2f} dB)**
    - Status: {'ACTIVE (U > 1)' if U > 1 else 'PASSIVE (U < 1)'}
    """)
    return K, S11, S12, S21, S22, U, delta, s11, s12, s21, s22


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Next:** [05 — MOSFET Small-Signal Analysis](05_mosfet_small_signal.py)
    """)
    return


if __name__ == "__main__":
    app.run()
