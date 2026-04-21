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
    # 06 — Load-Pull and Source-Pull Analysis

    This final notebook integrates the concepts of gain, stability, and power 
    matching into the practical design flow of an amplifier.

    ---

    ## 1. Linear vs. Non-Linear Load-Pull

    ### 1.1 Linear Load-Pull
    In the linear regime, the optimal load is the conjugate match ($\Gamma_L = \Gamma_{out}^*$). 
    The constant-gain contours are circles on the Smith chart.

    ### 1.2 Non-Linear Load-Pull
    In power amplifiers (PAs), the transistor operates in compression. The optimal load 
    shifts to maximize output power ($P_{sat}$) or efficiency ($PAE$). These contours 
    are often non-circular and must be determined via measurement or non-linear 
    simulation.

    ---

    ## 2. Design Flow

    A typical RF design follows these steps:
    1. **Stability Check**: Ensure $K > 1$ or define safe termination regions.
    2. **Source-Pull**: Systematically sweep $\Gamma_S$ to minimize noise figure or 
       maximize gain.
    3. **Load-Pull**: Systematically sweep $\Gamma_L$ to maximize power or efficiency.
    4. **Simultaneous Match**: Find the compromise point.

    ---

    ## 3. Combined Design Explorer

    In the final interactive, we combine the S-parameter model with stability 
    boundaries and gain contours to visualize a complete design space.
    """)
    return


@app.cell
def _(mo):
    # Parameters for the final integrated explorer
    s11_m = mo.ui.slider(0, 0.99, value=0.6, label="|S11|")
    s21_m = mo.ui.slider(0.1, 10, value=4.0, label="|S21|")
    s12_m = mo.ui.slider(0, 0.5, value=0.05, label="|S12|")
    s22_m = mo.ui.slider(0, 0.99, value=0.5, label="|S22|")
    
    mo.hstack([s11_m, s21_m, s12_m, s22_m])
    return s11_m, s12_m, s21_m, s22_m


@app.cell
def _(mo, np, s11_m, s12_m, s21_m, s22_m):
    # Logic to calculate stability and gain circles for the combined plot
    S11 = s11_m.value; S21 = s21_m.value; S12 = s12_m.value; S22 = s22_m.value
    
    delta = S11*S22 - S12*S21
    K = (1 - abs(S11)**2 - abs(S22)**2 + abs(delta)**2) / (2 * abs(S12*S21) + 1e-15)
    
    msg = S21/S12 if S12 > 0 else np.nan
    
    mo.md(f"""
    **Current Design Environment:**
    - K-factor: {K:.3f}
    - MSG: {10*np.log10(msg):.2f} dB if K=1
    """)
    return K, S11, S12, S21, S22, delta, msg


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **End of Series.** Return to the [Index](00_index.py).
    """)
    return


if __name__ == "__main__":
    app.run()
