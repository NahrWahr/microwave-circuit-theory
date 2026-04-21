# v1.0
import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return mo,


@app.cell
def _(mo):
    mo.md(r"""
    # Microwave Circuit Theory — Series Index

    This project provides a rigorous, first-principles exploration of microwave circuit theory, 
    spanning from basic two-port parameters to advanced stability analysis and device-level 
    figures of merit.

    ---

    ## 01. [Two-Port Fundamentals](01_two_port_fundamentals.py)
    *Foundational network representations and terminal conditions.*
    - **Port Conditions**: Current flow and linearity in N-port networks.
    - **Parameter Sets**: Rigorous definitions of **Z** (Impedance), **Y** (Admittance), and **ABCD** (Transmission) matrices.
    - **Reciprocity**: Mathematical conditions for reciprocal vs. active networks.

    ## 02. [Power, Waves, and S-Parameters](02_power_gain_definitions.py)
    *The transition from voltage/current to power waves.*
    - **Complex Power**: Time-domain instantaneous power and phasor representation.
    - **Available Power**: Derivation of the conjugate match condition ($Z_L = Z_S^*$).
    - **Power Waves**: Physical interpretation of incident ($a$) and reflected ($b$) variables.
    - **Scattering Matrix**: S-parameter definitions and properties (Unitary, Lossless).
    - **Signal Flow Graphs**: Analysis using **Mason's Non-Touching Loop Rule**.

    ## 03. [Power Gains and Stability](03_s_parameters_stability.py)
    *Bilateral analysis of gain and the onset of oscillation.*
    - **Gain Trio**: Operating ($G$), Available ($G_A$), and Transducer ($G_T$) power gains.
    - **Stability Criteria**: Derivation of **Rollett's $K$** factor and the **$\mu$-test**.
    - **Stability Circles**: Geometric mapping of unstable regions on the Smith Chart.
    - **Maximum Gain**: MAG and MSG (Maximum Available/Stable Gain) derivations.

    ## 04. [Mason's Unilateral Power Gain U](04_unilateral_power_gain.py)
    *Invariant figures of merit for active devices.*
    - **U-Parameter**: Derivation from activity/passivity conditions.
    - **Invariance Proof**: Proof that $U$ is invariant under lossless, reciprocal embedding.
    - **Speed Limits**: Relation to $f_{\max}$ and the $-20\text{ dB/dec}$ roll-off law.

    ## 05. [MOSFET Small-Signal Model](05_mosfet_small_signal.py)
    *High-frequency transistor physics and figures of merit.*
    - **Equivalent Circuit**: Intrinsic and extrinsic MOSFET models ($C_{gs}$, $C_{gd}$, $R_g$, etc.).
    - **$f_T$ vs. $f_{\max}$**: Analytical derivations of unity current-gain and power-gain frequencies.
    - **Technology Context**: Real-world performance metrics for GF 22FDX (FDSOI).

    ## 06. [Load-Pull and Stability In-Depth](06_loadpull_stability.py)
    *Practical amplifier design and optimization.*
    - **Load-Pull/Source-Pull**: Mapping constant gain and power contours.
    - **Efficiency**: Power-Added Efficiency (PAE) in linear and compressed regimes.
    - **Design Flow**: Five-step procedure for RF matching and stability verification.

    ---

    ### Quick Navigation

    | Index | Notebook Title | Key Metric |
    |:---:|:---|:---|
    | 01 | Two-Port Fundamentals | $Z, Y, ABCD$ |
    | 02 | Power, Waves, and S-Parameters | $\Gamma, S, \Delta_{SFG}$ |
    | 03 | Power Gains and Stability | $K, \mu, G_T$ |
    | 04 | Mason's Unilateral Power Gain | $U, f_{\max}$ |
    | 05 | MOSFET Small-Signal Model | $f_T, R_g, g_{ds}$ |
    | 06 | Load-Pull and Stability In-Depth | Contours, PAE |
    """)
    return


if __name__ == "__main__":
    app.run()
