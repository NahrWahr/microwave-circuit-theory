import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Power Gains, Signal Flow Graphs, and Mason's Gain

    This notebook builds upon the fundamentals of power waves and scattering matrices to define the essential tools for amplifier design. We introduce Signal Flow Graphs (SFG) and Mason's Rule to solve for network reflections, then rigorously define the three canonical power gains ($G, G_A, G_T$). Finally, we explore Mason's Unilateral Gain $U$, an invariant figure of merit whose boundary defines $f_{\max}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Signal Flow Graphs

    A **signal flow graph** (SFG) is a directed graph where:
    - **Nodes** represent wave variables ($a_1, b_1, a_2, b_2$, and the source/load waves)
    - **Branches** represent the S-parameters (or reflection coefficients) that relate them

    For a two-port with source reflection $\Gamma_S$ and load reflection $\Gamma_L$,
    the complete signal flow graph has five nodes and seven branches:

    <div style="text-align: center; margin: 20px 0;">
    <svg width="400" height="220" viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>

      <g font-family="sans-serif" font-size="16" fill="currentColor">
        <!-- Nodes -->
        <circle cx="50" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="50" y="65" text-anchor="middle">b<tspan dy="5" font-size="12">S</tspan></text>

        <circle cx="150" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="65" text-anchor="middle">a<tspan dy="5" font-size="12">1</tspan></text>

        <circle cx="150" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="150" y="165" text-anchor="middle">b<tspan dy="5" font-size="12">1</tspan></text>

        <circle cx="300" cy="60" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="65" text-anchor="middle">b<tspan dy="5" font-size="12">2</tspan></text>

        <circle cx="300" cy="160" r="18" fill="none" stroke="currentColor" stroke-width="2"/>
        <text x="300" y="165" text-anchor="middle">a<tspan dy="5" font-size="12">2</tspan></text>

        <!-- Paths -->
        <!-- bS to a1 -->
        <line x1="68" y1="60" x2="132" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="100" y="50" text-anchor="middle">1</text>

        <!-- a1 to b2 -->
        <line x1="168" y1="60" x2="282" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="50" text-anchor="middle">S<tspan dy="-5" font-size="12">21</tspan></text>

        <!-- a2 to b1 -->
        <line x1="282" y1="160" x2="168" y2="160" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="225" y="190" text-anchor="middle">S<tspan dy="-5" font-size="12">12</tspan></text>

        <!-- a1 to b1 -->
        <line x1="140" y1="75" x2="140" y2="145" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="130" y="115" text-anchor="end">S<tspan dy="-5" font-size="12">11</tspan></text>

        <!-- b1 to a1 -->
        <line x1="160" y1="145" x2="160" y2="75" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="170" y="115" text-anchor="start">Γ<tspan dy="-5" font-size="12">S</tspan></text>

        <!-- a2 to b2 -->
        <line x1="290" y1="145" x2="290" y2="75" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="280" y="115" text-anchor="end">S<tspan dy="-5" font-size="12">22</tspan></text>

        <!-- b2 to a2 -->
        <line x1="310" y1="75" x2="310" y2="145" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)"/>
        <text x="320" y="115" text-anchor="start">Γ<tspan dy="-5" font-size="12">L</tspan></text>
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
    ## 7. Mason's Non-Touching Loop Rule

    **Mason's non-touching loop rule** gives the transfer function $T = b_L/b_S$
    from any input node to any output node in terms of paths and loops:

    $$T = \frac{\sum_k P_k \Delta_k}{\Delta}$$

    where:
    - $P_k$ = gain of the $k$-th forward path from input to output
    - $\Delta = 1 - \sum_i L_i + \sum_{i<j} L_i L_j - \sum_{i<j<k} L_i L_j L_k + \cdots$
    - $L_i$ = gain of the $i$-th **first-order loop** (closed path returning to start)
    - $L_i L_j$ = product of **non-touching** loop pairs (loops that share no nodes)
    - $\Delta_k$ = the graph determinant $\Delta$ evaluated with all loops touching path $k$ removed

    ### 7.1 The SFG determinant for a terminated two-port

    The single forward path from $b_S$ to the load node $b_2$ has gain $P_1 = S_{21}$.

    The first-order loops are:
    - $L_1 = S_{11}\Gamma_S$ (input loop)
    - $L_2 = S_{22}\Gamma_L$ (output loop)
    - $L_3 = S_{21}S_{12}\Gamma_S\Gamma_L$ (feedback loop through both ports)

    The only non-touching loop pair is $(L_1, L_2)$ (the input and output loops share no nodes).

    The graph determinant is:
    $$\Delta_{\text{SFG}} = 1 - L_1 - L_2 - L_3 + L_1 L_2$$
    $$= 1 - S_{11}\Gamma_S - S_{22}\Gamma_L - S_{12}S_{21}\Gamma_S\Gamma_L + S_{11}S_{22}\Gamma_S\Gamma_L$$

    The denominator factors cleanly as:

    $$\boxed{\Delta_{\text{SFG}} = (1 - S_{11}\Gamma_S)(1 - S_{22}\Gamma_L) - S_{12}S_{21}\Gamma_S\Gamma_L}$$

    This determinant is the central quantity in bilateral two-port analysis — it reappears
    as the denominator of the transducer gain (notebook 03) and inside the stability
    conditions derived from $|\Gamma_{\text{in}}| < 1$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Deriving $\Gamma_{\text{in}}$ and $\Gamma_{\text{out}}$

    **Goal:** Find $\Gamma_{\text{in}} = b_1/a_1$ when port 2 is terminated in $\Gamma_L$.

    From the S-matrix equations and the load boundary condition $a_2 = \Gamma_L b_2$:

    $$b_2 = S_{21}a_1 + S_{22}\Gamma_L b_2 \implies b_2 = \frac{S_{21}a_1}{1 - S_{22}\Gamma_L}$$

    $$b_1 = S_{11}a_1 + S_{12}a_2 = S_{11}a_1 + S_{12}\Gamma_L b_2
          = S_{11}a_1 + \frac{S_{12}S_{21}\Gamma_L}{1 - S_{22}\Gamma_L}a_1$$

    Therefore:

    $$\boxed{\Gamma_{\text{in}} = \frac{b_1}{a_1} = S_{11} + \frac{S_{12}S_{21}\Gamma_L}{1 - S_{22}\Gamma_L}}$$

    By symmetry (swapping ports 1 ↔ 2, and $\Gamma_S \leftrightarrow \Gamma_L$):

    $$\boxed{\Gamma_{\text{out}} = \frac{b_2}{a_2} = S_{22} + \frac{S_{12}S_{21}\Gamma_S}{1 - S_{11}\Gamma_S}}$$

    An equivalent form, useful for stability derivations, uses the S-matrix determinant
    $\Delta = S_{11}S_{22} - S_{12}S_{21}$:

    $$\Gamma_{\text{in}} = \frac{S_{11} - \Delta\Gamma_L}{1 - S_{22}\Gamma_L}, \qquad
      \Gamma_{\text{out}} = \frac{S_{22} - \Delta\Gamma_S}{1 - S_{11}\Gamma_S}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Power Gains and Stability

    Notebook 02 established the vocabulary of power waves, the S-matrix, and signal flow
    graphs. With $\Gamma_{\text{in}}$, $\Gamma_{\text{out}}$, and the SFG determinant
    $\Delta_{\text{SFG}} = (1-S_{11}\Gamma_S)(1-S_{22}\Gamma_L)-S_{12}S_{21}\Gamma_S\Gamma_L$
    in hand, we can now define the three canonical two-port power gains, investigate when a
    two-port is stable, and find the maximum gain achievable from a stable device.

    ## 1. Three Power Gain Definitions — Y-Parameter Derivations

    We begin in Y-parameters because the algebra is lighter; the S-parameter forms follow
    in §2. Consider a two-port with Y-parameters driven by a source $(V_s, Z_s)$ terminated
    by load $Z_L$. Define port voltages $V_1$, $V_2$ and port currents $I_1$, $I_2$
    (sign convention: $I_1$ into port 1, $I_2$ out of port 2 into $Z_L$).

    ### 1.1 Operating Power Gain $G$

    **Definition:** ratio of power delivered to the load to power entering port 1.
    We adopt the standard convention where $I_1$ and $I_2$ both enter the ports. The power delivered to the load $Y_L$ (where $I_{L} = -I_2$) is:

    $$P_L = \frac{1}{2}\operatorname{Re}(-V_2 I_2^*) = \frac{1}{2} G_L |V_2|^2, \qquad P_{\text{in}} = \frac{1}{2}\operatorname{Re}(V_1 I_1^*) = \frac{1}{2} \operatorname{Re}(Y_{\text{in}}) |V_1|^2$$

    Using the Y-parameter equation at port 2, $I_2 = Y_{21}V_1 + Y_{22}V_2$. Since $I_2 = -V_2 Y_L$, we can find the voltage transfer ratio:

    $$-V_2 Y_L = Y_{21}V_1 + Y_{22}V_2 \implies V_2 = \frac{-Y_{21}}{Y_{22} + Y_L} V_1$$

    Substituting this into the $P_L$ expression and dividing by $P_{\text{in}}$, the $|V_1|^2$ term cancels out:

    $$\boxed{G = \frac{|Y_{21}|^2 G_L}{|Y_{22}+Y_L|^2 \operatorname{Re}(Y_{\text{in}})}}$$

    where $Y_{\text{in}} = Y_{11} - \frac{Y_{12}Y_{21}}{Y_{22}+Y_L}$ is the input admittance.

    **Key property:** $G$ depends on $Z_L$ but **not** on $Z_s$.

    ---

    ### 1.2 Available Power Gain $G_A$

    **Definition:** ratio of power available at port 2 to power available from the source.

    $$G_A \equiv \frac{P_{\text{avn}}}{P_{\text{avs}}}$$

    The power available from the source is $P_{\text{avs}} = \frac{|I_s|^2}{8 G_S}$, where $I_s$ is the Norton equivalent source current and $G_S = \operatorname{Re}(Y_S)$.
    At port 2, the available power is obtained by conjugate-matching the output, giving $P_{\text{avn}} = \frac{|I_{\text{sc,2}}|^2}{8 G_{\text{out}}}$.
    The short-circuit current at port 2 (when $V_2 = 0$) is $I_{\text{sc,2}} = Y_{21} V_1$.
    With port 2 shorted, $V_1 = \frac{I_s}{Y_{11} + Y_S}$, so the short-circuit current is $I_{\text{sc,2}} = \frac{Y_{21} I_s}{Y_{11} + Y_S}$.

    Therefore, the available power at the output is:

    $$P_{\text{avn}} = \frac{1}{8 G_{\text{out}}} \left| \frac{Y_{21} I_s}{Y_{11} + Y_S} \right|^2$$

    Taking the ratio $P_{\text{avn}} / P_{\text{avs}}$ causes the Norton source term $|I_s|^2 / 8$ to cancel perfectly:

    $$\boxed{G_A = \frac{|Y_{21}|^2 G_S}{|Y_{11}+Y_S|^2 \operatorname{Re}(Y_{\text{out}})}}$$

    where $Y_{\text{out}} = Y_{22} - \frac{Y_{12}Y_{21}}{Y_{11}+Y_S}$.

    **Key property:** $G_A$ depends on $Z_s$ but **not** on $Z_L$.
    This makes it the right figure of merit for noise analysis (Friis formula).

    ---

    ### 1.3 Transducer Power Gain $G_T$

    **Definition:** ratio of power delivered to the load to power available from the source.

    $$G_T \equiv \frac{P_L}{P_{\text{avs}}}$$

    This is the gain a vector network analyser (VNA) reports: it includes all mismatch
    losses at both ports. From the earlier derivations, we know:

    $$P_L = \frac{1}{2} G_L |V_2|^2 \qquad \text{and} \qquad P_{\text{avs}} = \frac{|I_s|^2}{8 G_S}$$

    To find $V_2$ in terms of the Norton source $I_s$, we solve the linear system $(Y + \text{\textbf{Y}}_{terminations})\mathbf{V} = \mathbf{I}$:

    $$
    \begin{bmatrix} Y_{11} + Y_S & Y_{12} \\ Y_{21} & Y_{22} + Y_L \end{bmatrix}
    \begin{bmatrix} V_1 \\ V_2 \end{bmatrix} =
    \begin{bmatrix} I_s \\ 0 \end{bmatrix}
    $$

    Using Cramer's rule, $V_2 = \frac{-Y_{21} I_s}{(Y_{11}+Y_S)(Y_{22}+Y_L) - Y_{12}Y_{21}}$. Substituting this into $P_L$:

    $$P_L = \frac{1}{2} G_L \frac{|Y_{21}|^2 |I_s|^2}{|(Y_{11}+Y_S)(Y_{22}+Y_L) - Y_{12}Y_{21}|^2}$$

    Dividing by $P_{\text{avs}}$ eliminates the $|I_s|^2$, yielding the **exact bilateral** formula valid for any terminations:

    $$\boxed{G_T = \frac{4 G_S G_L\, |Y_{21}|^2}{|(Y_{11}+Y_S)(Y_{22}+Y_L) - Y_{12}Y_{21}|^2}}$$

    *(Note: The equivalent formula in Z-parameters uses $4 R_s R_L |Z_{21}|^2$, but when using Y-parameters one must strictly use $G_S$ and $G_L$.)*

    ---

    ### 1.4 Inequalities and When Equality Holds

    From the definitions it follows that:

    $$G_T \leq G_A \quad \text{and} \quad G_T \leq G$$

    with equality:

    | Condition | Equality |
    |---|---|
    | $Z_s = Z_{\text{in}}^*$ (input conjugate-matched) | $G_T = G_A$ |
    | $Z_L = Z_{\text{out}}^*$ (output conjugate-matched) | $G_T = G$ |
    | Both simultaneously | $G_T = G_A = G = \text{MAG}$ |

    The chain of inequalities has a physical interpretation:
    $G_T$ is the actual gain; $G_A$ is the potential gain if the output were optimally loaded;
    $G$ is the gain given the actual input power (independent of source mismatch).
    """)
    return


# ---------------------------------------------------------------------------
# Section 2 — Three gain definitions in S-parameters
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Three Power Gain Definitions — S-Parameter Formulation

    With $\Gamma_{\text{in}}$, $\Gamma_{\text{out}}$ already derived from Mason's rule in
    notebook 02 §8, the S-parameter forms of the three gains fall out directly from the
    wave-power identity $P = \tfrac{1}{2}(|a|^2 - |b|^2)$.

    ### 2.1 Operating Power Gain $G$

    The operating gain involves only the power entering the network and the power delivered to the load.

    $$G \equiv \frac{P_L}{P_{\text{in}}} = \frac{\frac{1}{2}|b_2|^2 (1 - |\Gamma_L|^2)}{\frac{1}{2}|a_1|^2 (1 - |\Gamma_{\text{in}}|^2)}$$

    Substitute the transmission ratio $b_2 / a_1 = S_{21} / (1 - S_{22} \Gamma_L)$ (notebook 02):

    $$\boxed{G = \frac{1 - |\Gamma_L|^2}{1 - |\Gamma_{\text{in}}|^2} \left| \frac{S_{21}}{1 - S_{22} \Gamma_L} \right|^2 = \frac{|S_{21}|^2 (1 - |\Gamma_L|^2)}{(1 - |\Gamma_{\text{in}}|^2) |1 - S_{22} \Gamma_L|^2}}$$

    This establishes $G$ as independent of the source termination $\Gamma_S$.

    ### 2.2 Available Power Gain $G_A$

    Available gain leverages the power available from the source and the maximum power *available* from the network at port 2.
    It can be found by evaluating the total power transfer under a conjugate match at the load ($\Gamma_L = \Gamma_{\text{out}}^*$), or by symmetric derivation from port 1:

    $$\boxed{G_A = \frac{1 - |\Gamma_S|^2}{1 - |\Gamma_{\text{out}}|^2} \left| \frac{S_{21}}{1 - S_{11} \Gamma_S} \right|^2 = \frac{|S_{21}|^2 (1 - |\Gamma_S|^2)}{(1 - |\Gamma_{\text{out}}|^2) |1 - S_{11} \Gamma_S|^2}}$$

    This establishes $G_A$ as independent of the load termination $\Gamma_L$.

    ### 2.3 Transducer Power Gain $G_T$

    Transducer gain is the actual physical power flow in the overall system. Based on the mismatch factor formula (notebook 02 §2.4), the power entering the network perfectly correlates to the available power via:

    $$P_{\text{in}} = P_{\text{avs}} \frac{(1 - |\Gamma_S|^2)(1 - |\Gamma_{\text{in}}|^2)}{|1 - \Gamma_S \Gamma_{\text{in}}|^2}$$

    Using $P_L / P_{\text{in}} = G$, we easily find $G_T = P_L / P_{\text{avs}}$ by substituting $P_{\text{in}} / P_{\text{avs}}$ and $G$:

    $$G_T = \frac{P_{\text{in}}}{P_{\text{avs}}} G = \frac{(1 - |\Gamma_S|^2)(1 - |\Gamma_{\text{in}}|^2)}{|1 - \Gamma_S \Gamma_{\text{in}}|^2} \frac{|S_{21}|^2 (1 - |\Gamma_L|^2)}{(1 - |\Gamma_{\text{in}}|^2) |1 - S_{22} \Gamma_L|^2}$$

    Notice that $(1 - |\Gamma_{\text{in}}|^2)$ perfectly cancels. Grouping the denominator terms:

    $$|1 - \Gamma_S \Gamma_{\text{in}}|^2 |1 - S_{22}\Gamma_L|^2 = \left| (1 - \Gamma_S (S_{11} + \frac{S_{12}S_{21}\Gamma_L}{1-S_{22}\Gamma_L}))(1-S_{22}\Gamma_L) \right|^2 = |(1-S_{11}\Gamma_S)(1-S_{22}\Gamma_L) - S_{12}S_{21}\Gamma_S\Gamma_L|^2$$

    This is exactly the SFG determinant $\Delta_{\text{SFG}}$ derived via Mason's rule in
    notebook 02 §7 — cross-checking the two routes to the same denominator. The result is
    the elegant **exact bilateral transducer gain**:

    $$\boxed{G_T = \frac{P_L}{P_{\text{avs}}} = \frac{(1-|\Gamma_S|^2)\,|S_{21}|^2\,(1-|\Gamma_L|^2)}{|(1-S_{11}\Gamma_S)(1-S_{22}\Gamma_L) - S_{12}S_{21}\Gamma_S\Gamma_L|^2}}$$
    """)
    return


# ---------------------------------------------------------------------------
# Section 3 — Unilateral factorisation
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Unilateral Factorisation and Its Error Bound

    If we isolate the amplifier such that reverse transmission is negligible ($S_{12} = 0$),
    the complex denominator $\Delta_{\text{SFG}}$ decouples. The transducer gain separates
    into three independent multiplicative blocks:

    $$G_{TU} = \underbrace{\frac{1-|\Gamma_S|^2}{|1-S_{11}\Gamma_S|^2}}_{G_S} \cdot \underbrace{|S_{21}|^2}_{G_0} \cdot \underbrace{\frac{1-|\Gamma_L|^2}{|1-S_{22}\Gamma_L|^2}}_{G_L}$$

    - $G_S$: Input mismatch gain. Maximised when $\Gamma_S = S_{11}^*$.
    - $G_0$: Intrinsic forward device gain ($|S_{21}|^2$) between matched $Z_0$ ports.
    - $G_L$: Output mismatch gain. Maximised when $\Gamma_L = S_{22}^*$.

    The **unilateral figure of merit** mathematically confines the fraction of error introduced by dropping $S_{12}$:

    $$U_m = \frac{|S_{11} S_{12} S_{21} S_{22}|}{(1-|S_{11}|^2)(1-|S_{22}|^2)}$$

    > **Notation note.** The letter $U$ is used in two distinct senses in this series. Here
    > $U_m$ is the **unilateral figure of merit** — a dimensionless bound on the error
    > incurred by the unilateral approximation. In notebook 04 we encounter **Mason's
    > unilateral power gain $U$**, which is an embedding-invariant gain metric. They are
    > unrelated quantities despite sharing the same letter in older literature; we use
    > $U_m$ (for *merit*) here to avoid collision.

    The true bilateral $G_T$ is strictly bounded within the interval
    $\frac{1}{(1+U_m)^2} \leq \frac{G_T}{G_{TU}} \leq \frac{1}{(1-U_m)^2}$.
    """)
    return


# ---------------------------------------------------------------------------
# Section 4 — Stability setup
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    # Mason's Unilateral Power Gain U

    ## 1. Motivation — Why a New Gain Metric?

    We have already defined MAG (maximum available gain) in notebook 03 §8. MAG is
    useful but has a fundamental limitation: it **depends on the embedding network**.
    If you wrap a transistor in a lossless matching network, you can change MAG.

    S. J. Mason (1954) asked: is there a gain figure that is **invariant** under all
    lossless, reciprocal embeddings? The answer is yes, and it is called the
    **Unilateral Power Gain** $U$.

    Invariance means: no matter what lossless passive network surrounds the transistor,
    $U$ does not change. This makes $U$ a fundamental property of the device itself,
    not of the circuit.

    > **Notation warning.** The letter $U$ is used in two distinct senses in this
    > series. Notebook 03 §3 uses $U_m$ for the **unilateral figure of merit** — a
    > dimensionless bound on the error incurred when the unilateral approximation
    > $S_{12} \approx 0$ is applied to compute $G_T$. That $U_m$ is a property of the
    > S-parameter values and terminations. In this notebook, $U$ is **Mason's
    > unilateral power gain** — an embedding-invariant device figure of merit whose
    > 0 dB crossing defines $f_{\max}$. The two quantities share a letter in older
    > literature but are unrelated; we keep $U$ for Mason's invariant and $U_m$ for
    > the error bound.

    ---

    ## 2. Derivation of U from Activity Criterion (Sylvester's Criterion)

    Any linear two-port admittance matrix $\mathbf{Y}$ can be decomposed into a Hermitian part $\mathbf{Y}_H$ and a skew-Hermitian part $\mathbf{Y}_{SH}$:

    $$ \mathbf{Y}_H = \frac{\mathbf{Y} + \mathbf{Y}^\dagger}{2}, \quad \mathbf{Y}_{SH} = \frac{\mathbf{Y} - \mathbf{Y}^\dagger}{2} $$

    where $\mathbf{Y}^\dagger = (\mathbf{Y}^*)^T$ is the conjugate transpose. The real power dissipated by the two-port network for any port voltage vector $\mathbf{v}$ depends entirely on the Hermitian part:

    $$ P_{\text{diss}} = \operatorname{Re}(\mathbf{v}^\dagger \mathbf{Y} \mathbf{v}) = \mathbf{v}^\dagger \mathbf{Y}_H \mathbf{v} $$

    For a network to be strictly **passive**, it must dissipate power for all possible excitations. Mathematically, this means $\mathbf{Y}_H$ must be a **positive semi-definite** matrix. By **Sylvester's criterion**, a $2 \times 2$ Hermitian matrix is positive semi-definite if and only if all its principal minors are non-negative.

    For $\mathbf{Y}_H$:
    $$
    \mathbf{Y}_H = \begin{bmatrix}
    G_{11} & \frac{Y_{12} + Y_{21}^*}{2} \\
    \frac{Y_{21} + Y_{12}^*}{2} & G_{22}
    \end{bmatrix}
    $$
    where $G_{ij} = \operatorname{Re}(Y_{ij})$. The conditions for passivity are:
    - **Condition A:** $G_{11} \geq 0$
    - **Condition B:** $G_{22} \geq 0$
    - **Condition C:** $\det(\mathbf{Y}_H) = G_{11} G_{22} - \frac{|Y_{12} + Y_{21}^*|^2}{4} \geq 0$

    Thus, the total passivity condition in set theory and logic operations is the intersection: $A \cap B \cap C$.

    #### The Activity Criterion and $f_{\max}$

    A device is **active** if it is *not* passive. Using De Morgan's laws, the activity condition is the logical negation of passivity:
    $$ \neg(A \cap B \cap C) \equiv \neg A \cup \neg B \cup \neg C $$

    In a physical transistor at normal operating frequencies, the input and output conductances are inherently positive ($G_{11} > 0$ and $G_{22} > 0$), meaning $\neg A$ and $\neg B$ are strictly false. Therefore, the device relies exclusively on condition $\neg C$ to be active:
    
    $$ G_{11} G_{22} - \frac{|Y_{12} + Y_{21}^*|^2}{4} < 0 $$

    If we apply a lossless, reciprocal embedding to unilateralize the device, we force the new reverse transadmittance $Y'_{12} = 0$. Since $Y'_{11}$ and $Y'_{22}$ are unaffected in their real parts by lossless reactive feedback, $G'_{11} = G_{11}$ and $G'_{22} = G_{22}$. The new forward transadmittance becomes $Y'_{21} = Y_{21} - Y_{12}$.

    Applying Condition $\neg C$ to this *unilateralized* network, it remains active when:
    
    $$ G_{11} G_{22} - \frac{|Y_{21} - Y_{12}|^2}{4} < 0 $$

    Rearranging this inequality directly gives us the definition of **Mason's Unilateral Power Gain $U$**:

    $$ U = \frac{|Y_{21} - Y_{12}|^2}{4 G_{11} G_{22}} > 1 $$

    *(Note that for the unilateralized device, $G_{12}=0$, so $G_{11}G_{22} = \det(\operatorname{Re}(\mathbf{Y}))$. Thus, the generalized denominator for any $\mathbf{Y}$ is often written as $4 \det(\operatorname{Re}(\mathbf{Y}))$.)*

    At the critical boundary where the device transitions from active to passive under any possible embedding, the activity condition $\neg C$ strictly fails. This happens precisely at $U = 1$, which defines the **maximum frequency of oscillation**, $f_{\max}$.

    ---

    ## 3. Proof of Invariance (Mason's Argument)

    Mason's profound realization was that $U$ is an invariant under any **lossless, reciprocal embedding**. This means you can add lossless matching networks to the input and output, and lossless feedback between the ports, and $U$ will remain completely unchanged.

    A lossless, reciprocal embedding can be decomposed into three fundamental operations:
    1. **Parallel Lossless Feedback (Shunt Susceptance):** Connecting a pure susceptance $jB_f$ in parallel across the two ports.
    2. **Series Lossless Reactances:** Connecting pure reactances $jX_s$ in series with the ports.
    3. **Ideal Transformers:** Changing the impedance level or phase polarity.

    Let's rigorously prove invariance under parallel lossless feedback.
    If we add an admittance $Y_f = jB_f$ between port 1 and port 2, the admittance matrix updates as:

    $$
    \mathbf{Y}' = 
    \begin{bmatrix}
    Y_{11} + jB_f & Y_{12} - jB_f \\
    Y_{21} - jB_f & Y_{22} + jB_f
    \end{bmatrix}
    $$

    Now, let's recalculate the components of $U$ for the new matrix $\mathbf{Y}'$:
    - **Numerator:** The new non-reciprocal term is $Y'_{21} - Y'_{12} = (Y_{21} - jB_f) - (Y_{12} - jB_f) = Y_{21} - Y_{12}$. The numerator $|Y_{21} - Y_{12}|^2$ is **invariant**.
    - **Denominator:** The real parts of the new $\mathbf{Y}'$ matrix are:
      $$G'_{11} = \operatorname{Re}(Y_{11} + jB_f) = G_{11}$$
      $$G'_{22} = \operatorname{Re}(Y_{22} + jB_f) = G_{22}$$
      $$G'_{12} = \operatorname{Re}(Y_{12} - jB_f) = G_{12}$$
      $$G'_{21} = \operatorname{Re}(Y_{21} - jB_f) = G_{21}$$
      Because the added feedback $jB_f$ is purely imaginary, the real parts $G_{ij}$ are completely unchanged. Therefore, $\det(\operatorname{Re}(\mathbf{Y}')) = \det(\operatorname{Re}(\mathbf{Y}))$.

    Since both the numerator and the denominator are unchanged, **$U$ is invariant under lossless reciprocal parallel feedback**.

    Similar proofs hold for series reactances (using Z-parameters, where $U$ has an identical functional form) and ideal transformers (which scale currents and voltages but cancel out in the ratio of $U$). Because any lossless reciprocal embedding can be constructed from these elements, $U$ is a fundamental **topological invariant** of the active device.

    ---

    ## 4. Relation to Other Gain Metrics

    For an unconditionally stable device, the gain metrics are ordered as:

    $$
    U \geq \text{MAG} \geq \text{MSG} = \left|\frac{S_{21}}{S_{12}}\right|
    $$

    When a device is **unilateralized** (meaning a feedback network is designed specifically to make the new $Y'_{12} = 0$), the Maximum Available Gain (MAG) of this new network is exactly equal to $U$. Since $U$ is invariant, $U$ represents the **absolute maximum power gain** that can be extracted from the transistor using any combination of passive lossless matching and feedback.

    ---

    ## 5. U represented in S-Parameters

    While defined elegantly using Y parameters and Sylvester's criterion, RF measurements are typically captured as S-parameters. The mathematical conversion of $U$ into S-parameters yields an exact, unified expression:

    $$
    U = \frac{|S_{21}/S_{12} - 1|^2}{2K|S_{21}/S_{12}| - 2\operatorname{Re}(S_{21}/S_{12})}
    $$

    where $K$ is Rollett's stability factor (derived in notebook 03 §5):

    $$
    K = \frac{1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2}{2|S_{12}S_{21}|}
    $$
    and $\Delta = S_{11}S_{22} - S_{12}S_{21}$.

    **Key Observations from the S-parameter Form:**
    - If the device is purely unilateral ($S_{12} = 0$), the ratio $S_{21}/S_{12} \to \infty$. In this mathematical limit, $U \to \infty$. This implies a perfectly unilateral active device would have infinite maximum unilateral gain. This is why real physical devices always have some finite isolation $S_{12} \neq 0$.
    - The denominator combines the stability factor $K$ and the phase alignment of the forward/reverse transmission. If $K$ is very large (highly stable and lossy), the denominator grows, reducing $U$.
    - $U$ is generally computed directly from this expression or by first transforming S-parameters to Y-parameters and using the $Y_H$ based definition.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Interactive: U vs Frequency from S-Parameter Inputs

    Enter the magnitude and phase of each S-parameter at a reference frequency,
    then choose how they roll off with frequency (simple $1/f$ or $1/f^2$ model).
    The plot shows $U(f)$, MAG, MSG, and extracts $f_{\max}$.
    """)
    return


@app.cell
def _(mo):
    mo.md("### S-parameters at reference frequency f₀")
    f0_sl   = mo.ui.slider(1, 50,   step=1, value=10, label="f₀ (GHz)", show_value=True)
    s11_m   = mo.ui.slider(0.0, 0.99, step=0.01, value=0.55, label="|S₁₁| at f₀", show_value=True)
    s21_m   = mo.ui.slider(0.5, 15,   step=0.1,  value=4.0,  label="|S₂₁| at f₀", show_value=True)
    s12_m   = mo.ui.slider(0.0, 0.3,  step=0.005,value=0.04, label="|S₁₂| at f₀", show_value=True)
    s22_m   = mo.ui.slider(0.0, 0.99, step=0.01, value=0.40, label="|S₂₂| at f₀", show_value=True)
    s11_ph  = mo.ui.slider(-180, 180, step=5, value=-70, label="∠S₁₁ (°)", show_value=True)
    s21_ph  = mo.ui.slider(-180, 180, step=5, value=120, label="∠S₂₁ (°)", show_value=True)
    s12_ph  = mo.ui.slider(-180, 180, step=5, value=55,  label="∠S₁₂ (°)", show_value=True)
    s22_ph  = mo.ui.slider(-180, 180, step=5, value=-50, label="∠S₂₂ (°)", show_value=True)

    rolloff = mo.ui.dropdown(
        options={"S₂₁ ~ 1/f, S₁₂ ~ 1/f (both -20 dB/dec)": "1f_1f",
                 "S₂₁ ~ 1/f, S₁₂ ~ 1/f²  (U ~ 1/f²)": "1f_2f",
                 "Flat (no frequency dependence)": "flat"},
        value="S₂₁ ~ 1/f, S₁₂ ~ 1/f (both -20 dB/dec)",
        label="Frequency rolloff model",
    )

    mo.vstack([
        mo.hstack([f0_sl, rolloff]),
        mo.hstack([
            mo.vstack([s11_m, s11_ph]),
            mo.vstack([s21_m, s21_ph]),
            mo.vstack([s12_m, s12_ph]),
            mo.vstack([s22_m, s22_ph]),
        ], gap="1.5rem"),
    ])
    return (
        f0_sl,
        rolloff,
        s11_m,
        s11_ph,
        s12_m,
        s12_ph,
        s21_m,
        s21_ph,
        s22_m,
        s22_ph,
    )


@app.cell
def _(
    f0_sl,
    go,
    make_subplots,
    mo,
    np,
    rolloff,
    s11_m,
    s11_ph,
    s12_m,
    s12_ph,
    s21_m,
    s21_ph,
    s22_m,
    s22_ph,
):
    f0 = f0_sl.value * 1e9
    freqs_u = np.logspace(np.log10(f0 * 0.1), np.log10(f0 * 20), 800)
    ratio = freqs_u / f0

    # S-parameters at f0
    _S11_0 = s11_m.value * np.exp(1j * np.radians(s11_ph.value))
    _S21_0 = s21_m.value * np.exp(1j * np.radians(s21_ph.value))
    _S12_0 = s12_m.value * np.exp(1j * np.radians(s12_ph.value))
    _S22_0 = s22_m.value * np.exp(1j * np.radians(s22_ph.value))

    # Frequency scaling
    ro = rolloff.value
    if ro == "flat":
        s21_f = _S21_0 * np.ones_like(ratio)
        s12_f = _S12_0 * np.ones_like(ratio)
    elif ro == "1f_1f":
        s21_f = _S21_0 / ratio
        s12_f = _S12_0 / ratio
    else:  # 1f_2f
        s21_f = _S21_0 / ratio
        s12_f = _S12_0 / ratio**2

    # S11, S22: mild rolloff of reflection with frequency
    s11_f = _S11_0 * np.exp(-ratio * 0.1) * np.exp(1j * np.angle(_S11_0))
    s22_f = _S22_0 * np.exp(-ratio * 0.1) * np.exp(1j * np.angle(_S22_0))
    # Clip magnitudes
    s11_f = np.where(np.abs(s11_f) > 0.99, 0.99 * np.exp(1j*np.angle(s11_f)), s11_f)
    s22_f = np.where(np.abs(s22_f) > 0.99, 0.99 * np.exp(1j*np.angle(s22_f)), s22_f)

    # Convert S → Y
    Z0 = 50.0

    def s_to_y(S11, S21, S12, S22, z0=50.0):
        # Y = (I - S)(I + S)^{-1} / z0
        # For 2x2: analytic form
        denom = (1 + S11) * (1 + S22) - S12 * S21
        Y11 = ((1 - S11) * (1 + S22) + S12 * S21) / (z0 * denom)
        Y12 = -2 * S12 / (z0 * denom)
        Y21 = -2 * S21 / (z0 * denom)
        Y22 = ((1 + S11) * (1 - S22) + S12 * S21) / (z0 * denom)
        return Y11, Y12, Y21, Y22

    Y11_f, Y12_f, Y21_f, Y22_f = s_to_y(s11_f, s21_f, s12_f, s22_f)

    # Compute U
    num_U = np.abs(Y21_f - Y12_f)**2
    den_U = 4 * (np.real(Y11_f) * np.real(Y22_f) - np.real(Y12_f) * np.real(Y21_f))
    U_f = np.where(den_U > 1e-60, num_U / den_U, np.nan)
    U_db = 10 * np.log10(np.where(U_f > 1e-30, U_f, np.nan))

    # Rollett K and MAG
    Delta_f = s11_f * s22_f - s12_f * s21_f
    K_denom = 2 * np.abs(s12_f * s21_f)
    K_f = (1 - np.abs(s11_f)**2 - np.abs(s22_f)**2 + np.abs(Delta_f)**2) / np.where(K_denom > 1e-30, K_denom, np.nan)

    MSG_f  = np.abs(s21_f / np.where(np.abs(s12_f) > 1e-30, s12_f, np.nan))
    MSG_db = 10 * np.log10(np.where(MSG_f > 1e-30, MSG_f, np.nan))

    MAG_db = np.where(
        K_f > 1,
        MSG_db + 10*np.log10(np.where(K_f > 1, K_f - np.sqrt(np.where(K_f > 1, K_f**2 - 1, np.nan)), np.nan)),
        np.nan,
    )

    # Find fmax (U = 1, i.e. 0 dB)
    _valid = np.isfinite(U_db)
    _cross = np.where(_valid[:-1] & _valid[1:] & (U_db[:-1] >= 0) & (U_db[1:] < 0))[0]
    if len(_cross) > 0:
        _i = _cross[0]
        fmax_val = freqs_u[_i] + (freqs_u[_i+1]-freqs_u[_i]) * (-U_db[_i]) / (U_db[_i+1]-U_db[_i])
        fmax_str = f"{fmax_val/1e9:.2f} GHz"
    else:
        fmax_val = None
        fmax_str = "not in range"

    fGHz_u = freqs_u / 1e9

    fig_U = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Gain metrics (dB)", "Rollett K"])

    fig_U.add_trace(go.Scatter(x=fGHz_u, y=U_db, name="U (Mason)", line=dict(color="#00CC96", width=2.5)), row=1, col=1)
    fig_U.add_trace(go.Scatter(x=fGHz_u, y=MSG_db, name="MSG", line=dict(color="#EF553B", width=2, dash="dash")), row=1, col=1)
    fig_U.add_trace(go.Scatter(x=fGHz_u, y=MAG_db, name="MAG (K>1 only)", line=dict(color="#636EFA", width=2)), row=1, col=1)
    fig_U.add_hline(y=0, line=dict(color="white", dash="dot", width=1), row=1, col=1)
    if fmax_val:
        fig_U.add_vline(x=fmax_val/1e9, line=dict(color="yellow", dash="dash", width=1.5))
        fig_U.add_annotation(x=fmax_val/1e9, y=3, text=f"f_max={fmax_str}", showarrow=True,
            arrowhead=2, font=dict(color="yellow"), row=1, col=1)

    fig_U.add_trace(go.Scatter(x=fGHz_u, y=K_f, name="K (Rollett)", line=dict(color="#FFA15A", width=2)), row=2, col=1)
    fig_U.add_hline(y=1, line=dict(color="white", dash="dot", width=1), row=2, col=1)

    fig_U.update_xaxes(type="log", title_text="Frequency (GHz)", row=2, col=1)
    fig_U.update_yaxes(title_text="Gain (dB)", row=1, col=1)
    fig_U.update_yaxes(title_text="K", row=2, col=1)
    fig_U.update_layout(
        template="plotly_dark",
        height=600,
        title=f"Mason's U, MAG, MSG vs Frequency  |  f_max = {fmax_str}",
        legend=dict(orientation="h", y=-0.12),
    )
    mo.ui.plotly(fig_U)
    return (
        Delta_f,
        K_denom,
        K_f,
        MAG_db,
        MSG_db,
        MSG_f,
        U_db,
        U_f,
        Y11_f,
        Y12_f,
        Y21_f,
        Y22_f,
        Z0,
        den_U,
        f0,
        fGHz_u,
        fmax_str,
        fmax_val,
        fig_U,
        freqs_u,
        num_U,
        ratio,
        s11_f,
        s12_f,
        s21_f,
        s22_f,
        s_to_y,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. The −20 dB/decade Law

    For a MOSFET at frequencies well above $f_T / 10$ and well below package resonances,
    $U$ rolls off as $1/f^2$, i.e. **−20 dB/decade**. This is a direct consequence
    of the intrinsic transistor Y-parameters.

    Let's examine a simplified MOSFET small-signal model. At moderate to high frequencies, the intrinsic parameters are approximated by:

    $$
    Y_{11} \approx j\omega C_{gs}
    $$
    $$
    Y_{12} \approx -j\omega C_{gd}
    $$
    $$
    Y_{21} \approx g_m - j\omega C_{gd} \approx g_m
    $$
    $$
    Y_{22} \approx g_{ds} + j\omega (C_{gd} + C_{db})
    $$

    However, to see the $-20\text{ dB/decade}$ rolloff correctly, we must include the physical gate resistance $R_g$, which introduces a real part to the input admittance. A more rigorous analysis including $R_g$ gives:

    $$
    \operatorname{Re}(Y_{11}) \approx \omega^2 C_{gs}^2 R_g
    $$

    Now, evaluate the components of $U$:
    - **Numerator:** $|Y_{21} - Y_{12}|^2 \approx |g_m - j\omega C_{gd} - (-j\omega C_{gd})|^2 = g_m^2$
    - **Denominator:** $\operatorname{Re}(Y_{22}) \approx g_{ds}$ and $\operatorname{Re}(Y_{12}) \operatorname{Re}(Y_{21})$ is typically negligible or much smaller compared to $\operatorname{Re}(Y_{11}) \operatorname{Re}(Y_{22})$.

    Substituting these into Mason's formula:

    $$
    U \approx \frac{g_m^2}{4 \operatorname{Re}(Y_{11}) \operatorname{Re}(Y_{22})} \approx \frac{g_m^2}{4 (\omega^2 C_{gs}^2 R_g) g_{ds}}
    $$

    Notice the $\omega^2$ in the denominator! Since $\omega = 2\pi f$, this means:

    $$
    U(f) \propto \frac{1}{f^2}
    $$

    Taking $10 \log_{10}$ of this expression yields:

    $$
    10\log_{10} U(f) = 10\log_{10} \left( \text{const} \cdot \frac{1}{f^2} \right) = \text{const} - 20\log_{10} f
    $$

    This rigorously proves the **−20 dB/decade** slope of Unilateral Gain.

    We can set $U(f_{\max}) = 1$ and solve for $f_{\max}$:

    $$
    1 \approx \frac{g_m^2}{4 (2\pi f_{\max})^2 C_{gs}^2 R_g g_{ds}} \implies f_{\max} \approx \frac{g_m}{4\pi C_{gs}} \sqrt{\frac{1}{R_g g_{ds}}}
    $$

    Since the transit frequency is $f_T \approx \frac{g_m}{2\pi C_{gs}}$, we elegantly find:

    $$
    f_{\max} \approx \frac{f_T}{2} \sqrt{\frac{1}{R_g g_{ds}}}
    $$

    This allows $f_{\max}$ to be read off a Bode-like plot by simply finding the **0 dB crossing** of $U$, or by extrapolating a lower frequency measurement with a $-20\text{ dB/decade}$ line.

    ---

    ## 8. Physical Meaning and Significance of $f_{\max}$

    As seen in Section 2, the condition for activity requires Mason's Unilateral Gain to be greater than unity ($U > 1$). As frequency increases, parasitic delays and reactive loading inevitably cause $U$ to drop (governed by the $-20\text{ dB/decade}$ law). 

    The frequency at which $U(f_{\max}) = 1$ is profoundly significant because it is precisely where the activity condition $\neg C$ fails. 

    **At exactly $f_{\max}$**:
    - The maximum possible power gain of the transistor strictly equals 1 (0 dB).
    - The transistor can no longer amplify power, regardless of any mathematically conceivable conjugate matching or embedding network.
    - The device fundamentally transitions from being an **active power generator** to a **passive power dissipator**. No lossless feedback network can be constructed to make it oscillate.

    It represents the **absolute theoretical upper frequency limit** of the device's usefulness in an electronic circuit. By contrast, $f_T$ (the unity current gain frequency) is merely an operational metric measured with a short-circuit load; it completely ignores the practical necessity of matching impedance and transferring real power. 

    Because $f_{\max}$ defines the boundary of passivity via Sylvester's Criterion, high-performance RF designs aggressively maximize $f_{\max} > f_T$ by minimizing gate resistance $R_g$ (using multi-finger topologies) and parasitic output conductance $g_{ds}$ to preserve the positive definiteness of $\mathbf{Y}_H$ to the highest possible frequencies.

    ---

    **Previous:** [03 — Power Gains and Stability](03_s_parameters_stability.py)

    **Next:** [05 — MOSFET Small-Signal Model and f_T, f_max](05_mosfet_small_signal.py)
    """)
    return


if __name__ == "__main__":
    app.run()


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Previous:** [01 — Two-Port Fundamentals](01_two_port_fundamentals.py)

    **Next:** [03 — S-Parameters and Stability Analysis](03_s_parameters_stability.py)
    """)
    return


