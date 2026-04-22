import marimo

__generated_with = "0.23.1"
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
    ## 9. Summary

    | Concept | Symbol / formula |
    |---|---|
    | Complex power | $S = \tfrac{1}{2}\tilde V\tilde I^*$ |
    | Available power | $P_{\text{avs}} = \vert\tilde V_s\vert^2/(8R_s)$ |
    | Mismatch factor (Γ form) | $M = (1-\vert\Gamma_S\vert^2)(1-\vert\Gamma_L\vert^2)/\vert1-\Gamma_S\Gamma_L\vert^2$ |
    | Power waves | $a_i = (V_i+Z_0 I_i)/(2\sqrt{Z_0})$, $b_i = (V_i-Z_0 I_i)/(2\sqrt{Z_0})$ |
    | Net port power | $P_i = \tfrac{1}{2}\vert a_i\vert^2 - \tfrac{1}{2}\vert b_i\vert^2$ |
    | S-matrix | $\mathbf{b} = \mathbf{S}\mathbf{a}$ |
    | S-determinant | $\Delta = S_{11}S_{22} - S_{12}S_{21}$ |
    | SFG determinant | $\Delta_{\text{SFG}} = (1-S_{11}\Gamma_S)(1-S_{22}\Gamma_L) - S_{12}S_{21}\Gamma_S\Gamma_L$ |
    | Input reflection (arbitrary $\Gamma_L$) | $\Gamma_{\text{in}} = S_{11} + \frac{S_{12}S_{21}\Gamma_L}{1-S_{22}\Gamma_L}$ |
    | Output reflection (arbitrary $\Gamma_S$) | $\Gamma_{\text{out}} = S_{22} + \frac{S_{12}S_{21}\Gamma_S}{1-S_{11}\Gamma_S}$ |

    These are the building blocks. In notebook 03 we use them to define the three power
    gains ($G$, $G_A$, $G_T$), derive Rollett's stability factor $K$ and the μ-test,
    draw stability and gain circles, and compute MAG/MSG.

    ---

    **Next:** [03 — Power Gains and Stability](03_s_parameters_stability.py)
    """)
    return


if __name__ == "__main__":
    app.run()
