# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///

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


# ---------------------------------------------------------------------------
# Section 1 — Three gain definitions in Y-parameters
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    # S-Parameters and Stability Analysis

    With all power definitions and Mason's Gain established in Notebook 02, we now focus entirely on **stability**. This notebook covers the criteria for oscillation, derives Rollett's $K$ factor and the $\mu$-test, plots stability and gain circles, and computes Maximum Available Gain (MAG).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Stability — The Problem

    A two-port **oscillates** when, for some passive termination ($|\Gamma_S| < 1$,
    $|\Gamma_L| < 1$), the reflected wave exceeds the incident wave — i.e., the
    input or output reflection coefficient magnitude exceeds unity:

    $$|\Gamma_{\text{in}}| > 1 \quad \text{or} \quad |\Gamma_{\text{out}}| > 1$$

    A device is **unconditionally stable** if no passive source or load termination
    can cause $|\Gamma_{\text{in}}| \geq 1$ or $|\Gamma_{\text{out}}| \geq 1$.

    Physically, instability arises when $S_{12} \neq 0$ creates a feedback loop between
    output and input: if the loop gain $|S_{12}S_{21}\Gamma_S\Gamma_L|$ has the right
    phase, the denominator $|\Delta_{\text{SFG}}|$ of $G_T$ collapses to zero and the
    network sustains a wave with no input — an oscillation.
    """)
    return


# ---------------------------------------------------------------------------
# Section 5 — Rollett K derivation
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Rollett's Stability Factor $K$

    We require $|\Gamma_{\text{in}}| < 1$ for all $|\Gamma_L| < 1$.  Writing out
    $\Gamma_{\text{in}}$ in the $\Delta$ form from notebook 02 §8:

    $$\Gamma_{\text{in}} = S_{11} + \frac{S_{12}S_{21}\Gamma_L}{1 - S_{22}\Gamma_L}
      = \frac{S_{11}(1-S_{22}\Gamma_L) + S_{12}S_{21}\Gamma_L}{1 - S_{22}\Gamma_L}
      = \frac{S_{11} - \Delta\Gamma_L}{1 - S_{22}\Gamma_L}$$

    where $\Delta = S_{11}S_{22} - S_{12}S_{21}$ is the determinant of the S-matrix.

    The condition $|\Gamma_{\text{in}}| = 1$ becomes:
    $$|S_{11} - \Delta\Gamma_L| = |1 - S_{22}\Gamma_L|$$

    Squaring both sides and expanding using $|A-B|^2 = |A|^2 - AB^* - A^*B + |B|^2$:

    $$|S_{11}|^2 - \Delta\Gamma_L S_{11}^* - \Delta^*\Gamma_L^* S_{11} + |\Delta|^2|\Gamma_L|^2
      = 1 - S_{22}\Gamma_L - S_{22}^*\Gamma_L^* + |S_{22}|^2|\Gamma_L|^2$$

    Rearranging and grouping terms, we obtain the boundary equation:

    $$ \begin{aligned}
    (|\Delta|^2 - |S_{22}|^2)|\Gamma_L|^2
    &+ (S_{22}^* - \Delta^* S_{11})\Gamma_L^* \\
    &+ (S_{22} - \Delta S_{11}^*)\Gamma_L + |S_{11}|^2 - 1 = 0
    \end{aligned} $$

    This forms the equation of a circle in the $\Gamma_L$ plane — the **output stability circle**.
    In §7 we formally complete the square and read off:

    $$C_L = \frac{(S_{22} - \Delta S_{11}^*)^*}{|S_{22}|^2 - |\Delta|^2}, \qquad R_L = \frac{|S_{12}S_{21}|}{\left| |S_{22}|^2 - |\Delta|^2 \right|}$$

    **Derivation of the Rollett factor ($K$):**
    For **unconditional stability**, the stability circle must lie entirely *outside* the unit disk
    (assuming $|S_{11}| < 1$ so the origin is stable). This geometrically requires the distance to the
    centre to be strictly greater than the radius plus the unit disk radius ($1$):

    $$|C_L| - R_L > 1 \implies |C_L| > 1 + R_L$$

    Since both sides are positive, we square the inequality:
    $$|C_L|^2 > 1 + 2R_L + R_L^2 \implies |C_L|^2 - R_L^2 - 1 > 2R_L$$

    From the definitions of $C_L$ and $R_L$, we can analytically prove that (see §7):
    $$|C_L|^2 - R_L^2 = \frac{1 - |S_{11}|^2}{|S_{22}|^2 - |\Delta|^2}$$

    Substituting this identity into our inequality yields:
    $$\frac{1 - |S_{11}|^2}{|S_{22}|^2 - |\Delta|^2} - 1 > \frac{2|S_{12}S_{21}|}{\left| |S_{22}|^2 - |\Delta|^2 \right|}$$

    To clear the denominators safely without flipping the inequality, we observe the sign of the denominator. Since unconditional stability requires $|C_L|^2 - R_L^2 > 0$, and the numerator $1 - |S_{11}|^2 > 0$ (for a passively matched port), the denominator must be strictly positive:
    $$|S_{22}|^2 - |\Delta|^2 > 0$$

    Multiplying through by $(|S_{22}|^2 - |\Delta|^2)$:
    $$1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2 > 2|S_{12}S_{21}|$$

    Dividing by $2|S_{12}S_{21}|$, we rigorously arrive at the **Rollett criterion**:

    $$\boxed{K = \frac{1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2}{2|S_{12}S_{21}|} > 1
      \quad \text{AND} \quad |\Delta| < 1}$$

    *(Note: The auxiliary condition $|\Delta| < 1$ ensures that both ports are simultaneously unconditionally stable, preventing internal oscillation modes).*
    """)
    return


# ---------------------------------------------------------------------------
# Section 6 — μ-test
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. The μ-test (Edwards & Sinsky, 1992)

    A limitation of the Rollett criterion is that it requires **two** conditions
    ($K > 1$ and $|\Delta| < 1$).  Edwards and Sinsky showed that a single parameter
    suffices:

    $$\boxed{\mu = \frac{1 - |S_{11}|^2}{|S_{22} - \Delta S_{11}^*| + |S_{12}S_{21}|} > 1}$$

    *Derivation.* The device is unconditionally stable iff the output stability circle
    lies entirely outside the unit disk, which mathematically requires:

    $$|C_L| - R_L > 1$$

    We can multiply and divide by the conjugate term $(|C_L| + R_L)$:

    $$ \frac{|C_L|^2 - R_L^2}{|C_L| + R_L} > 1 $$

    From our previous derivations, the numerator is $|C_L|^2 - R_L^2 = \frac{1 - |S_{11}|^2}{|S_{22}|^2 - |\Delta|^2}$.
    Substituting $C_L$ and $R_L$ into the denominator:

    $$ \begin{aligned}
    |C_L| + R_L &= \frac{|S_{22} - \Delta S_{11}^*|}{\left| |S_{22}|^2 - |\Delta|^2 \right|} + \frac{|S_{12}S_{21}|}{\left| |S_{22}|^2 - |\Delta|^2 \right|}
    \end{aligned} $$

    Assuming $|S_{22}|^2 - |\Delta|^2 > 0$ for a stable device, the terms perfectly combine:

    $$ \frac{\frac{1 - |S_{11}|^2}{|S_{22}|^2 - |\Delta|^2}}{\frac{|S_{22} - \Delta S_{11}^*| + |S_{12}S_{21}|}{|S_{22}|^2 - |\Delta|^2}} > 1 $$

    The denominator terms cancel exactly, leaving the beautifully compact $\mu$-test criterion:

    $$ \boxed{\mu = \frac{1 - |S_{11}|^2}{|S_{22} - \Delta S_{11}^*| + |S_{12}S_{21}|} > 1} $$

    **Advantages of μ over K:**
    - $\mu > 1$ is both necessary **and** sufficient (single test)
    - $\mu$ has a geometric meaning: it is the **distance** from the centre of the
      Smith chart to the nearest point of the output stability circle
    - A larger $\mu$ means more stability margin
    """)
    return


# ---------------------------------------------------------------------------
# Section 7 — Stability circles full derivation
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Stability Circles — Full Derivation

    ### 7.1 Output stability circle

    We seek the locus of $\Gamma_L$ values for which $|\Gamma_{\text{in}}| = 1$
    (the boundary between stable and unstable regions).

    Starting from the boundary equation derived in §5:

    $$ \begin{aligned}
    (|\Delta|^2 - |S_{22}|^2)|\Gamma_L|^2
    &+ (S_{22}^* - \Delta^* S_{11})\Gamma_L^* \\
    &+ (S_{22} - \Delta S_{11}^*)\Gamma_L + |S_{11}|^2 - 1 = 0
    \end{aligned} $$

    We divide the entire equation by $-(|S_{22}|^2 - |\Delta|^2)$. Defining the real scalar denominator $D_L \equiv |S_{22}|^2 - |\Delta|^2$, we rewrite the equation:

    $$ \begin{aligned}
    |\Gamma_L|^2 &- \frac{S_{22}^* - \Delta^* S_{11}}{D_L}\Gamma_L^* \\
    &- \frac{S_{22} - \Delta S_{11}^*}{D_L}\Gamma_L = \frac{|S_{11}|^2 - 1}{D_L}
    \end{aligned} $$

    To cast this into the standard circle equation form $|\Gamma_L - C_L|^2 = R_L^2$, we define a complex variable representing the geometric centre:

    $$C_L = \frac{(S_{22} - \Delta S_{11}^*)^*}{D_L} = \frac{S_{22}^* - \Delta^* S_{11}}{D_L}$$

    Substituting this definition back into the boundary equation:

    $$|\Gamma_L|^2 - C_L \Gamma_L^* - C_L^* \Gamma_L = \frac{|S_{11}|^2 - 1}{D_L}$$

    **Completing the square.** We add $|C_L|^2$ to both sides. The left side collapses into a perfect absolute square:

    $$|\Gamma_L - C_L|^2 = \frac{|S_{11}|^2 - 1}{D_L} + |C_L|^2$$

    Equating this to the right side, the squared radius $R_L^2$ must be:

    $$R_L^2 = \frac{|S_{11}|^2 - 1}{D_L} + \frac{|S_{22} - \Delta S_{11}^*|^2}{D_L^2}$$

    Now, we rigorously expand the numerator of $R_L^2$:

    $$ \begin{aligned}
    R_L^2 &= \frac{(|S_{11}|^2 - 1)(|S_{22}|^2 - |\Delta|^2) + |S_{22}|^2 - \Delta^* S_{11}S_{22}^* - \Delta S_{11}^* S_{22} + |\Delta|^2|S_{11}|^2}{D_L^2} \\
          &= \frac{|S_{11}|^2|S_{22}|^2 - |S_{11}|^2|\Delta|^2 - |S_{22}|^2 + |\Delta|^2 + |S_{22}|^2 - \Delta^* S_{11}S_{22}^* - \Delta S_{11}^* S_{22} + |\Delta|^2|S_{11}|^2}{D_L^2} \\
          &= \frac{|S_{11}S_{22}|^2 - \Delta^* S_{11}S_{22}^* - \Delta S_{11}^* S_{22} + |\Delta|^2}{D_L^2}
    \end{aligned} $$

    Recognising that the numerator is exactly $|S_{11}S_{22} - \Delta|^2$, and substituting $\Delta = S_{11}S_{22} - S_{12}S_{21}$, we find:

    $$S_{11}S_{22} - \Delta = S_{12}S_{21} \implies R_L^2 = \frac{|S_{12}S_{21}|^2}{D_L^2}$$

    Taking the square root yields the formal definitions of the output stability circle:

    $$\boxed{C_L = \frac{(S_{22} - \Delta S_{11}^*)^*}{|S_{22}|^2 - |\Delta|^2}, \qquad
      R_L = \frac{|S_{12}S_{21}|}{\left| |S_{22}|^2 - |\Delta|^2 \right|}}$$

    ### 7.2 Input stability circle

    By symmetry (swap ports: $S_{11} \leftrightarrow S_{22}$, $\Gamma_S \leftrightarrow \Gamma_L$):

    $$\boxed{C_S = \frac{(S_{11} - \Delta S_{22}^*)^*}{|S_{11}|^2 - |\Delta|^2}, \qquad
      R_S = \left|\frac{S_{12}S_{21}}{|S_{11}|^2 - |\Delta|^2}\right|}$$

    ### 7.3 Determining the stable region

    The stability circle divides the $\Gamma$-plane into two regions.  To determine
    which side is stable, test **one known point** — typically $\Gamma_L = 0$ (the
    $Z_0$ termination):

    - At $\Gamma_L = 0$: $\Gamma_{\text{in}} = S_{11}$
    - If $|S_{11}| < 1$ (which is almost always true for a practical device), then
      $\Gamma_L = 0$ is in the **stable** region

    The origin is at distance $|C_L|$ from the circle centre:
    - If $|C_L| > R_L$: the origin is **outside** the stability circle → the interior
      of the circle is the unstable region
    - If $|C_L| < R_L$: the origin is **inside** the stability circle → the exterior
      of the circle (within the unit disk) is the unstable region

    When the stability circle lies entirely outside the unit disk ($|C_L| - R_L > 1$),
    the **entire Smith chart** is stable — this corresponds to unconditional stability.
    """)
    return


# ---------------------------------------------------------------------------
# Section 8 — MAG / MSG (requires K from §5)
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Maximum Available Gain (MAG) and Maximum Stable Gain (MSG)

    With $K$ now established as the stability factor, we can ask: for a stable device,
    what is the largest transducer gain achievable by *simultaneously* conjugate-matching
    both ports?

    ### 8.1 Simultaneous conjugate match conditions

    MAG is optimally achieved when both ports are simultaneously conjugate-matched to the network:
    $$\Gamma_S = \Gamma_{\text{in}}^* \quad \text{and} \quad \Gamma_L = \Gamma_{\text{out}}^*$$

    These form two coupled non-linear equations since $\Gamma_{\text{in}}$ relies on $\Gamma_L$ and vice versa:

    $$\Gamma_S = S_{11}^* + \frac{S_{12}^*S_{21}^*\Gamma_L^*}{1-S_{22}^*\Gamma_L^*}$$

    Substituting $\Gamma_L^* = \Gamma_{\text{out}}$ into this equation, and using the fundamental expansion $\Gamma_{\text{out}} = S_{22} + \dots$, we obtain a quadratic equation strictly in terms of $\Gamma_S$:

    $$C_1 \Gamma_S^2 - B_1 \Gamma_S + C_1^* = 0$$

    where the coefficients are given fully by the S-parameters and determinant $\Delta = S_{11}S_{22} - S_{12}S_{21}$:

    $$B_1 = 1 + |S_{11}|^2 - |S_{22}|^2 - |\Delta|^2, \qquad C_1 = S_{11} - \Delta S_{22}^*$$

    By symmetry, operating on the port 2 load side grants $C_2 \Gamma_L^2 - B_2 \Gamma_L + C_2^* = 0$, where:

    $$B_2 = 1 + |S_{22}|^2 - |S_{11}|^2 - |\Delta|^2, \qquad C_2 = S_{22} - \Delta S_{11}^*$$

    The solutions to these quadratic equations yield the optimal terminations needed:

    $$\Gamma_{S,\text{opt}} = \frac{B_1 - \operatorname{sgn}(B_1)\sqrt{B_1^2 - 4|C_1|^2}}{2C_1}$$
    $$\Gamma_{L,\text{opt}} = \frac{B_2 - \operatorname{sgn}(B_2)\sqrt{B_2^2 - 4|C_2|^2}}{2C_2}$$

    The sign is decisively chosen as $-\operatorname{sgn}(B_x)$ to ensure the optimal reflection coefficient magnitude rests neatly within the Smith chart unit circle ($|\Gamma| < 1$), which corresponds physically to a passive termination.

    ### 8.2 The discriminant connects back to $K$

    For a real, physically realisable matching network to exist, the term under the square
    root must be non-negative: $B_1^2 - 4|C_1|^2 \geq 0$.

    Through careful algebraic expansion, this critical discriminant factors into a beautifully symmetric term for both ports:

    $$B_1^2 - 4|C_1|^2 = B_2^2 - 4|C_2|^2 = (1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2)^2 - 4|S_{12}S_{21}|^2$$

    Factoring the right-hand side as $4|S_{12}S_{21}|^2(K^2-1)$ recovers the same $K$
    derived from the stability boundary in §5. So the simultaneous-conjugate-match
    condition is **exactly** the condition $K \geq 1$: a device is matchable for MAG iff
    it is unconditionally stable.

    ### 8.3 MAG formula

    Substituting the optimal terminations ($\Gamma_{S,\text{opt}}$ and $\Gamma_{L,\text{opt}}$) back into the $G_T$ formula from §2.3 yields the **Maximum Available Gain** (MAG). The complex algebraic expansion simplifies using $K$:

    $$\boxed{\text{MAG} = \left|\frac{S_{21}}{S_{12}}\right|\left(K - \sqrt{K^2-1}\right)}$$

    This formula reveals a physical balance:
    - The term $|S_{21}/S_{12}|$ represents a theoretical upper limit fundamentally determined by the isolation asymmetry of the active device.
    - The factor $(K-\sqrt{K^2-1}) \leq 1$ acts as a stability penalty, representing the fundamental limits on power transfer required to maintain device stability.

    ### 8.4 Maximum Stable Gain (MSG)

    If a device is potentially unstable ($K < 1$), the calculated MAG is mathematically undefined because the square root $\sqrt{K^2-1}$ becomes imaginary. Physically, an attempted simultaneous conjugate match forces the requisite terminations into the unstable region of the Smith chart, which causes the amplifier to break into oscillation.

    However, exactly at the boundary of unconditional stability ($K = 1$), the stability penalty term evaluates to $1$, yielding the **Maximum Stable Gain**:

    $$\text{MSG} = \left|\frac{S_{21}}{S_{12}}\right|$$

    This is the peak stable gain theoretically achievable. Pushing higher requires infinite mismatch resistive loading to forcefully rein in the inherent device instability.

    ### 8.5 Gain–stability hierarchy

    For an unconditionally stable device ($K > 1$), we observe the hierarchy:

    $$G_{0} \leq \text{MAG} \leq \text{MSG} = \left|\frac{S_{21}}{S_{12}}\right|$$

    The designer's strategy hinges strictly on this stability envelope:
    - If unconditionally stable ($K > 1$), match for true **MAG**.
    - If conditionally unstable ($K < 1$), purposefully engineer a power mismatch or insert intentional resistive damping networks to augment $K \geq 1$, reducing the network gain back to the native MSG.
    """)
    return


# ---------------------------------------------------------------------------
# Section 9 — Interactive: Gain + Stability Explorer
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Interactive: Power Flow and Gain Explorer

    Adjust the device S-parameters and source/load terminations. The tables and chart
    below compute every gain metric from first principles, including $K$, $\mu$, MAG,
    MSG, and the unilateral error bound.
    """)
    return


@app.cell
def _(mo):
    s11_db  = mo.ui.slider(-20, 0,   step=1,   value=-10, label="|S₁₁| (dB)",  show_value=True)
    s21_db  = mo.ui.slider(0,   20,  step=1,   value=10,  label="|S₂₁| (dB)",  show_value=True)
    s22_db  = mo.ui.slider(-20, 0,   step=1,   value=-12, label="|S₂₂| (dB)",  show_value=True)
    s12_db  = mo.ui.slider(-40, 0,   step=1,   value=-20, label="|S₁₂| (dB)",  show_value=True)
    ang_s21 = mo.ui.slider(-180, 180, step=5,  value=-90, label="∠S₂₁ (°)",    show_value=True)

    gS_mag  = mo.ui.slider(0, 0.99, step=0.01, value=0.0, label="|ΓS|",        show_value=True)
    gS_ang  = mo.ui.slider(-180, 180, step=5,  value=0,   label="∠ΓS (°)",     show_value=True)
    gL_mag  = mo.ui.slider(0, 0.99, step=0.01, value=0.0, label="|ΓL|",        show_value=True)
    gL_ang  = mo.ui.slider(-180, 180, step=5,  value=0,   label="∠ΓL (°)",     show_value=True)

    panel = mo.hstack([
        mo.vstack([mo.md("### Device"), s11_db, s21_db, s22_db, s12_db, ang_s21]),
        mo.vstack([mo.md("### Terminations"), gS_mag, gS_ang, gL_mag, gL_ang]),
    ], gap="2rem")
    panel
    return (
        ang_s21,
        gL_ang,
        gL_mag,
        gS_ang,
        gS_mag,
        panel,
        s11_db,
        s12_db,
        s21_db,
        s22_db,
    )


@app.cell
def _(ang_s21, gL_ang, gL_mag, gS_ang, gS_mag, mo, np, s11_db, s12_db, s21_db, s22_db):
    S11 = 10**(s11_db.value/20) * np.exp(1j * np.radians(-90))
    S21 = 10**(s21_db.value/20) * np.exp(1j * np.radians(ang_s21.value))
    S22 = 10**(s22_db.value/20) * np.exp(1j * np.radians(-110))
    S12 = 10**(s12_db.value/20) * np.exp(1j * np.radians(+30))

    GS = gS_mag.value * np.exp(1j * np.radians(gS_ang.value))
    GL = gL_mag.value * np.exp(1j * np.radians(gL_ang.value))

    Delta = S11*S22 - S12*S21

    # Input/output reflections (bilateral)
    Gin  = S11 + S12*S21*GL / (1 - S22*GL + 1e-30)
    Gout = S22 + S12*S21*GS / (1 - S11*GS + 1e-30)

    # Transducer gain (exact bilateral)
    GT_num = (1 - abs(GS)**2) * abs(S21)**2 * (1 - abs(GL)**2)
    GT_den = abs((1 - S11*GS) * (1 - S22*GL) - S12*S21*GS*GL)**2
    GT = GT_num / GT_den if GT_den > 1e-30 else 0.0

    # Unilateral approximation
    GS_factor = (1 - abs(GS)**2) / abs(1 - S11*GS)**2
    G0_factor = abs(S21)**2
    GL_factor = (1 - abs(GL)**2) / abs(1 - S22*GL)**2
    GT_uni = GS_factor * G0_factor * GL_factor

    # Available gain (optimise GL = Gout*)
    GL_opt = np.conj(Gout)
    if abs(GL_opt) >= 1.0:
        GL_opt = 0.9999 * GL_opt / abs(GL_opt)
    GA_num = (1 - abs(GS)**2) * abs(S21)**2 * (1 - abs(GL_opt)**2)
    GA_den = abs((1 - S11*GS)*(1 - S22*GL_opt) - S12*S21*GS*GL_opt)**2
    GA = GA_num / GA_den if GA_den > 1e-30 else 0.0

    # Operating gain (optimise GS = Gin*)
    GS_opt = np.conj(Gin)
    if abs(GS_opt) >= 1.0:
        GS_opt = 0.9999 * GS_opt / abs(GS_opt)
    G_op_num = (1 - abs(GS_opt)**2) * abs(S21)**2 * (1 - abs(GL)**2)
    G_op_den = abs((1 - S11*GS_opt)*(1 - S22*GL) - S12*S21*GS_opt*GL)**2
    G_op = G_op_num / G_op_den if G_op_den > 1e-30 else 0.0

    # MAG / MSG
    K_num = 1 - abs(S11)**2 - abs(S22)**2 + abs(Delta)**2
    K_den = 2*abs(S12*S21)
    K = K_num / K_den if K_den > 1e-30 else np.inf
    MSG = abs(S21) / abs(S12) if abs(S12) > 1e-30 else np.inf
    if K > 1:
        MAG = MSG * (K - np.sqrt(K**2 - 1))
    else:
        MAG = None

    # Unilateral error bound
    U_err = abs(S11) * abs(S12) * abs(S21) * abs(S22) / ((1-abs(S11)**2)*(1-abs(S22)**2) + 1e-30)

    def db(x):
        return 10*np.log10(max(x, 1e-30))

    mag_str = f"{db(MAG):.2f} dB" if MAG is not None else "N/A (K < 1)"

    mo.md(f"""
### Gain Summary

| Quantity | Linear | dB | Notes |
|---|---|---|---|
| $G_T$ (exact bilateral) | {GT:.4f} | **{db(GT):.2f}** | Actual transducer gain |
| $G_{{TU}}$ (unilateral) | {GT_uni:.4f} | {db(GT_uni):.2f} | Error: {db(GT_uni)-db(GT):+.2f} dB |
| $G_A$ (available, $\\Gamma_L = \\Gamma_{{\\text{{out}}}}^*$) | {GA:.4f} | {db(GA):.2f} | Upper bound on $G_T$ for this $\\Gamma_S$ |
| $G$ (operating, $\\Gamma_S = \\Gamma_{{\\text{{in}}}}^*$) | {G_op:.4f} | {db(G_op):.2f} | Upper bound on $G_T$ for this $\\Gamma_L$ |
| $G_S$ (source mismatch) | {GS_factor:.4f} | {db(GS_factor):.2f} | Max at $\\Gamma_S = S_{{11}}^*$ |
| $G_0 = |S_{{21}}|^2$ | {G0_factor:.4f} | {db(G0_factor):.2f} | Device gain at $Z_0$ terminations |
| $G_L$ (load mismatch) | {GL_factor:.4f} | {db(GL_factor):.2f} | Max at $\\Gamma_L = S_{{22}}^*$ |
| MSG | {MSG:.4f} | {db(MSG):.2f} | $= |S_{{21}}/S_{{12}}|$ |
| MAG | — | {mag_str} | Requires $K > 1$ |
| Rollett $K$ | {K:.4f} | — | $> 1$ = unconditionally stable |
| Unilateral merit $U_m$ | {U_err:.4f} | — | $G_{{TU}}$ within $\\pm${db(1/(1-U_err)**2):.2f} dB |

**Port reflections:**
$\\Gamma_{{\\text{{in}}}}$ = {abs(Gin):.4f} ∠ {np.degrees(np.angle(Gin)):.1f}°
$\\Gamma_{{\\text{{out}}}}$ = {abs(Gout):.4f} ∠ {np.degrees(np.angle(Gout)):.1f}°
""")
    return (
        Delta,
        G0_factor,
        GA,
        GL,
        GL_factor,
        GL_opt,
        GS,
        GS_factor,
        GT,
        GT_uni,
        Gin,
        Gout,
        K,
        MAG,
        MSG,
        S11,
        S12,
        S21,
        S22,
        U_err,
        db,
        mag_str,
    )


# ---------------------------------------------------------------------------
# Section 9 (cont'd) — Power flow bar chart
# ---------------------------------------------------------------------------


@app.cell
def _(G0_factor, GA, GL_factor, GS_factor, GT, GT_uni, MAG, db, go, mo, np):
    _labels = ["Pavs (source)", "×GS (input match)", "×G0 (device)", "×GL (output match)", "GT (delivered)"]
    _vals = [0.0, db(GS_factor), db(G0_factor), db(GL_factor), db(GT)]
    _cumulative = [0.0, db(GS_factor), db(GS_factor)+db(G0_factor),
                   db(GS_factor)+db(G0_factor)+db(GL_factor), db(GT)]

    _colors = ["#636EFA", "#00CC96" if GS_factor >= 1 else "#EF553B",
               "#00CC96" if G0_factor >= 1 else "#EF553B",
               "#00CC96" if GL_factor >= 1 else "#EF553B", "#AB63FA"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=_labels,
        y=[0, db(GS_factor), db(G0_factor), db(GL_factor), db(GT)],
        marker_color=_colors,
        text=[f"{v:.2f} dB" for v in [0, db(GS_factor), db(G0_factor), db(GL_factor), db(GT)]],
        textposition="outside",
    ))

    # Reference lines
    fig_bar.add_hline(y=db(GT_uni), line=dict(color="orange", dash="dash"), annotation_text=f"GTU={db(GT_uni):.1f}dB")
    if MAG is not None:
        fig_bar.add_hline(y=db(MAG), line=dict(color="yellow", dash="dot"), annotation_text=f"MAG={db(MAG):.1f}dB")
    if GA > 0:
        fig_bar.add_hline(y=db(GA), line=dict(color="cyan", dash="dashdot"), annotation_text=f"GA={db(GA):.1f}dB")

    fig_bar.update_layout(
        template="plotly_dark",
        title="Power Gain Budget (dB contributions)",
        yaxis_title="Gain contribution (dB)",
        height=420,
        showlegend=False,
    )
    mo.ui.plotly(fig_bar)
    return (fig_bar,)


# ---------------------------------------------------------------------------
# Section 10 — Gain circles derivation and visualisation
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Gain Circles — Derivation and Visualisation

    ### 10.1 Load gain circle derivation

    In the unilateral case, the load mismatch factor is:

    $$G_L(\Gamma_L) = \frac{1-|\Gamma_L|^2}{|1-S_{22}\Gamma_L|^2}$$

    The maximum is $G_{L,\text{max}} = 1/(1-|S_{22}|^2)$ at $\Gamma_L = S_{22}^*$.
    Define the **normalised** load gain parameter $g_L = G_L / G_{L,\text{max}} \in [0,1]$.

    Setting $G_L = g_L \cdot G_{L,\text{max}}$ and solving for the locus of $\Gamma_L$:

    $$\frac{1-|\Gamma_L|^2}{|1-S_{22}\Gamma_L|^2} = \frac{g_L}{1-|S_{22}|^2}$$

    Cross-multiplying and expanding $|1-S_{22}\Gamma_L|^2 = 1 - S_{22}^*\Gamma_L - S_{22}\Gamma_L^* + |S_{22}|^2|\Gamma_L|^2$:

    $$\left(1 - g_L|S_{22}|^2 + g_L|S_{22}|^2 - g_L\right)|\Gamma_L|^2 + g_L(S_{22}^*\Gamma_L + S_{22}\Gamma_L^*) = 1 - g_L$$

    Completing the square in $\Gamma_L$, this is a circle with:

    $$\text{Centre: } c_L = \frac{g_L S_{22}^*}{1-(1-g_L)|S_{22}|^2}$$

    $$\text{Radius: } r_L = \frac{\sqrt{1-g_L}\,(1-|S_{22}|^2)}{1-(1-g_L)|S_{22}|^2}$$

    **Special cases:**
    - $g_L = 1$: $r_L = 0$, point at $\Gamma_L = S_{22}^*$ ✓
    - $g_L = 0$: $r_L \to \infty$, $G_L = 0$ regardless of $\Gamma_L$ ✓
    - $|S_{22}| = 0$: all circles are centred at origin, $r_L = \sqrt{1-g_L}$ — concentric circles

    ### 10.2 Source gain circle

    By symmetry (swapping port 1 ↔ 2):

    $$c_S = \frac{g_S S_{11}^*}{1-(1-g_S)|S_{11}|^2}, \qquad
      r_S = \frac{\sqrt{1-g_S}\,(1-|S_{11}|^2)}{1-(1-g_S)|S_{11}|^2}$$
    """)
    return


@app.cell
def _(mo):
    gL_param = mo.ui.slider(0.0, 1.0, step=0.05, value=0.8,
                             label="gL (load gain, 1 = conjugate match)",
                             show_value=True)
    gS_param = mo.ui.slider(0.0, 1.0, step=0.05, value=0.8,
                             label="gS (source gain, 1 = conjugate match)",
                             show_value=True)
    show_src = mo.ui.radio(["Load circles", "Source circles", "Both"],
                            value="Load circles", label="Show:", inline=True)
    mo.hstack([gL_param, gS_param, show_src])
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


# ---------------------------------------------------------------------------
# Section 11 — Stability circles (second interactive, drives load-pull too)
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Interactive: Stability Circles

    The second interactive below uses a separate set of sliders for the device
    S-parameters (magnitude + phase) so you can probe bilateral/feedback-heavy devices
    where the gain explorer's log-magnitude sliders lose resolution. The same sliders
    drive the load-pull analysis in §12.
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

    # μ' (dual parameter, referenced to port 1)
    mu_prime_denom = abs(S11_c - Delta_c * np.conj(S22_c)) + abs(S12_c * S21_c)
    mu_prime = (1 - abs(S22_c)**2) / mu_prime_denom if mu_prime_denom > 1e-30 else np.inf

    stable = K_c > 1 and Delta_mag < 1
    mu_stable = mu > 1

    if stable:
        status = "**Unconditionally Stable** ✓"
    elif K_c > 0:
        status = "**Conditionally Stable** (potentially unstable regions exist)"
    else:
        status = "**Unstable** ✗"

    # MSG and MAG
    msg = abs(S21_c / S12_c) if abs(S12_c) > 1e-30 else np.inf
    if stable:
        mag = msg * (K_c - np.sqrt(K_c**2 - 1))
        mag_db = 10*np.log10(max(mag, 1e-30))
    else:
        mag = None
        mag_db = None

    msg_db = 10*np.log10(max(msg, 1e-30))

    # B1 auxiliary parameter
    B1 = 1 + abs(S11_c)**2 - abs(S22_c)**2 - abs(Delta_c)**2

    lines = [
        f"### Stability Analysis",
        f"",
        f"| Quantity | Value | Condition |",
        f"|---|---|---|",
        f"| $\\Delta = S_{{11}}S_{{22}}-S_{{12}}S_{{21}}$ | {abs(Delta_c):.4f} ∠{np.degrees(np.angle(Delta_c)):.1f}° | $|\\Delta| < 1$: {'✓' if Delta_mag < 1 else '✗'} |",
        f"| Rollett $K$ | **{K_c:.4f}** | $K > 1$: {'✓' if K_c > 1 else '✗'} |",
        f"| $\\mu$ (output) | **{mu:.4f}** | $\\mu > 1$: {'✓' if mu > 1 else '✗'} |",
        f"| $\\mu'$ (input) | **{mu_prime:.4f}** | $\\mu' > 1$: {'✓' if mu_prime > 1 else '✗'} |",
        f"| $B_1 = 1+|S_{{11}}|^2-|S_{{22}}|^2-|\\Delta|^2$ | {B1:.4f} | {'$> 0$ ✓' if B1 > 0 else '$< 0$ (check stability)'} |",
        f"| MSG = $|S_{{21}}/S_{{12}}|$ | {msg:.3f} ({msg_db:.2f} dB) | — |",
    ]
    if mag is not None:
        lines.append(f"| MAG | {mag:.3f} ({mag_db:.2f} dB) | — |")
    else:
        lines.append(f"| MAG | Not defined ($K < 1$) | — |")
    lines += [
        "",
        f"**Status (Rollett):** {status}",
        f"",
        f"**Status (μ-test):** {'**Unconditionally Stable** ✓' if mu_stable else '**Not unconditionally stable** — stability circles show safe regions'}",
        f"",
        f"*Cross-check:* The Rollett and μ-test criteria {'**agree**' if (stable == mu_stable) else '**disagree** (inspect parameters)'}.",
    ]
    mo.md("\n".join(lines))
    return (
        B1,
        Delta_c,
        Delta_mag,
        K_c,
        K_den_s,
        K_num_s,
        S11_c,
        S12_c,
        S21_c,
        S22_c,
        mag,
        mag_db,
        msg,
        msg_db,
        mu,
        mu_prime,
        stable,
        status,
    )


@app.cell
def _(Delta_c, S11_c, S12_c, S21_c, S22_c, go, mo, np):
    theta_sc = np.linspace(0, 2*np.pi, 400)

    # Stability circle centres and radii
    denom_L = abs(S22_c)**2 - abs(Delta_c)**2
    denom_S = abs(S11_c)**2 - abs(Delta_c)**2

    CL = np.conj(S22_c - Delta_c * np.conj(S11_c)) / denom_L if abs(denom_L) > 1e-10 else 0.0
    RL = abs(S12_c * S21_c / denom_L) if abs(denom_L) > 1e-10 else 0.0

    CS = np.conj(S11_c - Delta_c * np.conj(S22_c)) / denom_S if abs(denom_S) > 1e-10 else 0.0
    RS = abs(S12_c * S21_c / denom_S) if abs(denom_S) > 1e-10 else 0.0

    # Determine stable region (test Γ = 0)
    origin_outside_L = abs(CL) > RL
    origin_outside_S = abs(CS) > RS

    stable_region_L = "Origin outside circle → interior of circle is unstable" if origin_outside_L else "Origin inside circle → exterior of circle (within unit disk) is unstable"
    stable_region_S = "Origin outside circle → interior of circle is unstable" if origin_outside_S else "Origin inside circle → exterior of circle (within unit disk) is unstable"

    def draw_stability_circle(centre, radius, circle_label, circle_color):
        t = np.linspace(0, 2*np.pi, 400)
        z = centre + radius * np.exp(1j*t)
        return go.Scatter(x=z.real, y=z.imag, mode="lines", name=circle_label,
                          line=dict(color=circle_color, width=2.5))

    # Build Smith chart figure
    fig_stab = go.Figure()

    # Smith grid — constant resistance circles
    for _r in [0, 0.2, 0.5, 1, 2, 5]:
        _c = _r/(1+_r) + 0j
        _rad = 1/(1+_r)
        _t = np.linspace(0, 2*np.pi, 300)
        _z = _c + _rad * np.exp(1j*_t)
        _mask = _z.real**2 + _z.imag**2 <= 1.002
        fig_stab.add_trace(go.Scatter(x=_z.real[_mask], y=_z.imag[_mask], mode="lines",
            line=dict(color="rgba(150,150,150,0.25)", width=1), showlegend=False, hoverinfo="skip"))

    # Smith grid — constant reactance arcs
    for _x in [0.2, 0.5, 1, 2, 5, -0.2, -0.5, -1, -2, -5]:
        _c = complex(1, 1/_x); _rad = abs(1/_x)
        _t = np.linspace(0, 2*np.pi, 300)
        _z = _c + _rad * np.exp(1j*_t)
        _mask = _z.real**2 + _z.imag**2 <= 1.002
        fig_stab.add_trace(go.Scatter(x=_z.real[_mask], y=_z.imag[_mask], mode="lines",
            line=dict(color="rgba(150,150,150,0.25)", width=1), showlegend=False, hoverinfo="skip"))

    # Unit circle (|Γ| = 1 boundary)
    fig_stab.add_trace(go.Scatter(x=np.cos(theta_sc), y=np.sin(theta_sc), mode="lines",
        line=dict(color="rgba(220,220,220,0.6)", width=1.5), showlegend=False))

    # Stability circles
    fig_stab.add_trace(draw_stability_circle(CL, RL, "Output stab. circle (Γ_L)", "#EF553B"))
    fig_stab.add_trace(draw_stability_circle(CS, RS, "Input stab. circle (Γ_S)",  "#636EFA"))

    # S-parameter points
    for _pt, _name, _clr in [
        (S11_c, "S₁₁", "cyan"),
        (S22_c, "S₂₂", "orange"),
        (np.conj(S11_c), "S₁₁*", "lightcyan"),
        (np.conj(S22_c), "S₂₂*", "lightyellow"),
    ]:
        fig_stab.add_trace(go.Scatter(x=[_pt.real], y=[_pt.imag], mode="markers+text",
            name=_name, marker=dict(color=_clr, size=9),
            text=[_name], textposition="top right",
            textfont=dict(color=_clr, size=11)))

    fig_stab.update_layout(
        template="plotly_dark",
        title="Stability Circles on Smith Chart",
        xaxis=dict(title="Re(Γ)", range=[-2.0, 2.0], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(Γ)", range=[-2.0, 2.0]),
        height=620,
    )
    _stab_plot = mo.ui.plotly(fig_stab)

    mo.vstack([
        _stab_plot,
        mo.md(f"""
**Stability circle parameters:**

| Circle | Centre | Radius | Stable region |
|---|---|---|---|
| Output ($\\Gamma_L$) | {CL.real:.3f} + j{CL.imag:.3f} | {RL:.3f} | {stable_region_L} |
| Input ($\\Gamma_S$) | {CS.real:.3f} + j{CS.imag:.3f} | {RS:.3f} | {stable_region_S} |

*Red circle:* Output stability circle — the locus of $\\Gamma_L$ values making $|\\Gamma_{{\\text{{in}}}}| = 1$.
*Blue circle:* Input stability circle — the locus of $\\Gamma_S$ values making $|\\Gamma_{{\\text{{out}}}}| = 1$.
        """),
    ])
    return (
        CL,
        CS,
        RL,
        RS,
        denom_L,
        denom_S,
        fig_stab,
        theta_sc,
    )


# ---------------------------------------------------------------------------
# Section 12 — Load-pull analysis
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Load-Pull Analysis

    ### 12.1 Concept

    **Load-pull** is the technique of systematically varying the load impedance
    $Z_L$ (sweeping $\Gamma_L$ across the Smith chart) and measuring output power,
    gain, or PAE (power-added efficiency).  The result is a set of **constant-performance
    contours** on the Smith chart.

    ### 12.2 Unilateral approximation

    For $S_{12} \approx 0$, the transducer gain simplifies and the power delivered to
    the load is:

    $$P_{\text{out}} \propto |S_{21}|^2 \cdot \frac{1-|\Gamma_L|^2}{|1-S_{22}\Gamma_L|^2}
      \cdot P_{\text{avs}}$$

    The **optimal load** for maximum output power is $\Gamma_L = S_{22}^*$
    (conjugate match at the output). These are exactly the unilateral load gain circles
    drawn in §10.

    ### 12.3 Bilateral formula

    When $S_{12} \neq 0$, the output power depends on both $\Gamma_S$ and $\Gamma_L$.
    For a fixed source termination $\Gamma_S$, the power delivered to the load is:

    $$P_L = \frac{1}{2}|b_2|^2(1 - |\Gamma_L|^2)$$

    where $b_2/b_S = S_{21}/\Delta_{\text{SFG}}$ from Mason's rule (notebook 02 §7).

    The optimal load in the bilateral case is $\Gamma_L = \Gamma_{\text{out}}^*$,
    where $\Gamma_{\text{out}}$ depends on $\Gamma_S$.  This creates a coupled
    optimisation problem (the simultaneous conjugate match from §8).

    ### 12.4 Practical load-pull

    In a real PA (power amplifier), the device operates in **large-signal** mode where
    the S-parameters are power-dependent.  The optimal load for maximum **output power**
    differs from the optimal load for maximum **efficiency** — load-pull contours let
    designers visualise both simultaneously and choose an operating point that
    compromises between power and efficiency.

    The plot below shows normalised $P_{\text{out}}$ across the Smith chart using the
    unilateral (small-signal) formula.

    > **Important Note:** This plot exclusively shows the **power delivered to the load** ($P_{\text{out}}$) under linear, small-signal conditions. It **does not** show Power-Added Efficiency (PAE), drain efficiency, or large-signal gain compression. True efficiency contours require non-linear transistor models and large-signal harmonic balance simulation.
    """)
    return


@app.cell
def _(S21_c, S22_c, go, mo, np, s11m, s11p, s21m, s21p, s12m, s12p, s22m, s22p):
    # Grid over the unit disk in Γ-plane
    _N = 150
    _u = np.linspace(-0.98, 0.98, _N)
    _gL_re, _gL_im = np.meshgrid(_u, _u)
    _gL_grid = _gL_re + 1j * _gL_im

    # Mask to unit disk
    _in_disk = (np.abs(_gL_grid) < 1.0)

    _denom = np.abs(1 - S22_c * _gL_grid)**2
    _Pout_norm = np.abs(S21_c)**2 * (1 - np.abs(_gL_grid)**2) / np.where(_denom > 1e-30, _denom, np.nan)
    _Pout_norm[~_in_disk] = np.nan

    _Pout_db = 10 * np.log10(np.where(_Pout_norm > 1e-30, _Pout_norm, np.nan))
    _Pmax_db = 10 * np.log10(np.abs(S21_c)**2)  # at GL = S22*

    fig_lp = go.Figure()

    fig_lp.add_trace(go.Contour(
        x=_u, y=_u, z=_Pout_db,
        colorscale="Viridis",
        contours=dict(
            start=_Pmax_db - 12, end=_Pmax_db, size=1,
            showlabels=True, labelfont=dict(size=10, color="white"),
        ),
        colorbar=dict(title="P_out (dB)"),
        name="P_out contours",
    ))

    # Smith chart overlay — unit circle
    _theta_lp = np.linspace(0, 2*np.pi, 300)
    fig_lp.add_trace(go.Scatter(
        x=np.cos(_theta_lp), y=np.sin(_theta_lp), mode="lines",
        line=dict(color="white", width=1.5), showlegend=False,
    ))

    # Smith grid — constant resistance circles
    for _r_lp in [0.2, 0.5, 1, 2]:
        _c_lp = _r_lp/(1+_r_lp)
        _rad_lp = 1/(1+_r_lp)
        _z_lp = _c_lp + _rad_lp * np.exp(1j*_theta_lp)
        _m_lp = _z_lp.real**2 + _z_lp.imag**2 <= 1.001
        fig_lp.add_trace(go.Scatter(x=_z_lp.real[_m_lp], y=_z_lp.imag[_m_lp], mode="lines",
            line=dict(color="rgba(200,200,200,0.2)", width=1), showlegend=False))

    # Optimal point (S22*)
    _GL_opt = np.conj(S22_c)
    fig_lp.add_trace(go.Scatter(
        x=[_GL_opt.real], y=[_GL_opt.imag],
        mode="markers", name="Opt Γ_L = S₂₂*",
        marker=dict(color="red", size=12, symbol="star"),
    ))

    fig_lp.update_layout(
        template="plotly_dark",
        title=f"Normalised P_out Contours (Max: {_Pmax_db:.2f} dB)",
        xaxis=dict(title="Re(Γ_L)", range=[-1.05, 1.05], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(Γ_L)", range=[-1.05, 1.05], constrain="domain"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=650,
        height=650,
    )

    return mo.hstack([
        mo.ui.plotly(fig_lp),
        mo.vstack([
            mo.md("### Interactive Controls"),
            mo.md("Adjust the device S-parameters to observe the effect on the load-pull contours."),
            s11m, s11p,
            s21m, s21p,
            s12m, s12p,
            s22m, s22p
        ])
    ], gap="2rem")


# ---------------------------------------------------------------------------
# Section 13 — Unilateral error sweep
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Unilateral Error — How Bad is the Approximation?

    The unilateral gain $G_{TU}$ approximates the true $G_T$ by neglecting $S_{12}$.
    The **unilateral figure of merit**:

    $$U_m = \frac{|S_{11}||S_{12}||S_{21}||S_{22}|}{(1-|S_{11}|^2)(1-|S_{22}|^2)}$$

    bounds the error:

    $$\frac{1}{(1+U_m)^2} \leq \frac{G_T}{G_{TU}} \leq \frac{1}{(1-U_m)^2}$$

    The plot below sweeps $|S_{12}|$ (reverse isolation) from 0 to its current value
    and shows how both $G_T$ and the error bound evolve. (Uses the §9 gain-explorer
    sliders.)
    """)
    return


@app.cell
def _(GS, GL, S11, S21, S22, go, mo, np):
    _s12_range = np.linspace(0, 0.5, 200)
    _GT_sweep  = np.zeros(len(_s12_range))
    _GTU_sweep = np.zeros(len(_s12_range))
    _Uerr_sweep = np.zeros(len(_s12_range))

    _GS_f = (1 - abs(GS)**2) / abs(1 - S11*GS)**2
    _G0   = abs(S21)**2
    _GL_f = (1 - abs(GL)**2) / abs(1 - S22*GL)**2
    _GTU_val = _GS_f * _G0 * _GL_f

    for _i, _s12v in enumerate(_s12_range):
        _S12v = _s12v * np.exp(1j * np.radians(30))  # fixed phase
        _GT_num_v = (1-abs(GS)**2)*abs(S21)**2*(1-abs(GL)**2)
        _GT_den_v = abs((1-S11*GS)*(1-S22*GL) - _S12v*S21*GS*GL)**2
        _GT_sweep[_i] = _GT_num_v / _GT_den_v if _GT_den_v > 1e-30 else 0
        _GTU_sweep[_i] = _GS_f * _G0 * _GL_f  # unilateral doesn't change with S12
        _Uerr_sweep[_i] = (abs(S11)*_s12v*abs(S21)*abs(S22) /
                           ((1-abs(S11)**2)*(1-abs(S22)**2) + 1e-30))

    _GT_db  = 10*np.log10(np.where(_GT_sweep > 1e-30, _GT_sweep, np.nan))
    _GTU_db = 10*np.log10(max(_GTU_val, 1e-30)) * np.ones_like(_s12_range)
    _ub_db  = _GTU_db + 20*np.log10(np.where(1-_Uerr_sweep > 0, 1/(1-_Uerr_sweep), np.nan))
    _lb_db  = _GTU_db - 20*np.log10(1+_Uerr_sweep)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=_s12_range, y=_GT_db,  name="GT (exact bilateral)",
        line=dict(color="#00CC96", width=2.5)))
    fig_err.add_trace(go.Scatter(x=_s12_range, y=_GTU_db, name="GTU (unilateral approx)",
        line=dict(color="#636EFA", width=2, dash="dash")))
    fig_err.add_trace(go.Scatter(x=_s12_range, y=_ub_db, name="Upper error bound",
        line=dict(color="rgba(255,100,100,0.5)", width=1.5, dash="dot")))
    fig_err.add_trace(go.Scatter(x=_s12_range, y=_lb_db, name="Lower error bound",
        fill="tonexty", fillcolor="rgba(100,100,255,0.1)",
        line=dict(color="rgba(100,100,255,0.5)", width=1.5, dash="dot")))

    fig_err.update_layout(
        template="plotly_dark",
        title="Unilateral Approximation Error vs |S₁₂|",
        xaxis_title="|S₁₂| (reverse isolation)",
        yaxis_title="Gain (dB)",
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    mo.ui.plotly(fig_err)
    return (fig_err,)


# ---------------------------------------------------------------------------
# Section 14 — Design strategies
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Gain–Stability Trade-off and Design Strategies

    ### 14.1 Why does feedback cause instability?

    Instability arises when $S_{12} \neq 0$: the output signal feeds back to the input,
    creating a loop.  If the loop gain $|S_{12}S_{21}\Gamma_S\Gamma_L|$ approaches
    unity with the right phase, the denominator of the transducer gain
    $|(1-S_{11}\Gamma_S)(1-S_{22}\Gamma_L) - S_{12}S_{21}\Gamma_S\Gamma_L|$
    approaches zero — i.e., $G_T \to \infty$, which signals oscillation.

    The Rollett factor $K$ quantifies the "distance" from this catastrophe.

    ### 14.2 Techniques to improve stability

    | Technique | Effect on S-parameters | Effect on $K$ |
    |---|---|---|
    | **Resistive loading** (e.g. shunt $R$ at output) | Reduces $\|S_{22}\|$, absorbs reflected power | Increases $K$ |
    | **Source degeneration** (e.g. emitter/source inductance) | Reduces $S_{12}$ by adding negative feedback | Increases $K$ (reduces $\|S_{12}S_{21}\|$ in denominator) |
    | **Cascode topology** | Dramatically reduces $S_{12}$ (near-zero reverse isolation) | Large $K$; near-unilateral |
    | **Neutralisation** (cross-coupled capacitance) | Cancels $C_{gd}$ feedback path | Reduces $S_{12}$, increases $K$ |

    ### 14.3 The fundamental trade-off

    **Large $|S_{21}|$ with small $|S_{12}|$** tends to produce large $K$, because:

    $$K = \frac{1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2}{2|S_{12}S_{21}|}$$

    The numerator is bounded by approximately 1 (for well-matched devices), while the
    denominator $2|S_{12}S_{21}|$ is maximised when both $S_{12}$ and $S_{21}$ are large.
    But the designer typically wants large $S_{21}$ (gain), so **minimising $S_{12}$**
    is the primary lever for stability.

    For an unconditionally stable device ($K > 1$):

    $$G_0 = |S_{21}|^2 \leq \text{MAG} = \left|\frac{S_{21}}{S_{12}}\right|(K - \sqrt{K^2-1}) \leq \text{MSG} = \left|\frac{S_{21}}{S_{12}}\right|$$

    The designer's strategy:
    - If $K > 1$: design matching networks for simultaneous conjugate match → achieve **MAG**
    - If $K < 1$: either (a) accept a gain less than MSG with careful termination selection
      within the stable region, or (b) add resistive networks to increase $K$ above 1,
      at the cost of reduced gain
    """)
    return

