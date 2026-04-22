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
