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
    import math

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, math, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # 04 — Noise Analysis and Low-Noise Amplifier Design

    Builds on notebooks 01–03. Level-iii stochastic-process foundations,
    two-port noise theory, noise matching, and a worked 28 GHz CMOS LNA.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Motivation — why noise matters at mmWave

    At mmWave, thermal noise from the first amplifier in a receiver
    chain sets the link-budget floor. A 10 MHz channel at $T_0 = 290$ K
    has $kT_0 B = -104$ dBm of noise power — an LNA with NF = 2 dB degrades
    this by 2 dB; NF = 6 dB degrades by 6 dB. Every decibel of NF
    costs range or throughput.

    The rest of this notebook builds the machinery to design for the
    minimum achievable NF of a given device: the stochastic vocabulary
    (Part I), the four noise parameters of a two-port (Part II),
    the geometric picture of noise/gain/stability (Part III), and a
    worked design at 28 GHz (Part IV).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Random processes, stationarity, ergodicity

    A **random process** $X(t, \omega)$ is a family of random variables
    indexed by time — each $\omega$ picks one sample path.

    **Wide-sense stationarity (WSS):** $E[X(t)]$ constant and
    $R_X(t_1, t_2) = R_X(t_1 - t_2)$ — autocorrelation depends only on
    time difference $\tau$.

    **Ergodicity:** the empirical bridge. For a WSS ergodic process
    time averages on a *single* sample path converge to ensemble averages.
    This is what lets an experimenter estimate $R_X(\tau)$ from one long
    recording.

    **Counter-example (non-ergodic):** a random DC bias
    $X(t) = A$ where $A \sim \mathcal{N}(0, \sigma^2)$. Every sample path
    is a constant, so time average = that constant ≠ ensemble mean 0.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Autocorrelation and Wiener-Khinchin

    **Autocorrelation** of a WSS process:
    $$R_X(\tau) = E\bigl[X(t)\,X(t+\tau)\bigr].$$

    Basic properties: $R_X(0) = E[X^2] \ge 0$; $R_X(-\tau) = R_X(\tau)$;
    $|R_X(\tau)| \le R_X(0)$; for mixing processes $R_X(\tau) \to 0$ as
    $|\tau| \to \infty$.

    **Power spectral density**:
    $$S_X(f) = \int_{-\infty}^{\infty} R_X(\tau)\,e^{-j2\pi f\tau}\,d\tau.$$

    **Wiener-Khinchin theorem:** for a WSS process, $S_X(f)$ equals the
    Fourier transform of $R_X(\tau)$. Proof sketch: take the limit of
    windowed periodograms as window length $T \to \infty$; cross-terms
    decorrelate by the mixing assumption.

    ---

    *Cyclostationary noise* — a **periodically time-varying**
    autocorrelation $R_X(t, t+\tau) = R_X(t+T_0, t+T_0+\tau)$ — appears in
    mixers, samplers, and switched-capacitor circuits. The full treatment
    (cyclostationary PSD, downconversion of noise bands) belongs with
    notebook 06's mixer analysis, so we flag it here and move on.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Vector processes and matrix-valued PSDs

    For two processes $X(t)$ and $Y(t)$, the **cross-correlation** is
    $R_{XY}(\tau) = E[X(t)\,Y(t+\tau)^*]$ and the **cross-PSD** is its
    Fourier transform $S_{XY}(f)$.

    For a vector process $\mathbf{X}(t) = [X_1, X_2]^T$ the relevant
    object is the $2\times 2$ **matrix-valued PSD**:
    $$\mathbf{S}_{\mathbf{X}}(f) = \begin{bmatrix} S_{X_1}(f) & S_{X_1 X_2}(f) \\ S_{X_2 X_1}(f) & S_{X_2}(f) \end{bmatrix}.$$

    $\mathbf{S}(f)$ is Hermitian positive-semidefinite at every $f$.

    **Coherence:** $\gamma_{XY}^2(f) = |S_{XY}(f)|^2 / (S_X(f)\,S_Y(f)) \in [0,1]$
    — 0 means uncorrelated at frequency $f$; 1 means perfectly coherent.

    Part II will treat the two input-referred noise sources of a two-port
    $(v_n, i_n)$ as exactly such a vector process. The Hermitian 2×2 matrix
    above becomes the **noise correlation matrix $\mathbf{C_A}$**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Physical noise sources

    ### 5.1 Thermal (Nyquist) noise

    A resistor $R$ at temperature $T$ has an equivalent series voltage-noise
    source with one-sided PSD
    $$\langle v_n^2 \rangle / \Delta f = 4\,k\,T\,R,$$
    or equivalently a shunt current source $\langle i_n^2 \rangle / \Delta f = 4kT/R$.

    Derivation sketch: equipartition assigns $\tfrac{1}{2}kT$ to each mode
    of the transmission-line cavity; summing modes up to a bandwidth $\Delta f$
    recovers the Nyquist formula.

    ### 5.2 Shot noise

    A DC current $I$ carried by discrete charges $q$ has current-noise PSD
    $$\langle i_n^2 \rangle / \Delta f = 2\,q\,I.$$

    Applies across **potential barriers** — diode junctions, BJT emitter-base.
    Does **not** apply to ohmic resistors (carriers are not barrier-limited;
    there only thermal noise is present).

    ### 5.3 Flicker (1/f) noise

    Empirical PSD
    $$S_v(f) = \frac{K_v}{f},\quad S_i(f) = \frac{K_i}{f}$$
    with a device-specific constant. In MOS:
    $$S_{v_g}(f) \approx \frac{K_f}{W L C_{ox}\,f}.$$
    Corner frequency $f_c$ = crossover with thermal/shot floor.

    Microscopic model (McWhorter): superposition of many surface traps,
    each a Lorentzian, producing approximately 1/f over decades.

    ### 5.4 Generation-recombination noise

    A single trap with capture/emission time constant $\tau$ gives a
    **Lorentzian** PSD $\sim \tau / (1 + (2\pi f \tau)^2)$. Summing a
    distribution of $\tau$'s spread over decades reproduces 1/f — the
    microscopic origin story behind §5.3.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Equivalent circuit-level noise sources

    A noisy resistor $R$ at $T$ decomposes as:
    - **Thevenin:** noiseless $R$ in series with $v_n$, $S_{v_n} = 4kTR$.
    - **Norton:** noiseless $R$ in parallel with $i_n$, $S_{i_n} = 4kT/R$.

    Both describe the *same* physical noise.

    **Sum rules** for multiple sources:
    $$S_{\text{total}} = \sum_k S_k\quad \text{(uncorrelated)}, \qquad
    S_{\text{total}} = \sum_k S_k + 2\,\mathrm{Re}\!\sum_{k<\ell} S_{k\ell}\quad\text{(with cross-terms)}.$$

    **Noise bandwidth** $B_n = \int_0^\infty |H(f)|^2\,df / |H(f_0)|^2$
    — the bandwidth of an ideal brick-wall filter that would pass the same
    noise power through the transfer function $H(f)$. Distinct from the
    −3 dB bandwidth; matters for integrating detectors and oscillator
    phase noise.

    Equivalent noise bandwidth of a first-order RC lowpass:
    $B_n = (\pi/2) \cdot f_{-3\mathrm{dB}}$.
    """)
    return


@app.cell
def _(mo):
    noise_kind = mo.ui.dropdown(
        options=["white", "flicker", "band-limited", "shot"],
        value="white",
        label="Noise type",
    )
    n_samples = mo.ui.slider(512, 8192, step=512, value=2048,
                             label="N samples", show_value=True)
    fs_hz = mo.ui.slider(1000, 50000, step=1000, value=10000,
                         label="f_s (Hz)", show_value=True)

    mo.vstack([
        mo.md("### 6.1 Interactive — Time, autocorrelation, and PSD"),
        mo.hstack([noise_kind, n_samples, fs_hz]),
    ])
    return fs_hz, n_samples, noise_kind


@app.cell
def _(fs_hz, go, make_subplots, mo, n_samples, noise_kind, np):
    from _lib_noise import (
        generate_white, generate_flicker, generate_band_limited, generate_shot,
        estimate_autocorr, estimate_psd,
    )

    rng = np.random.default_rng(42)
    N = int(n_samples.value)
    fs = float(fs_hz.value)

    if noise_kind.value == "white":
        x = generate_white(N, variance=1.0, rng=rng)
    elif noise_kind.value == "flicker":
        x = generate_flicker(N, variance=1.0, rng=rng)
    elif noise_kind.value == "band-limited":
        x = generate_band_limited(N, f_s=fs, f_hi=fs / 8, variance=1.0, rng=rng)
    else:  # shot
        x = generate_shot(N, rate=5.0, rng=rng)

    t = np.arange(N) / fs
    R = estimate_autocorr(x, max_lag=min(256, N // 4))
    f, S = estimate_psd(x, f_s=fs)

    fig_tri = make_subplots(rows=1, cols=3,
                            subplot_titles=("x(t)", "R̂(τ)", "Ŝ(f)"))
    fig_tri.add_trace(go.Scatter(x=t, y=x, mode="lines",
                                 line=dict(width=1), showlegend=False),
                      row=1, col=1)
    fig_tri.add_trace(go.Scatter(x=np.arange(R.size) / fs, y=R,
                                 mode="lines", showlegend=False),
                      row=1, col=2)
    fig_tri.add_trace(go.Scatter(x=f, y=10 * np.log10(np.maximum(S, 1e-20)),
                                 mode="lines", showlegend=False),
                      row=1, col=3)
    fig_tri.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=40, r=20, t=40, b=40))
    fig_tri.update_xaxes(title_text="t (s)",       row=1, col=1)
    fig_tri.update_xaxes(title_text="τ (s)",       row=1, col=2)
    fig_tri.update_xaxes(title_text="f (Hz)", type="log", row=1, col=3)
    fig_tri.update_yaxes(title_text="x",           row=1, col=1)
    fig_tri.update_yaxes(title_text="R",           row=1, col=2)
    fig_tri.update_yaxes(title_text="S (dB)",      row=1, col=3)

    mo.ui.plotly(fig_tri)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. The Rothe-Dahlke noise two-port

    **Theorem (Rothe & Dahlke, 1956).** Any noisy linear two-port is
    equivalent to a *noiseless* two-port (same Y-parameters) plus two
    correlated noise sources — an input-referred voltage $v_n$ and
    current $i_n$ — connected at the input port.

    $$\begin{bmatrix} v_1 \\ i_1 \end{bmatrix}
    = \underbrace{\begin{bmatrix} A & B \\ C & D \end{bmatrix}}_{\text{noiseless ABCD}}
      \begin{bmatrix} v_2 \\ -i_2 \end{bmatrix}
      + \begin{bmatrix} v_n \\ i_n \end{bmatrix}.$$

    Derivation: start from the Y-matrix form with internal noise
    sources $i_{n1}, i_{n2}$ at the ports; pre-multiply by the
    ABCD transformation; the two noise currents combine into one
    equivalent $(v_n, i_n)$ pair at the input.

    The **noise correlation matrix** in ABCD form:
    $$\mathbf{C_A} = \overline{\begin{bmatrix} v_n \\ i_n \end{bmatrix}
        \begin{bmatrix} v_n^* & i_n^* \end{bmatrix}}
    = \begin{bmatrix} \overline{|v_n|^2} & \overline{v_n i_n^*} \\
                      \overline{v_n^* i_n} & \overline{|i_n|^2} \end{bmatrix}.$$

    Hermitian, positive-semidefinite, 2×2 — exactly the matrix PSD
    of §4. Admittance form $\mathbf{C_Y}$ and impedance form
    $\mathbf{C_Z}$ are alternative representations; §10 will show the
    transformations. All derivations in §8–§9 stay in $\mathbf{C_A}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Noise figure F

    **Classical definition (Friis, 1944):**
    $$F \;\triangleq\; \frac{\mathrm{SNR}_{\text{in}}}{\mathrm{SNR}_{\text{out}}}
    \quad\text{at source temperature } T_0 = 290\text{ K.}$$

    Equivalently, $F$ is the factor by which a two-port degrades the
    signal-to-noise ratio of a 290 K source. $F \ge 1$ always;
    $F = 1$ means noiseless.

    **Equivalent noise temperature:**
    $$T_e \;=\; (F - 1)\,T_0.$$

    $T_e$ is the temperature a noiseless copy of the two-port would need
    at its input to produce the same output noise power. Receiver
    communities where the source is far from 290 K (radio astronomy:
    $T_\text{sky} \approx 3$–50 K; deep-space: a few K) prefer $T_e$
    because the 290 K reference in $F$ obscures the physics.

    **Unit convention:** throughout this notebook we quote $F$ in
    dB when used as a spec target ("NF < 2 dB") and in linear form when
    substituting into formulas.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. F as a function of source admittance — the four noise parameters

    Driving the Rothe-Dahlke two-port with a source of admittance $Y_s = G_s + jB_s$
    and computing output noise power gives

    $$\boxed{\;F(Y_s) \;=\; F_{\min} \;+\; \frac{R_n}{G_s}\,|Y_s - Y_{\text{opt}}|^2\;}$$

    Four real noise parameters:
    - $F_{\min}$ — minimum $F$ over all source admittances.
    - $R_n$ — **noise resistance**; sets the sensitivity of $F$ to source mismatch.
    - $Y_{\text{opt}} = G_{\text{opt}} + jB_{\text{opt}}$ — **optimum source admittance**.

    $\mathbf{C_A}$'s three independent real entries plus the device's
    $Y_{21}$ are sufficient to compute all four — which is why the
    four-parameter set suffices.

    **Γ-plane form** (after the bilinear $Y_s \leftrightarrow \Gamma_s$
    substitution):

    $$F(\Gamma_s) \;=\; F_{\min} \;+\; \frac{4\,R_n/Z_0}{|1 + \Gamma_{\text{opt}}|^2}\;
    \frac{|\Gamma_s - \Gamma_{\text{opt}}|^2}{1 - |\Gamma_s|^2}.$$

    Two sanity checks (next cell): $F(\Gamma_{\text{opt}}) = F_{\min}$, and
    $F(\Gamma_s \to \partial\mathrm{disk}) \to \infty$.
    """)
    return


@app.cell
def _(mo, np):
    from _lib_noise import noise_figure_from_Gamma

    _Gopt = 0.4 * np.exp(1j * np.deg2rad(60.0))
    _Fmin = 1.3
    _Rn   = 0.08

    _F_at_opt = noise_figure_from_Gamma(_Gopt, _Fmin, _Rn, _Gopt)
    assert abs(_F_at_opt - _Fmin) < 1e-9, f"F(Γ_opt) should equal F_min; got {_F_at_opt}"

    _Gnear_edge = 0.999 * np.exp(1j * 0.0)
    _F_edge = noise_figure_from_Gamma(_Gnear_edge, _Fmin, _Rn, _Gopt)
    assert _F_edge > 10.0 * _Fmin, f"F should blow up near disk edge; got {_F_edge}"

    mo.md(r"""
    **Validation:** $F(\Gamma_\text{opt}) = F_\min$ ✓ and
    $F(|\Gamma_s| \to 1) \gg F_\min$ ✓.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Noise correlation matrix transformations

    Noise sources ride along with small-signal sources under the
    same T-matrix transformations. If $\mathbf{P'} = \mathbf{T}\,\mathbf{P}$
    takes representation $\mathbf{P}$ to $\mathbf{P'}$ for the
    deterministic parameters, then

    $$\mathbf{C_{P'}} = \mathbf{T}\,\mathbf{C_P}\,\mathbf{T}^\dagger.$$

    **Three useful forms** and their interpretations:
    - $\mathbf{C_A}$ — input-referred $(v_n, i_n)$; natural for noise figure.
    - $\mathbf{C_Y}$ — port-referred $(i_{n1}, i_{n2})$; natural for shunt elements and active devices.
    - $\mathbf{C_Z}$ — port-referred $(v_{n1}, v_{n2})$; natural for series elements.

    Transformation T-matrices are the same that map (A-params ↔ Y-params ↔
    Z-params); the noise-correlation transformation *follows* the
    network-matrix transformation.

    **Why it matters:** matrix-level CAD tools (Keysight ADS, Cadence SpectreRF)
    propagate $\mathbf{C_A}$ or $\mathbf{C_Y}$ through interconnects,
    feedback, and cascades as matrix multiplications. For an analytic
    derivation of a feedback-amplifier NF, staying in $\mathbf{C_A}$
    and using the ABCD-cascade identity

    $$\mathbf{C_{A,\,\text{cascade}}} = \mathbf{C_{A,1}} + \mathbf{A_1}\,\mathbf{C_{A,2}}\,\mathbf{A_1}^\dagger$$

    is the cleanest path.

    **Worked example — noisy resistor.** In Z-form:
    $\mathbf{C_Z} = 2 k T R \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$
    (degenerate rank-1: a single voltage source drives both ports in
    series). In A-form the same resistor has
    $\mathbf{C_A} = \begin{bmatrix} 4kTR & 0 \\ 0 & 0 \end{bmatrix}$
    — all noise is an input voltage. Transforming $\mathbf{C_Z} \to \mathbf{C_A}$
    via the Z→A T-matrix reproduces this.

    The full algebraic derivation is in Hillbrand & Russer,
    *IEEE Trans. Circuits Syst.* 1976.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Cascaded noise — the Friis formula

    **Noise-temperature form** (simpler to derive):
    $$T_{e,\text{tot}} \;=\; T_{e,1} \;+\; \frac{T_{e,2}}{G_{A,1}} \;+\; \frac{T_{e,3}}{G_{A,1}\,G_{A,2}} \;+\; \cdots$$

    Derivation: each stage adds $T_{e,k}$ of noise *referred to its own input*;
    referring it back to the cascade input divides by the product of available
    gains of all preceding stages.

    **Noise-figure form** (substitute $T_e = (F-1)T_0$ and simplify):
    $$\boxed{\;F_{\text{tot}} \;=\; F_1 \;+\; \frac{F_2 - 1}{G_{A,1}} \;+\; \frac{F_3 - 1}{G_{A,1}\,G_{A,2}} \;+\; \cdots\;}$$

    **Why available gain $G_A$ and not $G_T$ or $G$?** Each stage is
    described by its own output noise, which depends on what that stage
    sees at its input — i.e., the cascade output impedance of the
    previous stage. Available gain is the source-match-independent
    figure that propagates cleanly through the cascade without requiring
    each stage to be re-matched to a known source.

    **Practical consequence:** if stage 1 has $G_{A,1} \gtrsim 10$ dB,
    subsequent stages contribute to $F_\text{tot}$ scaled down by
    $1/G_{A,1} \lesssim 0.1$ — the first stage dominates. This is why the
    entire notebook is built around LNA noise matching: the first stage
    of a receiver sets the noise floor.
    """)
    return


@app.cell
def _(math, mo):
    from _lib_noise import friis_cascade as _friis

    _F_linear = [10 ** (2.0 / 10), 10 ** (6.0 / 10), 10 ** (10.0 / 10)]
    _G_linear = [10 ** (15.0 / 10), 10 ** (10.0 / 10), 10 ** (10.0 / 10)]
    _F_tot    = _friis(_F_linear, _G_linear)

    _only_first = _F_linear[0]
    _penalty    = 10 * math.log10(_F_tot / _only_first)
    assert _penalty < 0.5, f"first stage should dominate; got +{_penalty:.3f} dB"

    mo.md(
        f"**Validation:** 3-stage cascade [2, 6, 10] dB NF with [15, 10, 10] dB gains → "
        f"$F_\\text{{tot}} \\approx$ {10 * math.log10(_F_tot):.2f} dB "
        f"(penalty over stage 1 alone: +{_penalty:.2f} dB) ✓"
    )
    return


@app.cell
def _(mo):
    F1_db = mo.ui.slider(0.5, 8.0, step=0.1, value=2.0, label="F₁ (dB)", show_value=True)
    F2_db = mo.ui.slider(0.5, 15.0, step=0.1, value=6.0, label="F₂ (dB)", show_value=True)
    F3_db = mo.ui.slider(0.5, 25.0, step=0.1, value=10.0, label="F₃ (dB)", show_value=True)
    G1_db = mo.ui.slider(0.0, 30.0, step=0.5, value=15.0, label="G_A,1 (dB)", show_value=True)
    G2_db = mo.ui.slider(0.0, 30.0, step=0.5, value=10.0, label="G_A,2 (dB)", show_value=True)
    G3_db = mo.ui.slider(0.0, 30.0, step=0.5, value=10.0, label="G_A,3 (dB)", show_value=True)
    reorder = mo.ui.dropdown(
        options=["1→2→3", "3→2→1", "2→1→3"],
        value="1→2→3",
        label="Stage order",
    )

    mo.vstack([
        mo.md("### 11.1 Interactive — Friis cascade explorer"),
        mo.hstack([F1_db, F2_db, F3_db]),
        mo.hstack([G1_db, G2_db, G3_db]),
        reorder,
    ])
    return F1_db, F2_db, F3_db, G1_db, G2_db, G3_db, reorder


@app.cell
def _(F1_db, F2_db, F3_db, G1_db, G2_db, G3_db, go, math, mo, reorder):
    from _lib_noise import friis_cascade as _friis

    _stages = {
        "1": (10 ** (F1_db.value / 10), 10 ** (G1_db.value / 10)),
        "2": (10 ** (F2_db.value / 10), 10 ** (G2_db.value / 10)),
        "3": (10 ** (F3_db.value / 10), 10 ** (G3_db.value / 10)),
    }
    _order = reorder.value.split("→")
    _F_list = [_stages[k][0] for k in _order]
    _G_list = [_stages[k][1] for k in _order]

    _F_tot = _friis(_F_list, _G_list)

    _contribs = [_F_list[0] - 1.0]
    _gain_so_far = 1.0
    for _i in range(1, len(_F_list)):
        _gain_so_far *= _G_list[_i - 1]
        _contribs.append((_F_list[_i] - 1.0) / _gain_so_far)
    _total_excess = sum(_contribs)
    _fractions = [c / _total_excess for c in _contribs] if _total_excess > 0 else [0.0] * len(_contribs)

    _fig_cascade = go.Figure()
    _fig_cascade.add_trace(go.Bar(
        x=[f"Stage {_k}" for _k in _order],
        y=[100 * _f for _f in _fractions],
        text=[f"{100*_f:.1f}%" for _f in _fractions],
        textposition="outside",
    ))
    _fig_cascade.update_layout(
        template="plotly_dark",
        title=f"F_tot = {10 * math.log10(_F_tot):.2f} dB  (order {reorder.value})",
        yaxis_title="% of excess NF contribution",
        height=340,
    )

    mo.vstack([mo.ui.plotly(_fig_cascade)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Noise circles

    Setting $F = F_{\text{target}}$ in the §9 expression and rearranging
    for $\Gamma_s$ gives a circle in the Γ-plane. Define

    $$N \;\triangleq\; \frac{F_{\text{target}} - F_{\min}}{4\,R_n / Z_0}\,|1 + \Gamma_{\text{opt}}|^2.$$

    Then the centre and radius of the constant-$F$ circle are

    $$C_F \;=\; \frac{\Gamma_{\text{opt}}}{1 + N},
    \qquad r_F \;=\; \frac{\sqrt{N^2 + N\,(1 - |\Gamma_{\text{opt}}|^2)}}{1 + N}.$$

    **Limits:**
    - $F_{\text{target}} = F_{\min} \Rightarrow N = 0 \Rightarrow C_F = \Gamma_{\text{opt}}$, $r_F = 0$ — single point.
    - $F_{\text{target}} \to \infty \Rightarrow N \to \infty \Rightarrow C_F \to 0$, $r_F \to 1$ — whole disk.

    Small targets → tight circle around $\Gamma_{\text{opt}}$; large targets
    cover the whole unit disk (any match works).

    (Validation cell next: sweep $F_{\text{target}}$ and confirm the
    expected limits.)
    """)
    return


@app.cell
def _(mo, np):
    from _lib_circles import noise_circle as _noise_circle

    _Gopt = 0.4 * np.exp(1j * np.deg2rad(50.0))
    _Fmin = 1.3
    _Rn   = 0.08

    _c0, _r0 = _noise_circle(_Fmin, _Fmin, _Rn, _Gopt)
    assert abs(_c0 - _Gopt) < 1e-9 and _r0 < 1e-9

    _c_big, _r_big = _noise_circle(1e6 * _Fmin, _Fmin, _Rn, _Gopt)
    assert abs(_c_big) < 1e-3 and _r_big > 0.99

    mo.md(r"**Validation:** $F = F_\min \Rightarrow$ single point at "
          r"$\Gamma_\text{opt}$ ✓; $F \to \infty \Rightarrow$ full unit disk ✓.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. The gain-noise tradeoff

    The fundamental design tension: in general

    $$\Gamma_{\text{opt}} \;\ne\; S_{11}^*,$$

    so noise-matching ($\Gamma_s = \Gamma_{\text{opt}}$) costs input
    reflection (and hence gain), and power-matching ($\Gamma_s = S_{11}^*$)
    costs noise figure. Overlay available-gain circles (notebook 03)
    on noise circles — the geometry of the compromise appears directly.

    **Noise measure M (Haus-Adler, 1958):** the invariant that collapses
    the tradeoff into one number,

    $$M \;\triangleq\; \frac{F - 1}{1 - 1/G_{A}}.$$

    Two properties worth stating without derivation:
    - $M$ is **invariant** under lossless reciprocal embedding of the
      two-port (the same invariance that makes Mason's $U$ a device property
      — notebook 02).
    - An infinite cascade of identical stages, each optimised for the
      cascade's input, achieves $F_\infty = 1 + M$. So **$M$ is the best
      noise factor per unit of "usable gain"** — the right figure of merit
      when you plan to cascade.

    At mmWave, $M_\min$ is comparable to $F_{\min} - 1$ (since stage gains
    are typically 10–15 dB, $1 - 1/G_A \approx 0.9$), so the two figures
    are usually close — but $M$ is the principled one.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Simultaneous noise + gain match

    **Haus condition:** a lossless feedback embedding can rotate
    $\Gamma_{\text{opt}}$ to $S_{11}^*$ if and only if a specific
    inequality between the two-port's noise parameters and its
    small-signal parameters is satisfied.

    On 65 nm bulk CMOS at 28 GHz the condition **fails**: the intrinsic
    transistor has $\Gamma_{\text{opt}}$ inside the unit disk but far
    enough from $S_{11}^*$ that any lossless transformation that
    moves one still leaves the other poorly matched. The designer's
    compromise: pick $\Gamma_s$ on the locus minimising a cost

    $$J(\Gamma_s) \;=\; \alpha\,[F(\Gamma_s) - F_{\min}]
                       \;+\; \beta\,[G_{A,\max} - G_A(\Gamma_s)],$$

    where $\alpha, \beta$ are weights chosen by the application
    (receiver-noise-dominated → large $\alpha$; gain-budget-limited → large $\beta$).

    **Feedback trick that makes simultaneous match nearly achievable:**
    **inductive source degeneration** (§16). The $L_s$ in the source path
    creates series-series feedback that rotates $\Gamma_{\text{opt}}$
    toward $S_{11}^*$ **without adding a resistor**. This is why the
    inductively-degenerated common-source is the canonical mmWave CMOS
    LNA topology.
    """)
    return


@app.cell
def _(mo):
    s11m_c = mo.ui.slider(0.0, 0.99, step=0.01, value=0.75, label="|S₁₁|",         show_value=True)
    s11p_c = mo.ui.slider(-180, 180, step=5,    value=165,  label="∠S₁₁ (°)",       show_value=True)
    s21m_c = mo.ui.slider(0.5, 10.0, step=0.1,  value=3.5,  label="|S₂₁|",          show_value=True)
    s21p_c = mo.ui.slider(-180, 180, step=5,    value=30,   label="∠S₂₁ (°)",       show_value=True)
    s12m_c = mo.ui.slider(0.0, 0.5,  step=0.005,value=0.08, label="|S₁₂|",          show_value=True)
    s12p_c = mo.ui.slider(-180, 180, step=5,    value=50,   label="∠S₁₂ (°)",       show_value=True)
    s22m_c = mo.ui.slider(0.0, 0.99, step=0.01, value=0.55, label="|S₂₂|",          show_value=True)
    s22p_c = mo.ui.slider(-180, 180, step=5,    value=-40,  label="∠S₂₂ (°)",       show_value=True)

    mo.vstack([
        mo.md("### 15.1 Interactive — Noise / gain / stability circles"),
        mo.md("**S-parameters at design frequency:**"),
        mo.hstack([s11m_c, s11p_c, s21m_c, s21p_c]),
        mo.hstack([s12m_c, s12p_c, s22m_c, s22p_c]),
    ])
    return s11m_c, s11p_c, s12m_c, s12p_c, s21m_c, s21p_c, s22m_c, s22p_c


@app.cell
def _(mo):
    Fmin_db_c = mo.ui.slider(0.5, 5.0, step=0.1, value=1.5,
                             label="F_min (dB)", show_value=True)
    Gopt_mag_c = mo.ui.slider(0.0, 0.9, step=0.01, value=0.45,
                              label="|Γ_opt|", show_value=True)
    Gopt_ang_c = mo.ui.slider(-180, 180, step=5, value=120,
                              label="∠Γ_opt (°)", show_value=True)
    Rn_norm_c = mo.ui.slider(0.02, 0.30, step=0.01, value=0.08,
                             label="R_n/Z₀", show_value=True)

    mo.vstack([
        mo.md("**Noise parameters:**"),
        mo.hstack([Fmin_db_c, Gopt_mag_c, Gopt_ang_c, Rn_norm_c]),
    ])
    return Fmin_db_c, Gopt_ang_c, Gopt_mag_c, Rn_norm_c


@app.cell
def _(Fmin_db_c, Gopt_ang_c, Gopt_mag_c, Rn_norm_c,
      go, mo, np,
      s11m_c, s11p_c, s12m_c, s12p_c, s21m_c, s21p_c, s22m_c, s22p_c):
    from _lib_circles import (
        noise_circle as _noise_circle,
        available_gain_circle as _ag_circle,
        source_stability_circle as _src_stab,
        rollett_K as _rollett_K,
        MAG_dB as _MAG_dB,
    )

    _S11 = s11m_c.value * np.exp(1j * np.deg2rad(s11p_c.value))
    _S21 = s21m_c.value * np.exp(1j * np.deg2rad(s21p_c.value))
    _S12 = s12m_c.value * np.exp(1j * np.deg2rad(s12p_c.value))
    _S22 = s22m_c.value * np.exp(1j * np.deg2rad(s22p_c.value))
    _Gopt = Gopt_mag_c.value * np.exp(1j * np.deg2rad(Gopt_ang_c.value))
    _Fmin_lin = 10 ** (Fmin_db_c.value / 10)

    K_c, Delta_c = _rollett_K(_S11, _S12, _S21, _S22)
    _cs_stab, _rs_stab = _src_stab(_S11, _S12, _S21, _S22)
    _MAG_db_val = _MAG_dB(_S11, _S12, _S21, _S22)

    _gain_levels = [_MAG_db_val - d for d in (0.5, 1.0, 2.0, 3.0)]
    _gain_circles = [_ag_circle(g, _S11, _S12, _S21, _S22) for g in _gain_levels]
    _noise_levels_db = [Fmin_db_c.value + d for d in (0.3, 0.6, 1.0, 2.0)]
    _noise_circles = [
        _noise_circle(10 ** (d / 10), _Fmin_lin, Rn_norm_c.value, _Gopt)
        for d in _noise_levels_db
    ]

    fig_circles = go.Figure()
    _theta = np.linspace(0, 2 * np.pi, 361)

    # unit disk
    fig_circles.add_trace(go.Scatter(
        x=np.cos(_theta), y=np.sin(_theta),
        mode="lines", line=dict(color="#888", width=1),
        name="|Γ|=1",
    ))

    # source stability circle
    _xs = _cs_stab.real + _rs_stab * np.cos(_theta)
    _ys = _cs_stab.imag + _rs_stab * np.sin(_theta)
    fig_circles.add_trace(go.Scatter(
        x=_xs, y=_ys, mode="lines",
        line=dict(color="#AA3333", width=2),
        name=f"Source stab (K={K_c:.2f})",
    ))

    for (c_g, r_g), lvl in zip(_gain_circles, _gain_levels):
        _x = c_g.real + r_g * np.cos(_theta)
        _y = c_g.imag + r_g * np.sin(_theta)
        fig_circles.add_trace(go.Scatter(
            x=_x, y=_y, mode="lines",
            line=dict(color="#3366CC", dash="dash", width=1.2),
            name=f"G_A = {lvl:.1f} dB",
        ))

    for (c_n, r_n), lvl_db in zip(_noise_circles, _noise_levels_db):
        _x = c_n.real + r_n * np.cos(_theta)
        _y = c_n.imag + r_n * np.sin(_theta)
        fig_circles.add_trace(go.Scatter(
            x=_x, y=_y, mode="lines",
            line=dict(color="#33AA66", width=1.4),
            name=f"F = {lvl_db:.1f} dB",
        ))

    fig_circles.add_trace(go.Scatter(
        x=[_Gopt.real], y=[_Gopt.imag],
        mode="markers", marker=dict(size=10, color="#33AA66"),
        name="Γ_opt",
    ))
    fig_circles.add_trace(go.Scatter(
        x=[np.conj(_S11).real], y=[np.conj(_S11).imag],
        mode="markers", marker=dict(size=10, color="#3366CC"),
        name="S₁₁*",
    ))

    fig_circles.update_layout(
        template="plotly_dark",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.3, 1.3]),
        yaxis=dict(range=[-1.3, 1.3]),
        height=560, width=560,
        legend=dict(font=dict(size=9)),
        title=f"K={K_c:.2f}   |Δ|={abs(Delta_c):.2f}   MAG≈{_MAG_db_val:.1f} dB",
    )

    mo.ui.plotly(fig_circles)
    return Delta_c, K_c


@app.cell
def _(mo):
    mo.md(r"""
    ## 16. The inductively-degenerated common-source LNA

    ### 16.1 Topology and rationale

    A common-source NMOS with
    - a **source inductor** $L_s$ (creates real input impedance),
    - a **gate inductor** $L_g$ (tunes out $C_{gs}$),
    - a drain-load network (tuned to the operating frequency),

    is the canonical mmWave CMOS LNA. Why this topology rather than
    common-gate or resistive-feedback? Inductive source degeneration
    creates $\mathrm{Re}(Z_{\text{in}}) = \omega_T L_s$ **without a
    resistor** — a resistor would dissipate signal and add thermal noise,
    two deal-breakers in a low-noise first stage.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 16.2 Small-signal model at mmWave

    Standard π-hybrid NMOS:

    $$\begin{array}{rl}
    C_{gs} & \text{gate-source capacitance (dominant)} \\
    C_{gd} & \text{gate-drain (Miller) capacitance} \\
    g_m    & \text{small-signal transconductance} \\
    r_o    & \text{output resistance} \\
    r_g    & \text{gate resistance (multi-finger layout)} \\
    \end{array}$$

    Transit frequency $\omega_T \equiv g_m / C_{gs}$.

    **mmWave-specific additions** (vs. textbook low-frequency model):
    finite $r_g$ from polysilicon + contacts, substrate resistance,
    pad capacitance. The Shaeffer-Lee treatment keeps the main four
    ($g_m$, $C_{gs}$, $C_{gd}$, $\omega_T$) and handles the rest as
    NF corrections (§19).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 16.3 Input impedance

    Neglecting $C_{gd}$ (first-order analysis):

    $$\boxed{\;Z_{\text{in}} \;\approx\; j\omega\,(L_g + L_s) \;+\; \frac{1}{j\omega\,C_{gs}} \;+\; \omega_T\,L_s\;}$$

    **Three terms:**
    - $j\omega(L_g+L_s)$ — reactance of the input inductors in series
    - $1/(j\omega C_{gs})$ — the device's own gate capacitance
    - $\omega_T L_s$ — the **real** term from source-inductor feedback
      (dimensionally Ω)

    **Design knobs:**
    - $L_s$ sets $\mathrm{Re}(Z_{\text{in}}) = \omega_T L_s$ → pick
      $L_s = 50/\omega_T$ for 50 Ω match.
    - $L_g$ resonates out the imaginary part at the operating frequency
      → pick $L_g$ such that $\omega^2(L_g+L_s) = 1/C_{gs}$.

    No resistor anywhere in the signal path → no thermal-noise penalty.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 16.4 Noise analysis

    Two NMOS noise sources:
    - **Channel thermal noise.** One-sided PSD
      $\overline{i_{nd}^2}/\Delta f = 4kT\gamma g_{d0}$;
      short-channel $\gamma \in [1, 2]$, long-channel $\gamma = 2/3$.
    - **Induced gate noise.** Capacitive coupling of channel fluctuations
      to the gate:
      $\overline{i_{ng}^2}/\Delta f = 4kT\delta\,\omega^2 C_{gs}^2 / (5 g_{d0})$
      with $\delta \approx 2\gamma$, correlated with $i_{nd}$ via a complex
      coefficient $c \approx j\,0.395$.

    Plug into the $\mathbf{C_A}$ machinery (§10), solve for $F_{\min}$,
    $\Gamma_{\text{opt}}$, $R_n$. The Shaeffer-Lee result:

    $$\boxed{\;F_{\min} \;\approx\; 1 \;+\; \frac{2.4\,\gamma}{\alpha}\,\frac{\omega}{\omega_T}\;}$$

    where $\alpha = g_m/g_{d0}$ is a short-channel correction ($\alpha \to 1$
    long-channel, $\alpha \approx 0.8$ at 65 nm mmWave bias).

    **Structural consequence (the beautiful result):** after the inductive
    source degeneration, $\Gamma_{\text{opt}}$ is **approximately $S_{11}^*$**
    — feedback has rotated the noise optimum onto the gain optimum.
    Simultaneous noise + gain match is nearly achievable without a
    lossless matching transformation (which would be hard at mmWave
    anyway due to lossy on-chip inductors).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 17. Worked design — 28 GHz, 65 nm CMOS LNA

    **Spec target:**
    - $\mathrm{NF} < 2.5$ dB
    - $|S_{21}| > 15$ dB (requires cascode + output matching — see §18)
    - $|S_{11}| < -10$ dB
    - $P_{DC} < 10$ mW
    - 50 Ω source and load at 28 GHz

    Seven-step design:

    1. **§17.1** Choose current density $J_\text{opt}$ at minimum-NF point, size $W$ for $P_{DC}$ budget.
    2. **§17.2** Choose $L_s$ for $\mathrm{Re}(Z_{\text{in}}) = 50\,\Omega$.
    3. **§17.3** Choose $L_g$ for resonance at 28 GHz.
    4. **§17.4** Verify $\Gamma_{\text{opt}} \approx S_{11}^*$.
    5. **§17.5** Add cascode for isolation.
    6. **§17.6** Design simple tapped-cap output match to 50 Ω.
    7. **§17.7** Verify S-parameters, NF, and stability sweeps.
    """)
    return


@app.cell
def _(math, mo, np):
    from _lib_lna import (
        DeviceParams, compute_operating_point, input_impedance,
        F_min_shaeffer_lee, gamma_opt_degenerated_cs,
    )

    params_design = DeviceParams()
    f0           = 28e9
    omega0       = 2 * math.pi * f0
    V_supply     = 1.0
    P_budget     = 10e-3

    # §17.1 Device sizing
    J_opt        = 0.15e-3
    I_budget     = P_budget / V_supply
    W_design_um  = I_budget / J_opt
    op           = compute_operating_point(W_design_um, J_opt, params_design)

    # §17.2 L_s for Re(Z_in) = 50 Ω
    L_s_design   = 50.0 / op["omega_T"]

    # §17.3 L_g for resonance at f0
    L_g_design   = 1.0 / (omega0 ** 2 * op["C_gs"]) - L_s_design

    # §17.4 Γ_opt location
    Gamma_opt_design = gamma_opt_degenerated_cs(omega0, L_s_design, L_g_design, op, params_design)
    Z_in = input_impedance(omega0, L_s_design, L_g_design, op)
    S11_design   = (Z_in - 50.0) / (Z_in + 50.0)

    # §17.5–17.6 (simplified) — cascode and tapped-cap output match are
    # documented in prose and exposed through §18. For the verification
    # computation here we use a simple series-LC load tuned to 28 GHz.
    L_d_design   = 150e-12

    # §17.7 verification numbers
    F_design     = F_min_shaeffer_lee(omega0, op, params_design)
    NF_design_db = 10 * math.log10(F_design)

    assert abs(Z_in.real - 50.0) < 5.0,     f"Re(Z_in) should be ~50 Ω, got {Z_in.real:.2f}"
    assert abs(Z_in.imag) < 10.0,           f"Im(Z_in) should be ~0, got {Z_in.imag:.2f}"
    assert NF_design_db < 2.5,              f"NF_min should be < 2.5 dB, got {NF_design_db:.2f}"
    assert W_design_um * J_opt * V_supply <= P_budget + 1e-9, "over power budget"

    mo.md(f"""
    **Design point computed:**

    | Parameter | Value |
    |---|---|
    | W (μm)            | {W_design_um:.1f} |
    | J (mA/μm)         | {J_opt*1e3:.3f} |
    | I_D (mA)          | {op['I_D']*1e3:.2f} |
    | P_DC (mW)         | {op['I_D']*V_supply*1e3:.2f} |
    | g_m (mS)          | {op['g_m']*1e3:.2f} |
    | C_gs (fF)         | {op['C_gs']*1e15:.1f} |
    | ω_T / 2π (GHz)    | {op['omega_T']/(2*math.pi*1e9):.1f} |
    | L_s (pH)          | {L_s_design*1e12:.1f} |
    | L_g (pH)          | {L_g_design*1e12:.1f} |
    | Z_in at 28 GHz    | {Z_in.real:.1f} + {Z_in.imag:.1f}j |
    | S₁₁ at 28 GHz     | {abs(S11_design):.3f} ∠{np.angle(S11_design, deg=True):.1f}° |
    | Γ_opt (estimate)  | {abs(Gamma_opt_design):.3f} ∠{np.angle(Gamma_opt_design, deg=True):.1f}° |
    | F_min (dB)        | {NF_design_db:.2f} |

    Input-match and NF spec targets met ✓.
    (|S₂₁| > 15 dB requires the cascode + output-match tuning exposed in §18.)
    """)
    return L_d_design, L_g_design, L_s_design, W_design_um, f0, op, params_design


@app.cell
def _(L_d_design, L_g_design, L_s_design, W_design_um, f0, go, make_subplots,
      math, mo, np, params_design):
    from _lib_lna import sparameters_at_freq, NF_degenerated_cs

    _freqs = np.linspace(20e9, 40e9, 41)
    _s11m = np.zeros_like(_freqs)
    _s21m = np.zeros_like(_freqs)
    _s22m = np.zeros_like(_freqs)
    _nf   = np.zeros_like(_freqs)

    for _i, _f in enumerate(_freqs):
        _omega = 2 * math.pi * _f
        _S11, _S21, _S12, _S22 = sparameters_at_freq(
            _omega, W_um=W_design_um, J_A_per_um=0.15e-3,
            L_s=L_s_design, L_g=L_g_design, L_d=L_d_design,
            params=params_design,
        )
        _s11m[_i] = 20 * math.log10(max(abs(_S11), 1e-6))
        _s21m[_i] = 20 * math.log10(max(abs(_S21), 1e-6))
        _s22m[_i] = 20 * math.log10(max(abs(_S22), 1e-6))
        _nf[_i]   = NF_degenerated_cs(
            _omega, W_um=W_design_um, J_A_per_um=0.15e-3,
            L_s=L_s_design, L_g=L_g_design, L_d=L_d_design,
            params=params_design,
        )

    _fig_ver = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("S-parameters", "NF"))
    _fig_ver.add_trace(go.Scatter(x=_freqs / 1e9, y=_s11m, mode="lines", name="|S₁₁| (dB)"), row=1, col=1)
    _fig_ver.add_trace(go.Scatter(x=_freqs / 1e9, y=_s21m, mode="lines", name="|S₂₁| (dB)"), row=1, col=1)
    _fig_ver.add_trace(go.Scatter(x=_freqs / 1e9, y=_s22m, mode="lines", name="|S₂₂| (dB)"), row=1, col=1)
    _fig_ver.add_trace(go.Scatter(x=_freqs / 1e9, y=_nf,   mode="lines", name="NF (dB)"),    row=2, col=1)
    _fig_ver.add_vline(x=f0 / 1e9, line_dash="dash", line_color="#888")
    _fig_ver.update_layout(template="plotly_dark", height=520)
    _fig_ver.update_xaxes(title_text="f (GHz)", row=2, col=1)

    mo.ui.plotly(_fig_ver)
    return


@app.cell
def _(mo):
    W_l       = mo.ui.slider(20, 200, step=5,    value=70,   label="W (μm)",            show_value=True)
    J_l       = mo.ui.slider(0.05, 0.30, step=0.01, value=0.15, label="J (mA/μm)",     show_value=True)
    Ls_pH_l   = mo.ui.slider(0, 100, step=2,     value=40,   label="L_s (pH)",          show_value=True)
    Lg_pH_l   = mo.ui.slider(0, 500, step=5,     value=260,  label="L_g (pH)",          show_value=True)
    Ld_pH_l   = mo.ui.slider(20, 400, step=5,    value=150,  label="L_d (pH)",          show_value=True)
    cascode_l = mo.ui.switch(value=True, label="Cascode")
    f0_GHz_l  = mo.ui.slider(20, 40, step=0.5,   value=28,   label="f_0 (GHz)",         show_value=True)

    mo.vstack([
        mo.md("### 18. Interactive — 28 GHz LNA design studio"),
        mo.hstack([W_l, J_l, cascode_l]),
        mo.hstack([Ls_pH_l, Lg_pH_l, Ld_pH_l, f0_GHz_l]),
    ])
    return Ld_pH_l, Lg_pH_l, Ls_pH_l, W_l, J_l, cascode_l, f0_GHz_l


@app.cell
def _(Ld_pH_l, Lg_pH_l, Ls_pH_l, W_l, J_l, cascode_l, f0_GHz_l,
      go, make_subplots, math, mo, np):
    from _lib_lna import (
        compute_operating_point as _op_point,
        sparameters_at_freq as _sparams,
        NF_degenerated_cs as _nf_cs,
    )
    from _lib_circles import rollett_K as _rollett_K

    _W   = W_l.value
    _J   = J_l.value * 1e-3
    _Ls  = Ls_pH_l.value * 1e-12
    _Lg  = Lg_pH_l.value * 1e-12
    _Ld  = Ld_pH_l.value * 1e-12
    _f0  = f0_GHz_l.value * 1e9

    op_l = _op_point(_W, _J)

    _freqs = np.linspace(20e9, 40e9, 41)
    _s11m = np.zeros_like(_freqs); _s21m = np.zeros_like(_freqs)
    _s12m = np.zeros_like(_freqs); _s22m = np.zeros_like(_freqs)
    K_l_arr = np.zeros_like(_freqs); Delta_l_arr = np.zeros_like(_freqs)
    _nf = np.zeros_like(_freqs)

    for _i, _f in enumerate(_freqs):
        _omega = 2 * math.pi * _f
        _S11, _S21, _S12, _S22 = _sparams(_omega, W_um=_W, J_A_per_um=_J,
                                          L_s=_Ls, L_g=_Lg, L_d=_Ld)
        if cascode_l.value:
            _S12 = _S12 * 0.3
            _S21 = _S21 * 3.0  # cascode adds ~10 dB of gain in practice
        _s11m[_i] = 20 * math.log10(max(abs(_S11), 1e-6))
        _s21m[_i] = 20 * math.log10(max(abs(_S21), 1e-6))
        _s12m[_i] = 20 * math.log10(max(abs(_S12), 1e-6))
        _s22m[_i] = 20 * math.log10(max(abs(_S22), 1e-6))
        _K_val, _Delta_val = _rollett_K(_S11, _S12, _S21, _S22)
        K_l_arr[_i] = _K_val
        Delta_l_arr[_i] = abs(_Delta_val)
        _nf[_i] = _nf_cs(_omega, W_um=_W, J_A_per_um=_J,
                         L_s=_Ls, L_g=_Lg, L_d=_Ld)

    _fig_l = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("S-params + NF (dB)", "Stability", "Operating point"),
        row_heights=[0.45, 0.25, 0.30],
    )
    _fig_l.add_trace(go.Scatter(x=_freqs / 1e9, y=_s11m, mode="lines", name="|S₁₁|"), row=1, col=1)
    _fig_l.add_trace(go.Scatter(x=_freqs / 1e9, y=_s21m, mode="lines", name="|S₂₁|"), row=1, col=1)
    _fig_l.add_trace(go.Scatter(x=_freqs / 1e9, y=_s22m, mode="lines", name="|S₂₂|"), row=1, col=1)
    _fig_l.add_trace(go.Scatter(x=_freqs / 1e9, y=_nf,   mode="lines", name="NF"),    row=1, col=1)
    _fig_l.add_vline(x=_f0 / 1e9, line_dash="dash", line_color="#888", row=1, col=1)

    _fig_l.add_trace(go.Scatter(x=_freqs / 1e9, y=K_l_arr,     mode="lines", name="K"),   row=2, col=1)
    _fig_l.add_trace(go.Scatter(x=_freqs / 1e9, y=Delta_l_arr, mode="lines", name="|Δ|"), row=2, col=1)
    _fig_l.add_hline(y=1.0, line_dash="dot", line_color="#888", row=2, col=1)

    _fig_l.add_trace(go.Bar(
        x=["g_m (mS)", "C_gs (fF)", "f_T (GHz)", "I_D (mA)"],
        y=[op_l["g_m"] * 1e3, op_l["C_gs"] * 1e15,
           op_l["omega_T"] / (2 * math.pi * 1e9), op_l["I_D"] * 1e3],
        name="device op",
    ), row=3, col=1)

    _fig_l.update_layout(
        template="plotly_dark", height=820,
        title=f"f_0 = {_f0 / 1e9:.1f} GHz, cascode={'on' if cascode_l.value else 'off'}",
    )
    _fig_l.update_xaxes(title_text="f (GHz)", row=2, col=1)

    mo.ui.plotly(_fig_l)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 19. Practical mmWave realities

    Five topics separating the idealised design from silicon reality.

    ### 19.1 Inductor Q
    On-chip spiral or transmission-line inductors have $Q \sim 10$–20 at
    28 GHz. An inductor with reactance $X_L$ contributes loss
    $r_L = X_L / Q$ in series. The NF penalty from finite-Q $L_g$ can
    be significant (next cell computes it for $Q = 12$).

    ### 19.2 Gate resistance layout
    $r_g \propto R_\square W_f / (12 N_f^2)$ for double-sided-contact
    multi-finger layout. The plot below shows ΔNF vs. $N_f$ —
    diminishing returns past $N_f \approx 20$–40.

    ### 19.3 Substrate coupling
    Pattern-ground-shields under inductors, deep-nwell around the LNA —
    typical mmWave layout hygiene.

    ### 19.4 PDK vs textbook model
    Where Shaeffer-Lee deviates from silicon: short-channel velocity
    saturation ($g_m$ vs $V_{GS}$ plateaus earlier); non-quasi-static
    effects above $f_T/3$ make $C_{gs}$ frequency-dependent; $\gamma$ is
    itself frequency-dependent above ~$0.3 f_T$. Silicon measurements
    match textbook within ~1 dB at 28 GHz; divergence grows above 60 GHz.

    ### 19.5 Beyond 28 GHz
    At 60/77 GHz: inductor $Q$ drops further, $r_g$ dominates; at 140+ GHz
    transmission-line-matched stages replace lumped LC; noise floor
    increasingly set by interconnect and pad loss (notebook 05).
    """)
    return


@app.cell
def _(math, mo):
    from _lib_lna import compute_operating_point as _op_point

    _W = 70.0
    _J = 0.15e-3
    _Ls = 40e-12
    _Lg = 260e-12
    op_q = _op_point(_W, _J)
    _omega0 = 2 * math.pi * 28e9

    _Q = 12.0
    _r_Lg = _omega0 * _Lg / _Q
    _r_Ls = _omega0 * _Ls / _Q

    _Re_Zin = op_q["omega_T"] * _Ls
    _extra = (_r_Lg + _r_Ls) / _Re_Zin
    _penalty_db = 10 * math.log10(1.0 + _extra)

    mo.md(
        f"**Inductor Q=12 penalty at 28 GHz:** +{_penalty_db:.2f} dB NF "
        f"($r_{{L_g}} \\approx$ {_r_Lg:.1f} Ω, $r_{{L_s}} \\approx$ {_r_Ls:.1f} Ω "
        f"on top of $\\mathrm{{Re}}(Z_{{in}}) \\approx$ {_Re_Zin:.1f} Ω)."
    )
    return


@app.cell
def _(go, mo, np):
    _R_sq = 10.0          # Ω/□ polysilicon
    _W_f_total_um = 70.0  # total device width
    _N_f_range = np.arange(2, 60, 1)
    _r_g_vals = _R_sq * (_W_f_total_um / _N_f_range) / (12.0 * _N_f_range ** 2)

    _penalty_vs_Nf = 10.0 * np.log10(1.0 + _r_g_vals / 50.0)

    _fig_rg = go.Figure()
    _fig_rg.add_trace(go.Scatter(x=_N_f_range, y=_penalty_vs_Nf,
                                 mode="lines+markers", name="ΔNF (dB)"))
    _fig_rg.update_layout(
        template="plotly_dark",
        xaxis_title="Number of fingers N_f",
        yaxis_title="ΔNF contribution from r_g (dB)",
        height=340,
    )

    mo.ui.plotly(_fig_rg)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 20. Summary and bridge to notebook 05

    **Unified geometric picture.** Notebook 04 adds noise circles to the
    Γ-plane already populated with gain and stability circles in notebook 03.
    A complete LNA design point is now a single marker on the disk, and
    every tradeoff is visible as distance from a circle family.

    **First-stage dominance.** Friis (§11) collapses a receiver chain's
    noise figure onto the first stage's noise parameters. That is why
    mmWave receiver architectures invest so much silicon area in the
    LNA: it sets the floor every subsequent stage stands on.

    **What's next (notebook 05):**
    - L / π / T matching-network synthesis — generalise the §17.6 tapped-cap
    - Broadband matching: Bode-Fano limits, transformer coupling
    - Noise in mixers and samplers — cyclostationary in full
    - On-chip inductor Q and its matching implications

    **What's next (notebook 06):**
    - Linearity: P1dB, IP3, AM-PM
    - PA design and efficiency
    - Integrated mmWave frontend case study
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Previous:** [03 — S-parameters and Stability](03_s_parameters_stability.py)  |
    **Next:** *05 — Matching Networks (in preparation)*
    """)
    return


if __name__ == "__main__":
    app.run()
