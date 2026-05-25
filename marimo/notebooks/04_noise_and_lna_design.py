# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///
# v1.3

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
def _(math, np):
    # ── _lib_noise (inlined) ─────────────────────────────────────────────

    def generate_white(n, variance=1.0, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        return rng.normal(0.0, np.sqrt(variance), size=n)

    def generate_flicker(n, variance=1.0, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        white = rng.normal(0.0, 1.0, size=n)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs[0] = freqs[1]
        spec = spec / np.sqrt(freqs)
        shaped = np.fft.irfft(spec, n=n)
        shaped = shaped - shaped.mean()
        return shaped * np.sqrt(variance / shaped.var())

    def generate_band_limited(n, f_s, f_hi, variance=1.0, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        white = rng.normal(0.0, 1.0, size=n)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / f_s)
        spec[freqs > f_hi] = 0.0
        shaped = np.fft.irfft(spec, n=n)
        shaped = shaped - shaped.mean()
        return shaped * np.sqrt(variance / shaped.var())

    def generate_shot(n, rate, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        counts = rng.poisson(rate, size=n).astype(float)
        return counts - rate

    def estimate_autocorr(x, max_lag):
        n = x.size
        x = x - x.mean()
        full = np.correlate(x, x, mode="full") / n
        mid = full.size // 2
        return full[mid : mid + max_lag]

    def estimate_psd(x, f_s):
        n = x.size
        seg = min(1024, n)
        hop = seg // 2
        window = np.hanning(seg)
        segments = []
        for start in range(0, n - seg + 1, hop):
            chunk = x[start : start + seg] * window
            segments.append(np.abs(np.fft.rfft(chunk)) ** 2)
        if not segments:
            segments = [np.abs(np.fft.rfft(x * np.hanning(n))) ** 2]
        psd = np.mean(segments, axis=0) / (f_s * (window ** 2).sum())
        freqs = np.fft.rfftfreq(seg, d=1.0 / f_s)
        return freqs, psd

    def friis_cascade(F, G_A):
        assert len(F) == len(G_A)
        assert all(g > 0 for g in G_A)
        f_tot = F[0]
        cum_gain = 1.0
        for i in range(1, len(F)):
            cum_gain *= G_A[i - 1]
            f_tot += (F[i] - 1.0) / cum_gain
        return f_tot

    def noise_figure_from_Gamma(Gamma_s, F_min, R_n_norm, Gamma_opt):
        denom = (1.0 - abs(Gamma_s) ** 2) * abs(1.0 + Gamma_opt) ** 2
        return F_min + 4.0 * R_n_norm * abs(Gamma_s - Gamma_opt) ** 2 / denom

    # ── _lib_circles (inlined) ───────────────────────────────────────────

    def noise_circle(F_target, F_min, R_n_norm, Gamma_opt):
        assert F_target >= F_min
        N = (F_target - F_min) / (4.0 * R_n_norm) * abs(1.0 + Gamma_opt) ** 2
        center = Gamma_opt / (1.0 + N)
        radius = np.sqrt(N ** 2 + N * (1.0 - abs(Gamma_opt) ** 2)) / (1.0 + N)
        return center, float(radius)

    def available_gain_circle(G_target_dB, S11, S12, S21, S22):
        g_a = 10 ** (G_target_dB / 10) / (abs(S21) ** 2)
        Delta = S11 * S22 - S12 * S21
        K = (1.0 - abs(S11) ** 2 - abs(S22) ** 2 + abs(Delta) ** 2) / (2.0 * abs(S12 * S21))
        C1 = S11 - Delta * np.conj(S22)
        denom_factor = 1.0 + g_a * (abs(S11) ** 2 - abs(Delta) ** 2)
        center = g_a * np.conj(C1) / denom_factor
        discrim = 1.0 - 2.0 * K * abs(S12 * S21) * g_a + (abs(S12 * S21) * g_a) ** 2
        radius = np.sqrt(max(discrim, 0.0)) / abs(denom_factor)
        return center, float(radius)

    def source_stability_circle(S11, S12, S21, S22):
        Delta = S11 * S22 - S12 * S21
        denom = abs(S11) ** 2 - abs(Delta) ** 2
        assert abs(denom) > 1e-20
        center = np.conj(S11 - Delta * np.conj(S22)) / denom
        radius = abs(S12 * S21) / abs(denom)
        return center, float(radius)

    def load_stability_circle(S11, S12, S21, S22):
        Delta = S11 * S22 - S12 * S21
        denom = abs(S22) ** 2 - abs(Delta) ** 2
        assert abs(denom) > 1e-20
        center = np.conj(S22 - Delta * np.conj(S11)) / denom
        radius = abs(S12 * S21) / abs(denom)
        return center, float(radius)

    def rollett_K(S11, S12, S21, S22):
        Delta = S11 * S22 - S12 * S21
        K = (1.0 - abs(S11) ** 2 - abs(S22) ** 2 + abs(Delta) ** 2) / (2.0 * abs(S12 * S21))
        return float(K), Delta

    def MAG_dB(S11, S12, S21, S22):
        K, _ = rollett_K(S11, S12, S21, S22)
        if K > 1.0:
            return 10.0 * np.log10(abs(S21) / abs(S12) * (K - np.sqrt(K ** 2 - 1.0)))
        return 10.0 * np.log10(abs(S21) / abs(S12))

    # ── _lib_lna (inlined) ───────────────────────────────────────────────

    from dataclasses import dataclass as _dataclass

    K_BOLTZMANN = 1.380649e-23
    T0 = 290.0
    Q_ELEC = 1.602176634e-19

    @_dataclass(frozen=True)
    class DeviceParams:
        gm_per_id: float = 12.0
        cox_per_area: float = 1.7e-2
        cgs_per_W: float = 1.5e-15
        gamma: float = 1.4
        delta: float = 2.0 * 1.4
        c_corr: complex = 0.395j
        alpha_nq: float = 0.8

    def compute_operating_point(W_um, J_A_per_um, params=DeviceParams()):
        I_D = W_um * J_A_per_um
        g_m = params.gm_per_id * I_D
        C_gs = params.cgs_per_W * W_um
        omega_T = g_m / C_gs
        return {"I_D": I_D, "g_m": g_m, "C_gs": C_gs, "omega_T": omega_T}

    def input_impedance(omega, L_s, L_g, op):
        return (1j * omega * (L_s + L_g)
                + 1.0 / (1j * omega * op["C_gs"])
                + op["omega_T"] * L_s)

    def F_min_shaeffer_lee(omega, op, params=DeviceParams()):
        ratio = omega / op["omega_T"]
        return 1.0 + (2.4 * params.gamma / params.alpha_nq) * ratio

    def gamma_opt_degenerated_cs(omega, L_s, L_g, op, params=DeviceParams()):
        Zin = input_impedance(omega, L_s, L_g, op)
        Z0 = 50.0
        return np.conj((Zin - Z0) / (Zin + Z0))

    def sparameters_at_freq(omega, W_um, J_A_per_um, L_s, L_g, L_d,
                            R_load=50.0, params=DeviceParams()):
        op = compute_operating_point(W_um, J_A_per_um, params)
        Zin = input_impedance(omega, L_s, L_g, op)
        Z0 = 50.0
        S11 = (Zin - Z0) / (Zin + Z0)
        R_loop = Z0 + op["omega_T"] * L_s
        Q_gate = 1.0 / (omega * op["C_gs"] * R_loop)
        Z_load = 1j * omega * L_d + R_load
        S21 = -op["g_m"] * Z_load * Q_gate * Z0 / (Z0 + Z_load)
        S22 = (Z_load - Z0) / (Z_load + Z0)
        s12_mag = 10 ** ((-40 + 15 * (omega / (2 * math.pi * 40e9))) / 20)
        S12 = s12_mag + 0j
        return S11, S21, S12, S22

    def NF_degenerated_cs(omega, W_um, J_A_per_um, L_s, L_g, L_d,
                          R_load=50.0, params=DeviceParams()):
        op = compute_operating_point(W_um, J_A_per_um, params)
        F = F_min_shaeffer_lee(omega, op, params)
        Zin = input_impedance(omega, L_s, L_g, op)
        Gamma_s = 0.0 + 0j
        S11 = (Zin - 50.0) / (Zin + 50.0)
        Gamma_opt = np.conj(S11)
        R_n_norm = params.gamma / (params.alpha_nq * op["g_m"] * 50.0)
        denom = (1.0 - abs(Gamma_s) ** 2) * abs(1.0 + Gamma_opt) ** 2
        F_at_50 = F + 4.0 * R_n_norm * abs(Gamma_s - Gamma_opt) ** 2 / denom
        return 10.0 * math.log10(F_at_50)

    return (
        DeviceParams, F_min_shaeffer_lee, MAG_dB, NF_degenerated_cs,
        available_gain_circle, compute_operating_point, estimate_autocorr,
        estimate_psd, friis_cascade, gamma_opt_degenerated_cs,
        generate_band_limited, generate_flicker, generate_shot,
        generate_white, input_impedance, load_stability_circle,
        noise_circle, noise_figure_from_Gamma, rollett_K,
        source_stability_circle, sparameters_at_freq,
    )


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
    ## 2. Physical Foundations of Noise

    Two complementary stories run through this section, and they must
    agree because they describe the same physics:

    | View | What fluctuates | Limit it shines in |
    |---|---|---|
    | **Macroscopic** | continuous EMF / voltage | $kT \gg$ level spacing — equipartition, Nyquist |
    | **Microscopic** | discrete electron occupancy | individual carriers — Fermi-Dirac, partition noise, shot noise |

    §2.1–2.3 build the quantum-statistical foundation: canonical
    ensemble, Fermi-Dirac distribution, and the partition-variance
    factor $f(1-f)$ that is the microscopic source of *every* noise
    statement in this notebook. §2.4 derives the Nyquist formula in
    its classical (equipartition) form and notes the microscopic check
    via the Callen-Welton fluctuation-dissipation theorem (full
    Landauer derivation: §6.3). §2.5 takes the same machinery to the
    *counting* limit — shot noise and the thermal $\leftrightarrow$
    shot crossover. §2.6 catalogs the resulting noise mechanisms with
    cross-references.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.1 Quantum origin and the canonical ensemble

    **Discrete states from Schrödinger.** A confined electron obeys the
    time-independent Schrödinger equation,

    $$-\frac{\hbar^2}{2m}\nabla^2\psi + V(\mathbf{r})\psi = E\psi.$$

    For any binding potential $V(\mathbf{r})$ the spectrum is *discrete*:
    $E_1, E_2, E_3, \ldots$. Statistical mechanics then has a list of
    *single-particle states* to populate, not a continuous energy variable.

    **Pauli exclusion — the binary occupancy.** Electrons are fermions:
    at most one electron occupies any single-particle state. The
    occupation number of state $k$ is therefore a Bernoulli random
    variable,

    $$n_k \in \{0, 1\} \quad \text{[Pauli, axiom]}.$$

    Every noise statement that follows is a statement about fluctuations
    of these binary $n_k$ — there is no other underlying random variable
    at the level of single-particle states.

    **The canonical ensemble — temperature enters.** A system weakly
    coupled to a heat bath at temperature $T$ has, at equilibrium, the
    **Boltzmann weight**

    $$P(\text{microstate } s) = \frac{e^{-E_s/kT}}{Z}, \qquad Z = \sum_s e^{-E_s/kT} \quad \text{[canonical ensemble, definition]}.$$

    A microstate $s$ specifies every $n_k$; its energy is
    $E_s = \sum_k n_k\,\varepsilon_k$.

    > **Note — entropy and the operational meaning of $T$.**
    > The Gibbs entropy of the ensemble is
    > $\,S = -k\sum_s P_s \ln P_s = \langle E\rangle/T + k\ln Z\,$.
    > Equilibrium maximises $S$ at fixed $\langle E\rangle$; the relation
    > $\partial S/\partial E\big|_{N,V} = 1/T$ *defines* temperature:
    > $T^{-1}$ is the rate at which available phase-space volume grows
    > with energy. High $T$ spreads the ensemble broadly; low $T$ huddles
    > it in low-energy states. This single relation is the only fact
    > about $T$ used everywhere below.

    The canonical ensemble by itself does **not** give $\langle n_k\rangle$ —
    the constraint $\sum_k n_k = N$ couples states together. For the
    mean occupancy we need either explicit combinatorics (§2.2 Route A)
    or the grand-canonical factorisation (§2.2 Route B). Both routes
    converge to the same closed form, the Fermi-Dirac distribution.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.2 The Fermi-Dirac distribution — two derivations

    Both routes yield the same closed form:

    $$\boxed{f(\varepsilon) = \frac{1}{e^{(\varepsilon-\mu)/kT} + 1}} \quad \text{[Fermi-Dirac distribution, theorem]}.$$

    Route A makes the combinatorics explicit; Route B makes the role of
    the chemical potential $\mu$ transparent.

    ---

    #### Route A — combinatorial counting with Stirling and Lagrange

    **Set-up.** Group single-particle states into bins of approximately
    equal energy: bin $i$ has $g_i$ states (degeneracy) at energy
    $\varepsilon_i$, occupied by $n_i$ fermions with $0 \le n_i \le g_i$
    (Pauli).

    **Microstate count.** The number of ways to choose which $n_i$ of
    the $g_i$ states in bin $i$ are occupied is the binomial coefficient.
    The total number of microstates compatible with the occupation
    vector $\{n_i\}$ is therefore

    $$W(\{n_i\}) = \prod_i \binom{g_i}{n_i} = \prod_i \frac{g_i!}{n_i!\,(g_i - n_i)!}.$$

    **Entropy.** $S = k\ln W$ [Boltzmann, definition]. Stirling's
    approximation $\ln m! \approx m\ln m - m$ (valid for $m \gg 1$)
    gives

    $$\frac{S}{k} \approx \sum_i\Bigl[g_i\ln g_i - n_i\ln n_i - (g_i - n_i)\ln(g_i - n_i)\Bigr].$$

    **Maximise under constraints.** Fix $\sum_i n_i = N$ and
    $\sum_i n_i\varepsilon_i = E$. With Lagrange multipliers $\alpha$
    and $\beta$,

    $$\frac{\partial}{\partial n_i}\!\Bigl[\frac{S}{k} - \alpha\bigl(\textstyle\sum_j n_j - N\bigr) - \beta\bigl(\textstyle\sum_j n_j\varepsilon_j - E\bigr)\Bigr] = 0.$$

    Differentiating the entropy expression term-by-term,

    $$-\ln n_i + \ln(g_i - n_i) - \alpha - \beta\varepsilon_i = 0 \;\Longrightarrow\; \ln\!\frac{g_i - n_i}{n_i} = \alpha + \beta\varepsilon_i,$$

    which rearranges to

    $$\frac{\langle n_i\rangle}{g_i} = \frac{1}{e^{\alpha + \beta\varepsilon_i} + 1}.$$

    **Identifying the multipliers.** The first law $dE = T\,dS - \mu\,dN$
    combined with entropy maximisation gives $\partial S/\partial E\big|_N = 1/T$
    and $\partial S/\partial N\big|_E = -\mu/T$; matching to
    $\partial(S/k)/\partial n_i = \alpha + \beta\varepsilon_i$ yields

    $$\beta = \frac{1}{kT}, \qquad \alpha = -\frac{\mu}{kT}.$$

    Substituting recovers Fermi-Dirac.

    ---

    #### Route B — grand canonical factorisation

    **Why grand canonical here.** The canonical-ensemble constraint
    $\sum_k n_k = N$ couples all states. Switching to the **grand
    canonical** ensemble (variable $N$, fixed $\mu$) lets each
    single-particle state be its own two-level subsystem. The cost is
    that $N$ now fluctuates — but only by relative order $1/\sqrt{N}$,
    irrelevant for any bulk conductor.

    **Single-state grand partition function.** Each state $k$ has
    $n \in \{0,1\}$:

    $$\Xi_k = \sum_{n=0}^{1} e^{-(\varepsilon_k - \mu)n/kT} = 1 + e^{-(\varepsilon_k - \mu)/kT}.$$

    **Mean occupancy** from $\langle n_k\rangle
    = -\partial\ln\Xi_k / \partial(\varepsilon_k/kT)$ at fixed $\mu$:

    $$\langle n_k\rangle = \frac{e^{-(\varepsilon_k - \mu)/kT}}{1 + e^{-(\varepsilon_k - \mu)/kT}} = \frac{1}{e^{(\varepsilon_k - \mu)/kT} + 1} = f(\varepsilon_k). \;\;\square$$

    **Variance — the key result for noise.** Because $n_k \in \{0, 1\}$
    we have $n_k^2 = n_k$, so $\langle n_k^2\rangle = \langle n_k\rangle = f$:

    $$\boxed{\mathrm{Var}(n_k) = f(\varepsilon_k)\,\bigl(1 - f(\varepsilon_k)\bigr)} \quad \text{[partition variance, theorem]}.$$

    Distinct states are statistically independent in the grand canonical
    ensemble (the joint distribution factorises across $k$), so
    variances of region-occupancies add:

    $$\mathrm{Var}\!\left(\sum_{k\in\mathcal{R}} n_k\right) = \sum_{k\in\mathcal{R}} f_k\bigl(1 - f_k\bigr).$$

    This is the foundation of every "thermal" noise statement that
    follows.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.3 Partition noise and the classical limit

    **Where the noise lives — the Fermi edge.** The factor
    $f(\varepsilon)\bigl(1 - f(\varepsilon)\bigr)$ peaks at $\varepsilon = \mu$
    with value $1/4$. Far below $E_F$ ($\varepsilon \ll \mu$) $f \to 1$
    and the state is **Pauli-locked** — no fluctuations possible. Far
    above $E_F$ ($\varepsilon \gg \mu$) $f \to 0$ and the state is
    empty — no fluctuations either. **All occupancy fluctuations live in
    a $\sim kT$-wide window around $E_F$.**

    The window's *width* is set by $|\varepsilon - \mu| \lesssim kT$; its
    *height* is bounded above by $1/4$. Their product is universal:

    $$\int_{-\infty}^\infty f(\varepsilon)\bigl(1 - f(\varepsilon)\bigr)\, d\varepsilon = kT \quad \text{[Fermi-edge integral, theorem]}.$$

    *Proof.* Direct differentiation gives $f(1-f) = -kT\,df/d\varepsilon$.
    Then $\int -kT\,(df/d\varepsilon)\,d\varepsilon
    = kT\,[f(-\infty) - f(+\infty)] = kT\,(1 - 0) = kT$. $\square$

    This integral is the **microscopic origin of the $kT$ in thermal
    noise** — it counts the effective number of fluctuating channels at
    the Fermi edge, weighted by their individual partition variance. It
    reappears in §2.4 packaged inside the Landauer-Büttiker formula, where
    it converts directly into Nyquist's $4kTR$.

    ---

    **Classical (Maxwell-Boltzmann) limit.** When
    $(\varepsilon - \mu)/kT \gg 1$ for all relevant states (non-degenerate
    gas: low density or high $T$),

    $$f(\varepsilon) \approx e^{-(\varepsilon - \mu)/kT} \quad \text{[Maxwell-Boltzmann limit]},$$

    and three things happen:
    *(i)* occupancies of distinct states become effectively independent
    Poisson variables (since $f \ll 1$, $f(1-f) \approx f$);
    *(ii)* sums over discrete states $\sum_k$ pass to phase-space
    integrals $\int dp\,dq/h$ weighted by $e^{-H/kT}/Z$;
    *(iii)* for any continuous **quadratic** degree of freedom
    $H = \tfrac{1}{2}\alpha x^2$ the Gaussian phase-space integral gives

    $$\langle H\rangle = \frac{\int_{-\infty}^\infty \tfrac{1}{2}\alpha x^2\, e^{-\alpha x^2/2kT}\,dx} {\int_{-\infty}^\infty e^{-\alpha x^2/2kT}\,dx} = \frac{1}{2}kT \quad \text{[equipartition theorem]}.$$

    The bridge from quantum statistics to classical equipartition is
    therefore **two steps, not one**: *FD $\to$ MB* (non-degenerate limit
    that removes Pauli), then *MB $\to$ equipartition for continuous
    quadratic DoFs* (Gaussian integral on the Boltzmann weight). The
    literature sometimes elides this; the distinction matters when
    either step breaks (degenerate gas at low $T$; non-quadratic
    Hamiltonian).

    For a capacitor, $H_C = \tfrac{1}{2}C V_C^2$ is one quadratic
    degree of freedom, so

    $$\langle V_C^2\rangle = \frac{kT}{C}.$$

    This is the input to the Nyquist derivation of §2.4.

    > **Note — when is the classical limit valid for circuit noise?**
    > The classical limit requires $kT \gg \hbar\omega$ at the relevant
    > frequency $\omega$. At $T = 290$ K and $f = 1$ THz:
    > $kT \approx 25$ meV vs $\hbar\omega \approx 4$ meV — classical
    > with margin. The quantum correction is below 10% up to ~6 THz at
    > room temperature. Cryogenic operation at microwave frequencies
    > (e.g.\ 10 GHz at 0.5 K) is the regime where zero-point noise
    > becomes the floor; see §6.2.
    """)
    return


@app.cell
def _(go, make_subplots, mo, np):
    # FIG 2.3 — Fermi-Dirac, partition variance, and the counting limit
    _fig_fd = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Fermi-Dirac f(ε) — edge sharpens as T → 0",
            "Partition variance f(1−f) — peaks at ε = μ, area = kT",
            "Counting limit: σ/⟨N⟩ = 1/√⟨N⟩",
        ),
    )

    # Panels 1 & 2 — energy in units of (ε - μ)/kT_ref
    _e_axis = np.linspace(-8.0, 8.0, 600)
    _T_list = [0.5, 1.0, 2.5]
    _colors_T = ["#5599DD", "#33CC88", "#EE8833"]

    # Panel 1 — f(ε) at three temperatures
    for _Tv, _clr in zip(_T_list, _colors_T):
        _fv = 1.0 / (np.exp(_e_axis / _Tv) + 1.0)
        _fig_fd.add_trace(go.Scatter(
            x=_e_axis, y=_fv, mode="lines",
            name=f"T/T_ref = {_Tv:.1f}",
            line=dict(color=_clr, width=2),
            legendgroup=f"T{_Tv}",
            showlegend=True), row=1, col=1)
    _fig_fd.add_vline(x=0.0, line_dash="dot", line_color="#888888",
                      annotation_text="ε = μ",
                      annotation_position="top right", row=1, col=1)

    # Panel 2 — f(1-f) at same temperatures (height ≤ 1/4 universal, width ∝ T)
    for _Tv, _clr in zip(_T_list, _colors_T):
        _fv = 1.0 / (np.exp(_e_axis / _Tv) + 1.0)
        _var = _fv * (1.0 - _fv)
        _fig_fd.add_trace(go.Scatter(
            x=_e_axis, y=_var, mode="lines",
            line=dict(color=_clr, width=2, dash="dash"),
            legendgroup=f"T{_Tv}",
            showlegend=False), row=1, col=2)
    _fig_fd.add_hline(y=0.25, line_dash="dot", line_color="#888888",
                      annotation_text="max = 1/4",
                      annotation_position="top right", row=1, col=2)

    # Panel 3 — current normalized to mean at two carrier rates
    _rng_pl = np.random.default_rng(11)
    _t_axis_pl = np.linspace(0.0, 1.0, 200)
    _N_high = 200.0
    _N_low = 4.0
    _counts_hi = _rng_pl.poisson(_N_high, size=_t_axis_pl.size)
    _counts_lo = _rng_pl.poisson(_N_low,  size=_t_axis_pl.size)
    _I_hi_norm = _counts_hi / _N_high
    _I_lo_norm = _counts_lo / _N_low

    _fig_fd.add_trace(go.Scatter(
        x=_t_axis_pl, y=_I_hi_norm, mode="lines",
        name=f"⟨N⟩ = {int(_N_high)} (sampling)",
        line=dict(color="#5599DD", width=1.5),
        showlegend=True), row=1, col=3)
    _fig_fd.add_trace(go.Scatter(
        x=_t_axis_pl, y=_I_lo_norm, mode="lines+markers",
        name=f"⟨N⟩ = {int(_N_low)} (counting)",
        line=dict(color="#EE8833", width=1.5),
        marker=dict(size=4),
        showlegend=True), row=1, col=3)
    _fig_fd.add_hline(y=1.0, line_dash="dot", line_color="#888888",
                      row=1, col=3)

    _fig_fd.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=50, r=20, t=80, b=50),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)",
                    font=dict(size=9)),
    )
    _fig_fd.update_xaxes(title_text="(ε − μ) / kT_ref", row=1, col=1)
    _fig_fd.update_yaxes(title_text="f(ε)",             row=1, col=1)
    _fig_fd.update_xaxes(title_text="(ε − μ) / kT_ref", row=1, col=2)
    _fig_fd.update_yaxes(title_text="f(1 − f)",         row=1, col=2)
    _fig_fd.update_xaxes(title_text="window index (norm.)", row=1, col=3)
    _fig_fd.update_yaxes(title_text="N / ⟨N⟩",          row=1, col=3)

    mo.ui.plotly(_fig_fd)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.4 The Nyquist formula — macroscopic derivation, microscopic check

    Two derivations of $S_v = 4kTR$ exist: a *macroscopic* (equipartition
    + RC) proof, given in full below, and a *microscopic*
    (Landauer-Büttiker) proof from $f(1-f)$, sketched here and detailed
    in §6.3. They must agree by the **Callen-Welton fluctuation-
    dissipation theorem (FDT)**, which states that for any linear system
    in thermal equilibrium the noise PSD is set by the dissipative part
    of the response function:

    $$\boxed{S_v(\omega) = 4kT\,\mathrm{Re}\{Z(\omega)\}} \quad \text{[Callen-Welton FDT, theorem]}.$$

    Setting $Z = R$ gives $S_v = 4kTR$, independent of the microscopic
    mechanism that achieves the dissipation.

    #### Macroscopic derivation: equipartition + RC + capacitance trick

    **Step 1 — Equipartition of the capacitor.** From §2.3, equipartition
    assigns $\langle E_C\rangle = \tfrac{1}{2}kT$ to the capacitor's one
    quadratic degree of freedom, giving

    $$\langle V_C^2\rangle = \frac{kT}{C}.$$

    This holds for *any* network connected to $C$, provided the whole
    system is at temperature $T$.

    **Step 2 — Circuit model.** Thevenize the noisy resistor: noiseless
    $R$ in series with open-circuit noise voltage $v_n(t)$ of (unknown)
    flat PSD $S_v$. Load with capacitor $C$ — a first-order RC low-pass
    with transfer function

    $$H(f) = \frac{1}{1 + j 2\pi f RC}.$$

    **Step 3 — Noise bandwidth of the RC filter.** The mean-square
    voltage on $C$ integrates over all positive frequencies,

    $$\langle V_C^2\rangle = \int_0^\infty S_v\,|H(f)|^2\,df = S_v\int_0^\infty \frac{df}{1+(2\pi fRC)^2} = \frac{S_v}{4RC}.$$

    The integral evaluates to $\arctan(\infty)/(2\pi RC) = 1/(4RC)$; this
    is the **noise bandwidth** $B_n = 1/(4RC)$ — the bandwidth of an
    ideal brick-wall filter that would pass the same total noise power.
    It differs from $f_{-3} = 1/(2\pi RC)$ by the factor $\pi/2$.

    **Step 4 — Solve.** Equating Steps 1 and 3,

    $$\frac{S_v}{4RC} = \frac{kT}{C} \;\Longrightarrow\; \boxed{S_v = 4kT R} \quad \text{[Nyquist formula, theorem]}.$$

    The capacitance $C$ drops out — as it must, since the noise is a
    property of the resistor, not of the load.

    **Why "one mode".** The lumped RC has a single energy-storing
    element (the capacitor): one quadratic degree of freedom, one
    factor of $\tfrac{1}{2}kT$. A transmission line of length $\ell$
    supports modes spaced by $\Delta f = c/(2\ell)$; a sum over all
    modes recovers the same $S_v = 4kTR$. Both pictures agree because
    the total energy per mode is fixed by temperature, not geometry.

    #### Microscopic check from $f(1-f)$ — sketched

    A conductor of conductance $G = 1/R$ has, microscopically, current
    fluctuations driven by occupancy fluctuations of states near $E_F$
    — the partition noise of §2.3. The Landauer-Büttiker scattering
    formalism (full derivation: §6.3) gives, in equilibrium,

    $$S_I = \frac{8e^2}{h}\sum_n \mathcal{T}_n \int f(1-f)\,d\varepsilon = \frac{8e^2}{h}\sum_n \mathcal{T}_n \cdot kT = 4kT\,G,$$

    using the Fermi-edge integral from §2.3 and the Landauer
    conductance $G = (2e^2/h)\sum_n \mathcal{T}_n$ (spin-doubled).
    Inverting, $S_v = S_I/G^2 = 4kTR$ — **Nyquist recovered from pure
    occupancy statistics.**

    The two derivations consume *the same input*: equipartition's
    $\tfrac{1}{2}kT$ per quadratic mode and the FD integral
    $\int f(1-f)\,d\varepsilon = kT$ are the same $kT$ entering by
    different routes. The FDT is the formal guarantee that they must
    agree for any equilibrium system.

    #### Available noise power and the link-budget consequence

    A source $R_s$ delivers fraction $1/4$ of its open-circuit noise
    power into a matched load $R_L = R_s$:

    $$P_\text{avail} = \frac{\langle v_n^2\rangle}{4R_s}\,\Delta f = \frac{4kTR_s}{4R_s}\,\Delta f = kT\,\Delta f.$$

    At $T_0 = 290$ K: $N_0 \equiv kT_0 = -174$ dBm/Hz — a hard floor no
    passive network can beat.

    Receiver sensitivity in bandwidth $B$:

    $$S_{\min} = k T_0 B \cdot F \cdot \mathrm{SNR}_{\min},$$

    where $F$ is the receiver noise factor (linear). At $B = 100$ MHz,
    $\mathrm{SNR}_{\min} = 10$ dB: floor $= -174 + 80 + 10 = -84$ dBm;
    a 2 dB NF raises this to $-82$ dBm, 6 dB NF to $-78$ dBm — already
    four times more transmit power required to close the link.
    """)
    return


@app.cell
def _(go, make_subplots, mo, np):
    _fig_eq = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Boltzmann weight — equipartition theorem",
            "RC filter: noise bandwidth B_n = (π/2)·f₋₃dB",
        ),
    )

    # ── Boltzmann panel ──────────────────────────────────────────────────────
    # p(x) = e^{-x²/2}/√(2π): normalised Gaussian with α=kT=1
    # Integrand of ⟨x²⟩ is x²·p(x); its integral = 1, so ⟨H⟩ = ½αkT/α = ½kT
    _x = np.linspace(-4.5, 4.5, 800)
    _pdf = np.exp(-_x**2 / 2.0) / np.sqrt(2.0 * np.pi)
    _igrand = _x**2 * _pdf                      # integrand of ⟨x²⟩; ∫ dx = 1

    _fig_eq.add_trace(go.Scatter(
        x=_x, y=_pdf, mode="lines",
        name="p(x) = e^{−x²/2}/√(2π)",
        line=dict(color="#5599DD", width=2)), row=1, col=1)
    _fig_eq.add_trace(go.Scatter(
        x=_x, y=_igrand, mode="lines",
        name="x²·p(x)  — integrand of ⟨x²⟩",
        line=dict(color="#EE8833", width=2)), row=1, col=1)
    _fig_eq.add_annotation(
        text="∫ x²p dx = 1  ⟹  ⟨H⟩ = ½kT",
        x=2.5, y=0.28, xref="x", yref="y",
        font=dict(color="#33CC88", size=11), showarrow=False)

    # ── RC noise bandwidth panel ─────────────────────────────────────────────
    # |H(f)|² = 1/(1+(f/f₋₃)²), noise BW: ∫|H|²df = (π/2)·f₋₃ = 1/(4RC)
    _fn = np.linspace(0.0, 5.5, 700)
    _H2 = 1.0 / (1.0 + _fn**2)
    _Bn = np.pi / 2.0                           # B_n / f_{-3dB}

    _fig_eq.add_trace(go.Scatter(
        x=_fn, y=_H2, mode="lines",
        name="|H(f)|² = 1/(1+(f/f₋₃)²)",
        line=dict(color="#5599DD", width=2)), row=1, col=2)
    _fig_eq.add_trace(go.Scatter(
        x=[0.0, _Bn, _Bn], y=[1.0, 1.0, 0.0], mode="lines",
        name="Brick-wall (equal area)",
        line=dict(color="#33CC88", dash="dash", width=2)), row=1, col=2)
    _fig_eq.add_vline(x=1.0, line_dash="dot", line_color="#EE8833",
                      annotation_text="f₋₃dB", annotation_position="top right",
                      row=1, col=2)
    _fig_eq.add_annotation(
        text="Bₙ = π/2 · f₋₃dB", x=_Bn + 0.4, y=0.80,
        xref="x2", yref="y2",
        font=dict(color="#33CC88", size=11), showarrow=False)

    _fig_eq.update_layout(
        template="plotly_dark", height=360,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    _fig_eq.update_xaxes(title_text="x", row=1, col=1)
    _fig_eq.update_yaxes(title_text="amplitude", row=1, col=1)
    _fig_eq.update_xaxes(title_text="f / f₋₃dB", row=1, col=2)
    _fig_eq.update_yaxes(title_text="|H|²", row=1, col=2)

    mo.ui.plotly(_fig_eq)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.5 The counting limit — shot noise and the thermal↔shot crossover

    The Nyquist derivation of §2.4 treats the resistor's noise as a
    *continuous* random voltage — appropriate when the number of carriers
    per measurement window is huge ($\gg 1$). At a **potential barrier**
    (diode junction, BJT emitter, MOS subthreshold) each carrier crossing
    is an *independent event*, and the discreteness of the electron
    becomes visible in the noise.

    #### Poisson statistics at a barrier — derivation of $2qI$

    Consider $N$ independent carriers crossing a barrier in observation
    window $\tau$. The arrival process is Poisson:

    $$P(N) = \frac{\bar{N}^{N}\,e^{-\bar{N}}}{N!}, \qquad \langle N\rangle = \bar{N}, \quad \mathrm{Var}(N) = \bar{N} \quad \text{[Poisson, theorem]}.$$

    The time-averaged current is $I = qN/\tau$, so
    $\langle I\rangle = q\bar{N}/\tau$ and

    $$\mathrm{Var}(I) = \frac{q^2}{\tau^2}\,\mathrm{Var}(N) = \frac{q^2\bar{N}}{\tau^2} = \frac{q\,\langle I\rangle}{\tau}.$$

    Converting from variance over a single window of length $\tau$ to a
    one-sided PSD (Carson's theorem: a Poisson impulse train with rate
    $\lambda$ has flat current PSD $S_I = q^2\lambda \cdot 2 = 2q^2\lambda$,
    where $\lambda = \langle I\rangle/q$),

    $$\boxed{S_I = 2qI} \quad \text{[Schottky shot noise, theorem]}.$$

    This applies wherever carriers cross a barrier independently — diodes,
    BJTs in forward active, MOS in subthreshold, photodetectors. It does
    **not** apply to ohmic resistors: carriers there suffer many
    scatterings per transit, the crossings are not independent, and only
    thermal noise survives.

    > **Note — the "counting electrons" picture, quantitatively.** With
    > $I = 1$ μA and observation window $\tau = 1$ ns, $\bar{N} = I\tau/q
    > \approx 6\,000$ carriers per window. Relative fluctuation
    > $\sigma_N/\bar{N} = 1/\sqrt{\bar{N}} \approx 1.3\%$ — the current
    > looks smooth. At $I = 1$ pA the same window has $\bar{N} \approx 6$
    > carriers and $\sigma_N/\bar{N} \approx 40\%$ — the noise is dominated
    > by the integer count itself. This is what "we're counting electrons"
    > means. See panel 3 of the Fermi-Dirac figure above.

    #### The thermal ↔ shot crossover — coth interpolation

    A diode junction at bias $V$ in detailed balance has independent
    forward and reverse Poisson streams:

    $$I_+ = I_s e^{qV/kT}, \quad I_- = I_s, \quad I = I_+ - I_- = I_s(e^{qV/kT} - 1) \quad \text{[Shockley diode equation, theorem]}.$$

    Each stream is independently shot-noisy and they are uncorrelated;
    variances add:

    $$S_I = 2q(I_+ + I_-) = 2q I_s(e^{qV/kT} + 1).$$

    Two lines of algebra rewrite this in coth form. With $x = qV/2kT$,

    $$e^{qV/kT} + 1 = e^{2x} + 1 = e^x(e^x + e^{-x}), \qquad e^{qV/kT} - 1 = e^x(e^x - e^{-x}),$$

    so $(e^{qV/kT}+1)/(e^{qV/kT}-1) = \coth(x)$. Multiplying numerator and
    denominator of $S_I$ by $(I_+ - I_-)/(I_+ - I_-)$,

    $$\boxed{S_I = 2qI\,\coth\!\left(\frac{qV}{2kT}\right)} \quad \text{[thermal–shot crossover, theorem]}.$$

    **Limits.**
    - $qV \to 0$: $\coth(x) \to 1/x = 2kT/qV$, so
      $S_I \to 2qI\cdot 2kT/(qV) = 4kT\,(I/V)$. At $V \to 0$ the chord
      conductance $g_d = I/V$ equals the differential conductance
      $qI_s/kT$, so $S_I \to 4kT g_d$ — pure Johnson noise.
    - $qV \gg kT$: $\coth(x) \to 1$, so $S_I \to 2qI$ — pure Schottky shot.

    The crossover scale is $V \sim kT/q \approx 26$ mV at room
    temperature. Below this bias the device "looks thermal" (current is
    small, both streams contribute equally and average out); above it
    the device "looks shot" (one stream dominates and discrete crossings
    set the noise). The same coth form arises for any Poisson barrier in
    detailed balance — including tunnel junctions and orthodox-regime
    single-electron transistors. Full algebraic derivation: §6.4.

    > **Note — $kT/q$ as the universal voltage scale.** The factor
    > $kT/q \approx 26$ mV at $T = 290$ K controls
    > *(i)* the diode subthreshold slope ($\ln 10 \cdot kT/q = 60$ mV/decade),
    > *(ii)* the MOS weak-inversion slope,
    > *(iii)* the width of the Fermi edge where partition noise lives (§2.3),
    > *(iv)* the thermal $\leftrightarrow$ shot crossover bias derived above,
    > *(v)* the noise voltage scale of any resistor at thermal equilibrium:
    > $\sqrt{4kTR\cdot B} \sim kT/q$ whenever $I = V/R \sim q/\tau$ (one
    > carrier per measurement window).
    > Every solid-state noise statement reduces, on dimensional grounds,
    > to a multiple of $kT/q$ — because the Boltzmann factor
    > $e^{-\Delta E/kT}$ that controls *every* state-jump probability has
    > $kT$ as its natural energy scale, and $q$ converts to voltage.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.6 Physical noise sources — summary

    A short catalog of the noise mechanisms that propagate into circuit
    analysis, each cross-referenced to its first-principles origin above.

    #### 2.6.1 Thermal (Nyquist) noise

    A resistor $R$ at temperature $T$ has equivalent series voltage-noise
    PSD $S_v = 4kTR$ (or shunt current-noise PSD $S_i = 4kT/R$). White to
    the phonon cutoff (~10 THz in metals). Derivation: §2.4
    (macroscopic equipartition + RC, plus microscopic FDT/Landauer
    sketch). Full Landauer derivation from $f(1-f)$: §6.3.

    #### 2.6.2 Shot noise

    A DC current $I$ crossing a barrier has $S_i = 2qI$. Applies at
    diodes, BJT emitter, MOS subthreshold, photodetectors; **not** at
    ohmic resistors. Derivation: §2.5. The full diode formula
    $S_i = 2qI\,\coth(qV/2kT)$ interpolates to $4kT g_d$ at zero bias
    and $2qI$ at high bias (§2.5; algebra details: §6.4).

    #### 2.6.3 Flicker (1/f) noise

    Empirical PSD $S_v(f) = K_v/f$, dominant below the corner $f_c$.
    In MOS,

    $$S_{v_g}(f) \approx \frac{K_f}{W L C_{ox}\,f}.$$

    **Microscopic origin (McWhorter model).** A surface trap with
    capture/emission time $\tau$ contributes a Lorentzian
    $S_v(f) \propto \tau/(1 + (2\pi f\tau)^2)$ — the same generation-
    recombination shape as §2.6.4. Summing many traps whose $\tau$ is
    distributed uniformly in $\log\tau$ produces $S_v(f) \propto 1/f$
    over the spanned decades. The trap times follow Arrhenius
    $\tau \propto e^{E_a/kT}$ (the same Boltzmann state-jump probability
    from §2.5), so a uniform distribution of activation energies $E_a$
    maps to a uniform distribution of $\log\tau$. The "$kT$ controls
    state-jumps" picture of §2.5 reappears here as the *temporal*
    origin of $1/f$.

    #### 2.6.4 Generation-recombination (Lorentzian) noise

    A single trap with capture/emission time $\tau$ yields a Lorentzian
    PSD $S(f) \propto \tau/(1 + (2\pi f\tau)^2)$. Each capture/emission
    is a Poisson event (counting limit, §2.5) at rate $1/\tau$ set by
    the Boltzmann factor $e^{-E_a/kT}$. A distribution of $\tau$ over
    decades reproduces $1/f$ (§2.6.3) — the microscopic origin of
    flicker noise.
    """)
    return


@app.cell
def _(go, mo, np):
    # FIG 3 — Physical noise PSDs, log-log (all normalised to thermal floor = 1)
    _fv = np.logspace(-1, 7, 800)  # 0.1 Hz … 10 MHz
    _fc = 1e6   # flicker corner (Hz)
    _tau_trap = 1.0 / (2.0 * np.pi * 1e5)  # G-R trap at 100 kHz

    _thermal  = np.ones_like(_fv)
    _flicker  = _fc / _fv
    _gr       = (2.0 * _tau_trap) / (1.0 + (2.0 * np.pi * _fv * _tau_trap)**2)
    _gr      /= _gr.max() * 2.0   # peak normalised to 0.5
    _combined = _thermal + _flicker

    _fig_psd = go.Figure()
    _fig_psd.add_trace(go.Scatter(
        x=_fv, y=_thermal, mode="lines", name="Thermal (4kTR)",
        line=dict(color="#5599DD", width=2)))
    _fig_psd.add_trace(go.Scatter(
        x=_fv, y=_flicker, mode="lines", name="Flicker K_f / f  (f_c = 1 MHz)",
        line=dict(color="#EE8833", width=2, dash="dash")))
    _fig_psd.add_trace(go.Scatter(
        x=_fv, y=_gr, mode="lines", name="G-R Lorentzian (single trap, 100 kHz)",
        line=dict(color="#CC55AA", width=2, dash="dot")))
    _fig_psd.add_trace(go.Scatter(
        x=_fv, y=_combined, mode="lines", name="Thermal + flicker",
        line=dict(color="#33CC88", width=2)))
    _fig_psd.add_vline(
        x=_fc, line_dash="dot", line_color="#EE8833",
        annotation_text="f_c = 1 MHz", annotation_position="top left")
    _fig_psd.update_layout(
        template="plotly_dark", height=360,
        xaxis=dict(type="log", title="f (Hz)"),
        yaxis=dict(type="log", title="S(f)  [normalised to thermal floor]"),
        margin=dict(l=70, r=20, t=40, b=50),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)"),
    )
    mo.ui.plotly(_fig_psd)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Stochastic Modeling and Representations

### 3.1 Random processes, stationarity, ergodicity

    ### 2.0 Formal definition

    A **random process** (stochastic process) is a function

    $$X : \mathcal{T} \times \Omega \;\longrightarrow\; \mathbb{R},$$

    where $\mathcal{T} \subseteq \mathbb{R}$ is the time index set and
    $(\Omega, \mathcal{F}, P)$ is a probability space.

    Two interpretations of the same object:

    | Fix… | Vary… | Object |
    |---|---|---|
    | $\omega \in \Omega$ | $t \in \mathcal{T}$ | Sample path (realization) $x(t)$ |
    | $t \in \mathcal{T}$ | $\omega \in \Omega$ | Random variable $X(t)$ |

    A **noise waveform** observed on an oscilloscope is one sample path.
    The statistical description (mean, autocorrelation, PSD) lives in the
    ensemble — the collection of *all* such paths weighted by $P$.

    ---

    ### 2.1 Strict-sense stationarity (SSS)

    $X(t,\omega)$ is **strictly stationary** if for every $n$, every
    $(t_1,\ldots,t_n)$, and every shift $s$:

    $$F_{X(t_1),\ldots,X(t_n)}(x_1,\ldots,x_n) = F_{X(t_1+s),\ldots,X(t_n+s)}(x_1,\ldots,x_n).$$

    All finite-dimensional distributions are invariant under time shifts.
    SSS is the strongest stationarity condition and is rarely verifiable
    directly.

    ---

    ### 2.2 Wide-sense stationarity (WSS)

    $X(t,\omega)$ is **WSS** (or *weakly stationary*) if and only if two
    conditions hold:

    **Condition 1 — constant mean:**

    $$E[X(t)] = \mu \quad \text{for all } t.$$

    **Condition 2 — autocorrelation depends only on lag:**

    $$R_X(t_1, t_2) \;\triangleq\; E\bigl[X(t_1)\,X(t_2)\bigr] = R_X(t_1 - t_2) \equiv R_X(\tau), \quad \tau = t_1 - t_2.$$

    WSS is weaker than SSS: it constrains only the first two moments.
    The converse implication SSS $\Rightarrow$ WSS holds whenever
    $E[X(t)^2] < \infty$.

    **Special case — Gaussian processes:** for a Gaussian process WSS
    $\Leftrightarrow$ SSS, because Gaussian distributions are fully
    determined by their first two moments. This is why Gaussian noise
    models are so tractable: proving WSS is sufficient to conclude that
    the entire joint distribution is time-shift invariant.

    ---

    ### 2.3 Ergodicity

    **The question:** does a single, infinitely long sample path carry
    enough information to recover ensemble statistics?

    **Mean-ergodicity.** Define the time-average estimator

    $$\hat{\mu}_T \;\triangleq\; \frac{1}{T}\int_0^T X(t,\omega)\,dt.$$

    $X$ is **mean-ergodic** (in the mean-square sense) if

    $$\lim_{T\to\infty} E\bigl[|\hat{\mu}_T - \mu|^2\bigr] = 0.$$

    **Sufficient condition (mixing).** A straightforward computation gives

    $$E\bigl[|\hat{\mu}_T - \mu|^2\bigr] = \frac{1}{T}\int_{-T}^{T}\!\left(1 - \frac{|\tau|}{T}\right) C_X(\tau)\,d\tau,$$

    where $C_X(\tau) = R_X(\tau) - \mu^2$ is the autocovariance.
    This vanishes as $T \to \infty$ whenever

    $$\int_{-\infty}^{\infty} |C_X(\tau)|\,d\tau < \infty,$$

    i.e. the process "forgets" its past ($C_X(\tau) \to 0$ as
    $|\tau| \to \infty$). White noise and most thermal-noise models satisfy
    this condition trivially.

    **Autocorrelation-ergodicity.** Separately, the time-lag estimator

    $$\hat{R}_X(\tau) \;=\; \frac{1}{T}\int_0^T X(t)\,X(t+\tau)\,dt$$

    converges to $R_X(\tau)$ under an analogous mixing condition on the
    fourth-order cumulant. In practice: a wider bandwidth (more decorrelated
    samples per second) and a longer record both improve the estimate.

    **Practical implication.** Ergodicity is what licenses the spectrum
    analyser: the instrument replaces the ensemble average by a time average
    over one long sweep. If ergodicity fails — e.g., a systematic DC offset
    that varies slowly from device to device — the measurement is biased.

    ---

    ### 2.4 Counter-example: WSS but non-ergodic

    Let $A \sim \mathcal{N}(0, \sigma^2)$ be drawn once and held fixed.
    Define $X(t) = A$ for all $t$.

    **Ensemble statistics:**

    $$E[X(t)] = E[A] = 0, \quad R_X(t_1,t_2) = E[A^2] = \sigma^2 \quad \text{(independent of $\tau$)}.$$

    Both WSS conditions are satisfied — this is a WSS process.

    **Time average on a single path:**

    $$\hat{\mu}_T = \frac{1}{T}\int_0^T A\,dt = A.$$

    This is a random variable with distribution $\mathcal{N}(0,\sigma^2)$,
    not the constant $\mu = 0$.

    $$E\bigl[|\hat{\mu}_T - \mu|^2\bigr] = E[A^2] = \sigma^2 \quad \text{for all } T.$$

    The mean-square error does not vanish — **mean-ergodicity fails**.

    The autocovariance $C_X(\tau) = \sigma^2$ is constant and its integral
    diverges, violating the mixing condition. Physically: the process has
    infinite correlation time; the "noise" is just a random but static DC
    level, and no finite record can reveal the ensemble mean.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.2 Physical content of $R_X(\tau)$ and the PSD

    **Noise power from autocorrelation.** For a zero-mean process,
    $R_X(0) = E[X(t)^2]$ is the mean-square value — the total noise power.
    The Parseval identity for the Wiener-Khinchin pair gives

    $$P_\text{noise} = R_X(0) = \int_{-\infty}^{\infty} S_X(f)\,df = 2\int_0^{\infty} S_X(f)\,df.$$

    This is why PSD carries units of W/Hz (or V²/Hz, A²/Hz): integrate over
    any bandwidth $B$ to get the noise power in that band.

    **Correlation time and noise "memory".** The width of $R_X(\tau)$
    around $\tau = 0$ is the time scale over which the process is correlated
    with itself — how long the noise "remembers" its past value. Three
    canonical shapes:

    | Process | $R_X(\tau)$ | $S_X(f)$ | Memory |
    |---|---|---|---|
    | White | $\sigma^2 \delta(\tau)$ | $N_0$ (flat) | Zero |
    | Bandpass ($f_c$, BW $W$) | $\sigma^2 \,\mathrm{sinc}(W\tau)\cos(2\pi f_c\tau)$ | brick-wall at $f_c$ | $\sim 1/W$ |
    | Flicker (1/f) | $\sim 1/(|\tau|\log|\tau|)$ | $\propto 1/f$ | Long |

    **Why white noise is an idealisation.** A flat PSD $S_X = N_0$ for all
    $f$ implies $R_X(\tau) = N_0\delta(\tau)$ and infinite total power —
    physically impossible. Every real noise source has a cutoff: the phonon
    scattering rate in a metal resistor sets a cutoff near 10 THz; collision
    rates set similar cutoffs for shot noise. For any microwave frequency
    $f \ll 10$ THz, thermal noise is indistinguishable from white, so the
    idealisation is safe and universally used.

    **Measurement consequence — ergodicity in practice.** The time-average
    estimator $\hat{R}_X(\tau) = T^{-1}\int_0^T X(t)X(t+\tau)\,dt$ converges
    to $R_X(\tau)$ only for an ergodic process and only as $T \to \infty$.
    In practice this means: longer measurement sweeps give better PSD
    estimates, and cyclostationary processes (mixers, switched-capacitor
    circuits) need a time-averaged PSD — the standard spectrum analyser
    overestimates or underestimates noise depending on trigger phase.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.3 Autocorrelation and Wiener-Khinchin

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
def _(go, make_subplots, mo, np):
    # FIG 2 — Wiener-Khinchin canonical pairs (3 process types)
    _fig_wk = make_subplots(
        rows=2, cols=3,
        row_titles=("Autocorrelation R(τ)", "PSD S(f)"),
        column_titles=("White noise", "Band-limited (sinc)", "Exp-decay (Lorentzian)"),
    )
    _tau = np.linspace(-4.0, 4.0, 800)
    _f   = np.linspace(0.0, 4.0, 800)

    # Col 1: white noise — narrow Gaussian → flat PSD
    _sig = 0.12
    _R1  = np.exp(-_tau**2 / (2.0 * _sig**2))
    _S1  = np.ones_like(_f)
    _S1[_f > 3.0] = 0.0   # visual truncation only

    # Col 2: band-limited — sinc autocorrelation → brick-wall PSD
    _fc = 1.5
    _R2  = np.sinc(2.0 * _fc * _tau)
    _S2  = np.where(np.abs(_f) <= _fc, 1.0, 0.0)

    # Col 3: exponential-decay — Lorentzian PSD
    _tc = 0.6
    _R3  = np.exp(-np.abs(_tau) / _tc)
    _S3  = (2.0 * _tc) / (1.0 + (2.0 * np.pi * _f * _tc)**2)
    _S3 /= _S3.max()

    _col_colors = ["#5599DD", "#EE8833", "#33CC88"]
    for _col_i, (_R, _S, _clr) in enumerate(
        zip([_R1, _R2, _R3], [_S1, _S2, _S3], _col_colors), start=1
    ):
        _fig_wk.add_trace(
            go.Scatter(x=_tau, y=_R, mode="lines",
                       line=dict(color=_clr, width=2), showlegend=False),
            row=1, col=_col_i)
        _fig_wk.add_trace(
            go.Scatter(x=_f, y=_S, mode="lines",
                       line=dict(color=_clr, width=2), showlegend=False),
            row=2, col=_col_i)

    _fig_wk.update_layout(
        template="plotly_dark", height=440,
        margin=dict(l=60, r=20, t=80, b=40),
    )
    for _ci in range(1, 4):
        _fig_wk.update_xaxes(title_text="τ", row=1, col=_ci)
        _fig_wk.update_xaxes(title_text="f", row=2, col=_ci)
    for _ri in range(1, 3):
        _fig_wk.update_yaxes(title_text="R(τ)" if _ri == 1 else "S(f)", row=_ri, col=1)

    mo.ui.plotly(_fig_wk)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.4 Vector processes and matrix-valued PSDs

    For two processes $X(t)$ and $Y(t)$, the **cross-correlation** is
    $R_{XY}(\tau) = E[X(t)\,Y(t+\tau)^*]$ and the **cross-PSD** is its
    Fourier transform $S_{XY}(f)$.

    For a vector process $\mathbf{X}(t) = [X_1, X_2]^T$ the relevant
    object is the $2\times 2$ **matrix-valued PSD**:

    $$\mathbf{S}_{\mathbf{X}}(f) = \begin{bmatrix} S_{X_1}(f) & S_{X_1 X_2}(f) \\ S_{X_2 X_1}(f) & S_{X_2}(f) \end{bmatrix}.$$

    $\mathbf{S}(f)$ is Hermitian positive-semidefinite at every $f$.

    **Coherence:** $\gamma_{XY}^2(f) = |S_{XY}(f)|^2 / (S_X(f)\,S_Y(f)) \in [0,1]$
    — 0 means uncorrelated at frequency $f$; 1 means perfectly coherent.

    Part III will treat the two input-referred noise sources of a two-port
    $(v_n, i_n)$ as exactly such a vector process. The Hermitian 2×2 matrix
    above becomes the **noise correlation matrix $\mathbf{C_A}$**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.5 Equivalent circuit-level noise sources

    A noisy resistor $R$ at $T$ decomposes as:
    - **Thevenin:** noiseless $R$ in series with $v_n$, $S_{v_n} = 4kTR$.
    - **Norton:** noiseless $R$ in parallel with $i_n$, $S_{i_n} = 4kT/R$.

    Both describe the *same* physical noise.

    **Sum rules** for multiple sources:

    $$S_{\text{total}} = \sum_k S_k\quad \text{(uncorrelated)}, \qquad S_{\text{total}} = \sum_k S_k + 2\,\mathrm{Re}\!\sum_{k<\ell} S_{k\ell}\quad\text{(with cross-terms)}.$$

    **Noise bandwidth** $B_n = \int_0^\infty |H(f)|^2\,df / |H(f_0)|^2$
    — the bandwidth of an ideal brick-wall filter that would pass the same
    noise power through the transfer function $H(f)$. Distinct from the
    −3 dB bandwidth; matters for integrating detectors and oscillator
    phase noise.

    Equivalent noise bandwidth of a first-order RC lowpass (see Phase I for RC noise bandwidth panel):
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
def _(estimate_autocorr, estimate_psd, fs_hz, generate_band_limited,
      generate_flicker, generate_shot, generate_white, go, make_subplots,
      mo, n_samples, noise_kind, np):

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
    ### 3.6 N-port noise correlation matrix — rigorous derivation

    **Setup.** A linear noisy n-port is modelled as a noiseless Y-matrix
    $\mathbf{Y}$ with an additive column vector of Norton noise current sources
    $\mathbf{i}_n = [i_{n1},\ldots,i_{nn}]^T$ injected at each port:

    $$\mathbf{i} = \mathbf{Y}\,\mathbf{v} + \mathbf{i}_n.$$

    Each $i_{nk}$ is a zero-mean WSS random process; its cross-spectral density
    with $i_{nl}$ is the $(k,l)$ entry of the **Y-form noise correlation matrix**:

    $$[\mathbf{C}_Y]_{kl}(f) \;\triangleq\; \frac{\langle i_{nk}(f)\,i_{nl}^*(f)\rangle}{\Delta f}.$$

    **Hermitian positive-semidefiniteness.** $\mathbf{C}_Y$ is Hermitian by
    construction ($[\mathbf{C}_Y]_{kl} = [\mathbf{C}_Y]_{lk}^*$). For any
    $\mathbf{a} \in \mathbb{C}^n$:

    $$\mathbf{a}^\dagger \mathbf{C}_Y \mathbf{a} = \frac{\langle|\mathbf{a}^\dagger \mathbf{i}_n|^2\rangle}{\Delta f} \ge 0,$$

    so $\mathbf{C}_Y \succeq 0$. The diagonal entries are PSDs (non-negative);
    the off-diagonal entries are cross-spectral densities bounded by Cauchy-Schwarz.

    **Transformation law.** Suppose the network parameters transform as
    $\mathbf{P}' = \mathbf{T}\,\mathbf{P}$ (e.g.\ Y $\to$ Z, Y $\to$ ABCD).
    The same linear map applies to the noise sources:
    $\mathbf{s}'_n = \mathbf{T}\,\mathbf{s}_n$. Therefore:

    $$\boxed{\mathbf{C}_{P'} = \mathbf{T}\,\mathbf{C}_P\,\mathbf{T}^\dagger.}$$

    *Proof.* $\mathbf{C}_{P'} = \langle\mathbf{s}'_n(\mathbf{s}'_n)^\dagger\rangle/\Delta f
    = \mathbf{T}\langle\mathbf{s}_n\mathbf{s}_n^\dagger\rangle\mathbf{T}^\dagger/\Delta f
    = \mathbf{T}\,\mathbf{C}_P\,\mathbf{T}^\dagger$. $\square$

    The key T-matrices (2-port):
    - $\mathbf{C}_Z = \mathbf{Z}\,\mathbf{C}_Y\,\mathbf{Z}^\dagger$
      (using $\mathbf{T}_{Y\to Z} = \mathbf{Y}^{-1}$).
    - $\mathbf{C}_A = \mathbf{T}_{Y\to A}\,\mathbf{C}_Y\,\mathbf{T}_{Y\to A}^\dagger$
      with $\mathbf{T}_{Y\to A}$ the standard Y-to-ABCD bilinear matrix.

    **Noise wave (S-parameter) representation.** Define noise waves at port $k$
    (reference impedance $Z_0$):

    $$b_{nk} = \tfrac{1}{2}\!\left(\sqrt{Z_0}\,i_{nk} - \frac{v_{nk}}{\sqrt{Z_0}}\right).$$

    The map from $(v_{nk}, i_{nk})$ to $b_{nk}$ is linear; the wave correlation
    matrix $\mathbf{C}_b = \langle\mathbf{b}_n\mathbf{b}_n^\dagger\rangle/\Delta f$
    satisfies the same congruence rule and propagates under S-matrix cascading
    exactly as $\mathbf{C}_A$ propagates under ABCD cascading:

    $$\mathbf{C}_{A,\mathrm{cas}} = \mathbf{C}_{A,1} + \mathbf{A}_1\,\mathbf{C}_{A,2}\,\mathbf{A}_1^\dagger.$$

    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.7 Bosma's theorem — noise of a passive n-port from thermal equilibrium

    **Statement (Bosma, 1967).** A passive linear n-port at temperature $T$ has
    noise-wave correlation matrix:

    $$\mathbf{C}_b = kT \left( \mathbf{I} - \mathbf{S}\mathbf{S}^\dagger \right) \quad \text{[Theorem: Bosma's Fundamental Result]}$$

    **Derivation via Detailed Balance.**
    1. *Thermodynamic Equilibrium Constraints:* Let the n-port be terminated at all ports by matched loads at the same physical temperature $T$. The combined system is in thermodynamic equilibrium. According to the Nyquist-Rayleigh-Jeans relation, each port receives an incident noise wave $a_i$ carrying an available power spectral density of:

       $$\frac{\langle \mathbf{a} \mathbf{a}^\dagger \rangle}{\Delta f} = kT \mathbf{I}$$

    2. *Linear Superposition of Waves:* The outgoing wave vector $\mathbf{b}$ consists of the reflected and transmitted incident waves plus the internally generated noise wave vector $\mathbf{b}_n$:

       $$\mathbf{b} = \mathbf{S}\mathbf{a} + \mathbf{b}_n$$

       Since the internal noise sources of the network are independent of external terminations, $\mathbf{b}_n$ is uncorrelated with the incident wave vector $\mathbf{a}$:

       $$\langle \mathbf{b}_n \mathbf{a}^\dagger \rangle = \mathbf{0} \quad \text{and} \quad \langle \mathbf{a} \mathbf{b}_n^\dagger \rangle = \mathbf{0}$$

    3. *Detailed Balance Principle:* In thermodynamic equilibrium, the power outgoing from any port must equal the power incident on it to prevent net heat transfer:

       $$\frac{\langle \mathbf{b} \mathbf{b}^\dagger \rangle}{\Delta f} = \frac{\langle \mathbf{a} \mathbf{a}^\dagger \rangle}{\Delta f} = kT \mathbf{I}$$

    4. *Algebraic Derivation:* Substituting the superposition relation into the detailed balance expression:

       $$\frac{\langle (\mathbf{S}\mathbf{a} + \mathbf{b}_n)(\mathbf{S}\mathbf{a} + \mathbf{b}_n)^\dagger \rangle}{\Delta f} = kT \mathbf{I}$$

       Expanding the product:

       $$\frac{1}{\Delta f} \left( \mathbf{S}\langle \mathbf{a}\mathbf{a}^\dagger \rangle\mathbf{S}^\dagger + \mathbf{S}\langle \mathbf{a}\mathbf{b}_n^\dagger \rangle + \langle \mathbf{b}_n\mathbf{a}^\dagger \rangle\mathbf{S}^\dagger + \langle \mathbf{b}_n\mathbf{b}_n^\dagger \rangle \right) = kT \mathbf{I}$$

       Applying the lack of correlation ($\langle \mathbf{b}_n \mathbf{a}^\dagger \rangle = \mathbf{0}$) and the covariance of incident waves ($\langle \mathbf{a} \mathbf{a}^\dagger \rangle = kT \Delta f \mathbf{I}$):

       $$kT \mathbf{S}\mathbf{S}^\dagger + \mathbf{C}_b = kT \mathbf{I}$$

       Isolating $\mathbf{C}_b$ yields:

       $$\mathbf{C}_b = kT \left( \mathbf{I} - \mathbf{S}\mathbf{S}^\dagger \right) \quad \square$$

    **Consistency checks.**
    - *Lossless n-port:* unitarity $\mathbf{S}^\dagger\mathbf{S}=\mathbf{I}$ implies
      $\mathbf{S}\mathbf{S}^\dagger = \mathbf{I}$, so $\mathbf{C}_b = 0$.
      A lossless passive network generates no noise.
    - *Matched termination (1-port, $S_{11}=0$):* $C_b = kT$ — the load
      emits full thermal noise, consistent with Nyquist.
    - *Attenuator with loss $L$ ($|S_{21}\|^2 = 1/L$):* $[\mathbf{C}_b]_{22} = kT(1-1/L)$,
      exactly the noise required to set the NF of a passive attenuator at $T$
      equal to $L$ (§14.1).

    **Connection to the fluctuation-dissipation theorem.** Bosma's theorem is the
    S-parameter statement of the FDT: wherever there is dissipation there is noise,
    in fixed ratio $kT$. The matrix $\mathbf{I} - \mathbf{S}\mathbf{S}^\dagger \succeq 0$
    is the n-port's **dissipation matrix** — zero for lossless networks,
    positive-definite for dissipative ones.

    **Bound on noise redistribution.** Because $\mathbf{C}_b \succeq 0$ for any
    passive component, no passive linear network can reduce the total noise power
    below the Bosma bound. Squeezing noise from one quadrature into another is
    possible (§19.6), but requires an active or nonlinear element.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Two-Port Noise Theory and Parameters

### 4.1 Noise figure F

    **Classical definition (Friis, 1944):**

    $$F \;\triangleq\; \frac{\mathrm{SNR}_{\text{in}}}{\mathrm{SNR}_{\text{out}}} \quad\text{at source temperature } T_0 = 290\text{ K.}$$

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
    ### 4.2 NF of a passive two-port equals its insertion loss

    **Statement.** A passive, reciprocal, lossy two-port at the standard
    reference temperature $T_0 = 290$ K has noise factor

    $$F = L,$$

    where $L = 1/G_A \ge 1$ is the insertion loss (linear) of the two-port.

    **Derivation.** Let the two-port have available power gain $G_A = 1/L$.
    At $T_0$ the input noise power spectral density is $kT_0$. Since the
    network is passive and at $T_0$, its output is a thermally equilibrated
    load: the total output noise PSD is $kT_0$ regardless of the signal path.
    The signal, however, is reduced by $G_A = 1/L$. Hence

    $$F = \frac{\mathrm{SNR}_{\mathrm{in}}}{\mathrm{SNR}_{\mathrm{out}}} = \frac{P_{\mathrm{sig,in}} / (kT_0)}{G_A\,P_{\mathrm{sig,in}} / (kT_0)} = \frac{1}{G_A} = L.$$

    Alternatively: the network adds $(L-1)kT_0$ of noise referred to its input
    (since its output noise $kT_0$ maps back to $kT_0 / G_A = L\,kT_0$, and
    the source already contributes $kT_0$), giving $F = 1 + (L-1) = L$.

    **Practical consequences:**

    - Any lossy passive element (matching network, switch, transmission line)
      placed **before** the LNA raises the system NF by its insertion loss.
      An interconnect with 1 dB loss adds 1 dB to NF before the transistor
      noise even enters — this loss cannot be recovered by the LNA.
    - A lossy matching network with IL $= L$ placed **after** the LNA
      contributes $F_{\mathrm{match}} = L$ to the Friis sum, but divided by
      $G_{A,\mathrm{LNA}} \gg 1$, so its effect on system NF is negligible.
      **Lossy elements must go after the LNA, not before it.**
    - The same result applies to switches, phase shifters, and diplexers in
      the receive path — a phased-array front end with 3 dB of switch loss
      before the LNA has at least 3 dB of NF degradation, independent of LNA quality.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 The Rothe-Dahlke noise two-port

    **Theorem (Rothe & Dahlke, 1956).** Any noisy linear two-port is
    equivalent to a *noiseless* two-port (same Y-parameters) plus two
    correlated noise sources — an input-referred voltage $v_n$ and
    current $i_n$ — connected at the input port.

    $$\begin{bmatrix} v_1 \\ i_1 \end{bmatrix} = \begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} v_2 \\ -i_2 \end{bmatrix} + \begin{bmatrix} v_n \\ i_n \end{bmatrix}$$

    Derivation: start from the Y-matrix form with internal noise
    sources $i_{n1}, i_{n2}$ at the ports; pre-multiply by the
    ABCD transformation; the two noise currents combine into one
    equivalent $(v_n, i_n)$ pair at the input.

    The **noise correlation matrix** in ABCD form:

    $$\mathbf{C_A} = \left\langle\begin{bmatrix} v_n \\ i_n \end{bmatrix}\begin{bmatrix} v_n^* & i_n^* \end{bmatrix}\right\rangle = \begin{bmatrix} \overline{|v_n|^2} & \overline{v_n i_n^*} \\ \overline{v_n^* i_n} & \overline{|i_n|^2} \end{bmatrix}$$

    Hermitian, positive-semidefinite, 2×2 — exactly the matrix PSD
    of §4. Admittance form $\mathbf{C_Y}$ and impedance form
    $\mathbf{C_Z}$ are alternative representations; §10 will show the
    transformations. All derivations in §8–§9 stay in $\mathbf{C_A}$.

    **n-port generalisation.** For a linear noisy n-port, the same
    argument applies port-by-port: the noise behaviour is fully described
    by $n$ noise sources (voltages, currents, or noise waves) at the ports
    and their $n \times n$ correlation matrix $\mathbf{C}$. The matrix
    is Hermitian positive-semidefinite; its diagonal entries are the
    individual source PSDs and the off-diagonal entries carry the
    cross-correlations. For noise-wave representations (used in
    microwave measurement and CAD), the sources are incident noise waves
    $b_{n,k}$ at each port and the correlation matrix
    $\mathbf{C_b} = \overline{\mathbf{b}_n \mathbf{b}_n^\dagger}$
    transforms under S-matrix cascading in the same way $\mathbf{C_A}$
    transforms under ABCD cascading — enabling systematic noise analysis
    of multi-stage, multi-port mmWave systems.

    **Connection to S-parameter matrix transformations.** The signal
    parameters of a two-port can be expressed in any basis: S, Z, Y, H, G,
    or ABCD — related by standard bilinear matrix transformations
    (see notebook 03, §4). The noise correlation matrix transforms with
    the same T-matrix: if $\mathbf{P'} = \mathbf{T}\,\mathbf{P}$, then
    $\mathbf{C_{P'}} = \mathbf{T}\,\mathbf{C_P}\,\mathbf{T}^\dagger$.
    In practice: start from the measured S-parameters and noise figure
    data, convert to $\mathbf{C_A}$ via the S→ABCD transformation, then
    cascade in ABCD form, and convert back to S to read off system NF.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.4 Noise correlation matrix transformations

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
    ### 4.5 F as a function of source admittance — the four noise parameters

    **Uncorrelated Case Derivation (First Principles).**
    Consider a noisy linear two-port driven by a source with impedance $Z_S = R_S + j X_S$ at temperature $T$. We represent the noisy two-port by a noiseless network with two input-referred noise sources: a series voltage source $v_n$ and a shunt current source $i_n$. The source impedance is associated with a thermal noise voltage source $v_S$, where:
    
    $$\overline{v_S^2} = 4 k T \Delta f R_S \quad \text{[Axiom: Nyquist Thermal Noise]}$$
    
    The total equivalent open-circuit noise voltage $v_{\text{total}}$ in the input loop is:
    
    $$v_{\text{total}} = v_S + v_n + Z_S i_n$$
    
    Assuming the source noise $v_S$ is generated by a physically independent source, it is uncorrelated with the two-port noise sources $v_n$ and $i_n$. The mean-squared total voltage is:
    
    $$\overline{|v_{\text{total}}|^2} = \overline{v_S^2} + \overline{|v_n + Z_S i_n|^2} = \overline{v_S^2} + \overline{v_n^2} + |Z_S|^2 \overline{i_n^2} + 2\text{Re}\left( Z_S^* \overline{v_n i_n^*} \right)$$
    
    If the input-referred voltage and current noise sources are uncorrelated ($\overline{v_n i_n^*} = 0$):
    
    $$\overline{|v_{\text{total}}|^2} = \overline{v_S^2} + \overline{v_n^2} + (R_S^2 + X_S^2) \overline{i_n^2}$$
    
    The noise factor $F$ is defined as the ratio of the total equivalent noise power (or mean-squared voltage) to the noise power from the source alone:
    
    $$F \;\triangleq\; \frac{\overline{|v_{\text{total}}|^2}}{\overline{v_S^2}} = 1 + \frac{\overline{|v_n + Z_S i_n|^2}}{\overline{v_S^2}} \quad \text{[Definition]}$$
    
    Substituting the uncorrelated terms yields:
    
    $$F = 1 + \frac{\overline{v_n^2}}{4k T \Delta f R_S} + \frac{(R_S^2 + X_S^2)\overline{i_n^2}}{4k T \Delta f R_S} \quad \text{[Corollary: Final Derived Equation Result]}$$
    
    **Generalization to Correlated Case.**
    When $v_n$ and $i_n$ are correlated, we write $i_n = i_u + Y_c v_n$, where $Y_c = G_c + j B_c$ is the correlation admittance and $i_u$ is uncorrelated with $v_n$. Substituting this and converting to source admittance $Y_s = G_s + j B_s$ yields the general four-parameter expression:

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

    $$F(\Gamma_s) \;=\; F_{\min} \;+\; \frac{4\,R_n/Z_0}{|1 + \Gamma_{\text{opt}}|^2}\; \frac{|\Gamma_s - \Gamma_{\text{opt}}|^2}{1 - |\Gamma_s|^2}.$$

    Two sanity checks (next cell): $F(\Gamma_{\text{opt}}) = F_{\min}$, and
    $F(\Gamma_s \to \partial\mathrm{disk}) \to \infty$.
    """)
    return


@app.cell
def _(mo, noise_figure_from_Gamma, np):
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
    ### 4.6 Noise circles

    Setting $F = F_{\text{target}}$ in the §9 expression and rearranging
    for $\Gamma_s$ gives a circle in the Γ-plane. Define

    $$N \;\triangleq\; \frac{F_{\text{target}} - F_{\min}}{4\,R_n / Z_0}\,|1 + \Gamma_{\text{opt}}|^2.$$

    Then the centre and radius of the constant-$F$ circle are

    $$C_F \;=\; \frac{\Gamma_{\text{opt}}}{1 + N}, \qquad r_F \;=\; \frac{\sqrt{N^2 + N\,(1 - |\Gamma_{\text{opt}}|^2)}}{1 + N}.$$

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
def _(mo, noise_circle, np):
    _noise_circle = noise_circle

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
def _(go, mo, noise_circle, np):
    # FIG 5 — Static noise circles in Γ-plane
    _Fmin  = 1.3
    _Rn    = 0.08
    _Gopt  = 0.4 * np.exp(1j * np.radians(50.0))
    _th_c  = np.linspace(0.0, 2.0 * np.pi, 360)

    _circle_colors = ["#33CC88", "#5599DD", "#EE8833", "#CC55AA", "#FF6644"]
    _F_targets_dB  = [0.3, 0.6, 1.0, 2.0, 4.0]

    _fig_nc = go.Figure()

    # Unit circle
    _fig_nc.add_trace(go.Scatter(
        x=np.cos(_th_c), y=np.sin(_th_c),
        mode="lines", name="|Γ| = 1",
        line=dict(color="#555555", width=1.5, dash="dot")))

    # Noise circles
    for _dF_dB, _clr in zip(_F_targets_dB, _circle_colors):
        _Ft = _Fmin * 10.0**(_dF_dB / 10.0)
        _c, _r = noise_circle(_Ft, _Fmin, _Rn, _Gopt)
        _fig_nc.add_trace(go.Scatter(
            x=_c.real + _r * np.cos(_th_c),
            y=_c.imag + _r * np.sin(_th_c),
            mode="lines",
            name=f"F_min + {_dF_dB} dB",
            line=dict(color=_clr, width=2)))

    # Γ_opt marker
    _fig_nc.add_trace(go.Scatter(
        x=[_Gopt.real], y=[_Gopt.imag],
        mode="markers", name="Γ_opt",
        marker=dict(symbol="star", size=12, color="white")))

    _fig_nc.update_layout(
        template="plotly_dark", height=420,
        xaxis=dict(title="Re(Γ)", range=[-1.1, 1.1],
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(Γ)", range=[-1.1, 1.1]),
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    mo.ui.plotly(_fig_nc)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.7 Cascaded noise — the Friis formula

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
def _(friis_cascade, math, mo):
    _friis = friis_cascade

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
def _(F1_db, F2_db, F3_db, G1_db, G2_db, G3_db, friis_cascade, go, math, mo, reorder):
    _friis = friis_cascade

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
    ### 4.8 Generalized noise temperature in noise wave formulation

    **Theorem Statement.** The equivalent input-referred noise temperature $T_e$ of a two-port network connected to a source with reflection coefficient $\Gamma_S$ is expressed in quadratic form as:

    $$T_e = \frac{\boldsymbol{\alpha}\mathbf{C}_b\boldsymbol{\alpha}^\dagger}{k(1 - |\Gamma_S|^2)} \quad \text{[Corollary: Derived Generalized Noise Temperature Equation]}$$

    where $\boldsymbol{\alpha}$ is a row vector mapping the device noise waves to the input, and $\dagger$ denotes the conjugate transpose.

    **Derivation from First Principles.**
    1. *Outgoing Noise Wave Setup:* Let the noisy two-port be modeled as a noiseless network with scattering matrix $\mathbf{S}$ and internal noise wave vector $\mathbf{b}_n = [b_{n1}, b_{n2}]^T$. We connect a source with reflection coefficient $\Gamma_S$ to Port 1 and terminate Port 2 with a matched load ($\Gamma_L = 0$):

       $$a_1 = \Gamma_S b_1$$

       $$a_2 = 0$$

    2. *Wave Relations:*

       $$b_1 = S_{11} a_1 + S_{12} a_2 + b_{n1} = S_{11} \Gamma_S b_1 + b_{n1} \implies b_1 = \frac{b_{n1}}{1 - S_{11}\Gamma_S}$$

       The incident wave at Port 1 is:

       $$a_1 = \frac{\Gamma_S b_{n1}}{1 - S_{11}\Gamma_S}$$

       The outgoing noise wave at Port 2 is:

       $$b_2 = S_{21} a_1 + b_{n2} = \frac{S_{21}\Gamma_S}{1 - S_{11}\Gamma_S} b_{n1} + b_{n2}$$

    3. *Input Referral:* Divide the outgoing noise wave $b_2$ by the transmission gain term $\frac{S_{21}}{1 - S_{11}\Gamma_S}$ to refer it to the input:

       $$b_{n,\text{in}} = \frac{1 - S_{11}\Gamma_S}{S_{21}} b_2 = \Gamma_S b_{n1} + \frac{1 - S_{11}\Gamma_S}{S_{21}} b_{n2}$$

       Defining the row vector $\boldsymbol{\alpha}$:

       $$\boldsymbol{\alpha} = \begin{bmatrix} \Gamma_S & \frac{1 - S_{11}\Gamma_S}{S_{21}} \end{bmatrix}$$

       We obtain the equivalent input-referred noise wave:

       $$b_{n,\text{in}} = \boldsymbol{\alpha} \mathbf{b}_n$$

    4. *Available Noise Power Referral:* The mean-squared input-referred noise wave is:

       $$\overline{|b_{n,\text{in}}|^2} = \boldsymbol{\alpha} \langle \mathbf{b}_n \mathbf{b}_n^\dagger \rangle \boldsymbol{\alpha}^\dagger = \boldsymbol{\alpha} \mathbf{C}_b \boldsymbol{\alpha}^\dagger \Delta f$$

       To find the available power at the input, we divide by the source mismatch factor $(1 - |\Gamma_S|^2)$:

       $$P_{n,\text{in,avail}} = \frac{\overline{|b_{n,\text{in}}|^2}}{(1 - |\Gamma_S|^2) \Delta f} = \frac{\boldsymbol{\alpha} \mathbf{C}_b \boldsymbol{\alpha}^\dagger}{1 - |\Gamma_S|^2}$$

       Equating this to the available thermal noise power from an equivalent source at temperature $T_e$:

       $$k T_e = \frac{\boldsymbol{\alpha} \mathbf{C}_b \boldsymbol{\alpha}^\dagger}{1 - |\Gamma_S|^2} \implies T_e = \frac{\boldsymbol{\alpha}\mathbf{C}_b\boldsymbol{\alpha}^\dagger}{k(1 - |\Gamma_S|^2)} \quad \square$$

    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Low-Noise Amplifier Design

### 5.1 The gain-noise tradeoff & Noise measure M

    The fundamental design tension: in general

    $$\Gamma_{\text{opt}} \;\ne\; S_{11}^*,$$

    so noise-matching ($\Gamma_s = \Gamma_{\text{opt}}$) costs input
    reflection (and hence gain), and power-matching ($\Gamma_s = S_{11}^*$)
    costs noise figure. Overlay available-gain circles (notebook 03)
    on noise circles — the geometry of the compromise appears directly.

    **Noise measure M (Haus-Adler, 1958):** the invariant that collapses
    the tradeoff into one number:

    $$M \;\triangleq\; \frac{F - 1}{1 - 1/G_{A}} \quad \text{[Definition]}$$

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

    **Haus-Adler Optimization Theorem (Cascade Ordering).**
    For a cascade of two linear amplifier stages, Stage A and Stage B, with noise measures $M_A$ and $M_B$ respectively, the overall noise figure of the cascade is minimized if and only if the stage with the lower noise measure is placed first:

    $$M_A < M_B \implies F_{AB} < F_{BA} \quad \text{[Theorem: Haus-Adler Optimization Theorem Result]}$$

    *Derivation.*
    1. Let the noise measures of Stage A and Stage B be:

       $$M_A \triangleq \frac{F_A - 1}{1 - 1/G_A} \quad \text{and} \quad M_B \triangleq \frac{F_B - 1}{1 - 1/G_B} \quad \text{[Definition]}$$

    2. Using the Friis cascade formula, the overall noise factors for configurations $AB$ and $BA$ are:

       $$F_{AB} = F_A + \frac{F_B - 1}{G_A} \quad \text{and} \quad F_{BA} = F_B + \frac{F_A - 1}{G_B}$$

    3. We set the inequality for configuration $AB$ to be superior to $BA$:

       $$F_{AB} < F_{BA} \implies F_A + \frac{F_B - 1}{G_A} < F_B + \frac{F_A - 1}{G_B}$$

    4. Subtract 1 from both sides to express in terms of excess noise factors:

       $$(F_A - 1) + \frac{F_B - 1}{G_A} < (F_B - 1) + \frac{F_A - 1}{G_B}$$

    5. Group the $(F_A - 1)$ and $(F_B - 1)$ terms to simplify the inequality:

       $$(F_A - 1) - \frac{F_A - 1}{G_B} < (F_B - 1) - \frac{F_B - 1}{G_A}$$

    6. Factor the expressions:

       $$(F_A - 1) \left( 1 - \frac{1}{G_B} \right) < (F_B - 1) \left( 1 - \frac{1}{G_A} \right)$$

    7. Assuming active stages with available gains $G_A > 1$ and $G_B > 1$, the factors $\left(1 - \frac{1}{G_A}\right)$ and $\left(1 - \frac{1}{G_B}\right)$ are positive. Dividing by their product yields:

       $$\frac{F_A - 1}{1 - 1/G_A} < \frac{F_B - 1}{1 - 1/G_B}$$

       which, by definition, is:

       $$M_A < M_B \quad \square$$

    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.2 Simultaneous noise + gain match

    **Haus condition:** a lossless feedback embedding can rotate
    $\Gamma_{\text{opt}}$ to $S_{11}^*$ if and only if a specific
    inequality between the two-port's noise parameters and its
    small-signal parameters is satisfied.

    On 65 nm bulk CMOS at 28 GHz the condition **fails**: the intrinsic
    transistor has $\Gamma_{\text{opt}}$ inside the unit disk but far
    enough from $S_{11}^*$ that any lossless transformation that
    moves one still leaves the other poorly matched. The designer's
    compromise: pick $\Gamma_s$ on the locus minimising a cost

    $$J(\Gamma_s) \;=\; \alpha\,[F(\Gamma_s) - F_{\min}] \;+\; \beta\,[G_{A,\max} - G_A(\Gamma_s)],$$

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
    mo.md(r"""
    ### 5.3 Feedback and noise — lossy vs lossless

    Feedback changes $Y_{\mathrm{opt}}$, $R_n$, and (for lossy feedback) $F_{\min}$.
    The key distinction:

    ### Lossy feedback raises $F_{\min}$

    A feedback element with series resistance $r_F$ (e.g., a resistive
    shunt-feedback resistor $R_F$ at the input) contributes thermal noise
    $\overline{v_{n,F}^2}/\Delta f = 4kT r_F$ at the input. This excess
    noise is irreducible: it raises $F_{\min}$ by approximately

    $$\Delta F_{\min} \approx \frac{r_F}{R_s} \cdot \frac{1}{|1 - \beta A|^2},$$

    where $\beta A$ is the loop gain. No retuning of the source impedance
    removes this penalty because $F_{\min}$ is the minimum over all sources —
    a floor set by the device and its noise sources.

    **Example:** a transimpedance amplifier with a $200\,\Omega$ shunt-feedback
    resistor $R_F$ at 50 $\Omega$ source has a noise contribution
    $\Delta F \approx 4kT/R_F \cdot R_s \approx 1$ at low frequency —
    roughly doubling the noise figure. This is unavoidable for resistive feedback.

    ### Lossless (purely reactive) feedback preserves $F_{\min}$

    **Theorem (Haus and Adler, 1958).** $F_{\min}$ is invariant under any
    lossless, reciprocal embedding of a two-port. That is, adding reactive
    elements (inductors, capacitors, transmission-line stubs, ideal transformers)
    around the transistor cannot change $F_{\min}$.

    **Proof sketch.** The noise measure $M = (F-1)/(1-1/G_A)$ is invariant
    under lossless embedding (Haus-Adler). At the noise-optimal source,
    $F = F_{\min}$, and in the limit $G_A \to \infty$ (which is approached by
    choosing a lossless feedback that maximises available gain), $M \to F_{\min} - 1$.
    Since $M$ is invariant, $F_{\min}$ cannot change.

    **What lossless feedback does change:**

    - $\Gamma_{\mathrm{opt}}$ rotates to a new location on the Smith chart.
    - $R_n$ may decrease — tightening or loosening noise circles.
    - The imaginary part $B_{s,\mathrm{opt}}$ shifts; the real part $G_{s,\mathrm{opt}}$
      changes very little under purely reactive feedback.

    **Inductive source degeneration as the canonical example.**
    $L_s$ in the source path creates lossless series-series feedback:
    - $\Gamma_{\mathrm{opt}}$ rotates toward $S_{11}^*$ (§14).
    - $F_{\min}$ is unchanged (no resistor in the feedback path).
    - Once $L_s$ has a finite series resistance $r_{L_s} = \omega L_s / Q$,
      the feedback is no longer lossless and $F_{\min}$ rises by
      $\Delta F_{\min} \approx r_{L_s} / R_s$. This is why inductor Q is
      a hard constraint in mmWave LNA design.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.4 Noise sensitivity — minimizing $R_n$

    The noise figure formula

    $$F(\Gamma_s) = F_{\min} + \frac{4\,R_n/Z_0}{|1+\Gamma_{\mathrm{opt}}|^2} \frac{|\Gamma_s - \Gamma_{\mathrm{opt}}|^2}{1 - |\Gamma_s|^2}$$

    has two distinct roles for $R_n$:

    1. **Sensitivity:** for a fixed source mismatch $|\Gamma_s - \Gamma_{\mathrm{opt}}|$,
       $F - F_{\min} \propto R_n$. Large $R_n$ means steep noise circles — a small
       departure from $\Gamma_{\mathrm{opt}}$ incurs a large NF penalty.

    2. **Circle radii:** the noise circle for target $F_{\mathrm{target}}$ has
       $N \propto (F_{\mathrm{target}} - F_{\min})/R_n$. Small $R_n$ gives large $N$
       and therefore larger circles — more source impedance choices are within
       the same NF contour.

    **Design goal:** minimise $R_n$ (and its dual $G_n^0 = R_n G_{s,\mathrm{opt}}^2$,
    the noise conductance at the optimum point).

    **At the device level** (Shaeffer-Lee, §16.4):

    $$R_n \approx \frac{\gamma}{\alpha\,g_m\,Z_0}.$$

    Increasing $g_m$ (wider device, higher bias current $I_D$) reduces $R_n$
    at the cost of power. This is the fundamental power–noise-sensitivity tradeoff:
    a low-$R_n$ LNA requires high $g_m$, which demands high $I_D$.

    **At the circuit level:** series feedback (inductive degeneration) also
    modifies the effective $R_n$ seen from the input by changing the reflection
    between $\Gamma_s$ and the actual transistor gate. Specifically, source
    degeneration reduces the effective $R_n$ seen by the source, spreading the
    noise circles and relaxing the matching precision required.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.5 Wideband noise matching

    Single-frequency noise matching is straightforward: set $\Gamma_s(f_0) = \Gamma_{\mathrm{opt}}(f_0)$.
    Over a bandwidth, both $\Gamma_s(f)$ and $\Gamma_{\mathrm{opt}}(f)$ move, and
    the noise figure degrades away from $f_0$.

    **What determines the bandwidth of the noise match?**
    The rate at which $Y_{s,\mathrm{opt}}(f) = G_{s,\mathrm{opt}} + jB_{s,\mathrm{opt}}$
    moves with frequency is set by the Q of the optimum source admittance:

    $$Q_{\mathrm{opt}} \;=\; \left|\frac{B_{s,\mathrm{opt}}}{G_{s,\mathrm{opt}}}\right| \quad\text{or equivalently}\quad \left|\frac{X_{s,\mathrm{opt}}}{R_{s,\mathrm{opt}}}\right|.$$

    A high $Q_{\mathrm{opt}}$ means $Y_{\mathrm{opt}}$ has a large reactive component
    that sweeps rapidly across the Smith chart with frequency — the noise match is
    inherently narrowband. A low $Q_{\mathrm{opt}}$ (optimum source near the real axis)
    varies slowly — broadband noise match is achievable.

    **Three strategies:**

    1. **Minimize** $|B_{s,\mathrm{opt}} / G_{s,\mathrm{opt}}|$.
       Make $Y_{\mathrm{opt}}$ look like a near-real conductance. Inductive source
       degeneration does this: it shifts $B_{s,\mathrm{opt}}$ toward zero while
       leaving $G_{s,\mathrm{opt}}$ roughly unchanged, reducing $Q_{\mathrm{opt}}$.
       This is a second benefit of source degeneration, independent of input matching.

    2. **Reactive feedback to rotate $\Gamma_{\mathrm{opt}}$ toward the real axis.**
       Since $F_{\min}$ is invariant, the designer can choose feedback reactances
       to place $\Gamma_{\mathrm{opt}}$ on or near the real $\Gamma$ axis (i.e.,
       real $Y_{\mathrm{opt}}$) at the design frequency, minimizing
       $|B_{s,\mathrm{opt}} / G_{s,\mathrm{opt}}|$ and the rate of $\Gamma_{\mathrm{opt}}$
       variation with frequency.

    3. **Minimize $R_n$ (§14.3).**
       Even if $\Gamma_{\mathrm{opt}}$ moves, a small $R_n$ keeps the noise circles
       wide — the designer can accept a moderate $\Gamma_s \neq \Gamma_{\mathrm{opt}}$
       without incurring large NF penalty. This trades noise-match bandwidth against
       power consumption.

    **Bode-Fano limit analogy.** Just as there is a fundamental limit to the
    bandwidth over which a reactive network can match a complex load (Bode-Fano),
    there is an analogous bound on noise-match bandwidth set by the
    $Q_{\mathrm{opt}}$ of the device. It is derived from the same causality and
    analyticity constraints on the noise parameters. At 28 GHz, a 65 nm CMOS
    transistor with $Q_{\mathrm{opt}} \approx 1$ after source degeneration can
    achieve $F < F_{\min} + 0.5$ dB over roughly 20–25% relative bandwidth —
    consistent with the $\sim 6$ GHz 5G NR FR2 channel aggregation window.
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
        mo.md("### 5.6 Interactive — Noise / gain / stability circles"),
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
def _(Fmin_db_c, Gopt_ang_c, Gopt_mag_c, MAG_dB, Rn_norm_c,
      available_gain_circle, go, mo, noise_circle, np, rollett_K,
      s11m_c, s11p_c, s12m_c, s12p_c, s21m_c, s21p_c, s22m_c, s22p_c,
      source_stability_circle):
    _noise_circle = noise_circle
    _ag_circle = available_gain_circle
    _src_stab = source_stability_circle
    _rollett_K = rollett_K
    _MAG_dB = MAG_dB

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
    ### 5.7 The inductively-degenerated common-source LNA

    #### 5.7.1 Topology and rationale

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
    #### 5.7.2 Small-signal model at mmWave

    Standard π-hybrid NMOS:

    $$\begin{array}{rl} C_{gs} & \text{gate-source capacitance (dominant)} \\ C_{gd} & \text{gate-drain (Miller) capacitance} \\ g_m    & \text{small-signal transconductance} \\ r_o    & \text{output resistance} \\ r_g    & \text{gate resistance (multi-finger layout)} \\ \end{array}$$

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
    #### 5.7.3 Input impedance

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
    #### 5.7.4 Noise analysis

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
    ### 5.8 Worked design — 28 GHz, 65 nm CMOS LNA

    **Spec target:**
    - $\mathrm{NF} < 2.5$ dB
    - $|S_{21}| > 15$ dB (requires cascode + output matching — see §18)
    - $|S_{11}| < -10$ dB
    - $P_{DC} < 10$ mW
    - 50 Ω source and load at 28 GHz

    Seven-step design:

    1. **§5.8.1** Choose current density $J_\text{opt}$ at minimum-NF point, size $W$ for $P_{DC}$ budget.
    2. **§5.8.2** Choose $L_s$ for $\mathrm{Re}(Z_{\text{in}}) = 50\,\Omega$.
    3. **§5.8.3** Choose $L_g$ for resonance at 28 GHz.
    4. **§5.8.4** Verify $\Gamma_{\text{opt}} \approx S_{11}^*$.
    5. **§5.8.5** Add cascode for isolation.
    6. **§5.8.6** Design simple tapped-cap output match to 50 Ω.
    7. **§5.8.7** Verify S-parameters, NF, and stability sweeps.
    """)
    return


@app.cell
def _(DeviceParams, F_min_shaeffer_lee, compute_operating_point,
      gamma_opt_degenerated_cs, input_impedance, math, mo, np):
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
    #### 5.8.7 Design point computed:

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
def _(L_d_design, L_g_design, L_s_design, NF_degenerated_cs, W_design_um,
      f0, go, make_subplots, math, mo, np, params_design, sparameters_at_freq):

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
        mo.md("### 5.9 Interactive — 28 GHz LNA design studio"),
        mo.hstack([W_l, J_l, cascode_l]),
        mo.hstack([Ls_pH_l, Lg_pH_l, Ld_pH_l, f0_GHz_l]),
    ])
    return Ld_pH_l, Lg_pH_l, Ls_pH_l, W_l, J_l, cascode_l, f0_GHz_l


@app.cell
def _(Ld_pH_l, Lg_pH_l, Ls_pH_l, NF_degenerated_cs, W_l, J_l, cascode_l,
      compute_operating_point, f0_GHz_l, go, make_subplots, math, mo, np,
      rollett_K, sparameters_at_freq):
    _op_point = compute_operating_point
    _sparams = sparameters_at_freq
    _nf_cs = NF_degenerated_cs
    _rollett_K = rollett_K

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
    ### 5.10 Practical mmWave realities

    Five topics separating the idealised design from silicon reality.

    #### 5.10.1 Inductor Q
    On-chip spiral or transmission-line inductors have $Q \sim 10$–20 at
    28 GHz. An inductor with reactance $X_L$ contributes loss
    $r_L = X_L / Q$ in series. The NF penalty from finite-Q $L_g$ can
    be significant (next cell computes it for $Q = 12$).

    #### 5.10.2 Gate resistance layout
    $r_g \propto R_\square W_f / (12 N_f^2)$ for double-sided-contact
    multi-finger layout. The plot below shows ΔNF vs. $N_f$ —
    diminishing returns past $N_f \approx 20$–40.

    #### 5.10.3 Substrate coupling
    Pattern-ground-shields under inductors, deep-nwell around the LNA —
    typical mmWave layout hygiene.

    #### 5.10.4 PDK vs textbook model
    Where Shaeffer-Lee deviates from silicon: short-channel velocity
    saturation ($g_m$ vs $V_{GS}$ plateaus earlier); non-quasi-static
    effects above $f_T/3$ make $C_{gs}$ frequency-dependent; $\gamma$ is
    itself frequency-dependent above ~$0.3 f_T$. Silicon measurements
    match textbook within ~1 dB at 28 GHz; divergence grows above 60 GHz.

    #### 5.10.5 Beyond 28 GHz
    At 60/77 GHz: inductor $Q$ drops further, $r_g$ dominates; at 140+ GHz
    transmission-line-matched stages replace lumped LC; noise floor
    increasingly set by interconnect and pad loss (notebook 05).
    """)
    return


@app.cell
def _(compute_operating_point, math, mo):
    _op_point = compute_operating_point

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
    ### 5.11 NF budget consumption at high frequencies

    Three mechanisms eat into the NF budget as frequency rises:

    #### 5.11.1 Intrinsic device $F_{\min} \propto f/f_T$.
    Shaeffer-Lee (§16.4) gives

    $$F_{\min} \approx 1 + \frac{2.4\,\gamma}{\alpha}\,\frac{f}{f_T}.$$

    On 65 nm CMOS ($f_T \approx 200$ GHz, $\gamma = 1.4$, $\alpha = 0.8$):

    | $f$ | $F_{\min}$ | NF |
    |---|---|---|
    | 10 GHz  | 1.11 | 0.46 dB |
    | 28 GHz  | 1.29 | 1.12 dB |
    | 60 GHz  | 1.63 | 2.12 dB |
    | 140 GHz | 2.47 | 3.93 dB |

    #### 5.11.2 Matching-network ohmic loss
    An inductor of quality factor $Q$ has series resistance
    $r_L = \omega L / Q$. The NF penalty from a gate inductor $L_g$ is

    $$\Delta F_{L_g} \approx r_{L_g} / R_s.$$

    Since $Q$ is roughly frequency-independent at mmWave,
    $r_L \propto \omega$ and the loss penalty grows linearly with $f$.
    The source inductor $L_s$ contributes a similar term in the feedback path.

    #### 5.11.3 Passive interconnect and pad loss
    At 60 GHz and above, transmission-line feeds, bond wires, and pad
    capacitances add 0.2–0.5 dB of insertion loss before the transistor —
    equivalent to a noisy attenuator cascaded ahead of the LNA, raising NF
    by the same amount.

    The interactive below shows how these contributions stack from 10 to
    200 GHz for a representative 65 nm inductively-degenerated CS LNA.
    The total NF (green) uses the linear sum of input-referred excess noise:
    $F_{\mathrm{tot}} = F_{\min} + r_{L_s}/R_s + r_{L_g}/R_s + F_{\mathrm{intercon}} - 1$.
    """)
    return


@app.cell
def _(mo):
    fT_GHz_nb = mo.ui.slider(100, 400, step=10, value=200,
                              label="f_T (GHz)", show_value=True)
    Q_ind_nb  = mo.ui.slider(5, 25, step=1, value=12,
                              label="Inductor Q", show_value=True)
    gamma_nb  = mo.ui.slider(0.5, 2.0, step=0.1, value=1.4,
                              label="γ (channel noise coeff)", show_value=True)

    mo.vstack([
        mo.md("### 5.12 Interactive — NF budget breakdown vs frequency"),
        mo.hstack([fT_GHz_nb, Q_ind_nb, gamma_nb]),
    ])
    return fT_GHz_nb, Q_ind_nb, gamma_nb


@app.cell
def _(fT_GHz_nb, Q_ind_nb, gamma_nb, go, math, mo, np):
    _fT    = fT_GHz_nb.value * 1e9
    _Q_nb  = float(Q_ind_nb.value)
    _gamma_nb = gamma_nb.value
    _alpha_nb = 0.8
    _Ls_H_nb  = 40e-12
    _Lg_H_nb  = 260e-12
    _Rs_nb    = 50.0

    _freqs_nb = np.linspace(10e9, 200e9, 300)

    _Fmin_dev = 1.0 + (2.4 * _gamma_nb / _alpha_nb) * (_freqs_nb / _fT)
    _NF_dev   = 10.0 * np.log10(_Fmin_dev)

    _r_Ls_nb = 2 * math.pi * _freqs_nb * _Ls_H_nb / _Q_nb
    _r_Lg_nb = 2 * math.pi * _freqs_nb * _Lg_H_nb / _Q_nb
    _NF_Ls_nb = 10.0 * np.log10(1.0 + _r_Ls_nb / _Rs_nb)
    _NF_Lg_nb = 10.0 * np.log10(1.0 + _r_Lg_nb / _Rs_nb)

    _NF_intercon = 0.002 * (_freqs_nb / 1e9)
    _F_intercon  = 10**(_NF_intercon / 10)

    _F_total_nb  = _Fmin_dev + _r_Ls_nb / _Rs_nb + _r_Lg_nb / _Rs_nb + _F_intercon - 1.0
    _NF_total_nb = 10.0 * np.log10(np.maximum(_F_total_nb, 1.0))

    _fig_nb = go.Figure()
    _fig_nb.add_trace(go.Scatter(
        x=_freqs_nb / 1e9, y=_NF_dev, mode="lines",
        name="Device F_min", line=dict(color="#3366CC", width=2)))
    _fig_nb.add_trace(go.Scatter(
        x=_freqs_nb / 1e9, y=_NF_Ls_nb, mode="lines",
        name="L_s loss (finite Q)", line=dict(color="#CC6633", dash="dash")))
    _fig_nb.add_trace(go.Scatter(
        x=_freqs_nb / 1e9, y=_NF_Lg_nb, mode="lines",
        name="L_g loss (finite Q)", line=dict(color="#CC9933", dash="dot")))
    _fig_nb.add_trace(go.Scatter(
        x=_freqs_nb / 1e9, y=_NF_intercon, mode="lines",
        name="Interconnect / pad", line=dict(color="#888888")))
    _fig_nb.add_trace(go.Scatter(
        x=_freqs_nb / 1e9, y=_NF_total_nb, mode="lines",
        name="Total NF (approx)", line=dict(color="#33AA66", width=2.5)))
    for _fv, _lbl in [(28, "28 GHz"), (60, "60 GHz"), (140, "140 GHz")]:
        _fig_nb.add_vline(x=_fv, line_dash="dash", line_color="#555",
                          annotation_text=_lbl, annotation_position="top left")
    _fig_nb.update_layout(
        template="plotly_dark",
        xaxis_title="Frequency (GHz)",
        yaxis_title="NF contribution (dB)",
        height=420,
        title="NF budget vs frequency — 65 nm CMOS inductively-degenerated CS LNA",
    )

    mo.ui.plotly(_fig_nb)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Advanced Topics (Appendix)

### 6.1 Equipartition of noise into amplitude and phase quadratures

    #### Quadrature decomposition

    Every real bandpass process $n(t)$ with spectral support near $\pm f_0$
    can be written uniquely as

    $$n(t) = n_I(t)\cos(2\pi f_0 t) - n_Q(t)\sin(2\pi f_0 t),$$

    where $n_I$ and $n_Q$ are real **lowpass** (baseband) processes obtained by
    synchronous demodulation and low-pass filtering:

    $$n_I(t) = \mathrm{LPF}\!\left\{2\,n(t)\cos(2\pi f_0 t)\right\}, \qquad n_Q(t) = \mathrm{LPF}\!\left\{-2\,n(t)\sin(2\pi f_0 t)\right\}.$$

    #### Derivation of the quadrature PSDs

    Let the two-sided PSD of $n(t)$ be

    $$S_n(f) = \frac{N_0}{2}, \quad |f \pm f_0| < \frac{B}{2}, \qquad \text{zero elsewhere.}$$

    This represents bandpass white noise with **one-sided PSD** $N_0$
    (W/Hz, as measured by a spectrum analyser) in a bandwidth $B$ around $f_0$.
    The total bandpass power is

    $$\langle n^2\rangle = \int_{-\infty}^{\infty} S_n(f)\,df = \frac{N_0}{2}\cdot B + \frac{N_0}{2}\cdot B = N_0 B.$$

    Applying the standard frequency-shifting identity for quadrature components,
    the PSD of the lowpass envelope is

    $$S_{n_I}(f) = S_{n_Q}(f) = \bigl[S_n(f - f_0) + S_n(f + f_0)\bigr]_{|f|<B/2}.$$

    For $|f| < B/2$: $f - f_0$ sits inside the negative-frequency band
    ($|(f-f_0)+f_0|=|f|<B/2$) and $f + f_0$ sits inside the positive-frequency
    band ($|(f+f_0)-f_0|=|f|<B/2$), so both terms equal $N_0/2$:

    $$S_{n_I}(f) = \frac{N_0}{2} + \frac{N_0}{2} = N_0, \quad |f| < \frac{B}{2}.$$

    Integrating gives the quadrature variance:

    $$\langle n_I^2\rangle = \int_{-B/2}^{B/2} N_0\,df = N_0 B, \qquad \langle n_Q^2\rangle = N_0 B.$$

    The cross-PSD evaluates to $S_{n_I n_Q}(f) = j[S_n(f+f_0) - S_n(f-f_0)] = 0$
    for this symmetric bandpass spectrum, so

    $$\langle n_I n_Q\rangle = 0 \quad\text{(uncorrelated, and for Gaussian noise: independent).}$$

    **Power accounting:** with $n = n_I\cos - n_Q\sin$ and the $\cos/\sin$
    time-averaging factor of $\tfrac{1}{2}$:

    $$\langle n^2\rangle = \tfrac{1}{2}\langle n_I^2\rangle + \tfrac{1}{2}\langle n_Q^2\rangle = \tfrac{1}{2} N_0 B + \tfrac{1}{2} N_0 B = N_0 B \;\checkmark$$

    The equal variance $\langle n_I^2\rangle = \langle n_Q^2\rangle = N_0 B$
    — each quadrature carries the same noise power — is the statement of
    **equipartition between quadratures**.

    **Why "classical"?** At microwave frequencies $f_0 \sim \text{GHz}$, the
    quantum occupation number $\bar{n} = (e^{hf_0/kT}-1)^{-1} \approx kT/hf_0
    \gg 1$ (at room temperature $\bar{n} \approx 6000$), so the quantum
    zero-point term $\hbar\omega/2$ is negligible and the classical Nyquist
    result $N_0 = kT$ holds. "Classical equipartition" simply flags that we are
    in this regime; the quantum correction matters only above ~60 GHz at
    cryogenic temperatures.

    #### Amplitude and phase noise from the phasor picture

    Place the signal phasor at $Ae^{j0}$ (along the I-axis by choice of
    reference). Additive noise $n_I + jn_Q$ perturbs the tip:

    $$A + n_I + jn_Q \approx (A + n_I)\,e^{\,j\,n_Q/A} \quad\text{for } |n_Q| \ll A.$$

    Reading off the polar perturbation:

    $$\delta A = n_I \qquad (\text{radial, amplitude noise}),$$

    $$\delta\phi = \arctan\!\frac{n_Q}{A} \approx \frac{n_Q}{A} \qquad (\text{tangential, phase noise}).$$

    The injected noise power is the same in both quadratures ($N_0B$);
    the phase noise is *suppressed by the signal amplitude*:

    $$\langle\delta\phi^2\rangle = \frac{\langle n_Q^2\rangle}{A^2} = \frac{N_0 B}{A^2} = \frac{N_0 B}{2 P_\text{sig}}.$$

    No noise is removed; the geometry of the phasor plane converts equal noise
    in both directions into unequal *angular* and *radial* fluctuations.

    **Oscillator consequence.** A sustained oscillator has a nonlinear restoring
    force in amplitude (amplitude is clamped to the limit cycle) but no restoring
    force in phase. Equal noise power $N_0 B$ is injected into both quadratures;
    $\delta A$ is damped, while $\delta\phi$ accumulates as a random walk. The
    one-sided phase-noise PSD is

    $$\mathcal{L}(f_m) \approx \frac{kTF}{2 P_{\mathrm{sig}}} \cdot \frac{f_0^2}{Q_L^2\,f_m^2}$$

    (Leeson's formula). The $1/P_\text{sig}$ factor is exactly the $1/A^2$
    scaling derived above. The $1/f_m^2$ roll-off is the power spectrum of a
    phase random walk.

    **Practical consequence.** Any noise reduction scheme must contend with
    equipartition: redistributing noise from one quadrature to the other is
    possible (§19.6), but the product
    $\langle n_I^2\rangle\langle n_Q^2\rangle \ge (N_0 B)^2$
    cannot be reduced by a linear passive network (Bosma's theorem, §7.2).
    """)
    return


@app.cell
def _(go, mo, np):
    # FIG 4 — Phasor decomposition: equal I/Q noise → unequal amplitude/phase noise
    _A = 1.5
    _sig_n = 0.28
    _th = np.linspace(0.0, 2.0 * np.pi, 200)

    _fig = go.Figure()

    # ─── Grid and axes ───────────────────────────────────────────────────────
    _fig.add_hline(y=0, line_dash="solid", line_color="#444444", line_width=0.5)
    _fig.add_vline(x=0, line_dash="solid", line_color="#444444", line_width=0.5)

    # ─── Signal phasor (dominant, along I-axis) ───────────────────────────────
    _fig.add_trace(go.Scatter(
        x=[0, _A], y=[0, 0],
        mode="lines", name="Signal: A = 1.5",
        line=dict(color="white", width=3),
        showlegend=True))

    # Large dot at phasor tip
    _fig.add_trace(go.Scatter(
        x=[_A], y=[0],
        mode="markers",
        marker=dict(size=12, color="white"),
        showlegend=False))

    # ─── Noise cloud: equal variance in I and Q ──────────────────────────────
    # 1σ circle centered at phasor tip
    _fig.add_trace(go.Scatter(
        x=_A + _sig_n * np.cos(_th),
        y=_sig_n * np.sin(_th),
        mode="lines",
        name="1σ noise: ⟨n_I²⟩ = ⟨n_Q²⟩ = N₀B",
        line=dict(color="#BBBBBB", width=2, dash="solid"),
        showlegend=True))

    # ─── Amplitude noise: δA = n_I (radial perturbation) ──────────────────────
    # Horizontal double-headed arrow at phasor tip
    _fig.add_annotation(
        x=_A + _sig_n, y=0.0,
        ax=_A - _sig_n, ay=0.0,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowwidth=2.5, arrowcolor="#5599DD",
        text="", showarrow=True)
    _fig.add_annotation(
        text="<b>Amplitude</b><br>δA = n_I<br>(radial)",
        x=_A, y=-0.42,
        xref="x", yref="y",
        font=dict(color="#5599DD", size=13, family="monospace"),
        showarrow=False, bgcolor="#1a1a1a", bordercolor="#5599DD", borderwidth=1.5,
        borderpad=6)

    # ─── Phase noise: δφ ≈ n_Q/A (angular perturbation) ────────────────────
    # Show a sample angular displacement
    _sample_angle = np.arctan2(_sig_n * 0.9, _A)
    _fig.add_annotation(
        x=_A * np.cos(_sample_angle),
        y=_A * np.sin(_sample_angle),
        ax=_A, ay=0.0,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowwidth=2.5, arrowcolor="#EE8833",
        text="", showarrow=True)

    # Angular arc
    _arc_r = 0.35
    _arc_th = np.linspace(0, _sample_angle, 40)
    _fig.add_trace(go.Scatter(
        x=_arc_r * np.cos(_arc_th),
        y=_arc_r * np.sin(_arc_th),
        mode="lines",
        line=dict(color="#EE8833", width=2),
        showlegend=False))

    _fig.add_annotation(
        text="<b>Phase</b><br>δφ = n_Q/A<br>(angular)",
        x=0.65, y=0.25,
        xref="x", yref="y",
        font=dict(color="#EE8833", size=13, family="monospace"),
        showarrow=False, bgcolor="#1a1a1a", bordercolor="#EE8833", borderwidth=1.5,
        borderpad=6)

    # ─── Key insight ───────────────────────────────────────────────────────
    _fig.add_annotation(
        text="<b>Phasor geometry:</b> Circular noise cloud (equal I/Q power)<br>"
             "becomes unequal radial (A) and angular (φ) fluctuations",
        x=0.5, y=1.55,
        xref="paper", yref="y",
        font=dict(color="#33CC88", size=11),
        showarrow=False,
        bgcolor="#1a2a1a", bordercolor="#33CC88", borderwidth=1,
        borderpad=6)

    _fig.update_layout(
        template="plotly_dark",
        width=700, height=550,
        xaxis=dict(
            title="I-axis", scaleanchor="y", scaleratio=1,
            range=[-0.3, 2.2]),
        yaxis=dict(
            title="Q-axis",
            range=[-1.0, 1.75]),
        margin=dict(l=60, r=20, t=80, b=60),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.7)", bordercolor="#666", borderwidth=1),
        hovermode=False)

    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.2 Noise squeezing — redistribution between quadratures

    Equipartition (§6.2) distributes thermal noise equally between the I and Q
    quadratures: $\langle n_I^2\rangle = \langle n_Q^2\rangle = kTB$. The
    question of noise squeezing is whether a circuit can create an asymmetric
    split — reducing noise in one quadrature below $kTB$ at the cost of
    amplifying the other — analogous to squeezed light in quantum optics.

    #### 6.2.1 Classical squeezing via phase-sensitive amplification

    A parametric amplifier pumped at $2\omega_0$ (degenerate operation) acts
    phase-sensitively: it amplifies the I-quadrature by $\sqrt{G}$ and
    attenuates the Q-quadrature by $1/\sqrt{G}$. For thermal input noise:

    $$\langle n_I^2\rangle \to G\,kTB, \qquad \langle n_Q^2\rangle \to \frac{kTB}{G}.$$

    The product $\langle n_I^2\rangle\langle n_Q^2\rangle = (kTB)^2$ is
    preserved; total noise power is unchanged. This is **classical noise
    squeezing** — redistribution without destruction.

    At microwave frequencies this is implemented in Josephson Travelling-Wave
    Parametric Amplifiers (JTWPAs), which achieve $\sim 3$–5 dB of classical
    squeezing while adding near-quantum-limited noise to the amplified quadrature.

    **Bosma's constraint (§7.2).** For any passive network,
    $\mathbf{C}_b \succeq 0$ and the total noise power (trace of $\mathbf{C}_b$)
    cannot decrease. Squeezing requires an active (pumped) element: the pump
    supplies energy that redistributes quantum fluctuations between quadratures.

    #### 6.2.2 Quantum noise squeezing

    The quantum floor per quadrature is set by zero-point fluctuations:
    $\langle n_k^2\rangle_\text{vac} = \hbar\omega/2$. Below the **standard
    quantum limit (SQL)**, squeezing one quadrature below $\hbar\omega/2$
    violates no physical law, provided the conjugate quadrature satisfies the
    Heisenberg-Robertson uncertainty relation:

    $$\Delta n_I \cdot \Delta n_Q \ge \frac{\hbar\omega}{2}.$$

    Reaching the quantum regime at microwave frequencies requires
    $kT \lesssim \hbar\omega$, i.e.\ $T \lesssim T^* = \hbar\omega/k$.
    At $\omega/2\pi = 10$ GHz: $T^* \approx 0.5$ K. The thermal photon
    occupation number is

    $$\bar{n} = \frac{kT}{\hbar\omega} \approx \frac{4.0\times10^{-21}\,\mathrm{J}}{6.6\times10^{-24}\,\mathrm{J}} \approx 600 \quad \text{at } T = 290\text{ K},$$

    so room-temperature microwave fields are deeply classical. At dilution-fridge
    temperatures ($T \lesssim 20$ mK, $\bar{n} \lesssim 0.04$), quantum
    squeezing below the SQL has been demonstrated with Josephson Parametric
    Amplifiers (JPAs): $\sim 10$ dB of squeezing in one quadrature.

    #### 6.2.3 Comparison with optical squeezing

    In optics ($\lambda = 1\,\mu$m), $h\nu/kT \approx 50$ at 300 K — the
    field is near vacuum even without cooling. Optical parametric oscillators
    (OPOs) and four-wave mixing achieve quantum squeezing from the outset.

    | Domain | $kT/\hbar\omega$ at 300 K | Mechanism | Demonstrated squeezing |
    |---|---|---|---|
    | Optical ($\lambda = 1\,\mu$m) | $\approx 0.02$ | OPO, FWM | 15 dB (quantum) |
    | Microwave (10 GHz, 300 K) | $\approx 600$ | JTWPA (parametric) | 3–5 dB (classical) |
    | Microwave (10 GHz, 20 mK) | $\approx 0.04$ | JPA | $\sim 10$ dB (quantum) |

    **Implication for receiver design.** At room temperature the LNA noise floor
    is set by $kT$, not by quantum fluctuations. No passive or linear active
    circuit can beat this floor — only physical cooling of the front end can
    approach the quantum limit. This is why radio-telescope front ends use
    cryogenic LNAs ($T_e \approx 5$–20 K) and superconducting-qubit readout
    chains operate at dilution-fridge temperatures.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.3 Microscopic Nyquist — Landauer-Büttiker derivation

    Setup: a conductor connecting two reservoirs (left $L$, right $R$)
    at temperature $T$ and chemical potentials $\mu_L = \mu_R = \mu$
    (equilibrium). The conductor supports $M$ transverse modes with
    energy-dependent transmission probabilities
    $\{\mathcal{T}_n(\varepsilon)\}_{n=1}^M$.

    **Current per channel.** Each spin-degenerate channel carries a
    forward current $e\,v(\varepsilon)\,g(\varepsilon)\,n_L\,\mathcal{T}_n$
    and a backward $e\,v(\varepsilon)\,g(\varepsilon)\,n_R\,\mathcal{T}_n$,
    where $g(\varepsilon) = 1/(h\,v(\varepsilon))$ is the 1D density of
    states (one direction). The velocity-DoS product $v\cdot g = 1/h$ is
    the celebrated cancellation that makes 1D ballistic conductance
    universal. The net current is

    $$I = \frac{2e}{h}\sum_n \int \mathcal{T}_n(\varepsilon) \bigl[n_L(\varepsilon) - n_R(\varepsilon)\bigr]\,d\varepsilon,$$

    with $n_{L,R}(\varepsilon) = f(\varepsilon; \mu_{L,R}, T)$ Fermi-Dirac
    (the factor of 2 is spin degeneracy).

    **Linear-response conductance.** Expanding to first order in
    $V = (\mu_L - \mu_R)/e$ and using $-\partial f/\partial\varepsilon
    \to \delta(\varepsilon - \mu)$ at $T = 0$,

    $$G = \frac{dI}{dV} = \frac{2e^2}{h}\sum_n \mathcal{T}_n(\mu) \quad \text{[Landauer formula, theorem]}.$$

    **Equilibrium current noise.** Each channel is a two-terminal
    scattering problem with binomial partition statistics for each
    energy slice. The general (Lesovik-Levitov) result for the current
    noise PSD, with the same spin doubling as the current operator and
    in the one-sided PSD convention, is

    $$S_I = \frac{4e^2}{h}\sum_n \int \Bigl\{\mathcal{T}_n\bigl[n_L(1-n_L) + n_R(1-n_R)\bigr] + \mathcal{T}_n(1-\mathcal{T}_n)(n_L - n_R)^2\Bigr\}d\varepsilon.$$

    At equilibrium $n_L = n_R = f$, the second (quantum-partition) term
    vanishes and $n_L(1-n_L) + n_R(1-n_R) = 2f(1-f)$, so

    $$S_I^{(\text{eq})} = \frac{8e^2}{h}\sum_n \mathcal{T}_n \int f(\varepsilon)\bigl(1 - f(\varepsilon)\bigr)\,d\varepsilon.$$

    **Evaluation.** Using the universal Fermi-edge integral from §2.3,
    $\int f(1-f)\,d\varepsilon = kT$, and the Landauer conductance
    $G = (2e^2/h)\sum_n \mathcal{T}_n$,

    $$\boxed{S_I^{(\text{eq})} = \frac{8e^2}{h}\sum_n \mathcal{T}_n \cdot kT = 4kT\,G} \quad \text{[microscopic Nyquist, theorem]}.$$

    Inverting to a voltage source: $S_V = S_I/G^2 = 4kT/G = 4kTR$.
    **The Nyquist formula is recovered from purely microscopic
    partition statistics.**

    The non-equilibrium ($V \ne 0$) contribution
    $\propto \mathcal{T}_n(1-\mathcal{T}_n)$ — absent at $V = 0$ —
    is the *quantum partition noise* that distinguishes ballistic
    conductors from classical wires. For perfect transmission
    $\mathcal{T} = 1$ it vanishes (a fully open channel has zero
    partition noise); for tunnelling $\mathcal{T} \ll 1$ it reduces to
    the full Poisson shot noise $S_I = 2eI$ of §2.5. This single
    framework therefore *contains* both Nyquist (equilibrium) and
    Schottky (non-equilibrium tunnelling) as limits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.4 The diode noise crossover — algebra of the coth form

    The result $S_I = 2qI\coth(qV/2kT)$ from §2.5 has two equivalent
    expressions; the algebra below verifies their identity.

    **Form 1 — sum of forward and reverse Poisson streams.**
    Detailed balance at the junction:

    $$I_+ = I_s e^{qV/kT}, \quad I_- = I_s,$$

    $$I = I_+ - I_- = I_s\bigl(e^{qV/kT}-1\bigr).$$

    Each stream is independently Poisson at rate $I_\pm/q$. The streams
    are uncorrelated (independent barriers), so noise variances add:

    $$S_I = 2qI_+ + 2qI_- = 2q I_s\bigl(e^{qV/kT} + 1\bigr).$$

    **Form 2 — coth.** Define $x = qV/2kT$. Then

    $$e^{qV/kT} + 1 = e^{2x} + 1 = e^x\bigl(e^x + e^{-x}\bigr),$$

    $$e^{qV/kT} - 1 = e^{2x} - 1 = e^x\bigl(e^x - e^{-x}\bigr),$$

    giving $(e^{qV/kT}+1)/(e^{qV/kT}-1)
    = (e^x + e^{-x})/(e^x - e^{-x}) = \coth(x)$. Hence

    $$S_I = 2q I_s\bigl(e^{qV/kT}-1\bigr) \cdot \frac{e^{qV/kT}+1}{e^{qV/kT}-1} = 2qI\,\coth\!\left(\frac{qV}{2kT}\right). \;\;\square$$

    **Limits — algebra check.**

    *Zero bias.* For $x \to 0$, $\coth(x) \to 1/x = 2kT/(qV)$, so

    $$S_I \to 2qI \cdot \frac{2kT}{qV} = 4kT\,\frac{I}{V} = 4kT\,g_d,$$

    where $g_d \equiv I/V$ is the chord conductance. At $V \to 0$ the
    chord conductance equals the differential conductance
    $g_d = dI/dV\big|_{V=0} = qI_s/kT$, recovering Johnson noise of the
    small-signal conductance.

    *High bias.* For $x \gg 1$, $\coth(x) \to 1$, $S_I \to 2qI$ —
    pure Schottky shot. The crossover scale is $x \sim 1$, i.e.\
    $V \sim 2kT/q \approx 50$ mV at room temperature.

    **Universality.** The same coth form arises for any Poisson barrier
    obeying detailed balance — tunnel junctions, ballistic point
    contacts, single-electron transistors in the orthodox regime — and
    it generalises to the Landauer framework of §6.3 with
    $\mathcal{T}(1-\mathcal{T})$ replacing the binomial factor. It is
    the simplest closed-form expression of the principle that thermal
    noise and shot noise are limits of the same underlying barrier-
    crossing statistics, made quantitative in one line.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Summary and bridge to notebook 05

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
    **Next:** [05 — Matching Networks](05_matching_networks.py)
    """)
    return


if __name__ == "__main__":
    app.run()
