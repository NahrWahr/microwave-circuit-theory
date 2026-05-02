# v1.0
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
def _(np):
    # v1.0
    """Pure-Python helpers for oscillator phase noise analysis.

    Used by notebook 06 (Oscillators, VCOs, mmWave Phase Noise).
    Conventions: Hajimiri & Lee, *The Design of Low Noise Oscillators* (1999);
    Razavi, *RF Microelectronics* 2nd ed., Chapter 8;
    Demir, Mehrotra & Roychowdhury, IEEE TCAS-I 47(5), 2000.
    All frequencies in Hz unless suffixed; all angular frequencies in rad/s.
    """

    KB = 1.380649e-23   # Boltzmann (J/K)
    T0 = 290.0          # Reference temperature (K)
    KT0 = KB * T0       # 4.0e-21 J

    # ---------------------------------------------------------------------------
    # Leeson model
    # ---------------------------------------------------------------------------

    def leeson_pn(df, F, P_sig, Q_L, f0, df_corner_1f3=0.0, T=T0):
        """Leeson phase noise spectrum L(df) in dBc/Hz.

        df : offset frequency from carrier (Hz), array_like
        F  : empirical noise factor (linear, ≥1)
        P_sig : signal power at tank input (W)
        Q_L : loaded tank quality factor
        f0  : carrier frequency (Hz)
        df_corner_1f3 : 1/f^3 corner frequency (Hz); 0 disables flicker term
        """
        df = np.asarray(df, dtype=float)
        kT = KB * T
        # 1/f^2 region with Lorentzian roll-off at the half-bandwidth
        half_bw = f0 / (2.0 * Q_L)
        S_white = (2.0 * F * kT / P_sig) * (1.0 + (half_bw / df) ** 2)
        if df_corner_1f3 > 0.0:
            S = S_white * (1.0 + df_corner_1f3 / np.maximum(df, 1e-30))
        else:
            S = S_white
        # L = 10 log10(S/2) for SSB phase noise; absorb the 1/2 into prefactor
        L_db = 10.0 * np.log10(0.5 * S)
        return L_db


    def fom_oscillator(L_db, df, f0, P_dc_mw):
        """Standard oscillator FOM (dB).

        FOM = -L(df) + 20 log10(f0/df) - 10 log10(P_dc_mw / 1mW)
        Higher (less negative) is better.
        """
        return -L_db + 20.0 * np.log10(f0 / df) - 10.0 * np.log10(P_dc_mw)


    # ---------------------------------------------------------------------------
    # Van der Pol simulator (limit-cycle interactive)
    # ---------------------------------------------------------------------------

    def vdp_rhs(state, eps, omega0):
        x, v = state
        return np.array([v, eps * (1.0 - x * x) * v - omega0 * omega0 * x])


    def vdp_simulate(eps, omega0, x0, v0, t_end, dt, kick_time=None,
                    kick_dx=0.0, kick_dv=0.0):
        """RK4 simulation of a Van der Pol oscillator with an optional kick.

        Returns t, x, v arrays.
        """
        n = int(np.round(t_end / dt))
        t = np.linspace(0.0, t_end, n + 1)
        x = np.zeros(n + 1)
        v = np.zeros(n + 1)
        x[0], v[0] = x0, v0
        kicked = False
        for i in range(n):
            if (kick_time is not None) and (not kicked) and (t[i] >= kick_time):
                x[i] += kick_dx
                v[i] += kick_dv
                kicked = True
            s = np.array([x[i], v[i]])
            k1 = vdp_rhs(s, eps, omega0)
            k2 = vdp_rhs(s + 0.5 * dt * k1, eps, omega0)
            k3 = vdp_rhs(s + 0.5 * dt * k2, eps, omega0)
            k4 = vdp_rhs(s + dt * k3, eps, omega0)
            s_new = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            x[i + 1], v[i + 1] = s_new[0], s_new[1]
        return t, x, v


    def vdp_amplitude_phase(t, x, v, omega0):
        """Decompose VdP trajectory into instantaneous amplitude and phase
        relative to the unperturbed steady-state amplitude A_ss = 2.

        For VdP the limit-cycle radius (in (x, v/omega0) coordinates) is 2.
        """
        r = np.sqrt(x * x + (v / omega0) ** 2)
        # Phase relative to a uniformly rotating reference at omega0
        theta = np.arctan2(v / omega0, x)
        phi = np.unwrap(theta) + omega0 * t   # because clockwise convention
        return r, phi


    # ---------------------------------------------------------------------------
    # ISF and phase noise (Hajimiri-Lee)
    # ---------------------------------------------------------------------------

    def isf_from_harmonics(c0, c_array, theta_array, phase):
        """Reconstruct Γ(ω0 t) from a Fourier series.

        c0      : DC coefficient of Γ
        c_array : harmonic magnitudes c_1, c_2, …
        theta_array : harmonic phases θ_1, θ_2, …
        phase   : ω0 t array (rad), array_like

        Returns Γ(phase).
        """
        phase = np.asarray(phase, dtype=float)
        out = (c0 / 2.0) * np.ones_like(phase)
        for n_idx, (c_n, theta_n) in enumerate(zip(c_array, theta_array), start=1):
            out += c_n * np.cos(n_idx * phase + theta_n)
        return out


    def isf_rms_squared(c0, c_array):
        """Γ²_rms from Fourier coefficients (Parseval)."""
        return 0.25 * c0 * c0 + 0.5 * float(np.sum(np.asarray(c_array) ** 2))


    def pn_from_isf(df, c0, c_array, q_max, in_white, in_flicker_corner,
                    f0, df_corner_1f3=None):
        """Phase noise spectrum L(df) (dBc/Hz) from ISF and a single noise source.

        df : offset frequency (Hz), array_like
        c0, c_array : ISF Fourier coefficients
        q_max  : max charge displacement on tank cap (C)
        in_white : white noise current PSD (A²/Hz)
        in_flicker_corner : device 1/f corner frequency (Hz)
        f0 : carrier frequency (Hz)
        df_corner_1f3 : optional override for the 1/f^3 corner
        """
        df = np.asarray(df, dtype=float)
        gamma2_rms = isf_rms_squared(c0, c_array)
        omega = 2.0 * np.pi * df
        # 1/f^2 base from white noise
        S_phi_white = in_white * gamma2_rms / (2.0 * q_max * q_max * omega ** 2)
        # 1/f^3 from c0 upconverting flicker noise
        if df_corner_1f3 is None:
            # Hajimiri & Lee: f_{1/f^3} = f_{1/f, device} · (c0² / 4Γ²_rms)
            ratio = (0.25 * c0 * c0) / max(gamma2_rms, 1e-30)
            df_corner_1f3 = in_flicker_corner * ratio
        S_phi = S_phi_white * (1.0 + df_corner_1f3 / np.maximum(df, 1e-30))
        return 10.0 * np.log10(0.5 * S_phi)


    # ---------------------------------------------------------------------------
    # Tank Q models (mmWave)
    # ---------------------------------------------------------------------------

    def inductor_Q(f, L_H, sigma_S_per_m, t_ox_um, rho_sub_ohm_cm,
                  C_par_F, geom_k=1.0):
        """Lumped Q model for an on-chip inductor.

        Captures three loss mechanisms:
        (1) Series resistance R_s ∝ √f from skin effect.
        (2) Substrate eddy current loss, scaling as f² / ρ_sub.
        (3) Self-resonant collapse from parasitic shunt C_par.

        Parameters control: metal conductivity σ, oxide thickness t_ox (μm),
        substrate resistivity ρ_sub (Ω·cm). geom_k is a geometric form factor.
        Returns Q_L(f).
        """
        f = np.asarray(f, dtype=float)
        omega = 2.0 * np.pi * f
        mu0 = 4.0 * np.pi * 1e-7
        # Series resistance with skin effect (resistance grows as √f).
        # Reference resistance set by σ and a fixed geometric path.
        delta = np.sqrt(2.0 / (omega * mu0 * sigma_S_per_m))   # skin depth (m)
        R_s_ref = geom_k * 1.0e-3 / (sigma_S_per_m * 1e-9)     # ~Ω at low f
        R_s = R_s_ref * np.sqrt(f / 1e9)
        # Substrate eddy loss: induced image currents below the inductor.
        rho_sub_si = rho_sub_ohm_cm * 1e-2                     # Ω·m
        # Coupling to substrate weakens with thicker oxide; cube-law fit.
        R_sub = (omega * omega) * L_H * L_H * (t_ox_um * 1e-6) ** 3 / \
                (rho_sub_si * 1e-12)
        R_sub = np.maximum(R_sub, 1e-9)
        # Effective inductance reduction near SRF
        f_srf = 1.0 / (2.0 * np.pi * np.sqrt(L_H * C_par_F))
        attn = 1.0 - (f / f_srf) ** 2
        attn = np.where(attn > 0.02, attn, 0.02)               # clamp to avoid div0
        L_eff = L_H * attn
        # Total tank-referred series resistance
        R_total = R_s + R_sub
        Q = omega * L_eff / R_total
        Q = np.where(f < 0.95 * f_srf, Q, Q * (1.0 - f / f_srf) ** 2)
        return np.maximum(Q, 0.1)


    def varactor_Q(f, C_F, R_var_ohm):
        """Q of a series-R varactor: Q = 1/(ω C R_var).
        R_var grows weakly with frequency at mmWave; modelled as f^0.5.
        """
        f = np.asarray(f, dtype=float)
        omega = 2.0 * np.pi * f
        R_eff = R_var_ohm * np.sqrt(f / 1e9)
        return 1.0 / (omega * C_F * R_eff)


    def tank_Q(Q_L_arr, Q_C_arr):
        return 1.0 / (1.0 / Q_L_arr + 1.0 / Q_C_arr)


    # ---------------------------------------------------------------------------
    # Floquet helpers (numerical demonstration only)
    # ---------------------------------------------------------------------------

    def vdp_monodromy(eps, omega0, T_period=None, n_steps=2000):
        """Compute the monodromy matrix M = Φ(T,0) for the VdP limit cycle.

        Integrates two perturbed trajectories around one period, starting from
        the steady-state limit cycle, and reads off the linearised return map.
        Returns (M, eigvals).
        """
        # Run long enough to settle on the limit cycle
        t_settle = 80.0 / max(eps, 0.05)
        dt = 0.01
        _t, _x, _v = vdp_simulate(eps, omega0, 2.0, 0.0, t_settle, dt)
        x0_lc, v0_lc = _x[-1], _v[-1]

        # Estimate period from zero-crossings of x near steady state
        if T_period is None:
            T_period = 2.0 * np.pi / omega0   # leading-order approximation

        eps_pert = 1e-4
        cols = []
        for dx, dv in [(eps_pert, 0.0), (0.0, eps_pert)]:
            _, x_p, v_p = vdp_simulate(eps, omega0, x0_lc + dx, v0_lc + dv,
                                       T_period, T_period / n_steps)
            _, x_0, v_0 = vdp_simulate(eps, omega0, x0_lc, v0_lc,
                                       T_period, T_period / n_steps)
            dxT = x_p[-1] - x_0[-1]
            dvT = v_p[-1] - v_0[-1]
            cols.append([dxT / eps_pert, dvT / eps_pert])
        M = np.column_stack(cols)
        return M, np.linalg.eigvals(M)


    return (
        KB, KT0, T0,
        fom_oscillator,
        inductor_Q,
        isf_from_harmonics, isf_rms_squared,
        leeson_pn,
        pn_from_isf,
        tank_Q,
        varactor_Q,
        vdp_amplitude_phase,
        vdp_monodromy,
        vdp_simulate,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 06 — Oscillators, VCOs, and mmWave Phase Noise

    Once an LNA boosts a weak antenna signal and a matching network couples
    it cleanly to the next stage, the receive chain still needs a *local
    oscillator* to translate the RF signal down to baseband. The cleanliness
    of that local oscillator — its **phase noise** — sets the receiver's
    in-band signal-to-noise floor and limits how closely two channels can be
    spaced in frequency-division multiplexing.

    This notebook builds the theory of oscillator phase noise from three
    progressively sharper viewpoints:

    1. **Leeson's linear closed-loop model** — the historical baseline; gives
       the spectral *shape* but treats noise factor, loaded $Q$, and 1/f
       upconversion as empirical fit parameters.
    2. **Floquet / ISF nonlinear theory** — derives those parameters from
       first principles by linearising about the periodic orbit and
       projecting noise onto the unit Floquet eigenvector.
    3. **mmWave application** — applies the ISF picture to tank-$Q$
       degradation, topology selection (cross-coupled vs. Colpitts), and
       coupled-array architectures used at 28-60 GHz.

    The thread connecting all three: the noise on the oscillator output is
    not just the input noise filtered by a transfer function, but the
    response of a nonlinear periodic orbit to perturbations along its
    tangent direction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part I — Linear phase-noise theory
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Oscillator fundamentals

    An oscillator is a circuit whose linearised small-signal poles sit
    exactly on the $j\omega$ axis. Two equivalent perspectives describe the
    condition.

    **Barkhausen criterion (loop view).** Break the loop, inject a test
    signal, and require that round-trip gain is unity in magnitude and zero
    in phase at the oscillation frequency $\omega_0$:

    $$
    \boxed{\;
    A(j\omega_0)\,\beta(j\omega_0) \;=\; 1
    \;}
    $$

    Magnitude $|A\beta| = 1$ and phase $\arg(A\beta) = 0\pmod{2\pi}$.

    **Negative-resistance criterion (impedance view).** The active device
    presents a small-signal resistance $-R_{\text{dev}}$ in parallel with
    the tank's loss $R_p$. Steady-state oscillation requires the two to
    cancel:

    $$
    \boxed{\;
    -R_{\text{dev}} + R_p \;=\; 0
    \;}
    $$

    The tank's parallel resonance at $\omega_0 = 1/\sqrt{LC}$ then satisfies
    Kirchhoff's laws with no external excitation.

    **Why a real oscillator must be nonlinear.** A purely linear circuit
    with poles on $j\omega$ produces undamped oscillation whose amplitude
    is set by the initial condition — any noise on startup grows or
    persists with no preferred amplitude. Real oscillators stabilise
    amplitude either through *gain compression* (transconductance falls as
    the signal swing grows) or an *explicit limiter*. Both are nonlinear,
    and both are essential to phase-noise analysis: the same nonlinearity
    that fixes the amplitude is the mechanism that suppresses amplitude
    noise while leaving phase noise unchecked.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Leeson's model

    Leeson (1966) treats the oscillator as a closed loop comprising a tank
    (loaded quality factor $Q_L$) and an amplifier (noise factor $F$,
    delivering signal power $P_{\text{sig}}$ at the tank input). The
    derivation has three steps.

    **Step 1 — Tank as bandpass filter.** Near $\omega_0$, the loaded tank
    presents a one-pole bandpass response of half-bandwidth
    $\Delta\omega_{1/2} = \omega_0 / (2 Q_L)$. White noise at the amplifier
    output is filtered to the same bandwidth; the input-referred noise PSD
    of the active device is

    $$
    S_{n,\text{in}} \;=\; 2 F k T,
    $$

    where the factor of two accounts for both sidebands.

    **Step 2 — Closed-loop transfer.** Around resonance the loop transfer
    from input noise to output phase is

    $$
    H_\phi(j\Delta\omega) \;=\; \frac{1}{2\,j\,Q_L\,\Delta\omega/\omega_0}
    \;\Rightarrow\; |H_\phi|^2 \;=\;
    \frac{1}{4 Q_L^2}\left(\frac{\omega_0}{\Delta\omega}\right)^2.
    $$

    **Step 3 — Output spectrum.** The single-sideband phase noise relative
    to the carrier is then

    $$
    \boxed{\;
    \mathcal{L}(\Delta\omega)
    \;=\; 10\log_{10}\!\left[
        \frac{2 F k T}{P_{\text{sig}}}
        \left(1 + \frac{\omega_0^{\,2}}{4 Q_L^2 \Delta\omega^2}\right)
        \left(1 + \frac{\Delta\omega_{1/f^3}}{|\Delta\omega|}\right)
    \right]
    \;}
    $$

    The third factor (an empirical patch) raises the noise close to the
    carrier to capture device 1/f upconversion.

    **Three spectral regions.** The bracketed expression separates
    naturally:

    | Region | Slope | Mechanism |
    |--------|-------|-----------|
    | Close-in | $1/\Delta\omega^3$ | Flicker (1/f) noise upconverted by nonlinearity |
    | Mid-range | $1/\Delta\omega^2$ | White device noise filtered by the tank |
    | Far-out | flat | $2 F k T / P_{\text{sig}}$ noise floor |

    The transition between $1/\Delta\omega^3$ and $1/\Delta\omega^2$ is the
    *flicker corner* $\Delta\omega_{1/f^3}$; the transition between
    $1/\Delta\omega^2$ and the flat floor is at the half-bandwidth
    $\omega_0/(2Q_L)$.
    """)
    return


@app.cell
def _(go, leeson_pn, mo, np):
    # Static demonstration plot of the three Leeson regions
    _df = np.logspace(2, 8, 600)         # 100 Hz to 100 MHz offsets
    _f0 = 28e9
    _Q  = 12.0
    _F  = 4.0                            # 6 dB
    _Psig = 1e-3                         # 0 dBm
    _df_corner = 1e5                     # 100 kHz flicker corner
    _L_full = leeson_pn(_df, _F, _Psig, _Q, _f0, _df_corner)
    _L_white = leeson_pn(_df, _F, _Psig, _Q, _f0, 0.0)
    _floor_db = 10.0 * np.log10(0.5 * 2 * _F * 1.380649e-23 * 290 / _Psig)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_df, y=_L_full, mode="lines",
                              line=dict(color="#00CC96", width=2.5),
                              name="Leeson (with 1/f³)"))
    _fig.add_trace(go.Scatter(x=_df, y=_L_white, mode="lines",
                              line=dict(color="#636EFA", width=1.8, dash="dash"),
                              name="No flicker (1/f² + floor)"))
    _fig.add_hline(y=_floor_db, line=dict(color="#FFA15A", dash="dot"),
                   annotation_text="2FkT/P_sig floor",
                   annotation_position="bottom right")
    _fig.add_vline(x=_df_corner, line=dict(color="#EF553B", dash="dot"),
                   annotation_text="1/f³ corner")
    _fig.add_vline(x=_f0/(2*_Q), line=dict(color="#AB63FA", dash="dot"),
                   annotation_text="ω₀/2Q_L")
    _fig.update_layout(template="plotly_dark",
                       title=f"Leeson spectrum, f₀={_f0/1e9:.0f} GHz, Q_L={_Q:.0f}, F={10*np.log10(_F):.0f} dB, P=0 dBm",
                       xaxis_title="Offset frequency Δf (Hz)",
                       yaxis_title="ℒ(Δω) (dBc/Hz)",
                       xaxis_type="log",
                       height=420,
                       legend=dict(orientation="h", y=-0.2))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Limitations of Leeson

    Leeson is the right *shape* but the wrong *theory*. Each of the three
    parameters in the bracketed expression hides a closed-form question:

    - **$F$ — the noise factor.** This is *not* the Friis noise factor of
      notebook 04; that one applies to a one-port amplifier with linear
      gain. The Leeson $F$ absorbs everything the linear analysis misses,
      including white noise from devices that are biased near pinch-off
      most of the cycle, cyclostationary modulation of the noise PSD, and
      noise upconversion through the device's harmonic content. Asking
      "what is $F$ for my circuit?" without nonlinear theory is asking the
      experimentalist to fit a curve.
    - **$Q_L$ — the loaded $Q$.** A single quality factor implies a single
      loss mechanism. At mmWave the tank has at least four: metal series
      resistance (skin effect), substrate eddy currents, varactor series
      resistance, and parasitic shunt capacitance near self-resonance.
      Each scales differently with frequency, so $Q_L(\omega)$ is not a
      number but a function — and Leeson cannot say which loss mechanism
      to attack first to improve phase noise.
    - **$\Delta\omega_{1/f^3}$ — the 1/f corner.** This is *not* the
      device flicker corner. It depends on the *waveform shape* of the
      voltage across the tank: a symmetric (cross-coupled) waveform
      suppresses 1/f upconversion entirely, while an asymmetric (Colpitts)
      waveform does not. Leeson cannot tell you which waveform you have.

    Part II resolves all three by deriving phase noise directly from the
    periodic orbit using Floquet theory. The result is a computable noise
    factor, a spectrum that reveals which loss mechanism dominates, and an
    explicit upconversion mechanism that depends on circuit topology.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part II — Nonlinear oscillator theory
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Limit cycles and the amplitude/phase split

    The canonical nonlinear oscillator is the **Van der Pol equation**

    $$
    \boxed{\;
    \ddot{x} - \varepsilon\,(1 - x^2)\,\dot{x} + \omega_0^{\,2}\,x \;=\; 0
    \;}
    $$

    The damping term $-\varepsilon(1-x^2)\dot{x}$ is *negative* (energy
    injection) when $|x|<1$ and *positive* (dissipation) when $|x|>1$. Any
    nonzero initial state spirals onto a closed orbit in the $(x,\dot{x})$
    phase plane — the **limit cycle**. For small $\varepsilon$ the limit
    cycle is approximately circular with radius $A_{\text{ss}} = 2$ in the
    $(x, \dot{x}/\omega_0)$ plane.

    **Decomposing a noise kick.** Suppose the system sits on the limit
    cycle and receives an instantaneous perturbation $(\delta x, \delta v)$
    at some point along the orbit. Decompose the perturbation into two
    components:

    - **Amplitude component** — perpendicular to the orbit tangent. This
      pushes the trajectory off the limit cycle, where the nonlinear
      damping term acts to restore it. Amplitude perturbations decay
      exponentially with rate $\varepsilon/2$.
    - **Phase component** — tangent to the orbit. This advances or retards
      the orbit along itself. There is *no restoring force* — every point
      on the orbit is dynamically equivalent. The perturbation persists
      forever.

    The asymmetry between these two components is the entire content of
    phase-noise theory:

    $$
    \begin{aligned}
    \text{Amplitude noise:} &\quad \langle |\delta A(t)|^2 \rangle
        \to \text{const} \quad\text{(decays to balance)} \\
    \text{Phase noise:}     &\quad \langle |\delta\phi(t)|^2 \rangle
        \propto t           \quad\text{(diffusive growth)}
    \end{aligned}
    $$

    Diffusive phase yields a Lorentzian carrier line — equivalent to the
    $1/\Delta\omega^2$ region of Leeson, but now derived from the geometry
    of the orbit instead of the bandpass filter heuristic.
    """)
    return


@app.cell
def _(mo):
    eps_slider = mo.ui.slider(0.05, 1.5, step=0.05, value=0.3,
                              label="ε (nonlinearity)", show_value=True)
    kick_mag = mo.ui.slider(0.0, 1.5, step=0.05, value=0.6,
                            label="Kick magnitude", show_value=True)
    kick_angle = mo.ui.slider(0.0, 360.0, step=5.0, value=90.0,
                              label="Kick angle (°)", show_value=True)
    kick_time = mo.ui.slider(5.0, 30.0, step=1.0, value=12.0,
                             label="Kick time (rad/ω₀)", show_value=True)
    mo.md("**Interactive I — Limit-cycle visualiser** (Van der Pol)")
    return eps_slider, kick_angle, kick_mag, kick_time


@app.cell
def _(eps_slider, kick_angle, kick_mag, kick_time, mo):
    mo.hstack([eps_slider, kick_mag, kick_angle, kick_time], gap="2rem")
    return


@app.cell
def _(eps_slider, kick_angle, kick_mag, kick_time, np, vdp_simulate):
    _eps = eps_slider.value
    _omega0 = 1.0
    _t_end = 60.0
    _dt = 0.02
    _ang = np.deg2rad(kick_angle.value)
    _kx = kick_mag.value * np.cos(_ang)
    _kv = kick_mag.value * np.sin(_ang)

    # Run unperturbed
    t_u, x_u, v_u = vdp_simulate(_eps, _omega0, 0.1, 0.0, _t_end, _dt)
    # Run with kick at requested time
    t_p, x_p, v_p = vdp_simulate(_eps, _omega0, 0.1, 0.0, _t_end, _dt,
                                 kick_time=kick_time.value,
                                 kick_dx=_kx, kick_dv=_kv)
    # Decompose perturbed trajectory amplitude/phase relative to unperturbed
    r_u = np.sqrt(x_u ** 2 + (v_u / _omega0) ** 2)
    r_p = np.sqrt(x_p ** 2 + (v_p / _omega0) ** 2)
    th_u = np.unwrap(np.arctan2(v_u / _omega0, x_u))
    th_p = np.unwrap(np.arctan2(v_p / _omega0, x_p))
    dphi = th_p - th_u
    return dphi, r_p, r_u, t_p, t_u, v_p, v_u, x_p, x_u


@app.cell
def _(dphi, go, kick_time, make_subplots, mo, np, r_p, r_u, t_p, v_p, v_u, x_p,
      x_u):
    _fig = make_subplots(rows=2, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=("Phase plane", "Trajectories overlaid",
                                        "Amplitude perturbation r(t)",
                                        "Phase perturbation Δφ(t)"))
    # Phase plane: unperturbed orbit + perturbed orbit
    _fig.add_trace(go.Scatter(x=x_u[400:], y=v_u[400:], mode="lines",
                              name="unperturbed",
                              line=dict(color="#636EFA", width=1.5)),
                   row=1, col=1)
    _fig.add_trace(go.Scatter(x=x_p[400:], y=v_p[400:], mode="lines",
                              name="perturbed",
                              line=dict(color="#EF553B", width=1.5)),
                   row=1, col=1)
    # Mark kick instant
    _idx_kick = int(kick_time.value / 0.02)
    if _idx_kick < len(x_p) - 1:
        _fig.add_trace(go.Scatter(x=[x_p[_idx_kick]], y=[v_p[_idx_kick]],
                                  mode="markers",
                                  marker=dict(color="#FFD700", size=10,
                                              symbol="x"),
                                  name="kick", showlegend=True),
                       row=1, col=1)
    # Time series
    _fig.add_trace(go.Scatter(x=t_p, y=x_u, mode="lines", showlegend=False,
                              line=dict(color="#636EFA", width=1.2)),
                   row=1, col=2)
    _fig.add_trace(go.Scatter(x=t_p, y=x_p, mode="lines", showlegend=False,
                              line=dict(color="#EF553B", width=1.2)),
                   row=1, col=2)
    # Amplitude
    _fig.add_trace(go.Scatter(x=t_p, y=r_u, mode="lines", showlegend=False,
                              line=dict(color="#636EFA", width=1.2)),
                   row=2, col=1)
    _fig.add_trace(go.Scatter(x=t_p, y=r_p, mode="lines", showlegend=False,
                              line=dict(color="#EF553B", width=1.2)),
                   row=2, col=1)
    _fig.add_hline(y=2.0, line=dict(color="#AB63FA", dash="dot"),
                   row=2, col=1, annotation_text="A_ss")
    # Phase difference (unwrapped)
    _phase_late = dphi[_idx_kick + 50:] - dphi[_idx_kick + 50]
    _t_late = t_p[_idx_kick + 50:]
    _fig.add_trace(go.Scatter(x=_t_late, y=_phase_late, mode="lines",
                              showlegend=False,
                              line=dict(color="#00CC96", width=1.5)),
                   row=2, col=2)
    _fig.update_layout(template="plotly_dark", height=720,
                       legend=dict(orientation="h", y=-0.08))
    _fig.update_xaxes(title_text="x", row=1, col=1)
    _fig.update_yaxes(title_text="v/ω₀", row=1, col=1,
                      scaleanchor="x", scaleratio=1)
    _fig.update_xaxes(title_text="t·ω₀", row=1, col=2)
    _fig.update_yaxes(title_text="x(t)", row=1, col=2)
    _fig.update_xaxes(title_text="t·ω₀", row=2, col=1)
    _fig.update_yaxes(title_text="r(t)", row=2, col=1)
    _fig.update_xaxes(title_text="t·ω₀", row=2, col=2)
    _fig.update_yaxes(title_text="Δφ relative to LC (rad)", row=2, col=2)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Observe in the panels above:

    - In the **phase plane**, both trajectories settle on the same circular
      limit cycle. The kick (gold ×) displaces the perturbed orbit
      momentarily, but it is reabsorbed onto the cycle.
    - The **amplitude** $r(t)$ relaxes back to $A_{\text{ss}} = 2$
      exponentially after the kick — amplitude noise is filtered.
    - The **phase difference** $\Delta\phi(t)$ remains permanently offset
      after the kick — phase noise persists.

    Try a kick that is *purely tangential* to the limit cycle (angle 90°
    when the trajectory is on the +x axis): amplitude barely moves while
    Δφ jumps. Try a *radial* kick (angle 0°): amplitude jumps, phase
    barely moves. The decomposition is geometric, not statistical.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Equipartition and the noise floor

    Strip the active device away. A lossless $LC$ tank in thermal contact
    with a bath at temperature $T$ holds, by equipartition, $\tfrac{1}{2}
    k T$ in each quadratic degree of freedom of its energy. The tank has
    two such degrees: capacitor voltage and inductor current. Therefore

    $$
    \tfrac{1}{2} C \langle v_n^2 \rangle \;=\; \tfrac{1}{2} k T,
    \quad
    \tfrac{1}{2} L \langle i_n^2 \rangle \;=\; \tfrac{1}{2} k T,
    $$

    giving the **Nyquist tank-noise relations**

    $$
    \boxed{\;
    \langle v_n^2 \rangle \;=\; \frac{kT}{C},
    \qquad
    \langle i_n^2 \rangle \;=\; \frac{kT}{L}.
    \;}
    $$

    Total stored noise energy is $kT$, **independent of $L$ and $C$
    individually**. This is a hard floor: no choice of element values can
    reduce it.

    **Consequence for phase noise.** Improving phase noise requires
    increasing the *signal* energy stored in the tank, since phase noise
    scales as the ratio of noise to signal energy:

    $$
    E_{\text{sig}} \;=\; \frac{P_{\text{sig}} \, Q}{\omega_0},
    \qquad
    \frac{E_{\text{noise}}}{E_{\text{sig}}}
    \;=\; \frac{kT \, \omega_0}{P_{\text{sig}} \, Q}.
    $$

    The 1/$\Delta\omega^2$ phase-noise floor (in rad²/Hz) is set by this
    ratio further attenuated by $(Q\Delta\omega/\omega_0)^2$ inside the
    tank's bandwidth. Combining,

    $$
    S_\phi(\Delta\omega)
    \;\propto\; \frac{kT\,\omega_0}{P_{\text{sig}}\,Q^2\,\Delta\omega^2}.
    $$

    Hence the canonical figure of merit: at fixed $P_{\text{sig}}$,
    **phase noise improves as $1/Q^2$** — doubling $Q$ buys 6 dB of
    phase noise, while doubling $P_{\text{sig}}$ buys only 3 dB. At
    mmWave, where $Q$ collapses (§10), this $Q^2$ dependence is the
    binding constraint.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. State-space formulation

    Write the noisy oscillator as an $n$-dimensional system

    $$
    \boxed{\;
    \dot{\mathbf{x}}(t) \;=\; \mathbf{f}(\mathbf{x}(t)) \;+\;
        B(\mathbf{x}(t))\,\mathbf{u}(t)
    \;}
    $$

    with

    - $\mathbf{x}(t) \in \mathbb{R}^n$ — circuit state (capacitor voltages,
      inductor currents, gate voltages, …).
    - $\mathbf{f}(\mathbf{x})$ — the autonomous (noise-free) vector field.
    - $B(\mathbf{x})$ — the noise coupling matrix; columns are the
      directions in state space along which each independent noise source
      injects.
    - $\mathbf{u}(t)$ — vector of independent white-noise processes with
      PSD matrix $S_u$.

    The state-dependence of $B(\mathbf{x})$ encodes
    **cyclostationary noise**: a transistor's white-noise current is
    proportional to its bias current, which itself rides the periodic
    waveform. This is the same cyclostationary structure that appeared in
    notebook 05 §18-20 for mixers, and in notebook 04 for LNA noise. The
    notebook 06 application of it just produces a different number — phase
    noise instead of mixer SSB noise figure.

    **The unperturbed orbit.** With $\mathbf{u} = 0$ the system has a
    $T$-periodic solution

    $$
    \mathbf{x}_s(t) \;=\; \mathbf{x}_s(t + T),
    \qquad T \;=\; 2\pi/\omega_0.
    $$

    This is the limit cycle in $n$-dimensional state space. The next two
    sections linearise the perturbed system around this orbit.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Floquet theory

    Linearise about the periodic orbit by setting $\mathbf{x}(t) =
    \mathbf{x}_s(t) + \delta\mathbf{x}(t)$:

    $$
    \delta\dot{\mathbf{x}} \;=\; A(t)\,\delta\mathbf{x},
    \qquad A(t) \;=\;
    \left.\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right|_{\mathbf{x}_s(t)}.
    $$

    $A(t)$ is $T$-periodic. Define the state-transition matrix
    $\Phi(t,0)$ by $\dot{\Phi} = A(t)\Phi$, $\Phi(0,0) = I$. The
    **monodromy matrix** is the state transition over exactly one period:

    $$
    \boxed{\;
    M \;\equiv\; \Phi(T,\,0).
    \;}
    $$

    The eigenvalues $\mu_i$ of $M$ are the **Floquet multipliers** of the
    orbit. They classify how each direction in state space evolves over
    one period.

    **Floquet's theorem (the unit multiplier).** *Any* autonomous periodic
    orbit has at least one multiplier exactly equal to 1. The proof is one
    line: differentiate $\dot{\mathbf{x}}_s = \mathbf{f}(\mathbf{x}_s)$
    with respect to $t$ to get

    $$
    \frac{d}{dt}\dot{\mathbf{x}}_s \;=\;
    \frac{\partial\mathbf{f}}{\partial\mathbf{x}}\bigg|_{\mathbf{x}_s}
    \dot{\mathbf{x}}_s \;=\; A(t)\,\dot{\mathbf{x}}_s.
    $$

    So $\dot{\mathbf{x}}_s(t)$ satisfies the linearised equation, and
    $\dot{\mathbf{x}}_s(T) = \dot{\mathbf{x}}_s(0)$ by periodicity, which
    means $M \dot{\mathbf{x}}_s(0) = \dot{\mathbf{x}}_s(0)$. The
    eigenvector is the orbit tangent; the eigenvalue is unity.

    **Stability of the limit cycle.** All other multipliers must satisfy
    $|\mu_i| < 1$ for the limit cycle to be stable — the corresponding
    *amplitude modes* decay over a period, contracting the trajectory back
    onto the cycle. The unit multiplier is the algebraic reason phase
    perturbations *neither grow nor decay*: they neither dissipate (no
    restoring force along the tangent) nor blow up (the orbit is closed).
    They simply *accumulate*. This is the origin of the
    $1/\Delta\omega^2$ Lorentzian.
    """)
    return


@app.cell
def _(go, mo, np, vdp_monodromy):
    # Numerical demonstration: Floquet multipliers of the VdP limit cycle
    _eps_list = [0.1, 0.3, 0.6, 1.0]
    _multipliers = []
    for _eps in _eps_list:
        _M, _ev = vdp_monodromy(_eps, omega0=1.0)
        _multipliers.append(_ev)
    _multipliers = np.array(_multipliers)

    _fig = go.Figure()
    # Unit circle for context
    _theta = np.linspace(0, 2*np.pi, 200)
    _fig.add_trace(go.Scatter(x=np.cos(_theta), y=np.sin(_theta),
                              mode="lines", showlegend=False,
                              line=dict(color="rgba(220,220,220,0.5)",
                                        width=1, dash="dot")))
    _palette = ["#00CC96", "#636EFA", "#EF553B", "#FFD700"]
    for _i, (_eps, _ev) in enumerate(zip(_eps_list, _multipliers)):
        _fig.add_trace(go.Scatter(x=[_ev[0].real, _ev[1].real],
                                  y=[_ev[0].imag, _ev[1].imag],
                                  mode="markers",
                                  name=f"ε={_eps:.2f}",
                                  marker=dict(size=12, color=_palette[_i],
                                              line=dict(color="white", width=1))))
    _fig.add_vline(x=1.0, line=dict(color="#AB63FA", dash="dot"),
                   annotation_text="μ = 1 (phase mode)",
                   annotation_position="top")
    _fig.update_layout(template="plotly_dark",
                       title="Floquet multipliers of the Van der Pol limit cycle",
                       xaxis_title="Re μ", yaxis_title="Im μ",
                       xaxis=dict(range=[-0.2, 1.4],
                                  scaleanchor="y", scaleratio=1),
                       yaxis=dict(range=[-0.6, 0.6]),
                       height=420,
                       legend=dict(orientation="h", y=-0.2))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Numerical confirmation: integrating the VdP linearisation around one
    period and reading off the eigenvalues of $\Phi(T,0)$ yields, for each
    $\varepsilon$, exactly one multiplier at $\mu = 1$ (the phase mode)
    and one inside the unit disk (the amplitude mode). As $\varepsilon$
    grows, the amplitude multiplier is pushed deeper toward the origin —
    the cycle becomes more strongly attracting in the radial direction,
    while the phase mode is unchanged.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Adjoint eigenvectors and the ISF

    Floquet theory says *that* phase noise accumulates along the unit
    multiplier; we now compute *how much*. The construction is a textbook
    application of adjoint sensitivity.

    **Adjoint system.** For the linearised dynamics
    $\delta\dot{\mathbf{x}} = A(t)\delta\mathbf{x}$, the adjoint is

    $$
    \dot{\mathbf{y}} \;=\; -A(t)^{T}\,\mathbf{y}.
    $$

    Its state-transition matrix is $\Psi(t,0) = [\Phi(t,0)^{-1}]^T$, and
    over one period $\Psi(T,0) = (M^{-1})^T = (M^T)^{-1}$. The eigenvalues
    of $\Psi(T,0)$ are $1/\mu_i$ — and by the Floquet pairing the
    eigenvalue $1/\mu_1 = 1$ corresponds to a left eigenvector of $M$.

    **Phase sensitivity vector.** Let $\mathbf{v}_1(t)$ be the left
    eigenvector of $M$ with eigenvalue 1, propagated by the adjoint
    dynamics so that $\mathbf{v}_1$ is itself $T$-periodic. The
    biorthogonality relation between left and right eigenvectors fixes the
    normalisation:

    $$
    \boxed{\;
    \mathbf{v}_1(t)^T\,\dot{\mathbf{x}}_s(t) \;=\; 1 \quad\forall\,t.
    \;}
    $$

    **Projection onto the phase mode.** A noise impulse $B(\mathbf{x}_s(\tau))
    \,\mathbf{u}\,\delta(t-\tau)$ injects a perturbation $\delta\mathbf{x}_0
    = B(\mathbf{x}_s(\tau))\mathbf{u}$ at time $\tau$. The Floquet
    decomposition splits this into a phase component (along the tangent)
    and amplitude components (along the contracting modes). The
    **phase-mode amplitude** is precisely the projection through the left
    eigenvector:

    $$
    \delta\phi(\tau)
    \;=\; \mathbf{v}_1(\tau)^T\,B(\mathbf{x}_s(\tau))\,\mathbf{u}.
    $$

    The amplitude components decay; only $\delta\phi$ persists.

    **Definition of the ISF.** Hajimiri & Lee absorb the projection,
    coupling matrix, and a normalising charge $q_{\text{max}}$ (the peak
    charge displacement on the tank capacitor over one period) into a
    single dimensionless function:

    $$
    \boxed{\;
    \Gamma(\omega_0\tau)
    \;\equiv\; \frac{\mathbf{v}_1(\tau)^T\,B(\mathbf{x}_s(\tau))}
                    {q_{\text{max}}}.
    \;}
    $$

    $\Gamma$ is the **Impulse Sensitivity Function**. It is $T$-periodic,
    dimensionless when scaled by $q_{\text{max}}$, and tells you exactly
    how much the steady-state phase shifts in response to a unit-charge
    impulse delivered at phase $\omega_0\tau$ along the orbit.

    **Time-domain phase response.** A finite noise current $i_n(t)$ then
    produces

    $$
    \delta\phi(t)
    \;=\; \frac{1}{q_{\text{max}}}\!
        \int_{-\infty}^{t}\!\Gamma(\omega_0\tau)\,i_n(\tau)\,d\tau.
    $$

    All of phase-noise theory follows from this one expression.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. From ISF to phase-noise spectrum

    The ISF is $T$-periodic, so expand it as a Fourier series:

    $$
    \boxed{\;
    \Gamma(\omega_0 t) \;=\; \frac{c_0}{2}
    \;+\; \sum_{n=1}^{\infty} c_n\,\cos(n\omega_0 t + \theta_n).
    \;}
    $$

    A cyclostationary noise current with PSD $\overline{i_n^2}/\Delta f$
    drives the integral above. The standard derivation (Hajimiri & Lee,
    1999) decomposes the cosine multiplications into upconversion and
    downconversion terms and reads off the spectrum at offset
    $\Delta\omega$:

    $$\boxed{\; S_\phi(\Delta\omega) \;=\; \frac{\overline{i_n^2}/\Delta f}{2\,q_{\text{max}}^{\,2}\,\Delta\omega^2}\,\Gamma_{\text{rms}}^{\,2}, \quad \Gamma_{\text{rms}}^{\,2} \;=\; \frac{c_0^2}{4} + \sum_{n\ge 1}\frac{c_n^{\,2}}{2}. \;}$$

    **Interpretation by harmonic.** A noise component at $n\omega_0 +
    \Delta\omega$ enters the phase output with weight $c_n^2/2$ at
    offset $\Delta\omega$. Each harmonic of $\Gamma$ thus
    **downconverts** noise from a different frequency band:

    | Harmonic | Source band | Output region |
    |----------|-------------|---------------|
    | $c_0$ | DC (1/$f$ flicker) | upconverted to carrier → $1/\Delta\omega^3$ |
    | $c_1$ | $\omega_0 \pm \Delta\omega$ | direct → $1/\Delta\omega^2$ (dominant) |
    | $c_2$ | $2\omega_0 \pm \Delta\omega$ | even harmonic mixing |
    | $c_n$ | $n\omega_0 \pm \Delta\omega$ | usually negligible for $n \ge 3$ |

    **The 1/$f^3$ corner is computable.** Hajimiri & Lee showed that the
    flicker-induced corner in the *phase noise* spectrum is

    $$
    \boxed{\;
    \Delta\omega_{1/f^3}
    \;=\; \omega_{1/f,\text{device}}\,\frac{c_0^{\,2}/4}{\Gamma_{\text{rms}}^{\,2}},
    \;}
    $$

    where $\omega_{1/f,\text{device}}$ is the device flicker corner. **If
    $c_0 = 0$, the 1/$f^3$ region disappears.** This identity will drive
    the topology comparison in §12-13.

    **Reduction to Leeson.** Comparing with the Leeson formula identifies
    the effective noise factor:

    $$
    F_{\text{eff}}
    \;=\; \frac{\Gamma_{\text{rms}}^{\,2}}{q_{\text{max}}^{\,2}}
        \cdot \frac{1}{kT R_p}
        \cdot \sum_{\text{sources}} \overline{i_{n,k}^{\,2}}/\Delta f.
    $$

    All three Leeson empirical parameters — $F$, the loaded $Q$ via
    $R_p$, and the 1/$f^3$ corner — are now closed-form functions of the
    waveform shape (encoded in $c_n$) and the device noise PSD.
    """)
    return


@app.cell
def _(mo):
    isf_topology = mo.ui.radio(["Symmetric (cross-coupled)",
                                "Asymmetric (Colpitts-like)",
                                "Custom"],
                               value="Symmetric (cross-coupled)",
                               label="Waveform topology", inline=False)
    c0_slider = mo.ui.slider(-1.0, 1.0, step=0.05, value=0.0,
                             label="c₀ (DC)", show_value=True)
    c1_slider = mo.ui.slider(0.0, 2.0, step=0.05, value=1.0,
                             label="c₁", show_value=True)
    c2_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0,
                             label="c₂", show_value=True)
    c3_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0,
                             label="c₃", show_value=True)
    flicker_slider = mo.ui.slider(1e3, 1e7, step=1e3, value=1e5,
                                  label="Device 1/f corner (Hz)",
                                  show_value=True)
    in_slider = mo.ui.slider(-180.0, -150.0, step=1.0, value=-168.0,
                             label="i_n² PSD (dBA²/Hz)", show_value=True)
    mo.md("**Interactive II — ISF explorer**")
    return (c0_slider, c1_slider, c2_slider, c3_slider, flicker_slider,
            in_slider, isf_topology)


@app.cell
def _(c0_slider, c1_slider, c2_slider, c3_slider, flicker_slider, in_slider,
      isf_topology, mo):
    mo.vstack([
        mo.hstack([isf_topology], gap="2rem"),
        mo.hstack([c0_slider, c1_slider, c2_slider, c3_slider], gap="2rem"),
        mo.hstack([flicker_slider, in_slider], gap="2rem"),
    ])
    return


@app.cell
def _(c0_slider, c1_slider, c2_slider, c3_slider, isf_topology, np):
    # Topology presets override the sliders unless "Custom" is chosen
    if isf_topology.value == "Symmetric (cross-coupled)":
        _c0, _c1, _c2, _c3 = 0.0, 1.0, 0.0, 0.15
    elif isf_topology.value == "Asymmetric (Colpitts-like)":
        _c0, _c1, _c2, _c3 = 0.45, 0.9, 0.4, 0.15
    else:
        _c0 = c0_slider.value
        _c1 = c1_slider.value
        _c2 = c2_slider.value
        _c3 = c3_slider.value

    isf_c0 = _c0
    isf_cn = np.array([_c1, _c2, _c3])
    return isf_c0, isf_cn


@app.cell
def _(flicker_slider, go, in_slider, isf_c0, isf_cn, isf_from_harmonics,
      isf_rms_squared, make_subplots, mo, np, pn_from_isf):
    _phase = np.linspace(0, 2 * np.pi, 400)
    _theta_n = np.zeros_like(isf_cn)
    _gamma = isf_from_harmonics(isf_c0, isf_cn, _theta_n, _phase)
    _gamma2_rms = isf_rms_squared(isf_c0, isf_cn)
    _f0 = 28e9
    _df = np.logspace(2, 8, 400)
    # Cap c0 small to avoid 1/f3 corner exploding off-scale
    _q_max = 1e-12
    _i_n_white = 10 ** (in_slider.value / 10.0)
    _flicker_corner = flicker_slider.value
    _L_db = pn_from_isf(_df, isf_c0, isf_cn, _q_max, _i_n_white,
                        _flicker_corner, _f0)

    _fig = make_subplots(rows=1, cols=3, column_widths=[0.4, 0.2, 0.4],
                        subplot_titles=("Γ(ω₀ t)", "|cₙ| spectrum",
                                        "ℒ(Δω) phase noise"))
    _fig.add_trace(go.Scatter(x=_phase / (2*np.pi), y=_gamma, mode="lines",
                              line=dict(color="#00CC96", width=2),
                              showlegend=False),
                   row=1, col=1)
    _fig.add_hline(y=0.0, line=dict(color="white", dash="dot"),
                   row=1, col=1)
    _fig.update_xaxes(title_text="t / T", row=1, col=1)
    _fig.update_yaxes(title_text="Γ", row=1, col=1)

    _bar_x = ["c₀"] + [f"c{i}" for i in range(1, len(isf_cn) + 1)]
    _bar_y = [abs(isf_c0)] + list(np.abs(isf_cn))
    _bar_colors = ["#EF553B"] + ["#636EFA"] * len(isf_cn)
    _fig.add_trace(go.Bar(x=_bar_x, y=_bar_y, marker_color=_bar_colors,
                          showlegend=False),
                   row=1, col=2)
    _fig.update_yaxes(title_text="|cₙ|", row=1, col=2)

    _fig.add_trace(go.Scatter(x=_df, y=_L_db, mode="lines",
                              line=dict(color="#FFD700", width=2),
                              showlegend=False),
                   row=1, col=3)
    _fig.update_xaxes(title_text="Δf (Hz)", type="log", row=1, col=3)
    _fig.update_yaxes(title_text="ℒ(Δω) (dBc/Hz)", row=1, col=3)
    _fig.update_layout(template="plotly_dark", height=380,
                       title=f"Γ²_rms = {_gamma2_rms:.3f},  "
                             f"c₀ = {isf_c0:.3f}  "
                             f"({'1/f³ suppressed' if abs(isf_c0) < 1e-3 else '1/f³ active'})")
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Switch between **Symmetric** and **Asymmetric** to see the qualitative
    distinction:

    - **Symmetric.** $c_0 = 0$ identically (a half-wave-symmetric
      waveform $v(t+T/2) = -v(t)$ has only odd harmonics in $\Gamma$).
      The $1/\Delta\omega^3$ region disappears from $\mathcal{L}(\Delta\omega)$
      to first order; phase noise rolls cleanly along $1/\Delta\omega^2$
      until the floor.
    - **Asymmetric.** $c_0 \ne 0$ from the asymmetry between the rising and
      falling half-cycles. The $1/\Delta\omega^3$ region appears.

    Note also that increasing $c_2$ alone, with the rest held fixed,
    raises $\Gamma^2_{\text{rms}}$ proportionally and so raises the entire
    phase-noise spectrum — extra harmonics are extra ways to downconvert
    noise.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part III — mmWave oscillator considerations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Tank Q at mmWave

    The phase-noise floor scales as $kT\omega_0/(P_{\text{sig}}Q^2)$, so
    each factor of two in $Q$ buys 6 dB. At GHz frequencies with off-chip
    inductors $Q \sim 50\text{-}100$ is routine. At 28-60 GHz on a CMOS
    substrate, $Q \sim 8\text{-}15$ is typical and trending downward as
    frequency increases. The reasons are physical, not technological.

    **Inductor loss.** Three coupled mechanisms degrade $Q_L$:

    - **Skin effect.** AC current crowds into a surface layer of depth
      $\delta = \sqrt{2/(\omega\mu_0\sigma)}$. Series resistance grows as
      $R_s \propto \sqrt{f}$, so

      $$Q_L \;=\; \frac{\omega L}{R_s} \;\propto\; \sqrt{f}$$

      in the skin-effect regime. $Q_L$ rises with frequency until other
      mechanisms take over.
    - **Substrate eddy currents.** A spiral inductor over a low-resistivity
      silicon substrate induces image currents in the bulk. These create
      an *image inductance* (reducing the effective $L$) and an *image
      resistance* (adding loss). The loss scales roughly as
      $f^2/\rho_{\text{sub}}$ and is partially mitigated by a *patterned
      ground shield* — slotted metal that breaks the eddy-current loops.
    - **Self-resonance.** Parasitic shunt capacitance $C_{\text{par}}$
      between the spiral turns and to the ground plane forms an unwanted
      LC at $f_{\text{SRF}} = 1/(2\pi\sqrt{L C_{\text{par}}})$. As $f$
      approaches $f_{\text{SRF}}$, the effective inductance falls and
      the loss spikes. Useful frequency range is roughly $f <
      f_{\text{SRF}}/2$.

    Combined: $Q_L(f)$ rises with $\sqrt{f}$, peaks somewhere below SRF,
    and collapses near SRF.

    **Varactor (capacitor) loss.** A MOS varactor used for tuning has
    series channel resistance $R_{\text{var}}$:

    $$
    Q_C \;=\; \frac{1}{\omega\,C\,R_{\text{var}}}.
    $$

    $R_{\text{var}}$ grows with frequency (skin effect in the channel and
    fringing path), so $Q_C$ falls faster than $1/\omega$ at mmWave. By
    30 GHz, MOS varactor $Q$ is typically below the inductor $Q$ — the
    varactor becomes the bottleneck.

    **Combined tank.**

    $$
    \boxed{\;
    \frac{1}{Q_{\text{tank}}} \;=\; \frac{1}{Q_L} + \frac{1}{Q_C}.
    \;}
    $$

    The weakest element dominates. The interactive below shows which one
    that is for a given technology and operating frequency.
    """)
    return


@app.cell
def _(mo):
    sigma_slider = mo.ui.slider(2.0e7, 6.0e7, step=2.0e6, value=4.1e7,
                                label="σ_metal (S/m)", show_value=True)
    tox_slider = mo.ui.slider(2.0, 20.0, step=0.5, value=8.0,
                              label="t_ox (μm)", show_value=True)
    rho_slider = mo.ui.slider(1.0, 100.0, step=1.0, value=10.0,
                              label="ρ_sub (Ω·cm)", show_value=True)
    rvar_slider = mo.ui.slider(0.5, 10.0, step=0.5, value=2.5,
                               label="R_var @1GHz (Ω)", show_value=True)
    L_slider = mo.ui.slider(50.0, 1000.0, step=25.0, value=200.0,
                            label="L (pH)", show_value=True)
    C_slider = mo.ui.slider(20.0, 500.0, step=10.0, value=160.0,
                            label="C (fF)", show_value=True)
    fop_slider = mo.ui.slider(5.0, 80.0, step=1.0, value=28.0,
                              label="f_op (GHz) marker", show_value=True)
    mo.md("**Interactive III — Tank Q breakdown**")
    return (C_slider, L_slider, fop_slider, rho_slider, rvar_slider,
            sigma_slider, tox_slider)


@app.cell
def _(C_slider, L_slider, fop_slider, mo, rho_slider, rvar_slider,
      sigma_slider, tox_slider):
    mo.vstack([
        mo.hstack([sigma_slider, tox_slider, rho_slider], gap="2rem"),
        mo.hstack([rvar_slider, L_slider, C_slider, fop_slider], gap="2rem"),
    ])
    return


@app.cell
def _(C_slider, L_slider, fop_slider, go, inductor_Q, make_subplots, mo, np,
      rho_slider, rvar_slider, sigma_slider, tank_Q, tox_slider, varactor_Q):
    _f = np.logspace(9, 11, 400)
    _L_H = L_slider.value * 1e-12
    _C_F = C_slider.value * 1e-15
    _C_par = 0.5 * _C_F                      # parasitic shunt cap on inductor
    _Q_L = inductor_Q(_f, _L_H, sigma_slider.value, tox_slider.value,
                     rho_slider.value, _C_par)
    _Q_C = varactor_Q(_f, _C_F, rvar_slider.value)
    _Q_tank = tank_Q(_Q_L, _Q_C)

    _fop_hz = fop_slider.value * 1e9
    _idx = int(np.argmin(np.abs(_f - _fop_hz)))
    _Q_L_op = _Q_L[_idx]
    _Q_C_op = _Q_C[_idx]
    _Q_tank_op = _Q_tank[_idx]

    _fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=("Q vs frequency",
                                        "Loss share at f_op"))
    _fig.add_trace(go.Scatter(x=_f / 1e9, y=_Q_L, mode="lines",
                              line=dict(color="#636EFA", width=2),
                              name="Q_L (inductor)"),
                   row=1, col=1)
    _fig.add_trace(go.Scatter(x=_f / 1e9, y=_Q_C, mode="lines",
                              line=dict(color="#EF553B", width=2),
                              name="Q_C (varactor)"),
                   row=1, col=1)
    _fig.add_trace(go.Scatter(x=_f / 1e9, y=_Q_tank, mode="lines",
                              line=dict(color="#00CC96", width=2.5),
                              name="Q_tank"),
                   row=1, col=1)
    _fig.add_vline(x=fop_slider.value, line=dict(color="#FFD700", dash="dot"),
                   row=1, col=1, annotation_text=f"f_op={fop_slider.value:.0f} GHz")
    _fig.update_xaxes(title_text="f (GHz)", row=1, col=1)
    _fig.update_yaxes(title_text="Q", type="log", row=1, col=1)

    # Stacked bar: 1/Q contributions at f_op
    _share_L = (1.0 / _Q_L_op) / (1.0 / _Q_tank_op)
    _share_C = (1.0 / _Q_C_op) / (1.0 / _Q_tank_op)
    _fig.add_trace(go.Bar(x=["1/Q_tank"], y=[_share_L * 100.0],
                          name="inductor", marker_color="#636EFA"),
                   row=1, col=2)
    _fig.add_trace(go.Bar(x=["1/Q_tank"], y=[_share_C * 100.0],
                          name="varactor", marker_color="#EF553B"),
                   row=1, col=2)
    _fig.update_xaxes(title_text="Loss component", row=1, col=2)
    _fig.update_yaxes(title_text="% of total loss", row=1, col=2)
    _fig.update_layout(template="plotly_dark", barmode="stack", height=440,
                       title=f"Q_L = {_Q_L_op:.1f} | Q_C = {_Q_C_op:.1f} | "
                             f"Q_tank = {_Q_tank_op:.1f}",
                       legend=dict(orientation="h", y=-0.2))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Push the operating frequency past the inductor's SRF (~the peak of
    $Q_L$) and watch $Q_{\text{tank}}$ collapse — the inductor stops
    behaving inductively. Below the SRF, in the skin-effect regime,
    $Q_L$ rises with $\sqrt{f}$ but is overtaken by the varactor's
    $1/(\omega C R_{\text{var}})$ falling faster. The crossover frequency
    where loss leadership swaps from inductor to varactor (visible in the
    stacked bar) is the **target operating regime** for mmWave VCO design:
    above it, varactor research dominates; below it, inductor research
    dominates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. The phase-noise figure of merit

    To compare oscillators across frequency and power, the standard FOM
    normalises out the trivial $\omega_0$ and $P_{\text{DC}}$ scaling:

    $$\boxed{\; \text{FOM} \;=\; -\mathcal{L}(\Delta\omega) + 20\log_{10}\!\frac{f_0}{\Delta f} - 10\log_{10}\!\frac{P_{\text{DC}}}{1\,\text{mW}} \;\text{(dB)}. \;}$$

    Higher (less negative) is better. A typical good 5 GHz CMOS VCO
    achieves FOM ≈ 195 dB; a 28 GHz design often falls to ≈ 185 dB; at
    60 GHz, ≈ 175 dB is competitive. Each 10 dB drop is a factor-of-ten
    increase in the noise-energy-to-signal-energy ratio at fixed offset.

    **Why FOM degrades at mmWave.** From §5,
    $S_\phi \propto kT\omega_0/(P_{\text{sig}}Q^2)$. At fixed
    $P_{\text{sig}}$ and varying $f_0$, FOM behaves as:

    $$
    \text{FOM}
    \;\sim\; \text{const} + 20\log_{10} Q.
    $$

    A drop in $Q$ from 50 to 10 takes 14 dB straight off the FOM. The
    only way to recover it is more power (which improves FOM only as
    $-10\log_{10} P_{\text{DC}}$ in the formula because phase noise itself
    falls as $1/P_{\text{sig}}$). Hence the **Q-limited ceiling**:
    published mmWave FOM ceilings reflect process-limited tank Q, not
    designer skill.
    """)
    return


@app.cell
def _(go, mo, np):
    # Schematic FOM ceiling vs. frequency, illustrating Q²-limited fall-off.
    _f = np.linspace(5.0, 100.0, 60)            # GHz
    # Empirical fit: FOM_ceiling ≈ 200 - 20 log10(f/5) for Q ∝ f^{-1/2}
    _Q_at_f = 50.0 * (5.0 / _f) ** 0.5
    _fom_ceiling = 200.0 + 20.0 * np.log10(_Q_at_f / 50.0)

    # Some published reference points (illustrative, not from a specific paper)
    _refs_f = np.array([5.0, 10.0, 28.0, 39.0, 60.0, 77.0])
    _refs_fom = np.array([196.0, 192.0, 184.0, 182.0, 178.0, 175.0])

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_f, y=_fom_ceiling, mode="lines",
                              line=dict(color="#00CC96", width=2.5,
                                        dash="dash"),
                              name="Q²-limited ceiling"))
    _fig.add_trace(go.Scatter(x=_refs_f, y=_refs_fom, mode="markers",
                              marker=dict(color="#FFD700", size=10),
                              name="published designs (illustrative)"))
    _fig.update_layout(template="plotly_dark",
                       title="Oscillator FOM vs. carrier frequency",
                       xaxis_title="f₀ (GHz)",
                       yaxis_title="FOM (dB)",
                       xaxis_type="log",
                       height=420,
                       legend=dict(orientation="h", y=-0.2))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Cross-coupled LC topology

    The standard mmWave VCO architecture is the differential cross-coupled
    pair (NMOS, PMOS, or complementary). The two transistors mutually
    drive each other's gates from the opposite drain, providing the
    negative resistance $-2/g_m$ across a differential tank.

    **Operation.** Each transistor's drain is the gate of the other:
    a positive swing on $V_o^+$ drives M1's $g_m$ to push current that
    pulls $V_o^-$ low, which in turn drives M2 to pull $V_o^+$ higher.
    The differential pair presents an effective negative resistance
    $-2/g_m$ across the tank; oscillation starts when

    $$
    g_m \;>\; \frac{1}{R_p}\;\;\text{(parallel tank loss)}.
    $$

    A factor-of-three margin is typical to ensure startup over corners.

    **ISF analysis.** The tank voltage waveform of an ideal cross-coupled
    pair has half-wave symmetry: $v(t + T/2) = -v(t)$. The ISF inherits
    this symmetry, $\Gamma(\omega_0 t + \pi) = -\Gamma(\omega_0 t)$,
    which forces all even Fourier coefficients to zero:

    $$
    \boxed{\;
    c_0 = c_2 = c_4 = \cdots = 0
    \;\Longleftrightarrow\;
    v(t+T/2) = -v(t).
    \;}
    $$

    Two consequences for phase noise:

    - **No 1/$f^3$ from device flicker.** Since $c_0 = 0$, the
      Hajimiri-Lee corner $\Delta\omega_{1/f^3} \propto c_0^2$ collapses
      to zero. To first order in waveform symmetry, 1/f device noise
      *does not upconvert*. (In practice mismatch and finite tail
      impedance break the symmetry, leaving a residual 1/f³ region
      typically 10-15 dB below an asymmetric topology's.)
    - **Tail current noise.** The tail device's noise spectral
      contribution at $2\omega_0$ couples to phase noise through $c_2$.
      For a perfectly symmetric waveform $c_2 = 0$ and tail noise
      vanishes — but symmetry is fragile. Practical designs add an
      **LC tail filter** (a parallel resonator at $2\omega_0$ in series
      with the tail current source) to short out residual tail-noise
      coupling.

    The cross-coupled topology is the standard mmWave VCO architecture
    precisely because its symmetry zeros $c_0$ — close-in phase noise
    matters more than far-out at most carrier offsets of interest.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Colpitts topology

    The Colpitts oscillator uses a single transistor and a capacitive
    voltage divider $C_1, C_2$ for feedback. The startup condition

    $$
    g_m \;>\; \frac{(C_1 + C_2)^2}{\omega_0^{\,2}\,C_1^{\,2}\,C_2\,R_p}
    $$

    requires a higher $g_m$ than cross-coupled, but the topology has only
    one active device — useful when transistor count or DC headroom is
    constrained.

    **ISF analysis.** The transistor in a Colpitts oscillator conducts
    in a *narrow pulse* near the voltage minimum (class-C-like
    operation), not continuously. The drain-current waveform is sharply
    asymmetric — and so is the resulting ISF:

    - **Non-zero $c_0$.** Asymmetry guarantees a finite DC component in
      $\Gamma$. Device 1/f noise *is* upconverted to 1/$f^3$ phase noise.
      Worse close-in phase noise than cross-coupled.
    - **Pulsed conduction reduces $\Gamma_{\text{rms}}$.** The transistor
      is OFF most of the cycle and injects noise only during a short
      conduction window. If the pulse aligns with the minimum of
      $|\Gamma(\omega_0 t)|$ (which it does, by construction, since the
      conduction occurs at the voltage minimum where the orbit is most
      "vertical"), the *cyclostationary weighting* reduces the
      effective $\Gamma_{\text{rms}}$.

    Net result: Colpitts is **worse for 1/$f^3$ noise but potentially
    better for the 1/$f^2$ floor**. Whether this is a win depends on
    application. Frequency synthesisers care about close-in phase noise
    (loop bandwidth limits how well a PLL can track 1/$f^3$), so
    cross-coupled wins. Sampling oscillators or far-out blocking
    requirements care about high-offset phase noise, where Colpitts can
    be competitive.

    **Topology comparison.**

    | Criterion              | Cross-coupled LC      | Colpitts             |
    |------------------------|-----------------------|----------------------|
    | Active devices         | 2 (differential pair) | 1 (single-ended)     |
    | Startup $g_m$          | $g_m > 1/R_p$         | $g_m > (C_1{+}C_2)^2/(\omega_0^2 C_1^2 C_2 R_p)$ |
    | Symmetry of waveform   | Half-wave symmetric   | Asymmetric (pulsed)  |
    | $c_0$                  | 0 (to first order)    | nonzero              |
    | 1/$f^3$ corner         | Strongly suppressed   | Present              |
    | $\Gamma^2_{\text{rms}}$ | Moderate              | Lower (cyclostationary) |
    | High-offset PN         | Standard              | Can be lower         |
    | Tail noise sensitivity | Needs LC tail filter  | N/A                  |
    | Tuning range           | Wide                  | Moderate             |
    | mmWave use             | Default architecture  | Selective use (V-band+) |
    """)
    return


@app.cell
def _(mo):
    pn_topology = mo.ui.radio(["Cross-coupled (c₀=0)",
                               "Colpitts (c₀≠0)"],
                              value="Cross-coupled (c₀=0)",
                              label="Topology", inline=True)
    pn_Q = mo.ui.slider(3.0, 60.0, step=1.0, value=12.0,
                        label="Q_tank", show_value=True)
    pn_f0 = mo.ui.slider(5.0, 100.0, step=1.0, value=28.0,
                         label="f₀ (GHz)", show_value=True)
    pn_Pdc = mo.ui.slider(2.0, 20.0, step=0.5, value=10.0,
                          label="P_DC (mW)", show_value=True)
    pn_F = mo.ui.slider(2.0, 12.0, step=0.5, value=4.0,
                        label="Leeson F (linear)", show_value=True)
    pn_corner = mo.ui.slider(1e3, 1e7, step=1e3, value=1e5,
                             label="device 1/f corner (Hz)", show_value=True)
    mo.md("**Interactive IV — Phase-noise budget tool: Leeson vs. ISF**")
    return pn_F, pn_Pdc, pn_Q, pn_corner, pn_f0, pn_topology


@app.cell
def _(mo, pn_F, pn_Pdc, pn_Q, pn_corner, pn_f0, pn_topology):
    mo.vstack([
        mo.hstack([pn_topology], gap="2rem"),
        mo.hstack([pn_Q, pn_f0, pn_Pdc], gap="2rem"),
        mo.hstack([pn_F, pn_corner], gap="2rem"),
    ])
    return


@app.cell
def _(KT0, fom_oscillator, go, leeson_pn, mo, np, pn_F, pn_Pdc, pn_Q,
      pn_corner, pn_f0, pn_from_isf, pn_topology):
    _df = np.logspace(2, 8, 600)
    _f0 = pn_f0.value * 1e9
    _Q = pn_Q.value
    _Pdc_W = pn_Pdc.value * 1e-3
    _Psig = 0.5 * _Pdc_W                    # ~50% drain efficiency assumption

    # Leeson curve
    _L_leeson = leeson_pn(_df, pn_F.value, _Psig, _Q, _f0, pn_corner.value)

    # ISF curve: pick c0/c_n by topology, target same far-out floor as Leeson
    if pn_topology.value.startswith("Cross"):
        _c0 = 0.0
        _cn = np.array([1.0, 0.0, 0.15])
    else:
        _c0 = 0.45
        _cn = np.array([0.9, 0.4, 0.15])

    # Calibrate i_n_white so the white-noise floor of ISF model matches Leeson
    _gamma2_rms = 0.25 * _c0 ** 2 + 0.5 * float(np.sum(_cn ** 2))
    # Compare 2FkT/Psig (Leeson floor PSD) to in_white * gamma2_rms / (2 q_max^2 * (omega_floor)^2)
    # Choose q_max so that L floor matches at Δf = f0/2Q
    _q_max = 1e-12
    _omega_floor = 2 * np.pi * (_f0 / (2 * _Q))
    _S_floor_leeson = 2.0 * pn_F.value * KT0 / _Psig  # 1/f^2 region's PSD constant scaling
    _i_n_white = _S_floor_leeson * 2.0 * _q_max ** 2 * _omega_floor ** 2 / _gamma2_rms
    _L_isf = pn_from_isf(_df, _c0, _cn, _q_max, _i_n_white,
                         pn_corner.value, _f0)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_df, y=_L_leeson, mode="lines",
                              line=dict(color="#636EFA", width=2.2),
                              name="Leeson"))
    _fig.add_trace(go.Scatter(x=_df, y=_L_isf, mode="lines",
                              line=dict(color="#00CC96", width=2.2,
                                        dash="dash"),
                              name=f"ISF ({pn_topology.value})"))
    _fig.add_vline(x=_f0/(2*_Q), line=dict(color="#AB63FA", dash="dot"),
                   annotation_text="ω₀/2Q")
    _fig.add_vline(x=pn_corner.value, line=dict(color="#EF553B", dash="dot"),
                   annotation_text="device 1/f")
    _fig.update_layout(template="plotly_dark",
                       title=(f"f₀={pn_f0.value:.0f} GHz, Q={_Q:.0f}, "
                              f"P_DC={pn_Pdc.value:.1f} mW, "
                              f"F={10*np.log10(pn_F.value):.1f} dB"),
                       xaxis_title="Δf (Hz)", xaxis_type="log",
                       yaxis_title="ℒ(Δω) (dBc/Hz)",
                       height=440,
                       legend=dict(orientation="h", y=-0.2))
    mo.ui.plotly(_fig)

    # FOM at 1 MHz offset
    _idx_1MHz = int(np.argmin(np.abs(_df - 1e6)))
    fom_leeson = fom_oscillator(_L_leeson[_idx_1MHz], 1e6, _f0,
                                pn_Pdc.value)
    fom_isf = fom_oscillator(_L_isf[_idx_1MHz], 1e6, _f0, pn_Pdc.value)
    return fom_isf, fom_leeson


@app.cell
def _(fom_isf, fom_leeson, mo, pn_Q, pn_f0):
    mo.md(f"""
| Metric | Leeson | ISF |
|---|---|---|
| FOM @ 1 MHz offset | {fom_leeson:.1f} dB | {fom_isf:.1f} dB |
| Tank $Q$ | {pn_Q.value:.0f} | {pn_Q.value:.0f} |
| Carrier $f_0$ | {pn_f0.value:.0f} GHz | {pn_f0.value:.0f} GHz |

Switch the topology toggle to see the close-in difference: cross-coupled
($c_0 = 0$) **flattens the 1/$f^3$ region** to follow the white-noise
1/$f^2$ slope all the way down, while Colpitts retains a visible flicker
upturn close to the carrier. The white-noise floor is the same in both
because the calibration sets it from $2 F kT / P_{{\\text{{sig}}}}$.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Coupled oscillator arrays

    A single LC tank is rarely the end of the story at mmWave. Two
    architectural patterns recur:

    **Superharmonic injection locking.** A high-frequency reference
    oscillator at $2 f_0$ injects current into the common-mode node of an
    $f_0$ oscillator. The $f_0$ oscillator's even-mode susceptibility
    (its $c_2$ Fourier coefficient of $\Gamma$) couples the injection to
    its phase, locking the $f_0$ oscillator's phase to half the
    injected phase. The Adler equation governs the locking transient and
    bandwidth; full phase-plane dynamics are explored in the companion
    interactive app `interactive/vco_pulling.py`.

    Within the locking bandwidth, the slave's phase noise tracks the
    reference divided by $N^2$ where $N$ is the harmonic ratio
    (here $N = 2$). For an injection-locked frequency divider, the slave
    inherits the reference's close-in noise but its own white-noise floor
    far from the carrier.

    **N-element coupled arrays.** Couple $N$ identical oscillators
    symmetrically (resistive networks, transformer coupling, or shared
    injection). In the in-phase mode:

    $$
    \boxed{\;
    P_{\text{out}} \propto N^2,
    \qquad
    P_{\text{noise}} \propto N
    \;\Longrightarrow\;
    \mathcal{L}_{\text{array}}
    \;=\; \mathcal{L}_{\text{single}} - 10\log_{10} N
    \;}
    $$

    Phase noise improves by 10 log $N$ dB. The cost is the coupling
    network: extra metal area, additional loss paths, and noise from the
    coupling elements themselves. Optimal coupling strength minimises the
    sum of tank noise (improves with stronger coupling) and coupling
    noise (worsens with stronger coupling) — a typical optimum coupling
    coefficient is around $k \sim 0.1$.

    **Application: phased-array LO distribution.** A 28 GHz or 60 GHz
    phased array needs a coherent LO at every front-end. Two architectures
    dominate:

    1. **Centralised distribution.** One high-Q VCO at $f_0$ feeds a
       buffer tree to all front-ends. Long lines, skin-effect loss, and
       distribution-network phase noise.
    2. **Distributed injection-locking.** A central reference at $2f_0$
       (or $f_0/2$) feeds an injection-locked frequency divider (or
       multiplier) at each front-end. Each ILFD inherits the reference
       phase noise; the local $f_0$ oscillator only contributes outside
       its locking bandwidth. Superior far-out phase noise per element
       at the cost of layout complexity.

    Modern 60-77 GHz radar transmitters typically use option 2 with a
    central V-band reference and ILFDs at each antenna element.

    See `interactive/vco_pulling.py` for an interactive treatment of
    Adler equation dynamics, locking range, and pulling transients.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part IV — Wrap-up
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 15. Summary and bridge to notebook 07

    The logical chain through notebook 06:

    1. **Leeson** sets the spectral shape — three regions ($1/\Delta\omega^3$,
       $1/\Delta\omega^2$, floor) — but treats $F$, $Q_L$, and the 1/$f^3$
       corner as empirical fit parameters.
    2. **Limit-cycle analysis** (Van der Pol) shows geometrically why
       phase noise diffuses while amplitude noise is restored: the
       limit cycle is closed, so motion along its tangent has no
       restoring force.
    3. **Equipartition** sets the absolute noise floor: $kT$ stored in the
       tank, regardless of $L,C$. Phase noise scales as $kT\omega_0/
       (P_{\text{sig}} Q^2)$ — hence the $Q^2$ figure of merit.
    4. **Floquet theory** identifies the unit eigenvalue of the
       monodromy matrix as the algebraic origin of the persistent phase
       mode, with all other multipliers strictly inside the unit disk.
    5. **Adjoint sensitivity / ISF** projects noise onto the unit
       eigenvector. The ISF $\Gamma(\omega_0 t)$ is computable from the
       waveform; its Fourier coefficients $c_n$ map directly to which
       noise band downconverts to phase noise. $c_0 \ne 0$ → 1/$f^3$;
       $c_0 = 0$ → no 1/$f^3$.
    6. **mmWave application.** Tank $Q$ is the binding constraint
       ($\sqrt{f}$ skin effect, $f^2$ substrate eddy, varactor $1/(\omega
       C R_{\text{var}})$, SRF collapse). Cross-coupled differential
       topology zeros $c_0$ by half-wave symmetry; Colpitts trades 1/$f^3$
       for lower $\Gamma^2_{\text{rms}}$. Coupled arrays improve PN as
       $-10\log N$ at the cost of coupling-network complexity.

    **Bridge to notebook 07.** Notebook 04 (LNA) handled receive-path
    noise; notebook 05 (matching) handled passive interfaces and
    mixer cyclostationarity; notebook 06 (this one) handled local
    oscillator phase noise. Notebook 07 will close the transmit
    chain: PA design, large-signal nonlinearity ($P_{1\text{dB}}$,
    $\text{IIP}_3$, AM-PM conversion), efficiency (Class A/B, Doherty,
    outphasing), and the full mmWave transceiver budget that combines
    LNA noise, mixer images, VCO phase noise, and PA distortion into a
    single error-vector-magnitude requirement.

    **Concept dependency map for notebook 07:**

    ```
    06 §5  P_sig·Q²  ──►  07 PA bias-Q tradeoff at mmWave
    06 §9  ISF c₀    ──►  07 AM-PM conversion (waveform asymmetry → flicker)
    06 §10 Tank Q    ──►  07 PA matching network Q vs. efficiency
    06 §13 Topology  ──►  07 Class-AB/E/F duty-cycle waveform shaping
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Previous:** [05 — Matching Networks](05_matching_networks.py)  |
    **Next:** *07 — PA Design and Linearity (in preparation)*
    """)
    return


if __name__ == "__main__":
    app.run()
