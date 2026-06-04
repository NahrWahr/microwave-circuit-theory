# v1.2
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


    # -----------------------------------------------------------------------
    # Dynamical-systems helpers — Colpitts route to chaos (§10–§12)
    # Kennedy normalized model (TCAS-I 1994); validated to reproduce the
    # period-doubling cascade and a positive largest Lyapunov exponent.
    # -----------------------------------------------------------------------

    def colpitts_deriv(x1, x2, x3, g, Q, k):
        """Normalized Colpitts vector field (component-wise; scalar or array).

        State (x1, x2, x3) ~ (C1 voltage, C2 voltage, L current). The single
        nonlinearity n(x2) = exp(-x2) - 1 is the transistor's compressive
        driving-point characteristic — the same class as the §4 MOSFET square
        law. g is loop gain, Q the tank quality factor, k = C2/(C1+C2).
        """
        nl = np.expm1(-np.clip(x2, -50.0, 50.0))           # exp(-x2) - 1
        return ((g / (Q * (1.0 - k))) * (-nl + x3),
                (g / (Q * k)) * x3,
                -(Q * k * (1.0 - k) / g) * (x1 + x2) - x3 / Q)

    def colpitts_step(x1, x2, x3, g, Q, k, dt):
        """One RK4 step of the normalized Colpitts field."""
        a1, a2, a3 = colpitts_deriv(x1, x2, x3, g, Q, k)
        b1, b2, b3 = colpitts_deriv(x1 + .5*dt*a1, x2 + .5*dt*a2, x3 + .5*dt*a3, g, Q, k)
        c1, c2, c3 = colpitts_deriv(x1 + .5*dt*b1, x2 + .5*dt*b2, x3 + .5*dt*b3, g, Q, k)
        e1, e2, e3 = colpitts_deriv(x1 + dt*c1, x2 + dt*c2, x3 + dt*c3, g, Q, k)
        return (x1 + dt/6*(a1+2*b1+2*c1+e1),
                x2 + dt/6*(a2+2*b2+2*c2+e2),
                x3 + dt/6*(a3+2*b3+2*c3+e3))

    def colpitts_trajectory(g, Q=1.38, k=0.5, dt=0.012, nsteps=12000, trans=0.5):
        """Single post-transient trajectory for the live phase-portrait view."""
        x1 = x2 = x3 = 0.1
        nt = int(trans * nsteps)
        for _ in range(nt):
            x1, x2, x3 = colpitts_step(x1, x2, x3, g, Q, k, dt)
        n = nsteps - nt
        X1 = np.empty(n); X2 = np.empty(n); X3 = np.empty(n)
        for i in range(n):
            x1, x2, x3 = colpitts_step(x1, x2, x3, g, Q, k, dt)
            X1[i], X2[i], X3[i] = x1, x2, x3
        return X1, X2, X3

    def poincare_section(X1, X2, X3, axis=2, level=None, direction=1):
        """Crossings of the plane coord[axis]=level; returns the two in-plane coords.

        A limit cycle pierces the plane in a finite set of points; a strange
        attractor pierces it in a fractal set. This is the section that, stacked
        over the bifurcation parameter, becomes the bifurcation diagram.
        """
        arrs = (X1, X2, X3)
        a = arrs[axis]
        if level is None:
            level = float(np.median(a))
        s = a - level
        if direction >= 0:
            idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
        else:
            idx = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
        if idx.size == 0:
            return np.array([]), np.array([])
        frac = -s[idx] / (s[idx + 1] - s[idx])
        out = [arrs[i][idx] + frac * (arrs[i][idx + 1] - arrs[i][idx]) for i in range(3)]
        rest = [out[i] for i in range(3) if i != axis]
        return rest[0], rest[1]

    def colpitts_bifurcation(gvals, Q=1.38, k=0.5, dt=0.012, nsteps=20000, trans=0.5):
        """Bifurcation data: steady-state x1-maxima vs loop gain g.

        Vectorized across the g-ensemble. Each vertical slice is a 1-D Poincaré
        section (the ẋ1=0, ẍ1<0 surface). Returns (g_points, x1max_points).
        """
        g = np.asarray(gvals, float)
        x1 = np.full(g.size, 0.1); x2 = x1.copy(); x3 = x1.copy()
        nt = int(trans * nsteps)
        for _ in range(nt):
            x1, x2, x3 = colpitts_step(x1, x2, x3, g, Q, k, dt)
        p2 = p1 = None
        g_out, x_out = [], []
        for _ in range(nsteps - nt):
            x1, x2, x3 = colpitts_step(x1, x2, x3, g, Q, k, dt)
            if p1 is not None and p2 is not None:
                m = (p1 > p2) & (p1 > x1) & np.isfinite(p1)
                j = np.where(m)[0]
                g_out.append(g[j]); x_out.append(p1[j])
            p2, p1 = p1, x1
        return (np.concatenate(g_out) if g_out else np.array([]),
                np.concatenate(x_out) if x_out else np.array([]))

    def colpitts_lyapunov(gvals, Q=1.38, k=0.5, dt=0.012, nsteps=18000,
                          d0=1e-9, renorm=8, trans=0.4):
        """Largest Lyapunov exponent vs g (Benettin two-trajectory method).

        lambda1 > 0 is the operational definition of chaos. Vectorized across g.
        """
        g = np.asarray(gvals, float)
        x1 = np.full(g.size, 0.1); x2 = x1.copy(); x3 = x1.copy()
        nt = int(trans * nsteps)
        for _ in range(nt):
            x1, x2, x3 = colpitts_step(x1, x2, x3, g, Q, k, dt)
        y1, y2, y3 = x1 + d0, x2.copy(), x3.copy()
        acc = np.zeros(g.size); cnt = 0
        for i in range(nsteps - nt):
            x1, x2, x3 = colpitts_step(x1, x2, x3, g, Q, k, dt)
            y1, y2, y3 = colpitts_step(y1, y2, y3, g, Q, k, dt)
            if (i + 1) % renorm == 0:
                dx, dy, dz = y1 - x1, y2 - x2, y3 - x3
                d = np.sqrt(dx*dx + dy*dy + dz*dz)
                d = np.where(d > 0, d, np.nan)
                acc += np.log(d / d0); cnt += 1
                s = d0 / d
                y1, y2, y3 = x1 + dx*s, x2 + dy*s, x3 + dz*s
        return acc / (cnt * renorm * dt)


    return (
        KB, KT0, T0,
        colpitts_bifurcation, colpitts_deriv, colpitts_lyapunov,
        colpitts_step, colpitts_trajectory,
        fom_oscillator,
        inductor_Q,
        isf_from_harmonics, isf_rms_squared,
        leeson_pn,
        pn_from_isf,
        poincare_section,
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

    This notebook builds the theory of oscillator phase noise — and the
    nonlinear dynamics underneath it — from four progressively sharper
    viewpoints:

    1. **Leeson's linear closed-loop model** — the historical baseline; gives
       the spectral *shape* but treats noise factor, loaded $Q$, and 1/f
       upconversion as empirical fit parameters.
    2. **Floquet / ISF nonlinear theory** — derives those parameters from
       first principles by linearising about the periodic orbit and
       projecting noise onto the unit Floquet eigenvector.
    3. **Nonlinear-dynamics view** — treats the oscillator as a state-space
       flow: the device's reactance count fixes the dimension and hence which
       behaviours are reachable (limit cycle vs. chaos), the cycle is born at a
       Hopf bifurcation, phase diffusion sets a finite linewidth, and phase
       reduction unifies the ISF with injection locking.
    4. **mmWave application** — applies the ISF picture to tank-$Q$
       degradation, topology selection (cross-coupled vs. Colpitts), and
       coupled-array architectures used at 28-60 GHz.

    The thread connecting them all: the noise on the oscillator output is
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
    ## 4. From the square law to the van der Pol limit cycle

    Part I asserted that a real oscillator *must* be nonlinear (§1) without
    writing the nonlinearity down. This section derives the governing
    equation of the standard cross-coupled LC oscillator from the **MOSFET
    square law**, shows it is exactly the **van der Pol equation**, and uses
    **Liénard's theorem** to prove a unique steady oscillation exists.
    Everything in Part II (Floquet, ISF) then rests on a *derived* equation,
    not a postulated one.

    ### 4.1 The device nonlinearity [constitutive law]

    A long-channel MOSFET in saturation obeys

    $$ \boxed{\; I_D = \tfrac{1}{2}\mu_n C_{ox}\tfrac{W}{L}(V_{GS}-V_{TH})^2 \equiv \tfrac{k}{2}(V_{GS}-V_{TH})^2, \quad V_{GS}\ge V_{TH}. \;} $$

    with $k \equiv \mu_n C_{ox} W/L$ (A/V²) and $I_D = 0$ below threshold.
    This single quadratic relation is the only nonlinearity required. (At
    mmWave the devices are short-channel and partly velocity-saturated,
    softening the law toward $I_D \propto V_{ov}$; the *shape* — odd,
    compressive — survives, and §4.5 shows that shape is all the theorem
    needs.)

    ### 4.2 The differential pair is an odd, saturating transconductor

    Source-couple two identical devices to a tail current $I_{SS}$ and drive
    them differentially with $V_{id} = V_{GS1}-V_{GS2}$. The tail sums the
    currents while the square law sets each:

    $$ I_{D1}+I_{D2} = I_{SS}, \qquad I_{D1,2} = \tfrac{k}{2}V_{ov1,2}^2. $$

    From $\sqrt{I_{D1}}-\sqrt{I_{D2}} = \sqrt{k/2}\,V_{id}$, squaring and
    using the sum gives $2\sqrt{I_{D1}I_{D2}} = I_{SS}-\tfrac{k}{2}V_{id}^2$;
    then $\Delta I = (\sqrt{I_{D1}}-\sqrt{I_{D2}})(\sqrt{I_{D1}}+\sqrt{I_{D2}})$
    yields the exact large-signal transfer characteristic

    $$ \boxed{\; \Delta I(V_{id}) \equiv I_{D1}-I_{D2} = k\,V_{id}\sqrt{\tfrac{I_{SS}}{k}-\tfrac{V_{id}^2}{4}}, \quad |V_{id}| \le V_{id,\max}=\sqrt{\tfrac{2I_{SS}}{k}}. \;} $$

    Beyond $V_{id,\max}$ one device cuts off and the pair **hard-clips** to
    $\Delta I = \pm I_{SS}$. Three properties matter:

    - **Odd:** $\Delta I(-V_{id}) = -\Delta I(V_{id})$ — no DC term, no even
      harmonics (the same symmetry that zeros $c_0$ in §17).
    - **Compressive:** the slope falls monotonically from the origin to zero
      at $V_{id,\max}$.
    - **Small-signal transconductance:** $g_m \equiv \partial \Delta I / \partial V_{id}\,\big|_0 = \sqrt{k\,I_{SS}}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 Cubic expansion and the cross-coupled negative resistance

    For swings well inside $V_{id,\max}$, expand the square root:

    $$ \Delta I = g_m V_{id}\sqrt{1-\tfrac{V_{id}^2}{4I_{SS}/k}} = g_m V_{id} - \beta V_{id}^3 + O(V_{id}^5), \qquad \beta = \tfrac{g_m k}{8 I_{SS}} = \tfrac{k^{3/2}}{8\sqrt{I_{SS}}} > 0. $$

    A positive odd cubic: a linear transconductance that **compresses** as
    the swing grows. Now close the loop. Cross-couple the pair — each drain
    drives the opposite gate — across a differential tank, and take the
    half-circuit: one tank node (single-ended-equivalent $C$, $L$, $R_p$ to
    the common-mode virtual ground) carries half the differential current,
    $\Delta I/2$, and swings to $v/2$, with $v$ the differential tank
    voltage. The pair sources current *in phase* with the voltage — a
    **negative resistance**. Working the push–pull bookkeeping, the
    small-signal differential resistance presented to the tank is

    $$ \boxed{\; R_{\text{dev}} = -\,\frac{2}{g_m}, \;} $$

    matching §17. Equivalently the pair injects a differential current
    $i_{\text{act}}(v) = g_m v - \beta v^3$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.4 The tank equation is van der Pol

    Apply KCL at the half-circuit node (capacitor + inductor + loss =
    injected signal current), with node voltage $v/2$ and injected current
    $\Delta I/2 = \tfrac12(g_m v - \beta v^3)$. The factors of two cancel:

    $$ g_m v - \beta v^3 = C\dot v + \frac{v}{R_p} + \frac{1}{L}\!\int v\,dt. $$

    Differentiate once to clear the integral and collect $\dot v$:

    $$ \boxed{\; C\ddot v + \big[(\tfrac{1}{R_p}-g_m) + 3\beta v^2\big]\dot v + \frac{v}{L} = 0. \;} $$

    Divide by $C$, set $\omega_0^2 = 1/(LC)$, and normalise the amplitude by
    $V_a = \sqrt{(g_m-1/R_p)/3\beta}$ through $x = v/V_a$:

    $$ \boxed{\; \ddot x - \varepsilon(1-x^2)\dot x + \omega_0^2 x = 0, \qquad \varepsilon = \frac{g_m - 1/R_p}{C}. \;} \quad\text{[van der Pol]} $$

    Van der Pol — usually asserted as canonical — is here a derived result.
    **Start-up** needs $\varepsilon>0 \Leftrightarrow g_m > 1/R_p$ — the §17
    condition, here a consequence of the sign of the linear damping. The
    **steady amplitude** is $x_{\text{ss}}=2$ (to leading order in
    $\varepsilon$), i.e. $v_{\text{ss}}=2V_a$ — the dimensionless
    $A_{\text{ss}}=2$ used in §4.7, now with physical units. Finally, the
    **nonlinearity strength** relative to resonance is set by the tank
    quality factor:

    $$ \boxed{\; \frac{\varepsilon}{\omega_0} = \frac{g_m-1/R_p}{\omega_0 C} = \frac{g_m R_p - 1}{Q}, \qquad Q = \omega_0 R_p C = \frac{R_p}{\omega_0 L}. \;} $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.5 Why a steady oscillation exists: Liénard's theorem

    Van der Pol is a special case of the **Liénard equation** [definition]
    $\ddot x + f(x)\dot x + g(x)=0$ with $f(x)=\varepsilon(x^2-1)$ and
    $g(x)=\omega_0^2 x$. The **Liénard transformation** $F(x)=\int_0^x f\,ds$,
    $\,y=\dot x + F(x)$ converts it to the planar system

    $$ \dot x = y - F(x), \qquad \dot y = -g(x), \qquad F(x) = \varepsilon\big(\tfrac{x^3}{3}-x\big). $$

    **Liénard's theorem (1928)** [theorem]. If $f,g\in C^1$ with $f$ even,
    $g$ odd, $g(x)>0$ for $x>0$, and $F$ has a single positive root at $x=a$,
    is negative on $(0,a)$, and rises monotonically to $+\infty$ on
    $(a,\infty)$, then the system has **exactly one limit cycle, and it is
    globally asymptotically stable**.

    *Verification.* $f=\varepsilon(x^2-1)$ is even; $g=\omega_0^2 x$ is odd
    and positive for $x>0$; $F=\tfrac{\varepsilon}{3}x(x^2-3)$ has its single
    positive root at $a=\sqrt3$, is negative on $(0,\sqrt3)$, and rises
    monotonically thereafter ($F'=f>0$ for $x>1$). All hypotheses hold, so a
    unique, globally attracting limit cycle exists. **This is the precise
    sense in which the oscillator is "solvable": a periodic steady state
    provably exists, is unique, and is reached from almost any initial
    condition** — the circuit self-starts from noise and converges to one
    waveform regardless of the turn-on transient. (Van der Pol has no
    elementary closed-form solution; Liénard guarantees the orbit without
    one.)

    *Proof sketch.* Define $E = \tfrac12 y^2 + G(x)$ with $G(x)=\int_0^x g$.
    Along trajectories $\dot E = y\dot y + g\dot x = -g(x)F(x)$. Since
    $g(x)>0$ for $x>0$ while $F<0$ on $(0,a)$ and $F>0$ beyond, small orbits
    gain energy and large orbits lose it — the negative-resistance /
    dissipation balance, now exact. **Existence:** the origin is the only
    equilibrium (an unstable focus for $\varepsilon>0$); build an annulus
    whose inner boundary trajectories cross outward and whose outer boundary
    they cross inward (using the *cycle-averaged* $\oint\dot E\,dt$ for large
    orbits), and Poincaré–Bendixson forces a periodic orbit inside.
    **Uniqueness** is the standard comparison argument: the energy gain per
    revolution $\oint\dot E\,dt$ is strictly monotone in orbit amplitude
    (monotonicity of $F$ for $x>a$), so the closure condition
    $\oint\dot E\,dt=0$ holds at one amplitude only.

    The payoff is generality. The cubic was a Taylor truncation, but the
    hypotheses hold for *any* odd, compressive, saturating transconductor —
    including the exact $\Delta I(V_{id})$ with its hard clip. **Liénard is
    what licenses modelling the cross-coupled oscillator as van der Pol:**
    the exact device curve and the cubic share the one feature the theorem
    needs, so both possess the same unique limit cycle.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.6 Quasi-sinusoidal vs. relaxation: the role of Q

    The single ratio $\varepsilon/\omega_0 = (g_m R_p - 1)/Q$ sets the shape
    of the limit cycle. For $\varepsilon/\omega_0 \ll 1$ the cycle is nearly
    circular and the output is a clean sinusoid at $\approx\omega_0$; for
    $\varepsilon/\omega_0 \gtrsim 1$ the trajectory clings to the branches
    $y=F(x)$ and jumps between them — sharp, non-sinusoidal **relaxation**
    edges.

    The point for circuit design: a cross-coupled LC oscillator lives in the
    **quasi-sinusoidal** regime *regardless of how hard the pair switches*.
    Even when the differential current is fully steered into a square wave
    (the clip of §4.2), the high-$Q$ tank filters it back to a sinusoidal
    voltage. With mmWave values $g_m R_p\approx 3$, $Q\approx 10$ (§15),
    $\varepsilon/\omega_0\approx 0.2$ — firmly quasi-sinusoidal. **Relaxation
    requires low $Q$** — the absence of a sharp resonant tank — which is the
    regime of ring and RC oscillators. Those have no $\omega_0^2 x$ restoring
    term, are *not* Liénard systems, and lie outside this derivation.

    Two axes are easy to conflate; they are independent:

    | Axis | Set by | Cross-coupled LC |
    |------|--------|------------------|
    | Waveform shape (sinusoid ↔ relaxation) | $\varepsilon/\omega_0 = (g_m R_p-1)/Q$ | quasi-sinusoidal (high $Q$) |
    | Amplitude limit (current- ↔ voltage-limited) | swing vs. supply rail | either — both stay quasi-sinusoidal |

    The interactive below builds the chain end to end: device curve →
    cubic + clip → Liénard $F(x)$ → limit cycle.
    """)
    return


@app.cell
def _(mo):
    iss_slider = mo.ui.slider(0.5, 5.0, step=0.1, value=2.0,
                              label="I_SS (mA)", show_value=True)
    kdev_slider = mo.ui.slider(2.0, 40.0, step=1.0, value=10.0,
                               label="k (mA/V²)", show_value=True)
    rp_slider = mo.ui.slider(100.0, 800.0, step=10.0, value=300.0,
                             label="R_p (Ω)", show_value=True)
    Qd_slider = mo.ui.slider(2.0, 40.0, step=1.0, value=12.0,
                             label="Tank Q", show_value=True)
    mo.md("**Interactive 0 — From device curve to limit cycle**")
    return Qd_slider, iss_slider, kdev_slider, rp_slider


@app.cell
def _(Qd_slider, iss_slider, kdev_slider, mo, rp_slider):
    mo.hstack([iss_slider, kdev_slider, rp_slider, Qd_slider], gap="2rem")
    return


@app.cell
def _(Qd_slider, go, iss_slider, kdev_slider, make_subplots, mo, np, rp_slider,
      vdp_simulate):
    _Iss = iss_slider.value * 1e-3
    _k = kdev_slider.value * 1e-3
    _Rp = rp_slider.value
    _Q = Qd_slider.value
    _gm = np.sqrt(_k * _Iss)
    _gmRp = _gm * _Rp
    _beta = _gm * _k / (8.0 * _Iss)
    _Vmax = np.sqrt(2.0 * _Iss / _k)

    # Panel 1 — transfer characteristic: exact (clipped), cubic, linear
    _Vid = np.linspace(-1.5 * _Vmax, 1.5 * _Vmax, 600)
    _inside = np.abs(_Vid) <= _Vmax
    _arg = np.clip(_Iss / _k - _Vid ** 2 / 4.0, 0.0, None)
    _dI_exact = np.where(_inside, _k * _Vid * np.sqrt(_arg),
                         np.sign(_Vid) * _Iss)
    _dI_cubic = _gm * _Vid - _beta * _Vid ** 3
    _dI_lin = _gm * _Vid

    # van der Pol parameters (normalised omega0 = 1)
    _eps_w0 = (_gmRp - 1.0) / _Q
    _omega0 = 1.0
    _eps = _eps_w0 * _omega0

    # Panel 2 — Liénard F(x) and damping f(x)
    _x = np.linspace(-3.0, 3.0, 400)
    _F = _eps * (_x ** 3 / 3.0 - _x)
    _fd = _eps * (_x ** 2 - 1.0)

    # Panel 3 — limit cycle (start on the cycle so small-eps cases settle)
    _osc = _gmRp > 1.0
    if _osc:
        _t, _xx, _vv = vdp_simulate(max(_eps, 1e-3), _omega0, 2.0, 0.0,
                                    50.0, 0.01)
        _n0 = int(len(_xx) * 0.5)
        _lc_x, _lc_y = _xx[_n0:], _vv[_n0:] / _omega0
    else:
        _lc_x, _lc_y = np.array([0.0]), np.array([0.0])

    _fig = make_subplots(rows=1, cols=3, column_widths=[0.36, 0.30, 0.34],
                         subplot_titles=("ΔI(V_id): exact vs cubic",
                                         "Liénard F(x) and damping f(x)",
                                         "Limit cycle (x, ẋ/ω₀)"))
    _fig.add_trace(go.Scatter(x=_Vid * 1e3, y=_dI_exact * 1e3, mode="lines",
                              name="exact (clipped)",
                              line=dict(color="#00CC96", width=2.5)),
                   row=1, col=1)
    _fig.add_trace(go.Scatter(x=_Vid * 1e3, y=_dI_cubic * 1e3, mode="lines",
                              name="cubic",
                              line=dict(color="#EF553B", width=1.8,
                                        dash="dash")),
                   row=1, col=1)
    _fig.add_trace(go.Scatter(x=_Vid * 1e3, y=_dI_lin * 1e3, mode="lines",
                              name="linear g_m",
                              line=dict(color="#AB63FA", width=1.2,
                                        dash="dot")),
                   row=1, col=1)
    _fig.update_xaxes(title_text="V_id (mV)", row=1, col=1)
    _fig.update_yaxes(title_text="ΔI (mA)",
                      range=[-1.3 * _Iss * 1e3, 1.3 * _Iss * 1e3],
                      row=1, col=1)

    _fig.add_trace(go.Scatter(x=_x, y=_F, mode="lines", name="F(x)",
                              line=dict(color="#636EFA", width=2.5)),
                   row=1, col=2)
    _fig.add_trace(go.Scatter(x=_x, y=_fd, mode="lines", name="f(x)=ε(x²−1)",
                              line=dict(color="#FFA15A", width=1.8,
                                        dash="dash")),
                   row=1, col=2)
    _fig.add_hline(y=0.0, line=dict(color="white", width=1, dash="dot"),
                   row=1, col=2)
    _fig.update_xaxes(title_text="x", row=1, col=2)
    _fig.update_yaxes(title_text="F(x), f(x)", row=1, col=2)

    if _osc:
        _fig.add_trace(go.Scatter(x=_lc_x, y=_lc_y, mode="lines",
                                  showlegend=False,
                                  line=dict(color="#FFD700", width=2)),
                       row=1, col=3)
    _fig.update_xaxes(title_text="x", row=1, col=3)
    _fig.update_yaxes(title_text="ẋ/ω₀", row=1, col=3)

    _regime = ("quasi-sinusoidal" if _eps_w0 < 0.5
               else "weakly nonlinear" if _eps_w0 < 1.5 else "relaxation")
    _Va = np.sqrt(max(_gm - 1.0 / _Rp, 0.0) / (3.0 * _beta))
    _status = (f"g_m={_gm*1e3:.2f} mS · g_mR_p={_gmRp:.2f} "
               f"({'oscillates' if _osc else 'no start-up (g_mR_p<1)'}) · "
               f"ε/ω₀={_eps_w0:.2f} ({_regime}) · V_a={_Va*1e3:.0f} mV")
    _fig.update_layout(template="plotly_dark", height=400, title=_status,
                       legend=dict(orientation="h", y=-0.25))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.7 Amplitude and phase: the two responses to a perturbation

    The limit cycle of §4.4–4.5 is the steady oscillation itself.
    Phase-noise theory turns on a single question about it: when a noise
    impulse knocks the state off the orbit, what survives? Recall that the
    damping term $-\varepsilon(1-x^2)\dot{x}$ is *negative* (energy
    injection) for $|x|<1$ and *positive* (dissipation) for $|x|>1$ — the
    mechanism that pulls any nearby trajectory back onto the cycle (radius
    $A_{\text{ss}} = 2$ in the $(x, \dot{x}/\omega_0)$ plane).

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
    mmWave, where $Q$ collapses (§15), this $Q^2$ dependence is the
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
    waveform. This is the same cyclostationary structure defined in
    notebook 04 §3.8 and applied to mixers/samplers in §4.9. The notebook 06
    application of it just produces a different number — phase noise instead
    of mixer SSB noise figure.

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
    the topology comparison in §17-18.

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

    # Part III — Nonlinear dynamics: dimension, bifurcations, chaos, synchronization

    Parts I and II are a *local* theory. Leeson linearises the loop; Floquet
    and the ISF linearise the vector field about the periodic orbit and ask
    how small noise perturbs it. Neither asks where the orbit *came from*,
    whether it is the *only* one, or whether it can *break*. Part III is the
    global nonlinear theory that answers those questions, organised around a
    single object: the active device as a nonlinear **two-port** driving a
    **state vector** through an ODE. The dimension of that state vector
    decides — before any detailed analysis — which qualitative behaviours a
    topology can exhibit at all (§10). We then watch the limit cycle be *born*
    as a parameter crosses a threshold (Hopf bifurcation, §11), be *destroyed*
    into chaos when the dimension allows it (§12), see how the same zero-mode
    that carries phase noise makes the phase *diffuse* (§13), and how one
    oscillator *locks* to another (§14).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. The oscillator as a two-port: the state vector and its dimension

    ### 10.1 The active device as a nonlinear two-port

    Every oscillator factors into an **active device** that supplies energy
    and an **embedding network** (tank, feedback divider, bias) that selects
    the frequency and closes the loop. Model the device as a two-port. For the
    ideal transconductor of §4 the constitutive law is *memoryless*,
    $i_{\text{out}} = f(v_{\text{in}})$ — the MOSFET square law, or the bipolar
    exponential of §12 — and the embedding network is linear and reactive.

    Small-signal, the device presents an *amplitude-dependent* admittance
    $Y_{\text{dev}}(A,\omega)$ across the embedding admittance
    $Y_{\text{emb}}(\omega)$, and a steady oscillation is their balance:

    $$\boxed{\,Y_{\text{dev}}(A,\omega_0) + Y_{\text{emb}}(\omega_0) = 0\,}\qquad\text{[oscillation condition]}$$

    a complex equation whose two real parts fix two unknowns. The
    **conductance balance** ($\operatorname{Re}$) sets the amplitude $A$: the
    device's negative conductance, which *compresses* as $A$ grows, must
    exactly cancel the embedding loss. The **susceptance balance**
    ($\operatorname{Im}$) sets the frequency $\omega_0$. This is the
    frequency-domain (Kurokawa / Barkhausen) view. At start-up ($A\to0$) the
    small-signal $-G_{\text{dev}}$ must *over*-cancel the loss; §17 derives
    $-2/g_m$ for the cross-coupled pair and §18 derives
    $-g_m/(\omega_0^2 C_1 C_2)$ for Colpitts. The time-domain counterpart of
    the same physics is a state-space ODE — and that is where *dimension*
    enters.

    ### 10.2 The state vector and its dimension

    The instantaneous state of a circuit is exactly the energy stored in its
    reactive elements: one coordinate per independent **capacitor voltage** and
    **inductor current**. A memoryless device adds *no* state — its law
    $i=f(v)$ is an algebraic constraint, not a differential one. Hence the
    autonomous field of §6, $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, has

    $$\boxed{\,n \;=\; \#\{\text{independent } C\} \;+\; \#\{\text{independent } L\}\,}\qquad\text{[state dimension]}$$

    assembled from KCL/KVL plus the device law. (§6 wrote
    $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}) + B\mathbf{u}$ abstractly for the
    noise analysis; here $\mathbf{x}$ and $n$ are pinned to the physical
    reactances of the device's two-port embedding.) A single differential
    $LC$ tank gives $n=2$; two divider capacitors plus a tank inductor give
    $n=3$. That integer is not bookkeeping — it *caps* the dynamics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 10.3 What each dimension permits

    A continuous autonomous flow cannot let trajectories cross — solutions are
    unique. That single fact, dimension by dimension, bounds the
    $t\to\infty$ behaviour:

    - **$n=1$.** $\dot x = f(x)$ is monotone between its zeros; the state can
      only crawl to a fixed point. No oscillation.
    - **$n=2$ (planar).** Fixed points and limit cycles — and nothing else.

    **Poincaré–Bendixson theorem** *[theorem].* For a $C^1$ planar system, a
    bounded forward trajectory whose $\omega$-limit set contains no equilibrium
    converges to a periodic orbit.

    The reason is topological. A closed orbit cuts the plane into an inside and
    an outside (Jordan curve theorem); because trajectories cannot cross, a
    bounded orbit is *trapped* — it has nowhere to accumulate but onto a cycle.
    There is no room in the plane for the **stretch-and-fold** that aperiodic
    motion requires.

    - **$n\ge3$.** The third coordinate is exactly the room the folded sheets
      need to pass over one another without intersecting. Quasiperiodic tori
      and **chaos** (sensitive dependence; a positive Lyapunov exponent, §12)
      become *possible* — not guaranteed, merely no longer forbidden.

    | Topology | Independent reactances | $n$ | Reachable long-term dynamics |
    |---|---|---|---|
    | Cross-coupled $LC$ (§17) | one differential $L$, one $C$ | $2$ | fixed point or limit cycle — **chaos provably impossible** |
    | Colpitts (§18) | $C_1,\;C_2,\;L$ | $3$ | limit cycle; at high gain, period-doubling to chaos |
    | Ring, $N$ stages | $N$ node capacitances | $N$ | $N$-dimensional; multi-phase and relaxation modes |

    This is the structural reason a cross-coupled pair — *however hard it
    switches* — stays a clean periodic oscillator: its van der Pol reduction
    (§4) lives in the plane, where Poincaré–Bendixson forbids anything worse
    than a limit cycle. The Colpitts carries one extra reactive degree of
    freedom (the second divider capacitor), and that third dimension is
    precisely what lets it period-double into chaos (§12). The extra capacitor
    is not a parasitic detail; it is the door to a qualitatively different
    dynamics.

    **A caveat on idealisation.** Real cross-coupled oscillators carry
    parasitics — tail-node, varactor, and gate capacitances — that formally
    raise $n$ above two, so "no chaos" is exact only for the idealised
    two-state model. What rescues it in practice: a well-designed tank keeps
    those parasitics as *fast, strongly damped* modes (Floquet multipliers
    deep inside the unit disk, §7) that contract away within a cycle and never
    act as independent dimensions. The 2-D van der Pol picture is accurate
    *because* the design forces the extra dimensions to be inert.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Hopf bifurcation: the birth of the limit cycle

    §4 produced a limit cycle and §10 guaranteed (in the plane) that it is the
    only attractor. Neither said *how* it appears as the designer turns up the
    gain. That event has a universal name.

    ### 11.1 Start-up is an eigenvalue crossing

    Linearise the field about the quiescent equilibrium (oscillator off). For
    the van der Pol tank of §4.4, $\ddot x - \varepsilon\dot x + \omega_0^2 x = 0$
    about $x=0$, the characteristic roots are

    $$ s = \tfrac{\varepsilon}{2} \pm j\sqrt{\omega_0^2 - \varepsilon^2/4} \;\approx\; \tfrac{\varepsilon}{2} \pm j\omega_0. $$

    A complex-conjugate pair sits at real part
    $\alpha = \varepsilon/2 = (g_m - 1/R_p)/(2C)$. As the gain rises the pair
    drifts rightward and crosses the imaginary axis exactly at $g_m = 1/R_p$ —
    the §17 start-up condition, read geometrically: **two poles cross into the
    right half-plane at $\pm j\omega_0$.**

    $$\boxed{\;\alpha(\mu_c)=0,\quad \omega(\mu_c)=\omega_0\neq0,\quad \alpha'(\mu_c)>0\;}\qquad\text{[Hopf conditions]}$$

    ### 11.2 The normal form is Stuart–Landau

    Near a Hopf point every system collapses to one universal equation, by two
    reductions. A **center-manifold** reduction discards the fast contracting
    modes — legitimate because at onset only the crossing pair is marginal and
    everything else is strongly damped (the §10.3 inertness again) — leaving a
    2-D flow. A **normal-form** reduction (near-identity coordinate changes that
    delete every *non-resonant* nonlinear term) leaves, at cubic order, the one
    term that cannot be removed:

    $$\boxed{\;\dot A = (\alpha + j\omega)\,A - \beta\,|A|^2 A\;}\qquad\text{[Hopf normal form / Stuart–Landau]}$$

    with $A\in\mathbb{C}$ the slowly-varying envelope and $\beta$ the cubic
    coefficient. Writing $A = R\,e^{j\phi}$ splits it:

    $$ \dot R = \alpha R - (\operatorname{Re}\beta)\,R^3, \qquad \dot\phi = \omega - (\operatorname{Im}\beta)\,R^2. $$

    The amplitude has a restoring term and settles; the phase has *none* and
    drifts freely. This is exactly the amplitude-stable / phase-free split of
    §4.7 and the unit Floquet multiplier of §7 — here obtained as the generic
    structure of *any* Hopf, not a quirk of van der Pol.

    ### 11.3 Soft vs hard start-up

    The steady amplitude is

    $$\boxed{\;R_{ss} = \sqrt{\alpha/\operatorname{Re}\beta}\;\propto\;\sqrt{\mu-\mu_c}\;}\qquad\text{[supercritical amplitude law]}$$

    — the square-root growth that the §4.4 result
    $V_a=\sqrt{(g_m-1/R_p)/3\beta}$ already shows (there $\mu=g_m$,
    $\mu_c=1/R_p$, $\beta$ the device cubic of §4.3; the dimensionless
    $A_{ss}=2$ is the same cycle normalised). The *sign* of
    $\operatorname{Re}\beta$ — equivalently the **first Lyapunov coefficient**
    $\ell_1\propto-\operatorname{Re}\beta$ — splits two worlds:

    - **Supercritical ($\operatorname{Re}\beta>0$).** A small stable cycle
      grows continuously from zero as $\mu$ passes $\mu_c$. **Soft** start-up:
      oscillation builds gracefully out of noise. The well-designed regime.
    - **Subcritical ($\operatorname{Re}\beta<0$).** The cubic term is
      *destabilising*; the bifurcating cycle is unstable, and only a
      higher-order (quintic / hard-saturation) term bounds the amplitude. A
      stable large-amplitude cycle then coexists with the stable "off" state
      over a range of $\mu$ — **hysteresis** and **hard** excitation (needs a
      kick; once running, keeps running below $\mu_c$). Its dynamical cousin
      **squegging** — intermittent start–stop bursts — appears when a slow
      bias loop drags $\mu$ back and forth across a subcritical onset.

    Design reading: a current-limited oscillator with a slow tail-bias loop is
    the classic squegger; the cure is a guaranteed supercritical onset
    ($\operatorname{Re}\beta>0$) and an amplitude-control loop fast compared
    with the tank.
    """)
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Hopf bifurcation diagrams: steady amplitude vs control parameter μ.
    _mu = np.linspace(-0.4, 0.6, 400)
    _fig = make_subplots(rows=1, cols=2,
                         subplot_titles=("Supercritical (soft)",
                                         "Subcritical (hard, hysteretic)"))

    # --- Supercritical: Ṙ = μR - R³  →  R_ss = √μ for μ>0 ---
    _pos = _mu > 0
    _fig.add_trace(go.Scatter(x=_mu[_mu <= 0], y=0*_mu[_mu <= 0], mode="lines",
                              line=dict(color="#00CC96", width=3),
                              name="stable equilibrium"), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_mu[_pos], y=0*_mu[_pos], mode="lines",
                              line=dict(color="#EF553B", width=2, dash="dash"),
                              name="unstable equilibrium"), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_mu[_pos], y=np.sqrt(_mu[_pos]), mode="lines",
                              line=dict(color="#00CC96", width=3),
                              showlegend=False), row=1, col=1)

    # --- Subcritical: Ṙ = μR + R³ - R⁵  →  μ + R² - R⁴ = 0 ---
    _a, _b = 1.0, 1.0
    _disc = _a**2 + 4*_b*_mu                  # real branches where ≥ 0
    _ok = _disc >= 0
    _Rup2 = (_a + np.sqrt(np.where(_ok, _disc, np.nan))) / (2*_b)   # stable upper
    _Rlo2 = (_a - np.sqrt(np.where(_ok, _disc, np.nan))) / (2*_b)   # unstable lower
    _mu_sn = -_a**2/(4*_b)                     # saddle-node fold
    _fig.add_trace(go.Scatter(x=_mu[_mu <= 0], y=0*_mu[_mu <= 0], mode="lines",
                              line=dict(color="#00CC96", width=3),
                              showlegend=False), row=1, col=2)
    _fig.add_trace(go.Scatter(x=_mu[_mu > 0], y=0*_mu[_mu > 0], mode="lines",
                              line=dict(color="#EF553B", width=2, dash="dash"),
                              showlegend=False), row=1, col=2)
    _msk_up = _ok & (_Rup2 >= 0)
    _fig.add_trace(go.Scatter(x=_mu[_msk_up], y=np.sqrt(_Rup2[_msk_up]),
                              mode="lines", line=dict(color="#00CC96", width=3),
                              showlegend=False), row=1, col=2)
    _msk_lo = _ok & (_Rlo2 >= 0)
    _fig.add_trace(go.Scatter(x=_mu[_msk_lo], y=np.sqrt(_Rlo2[_msk_lo]),
                              mode="lines", line=dict(color="#EF553B", width=2,
                                                      dash="dash"),
                              showlegend=False), row=1, col=2)
    _fig.add_vrect(x0=_mu_sn, x1=0, fillcolor="rgba(255,215,0,0.12)",
                   line_width=0, row=1, col=2,
                   annotation_text="hysteresis", annotation_position="top")

    for _c in (1, 2):
        _fig.add_vline(x=0, line=dict(color="#AB63FA", dash="dot"), row=1, col=_c)
        _fig.update_xaxes(title_text="μ − μ_c", row=1, col=_c)
    _fig.update_yaxes(title_text="steady amplitude R_ss", row=1, col=1)
    _fig.update_layout(template="plotly_dark", height=380,
                       title="Hopf bifurcation: green = stable, red dashed = unstable",
                       legend=dict(orientation="h", y=-0.25))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Lyapunov exponents and the route to chaos

    §7 measured how perturbations grow over *one period* (Floquet
    multipliers). A chaotic orbit has no period, so we need the rate per unit
    *time*, averaged along the trajectory.

    ### 12.1 From Floquet multipliers to Lyapunov exponents

    Track an infinitesimal perturbation $\delta\mathbf{x}(t)$ along a
    trajectory. Its long-time exponential rate along the $i$-th direction is
    the **Lyapunov exponent**

    $$\boxed{\;\lambda_i = \lim_{t\to\infty}\frac{1}{t}\,\ln\frac{\lVert\delta\mathbf{x}_i(t)\rVert}{\lVert\delta\mathbf{x}_i(0)\rVert}\;}\qquad\text{[definition]}$$

    For a *periodic* orbit this is not new information: a Floquet multiplier
    $\mu_i$ acts once per period $T$, so $\lambda_i = \tfrac{1}{T}\ln|\mu_i|$.
    The §7 unit multiplier $\mu_1=1$ becomes the **zero exponent**
    $\lambda_1=0$ — the phase direction, neither growing nor decaying (the same
    marginal mode that diffuses phase noise, §13); the contracting amplitude
    multipliers $|\mu_i|<1$ become negative exponents. Lyapunov exponents just
    extend this rate to orbits that never close.

    ### 12.2 The spectrum is the universal classifier

    Order the exponents $\lambda_1\ge\lambda_2\ge\dots\ge\lambda_n$. Their
    *sign pattern* names the attractor — independent of circuit, device, or
    topology:

    | Attractor | Signature | Geometry |
    |---|---|---|
    | Stable equilibrium | $(-,-,\dots)$ | point |
    | Limit cycle | $(0,-,\dots)$ | closed loop |
    | Quasiperiodic ($k$-torus) | $(0,\dots,0,-,\dots)$ with $k$ zeros | torus $T^k$ |
    | **Chaos** | $(+,0,-,\dots)$ | strange attractor |

    Two invariants come for free. The single **zero** is mandatory for any
    non-equilibrium flow (time-translation along the orbit is always marginal —
    the Floquet unit multiplier). And for a *dissipative* circuit the sum
    $\sum_i\lambda_i = \langle\nabla\!\cdot\mathbf{f}\rangle < 0$: phase-space
    volume contracts, so a positive $\lambda_1$ *forces* a strongly negative
    partner to balance it. For the Colpitts this is exact —
    $\nabla\!\cdot\mathbf{f} = -1/Q$ identically, so $\sum_i\lambda_i = -1/Q$,
    and in the chaotic regime ($Q=1.38$) a measured $\lambda_1\approx+0.08$
    pins the third exponent at $\lambda_3\approx-0.80$. That is the
    **stretch-and-fold** of §10.3 made quantitative — expansion in one
    direction, net volume contraction — and it needs $n\ge3$. A positive largest exponent is the operational

    $$\boxed{\;\lambda_1 > 0 \;\Longleftrightarrow\; \text{chaos}\;}\qquad\text{[definition]}$$

    neighbouring trajectories diverge exponentially, so the state is
    unpredictable beyond a horizon $\sim 1/\lambda_1$ even though the dynamics
    are perfectly deterministic.

    ### 12.3 Period-doubling and the chaotic Colpitts

    Which oscillators can reach $\lambda_1>0$? By §10.3 not the planar
    cross-coupled pair — but the **Colpitts** ($n=3$) can. Its normalized
    state equations — the §18 Colpitts written in dimensionless form — are

    $$ \dot x_1 = \tfrac{g}{Q(1-k)}\big(x_3 - n(x_2)\big),\quad \dot x_2 = \tfrac{g}{Qk}\,x_3,\quad \dot x_3 = -\tfrac{Qk(1-k)}{g}(x_1+x_2)-\tfrac{x_3}{Q}, $$

    with one nonlinearity $n(x_2)=e^{-x_2}-1$ (the transistor's compressive
    driving-point law), loop gain $g$, tank quality $Q$, capacitive-divider
    ratio $k=C_2/(C_1+C_2)$. State $(x_1,x_2,x_3)$ tracks the two capacitor
    voltages and the inductor current — the three reactances of §10.2.

    Hold $Q=1.38$, $k=0.5$ and raise $g$. The limit cycle does not break all at
    once; it **period-doubles** — one loop becomes two, two become four, the
    cascade accumulating geometrically (successive doublings shrink by the
    universal **Feigenbaum** ratio $\delta\approx4.669$) until at a finite $g$
    the period is infinite and the orbit is chaotic. Beyond that, bands of
    chaos alternate with **periodic windows** where a low-period cycle briefly
    returns. The interactive below computes both diagnostics — the bifurcation
    diagram (steady $x_1$-maxima vs $g$) and the largest Lyapunov exponent
    $\lambda_1(g)$ — and they agree.

    **Universality — why the device law barely matters.** The period-doubling
    route and the constant $\delta$ depend only on the return map having a
    smooth quadratic-like fold, *not* on whether the device obeys the bipolar
    exponential or the §4 MOSFET square law. A MOSFET Colpitts cascades the
    same way; only the threshold values of $g$ shift. The exponential model is
    therefore not a detour from the square-law spine — it is the same
    universality class, and it is the **dimension count of §10.3**, not the
    constitutive detail, that made chaos reachable here and impossible for the
    cross-coupled pair.

    **Engineering note.** Chaos in a VCO is a *failure* mode, not a feature: it
    lives at high loop gain and strong nonlinearity — the corner a
    clean-spectrum design stays away from. The point of seeing it is to know
    the boundary, and to recognise its symptoms (a broadband, noise-like output
    spectrum; subharmonics at $f_0/2,\,f_0/4,\dots$) as *dynamics*, not as a
    broken measurement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Capstone — the Colpitts route to chaos.** Sweep the loop gain $g$. The
    live view integrates one trajectory (the 3-D attractor and the red dots
    where it pierces the Poincaré plane $x_3=\langle x_3\rangle$); press the
    button to compute the *global* bifurcation diagram and Lyapunov spectrum
    across $g$. Watch a clean limit cycle (one set of Poincaré dots,
    $\lambda_1\le0$) period-double and dissolve into a strange attractor
    ($\lambda_1>0$). The vertical purple line marks the slider's $g$.
    """)
    return


@app.cell
def _(mo):
    g_slider = mo.ui.slider(1.5, 6.0, step=0.05, value=3.0,
                            label="loop gain g", show_value=True)
    sweep_btn = mo.ui.run_button(label="▶ Compute bifurcation + Lyapunov sweep")
    mo.md("**Interactive — Colpitts bifurcation explorer**")
    return g_slider, sweep_btn


@app.cell
def _(g_slider, mo, sweep_btn):
    mo.hstack([g_slider, sweep_btn], gap="2rem", justify="start")
    return


@app.cell
def _(colpitts_trajectory, g_slider, go, make_subplots, mo, np, poincare_section):
    # Live single-trajectory view (cheap): 3-D attractor + Poincaré + x1(t).
    _g = g_slider.value
    _X1, _X2, _X3 = colpitts_trajectory(_g)
    _dt = 0.012
    _t = np.arange(_X1.size) * _dt
    _lvl = float(np.median(_X3))
    _px, _py = poincare_section(_X1, _X2, _X3, axis=2, level=_lvl, direction=1)

    # crude period label from distinct steady-state x1-maxima levels
    _int = _X1[1:-1]
    _mx = _int[(_int > _X1[:-2]) & (_int > _X1[2:])]
    if _mx.size:
        _tail = np.sort(_mx[-200:])
        _levels = 1 + int(np.sum(np.diff(_tail) > 0.05 * (np.ptp(_tail) + 1e-9)))
    else:
        _levels = 0
    _label = (f"≈ period-{_levels}" if 0 < _levels <= 6
              else "aperiodic / chaotic-looking")

    _fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44],
                         specs=[[{"type": "scene"}, {"type": "xy"}]],
                         subplot_titles=("Phase-space attractor",
                                         "x₁(t)"))
    _fig.add_trace(go.Scatter3d(x=_X1, y=_X2, z=_X3, mode="lines",
                                line=dict(width=2, color="#00CC96"),
                                showlegend=False), row=1, col=1)
    if _px.size:
        _fig.add_trace(go.Scatter3d(x=_px, y=_py, z=np.full(_px.size, _lvl),
                                    mode="markers",
                                    marker=dict(size=3, color="#EF553B"),
                                    name="Poincaré"), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_t, y=_X1, mode="lines",
                              line=dict(color="#FFD700", width=1.2),
                              showlegend=False), row=1, col=2)
    _fig.update_xaxes(title_text="t (normalized)", row=1, col=2)
    _fig.update_yaxes(title_text="x₁", row=1, col=2)
    _fig.update_layout(template="plotly_dark", height=440,
                       scene=dict(xaxis_title="x₁", yaxis_title="x₂",
                                  zaxis_title="x₃"),
                       title=f"Colpitts at g = {_g:.2f}  (Q=1.38, k=0.5): {_label}",
                       legend=dict(orientation="h", y=-0.08))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(colpitts_bifurcation, colpitts_lyapunov, np, sweep_btn):
    # Expensive global sweep — gated by the run button (out of the live path).
    if sweep_btn.value:
        bif_g, bif_x = colpitts_bifurcation(np.linspace(1.5, 6.0, 64))
        lyap_g = np.linspace(1.5, 6.0, 40)
        lyap_l = colpitts_lyapunov(lyap_g)
    else:
        bif_g = bif_x = lyap_g = lyap_l = None
    return bif_g, bif_x, lyap_g, lyap_l


@app.cell
def _(bif_g, bif_x, g_slider, go, lyap_g, lyap_l, make_subplots, mo):
    if bif_g is None:
        _out = mo.md("*Press **▶ Compute bifurcation + Lyapunov sweep** above "
                     "to run the global sweep (~7 s native; slower in-browser).*")
    else:
        _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.62, 0.38], vertical_spacing=0.09,
                             subplot_titles=("Bifurcation: steady x₁-maxima vs g",
                                             "Largest Lyapunov exponent λ₁(g)"))
        _fig.add_trace(go.Scattergl(x=bif_g, y=bif_x, mode="markers",
                                    marker=dict(size=2, color="#00CC96",
                                                opacity=0.45),
                                    showlegend=False), row=1, col=1)
        _fig.add_trace(go.Scatter(x=lyap_g, y=lyap_l, mode="lines",
                                  line=dict(color="#FFD700", width=2),
                                  showlegend=False), row=2, col=1)
        _fig.add_hline(y=0.0, line=dict(color="white", width=1, dash="dot"),
                       row=2, col=1)
        for _r in (1, 2):
            _fig.add_vline(x=g_slider.value,
                           line=dict(color="#AB63FA", width=1.5, dash="dash"),
                           row=_r, col=1)
        _fig.update_yaxes(title_text="x₁ maxima", row=1, col=1)
        _fig.update_yaxes(title_text="λ₁", row=2, col=1)
        _fig.update_xaxes(title_text="loop gain g", row=2, col=1)
        _fig.update_layout(template="plotly_dark", height=520,
                           title="Period-doubling cascade and λ₁  "
                                 "(green = orbit maxima, gold = λ₁)")
        _out = mo.ui.plotly(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Stochastic phase dynamics: Fokker–Planck and the Lorentzian

    §8–§9 computed how much a single noise impulse shifts the phase. Feeding
    *continuous* white noise into that phase makes the phase a random process —
    and pinning down its statistics resolves the paradox left open in §3:
    Leeson's spectrum diverges at the carrier.

    ### 13.1 The phase is Brownian motion

    From §8, white noise $i_n(t)$ drives the phase through the ISF,
    $\dot\phi = \Gamma(\omega_0 t)\,i_n(t)/q_{\max}$. The phase direction is
    the zero-Lyapunov / unit-Floquet mode (§7, §12.1): there is *no restoring
    force*, so $\phi$ never relaxes — it **accumulates**. Integrating white
    noise is a Wiener process, and averaging $\Gamma^2$ over a cycle makes the
    phase variance grow *linearly*:

    $$\boxed{\;\langle\Delta\phi^2(\tau)\rangle = 2D\,|\tau|,\qquad D = \frac{\Gamma^2_{\text{rms}}}{2\,q_{\max}^2}\,\overline{\left(\frac{i_n^2}{\Delta f}\right)}\;}\qquad\text{[phase diffusion]}$$

    with $D$ the **phase diffusion constant** — built from exactly the
    $\Gamma^2_{\text{rms}}$ and noise PSD that §9 already used. The
    Fokker–Planck equation for the phase density is pure diffusion,
    $\partial_t p = D\,\partial_\phi^2 p$: a spreading Gaussian, no drift in
    the co-rotating frame.

    ### 13.2 Two spectra, kept distinct

    Fourier-transforming $\phi$ directly gives the **phase PSD** of a Wiener
    process, $S_\phi(\Delta\omega) = 2D/\Delta\omega^2$, which *diverges* as
    $\Delta\omega\to0$. That divergence is the §3 Leeson pathology — infinite
    power at the carrier — and it is spurious, because $\phi$ is *not*
    stationary (its variance is unbounded), so $S_\phi$ is not what a spectrum
    analyser measures.

    The measured quantity is the PSD of the **output**
    $v(t)=A\cos(\omega_0 t+\phi(t))$, which *is* stationary. With $\phi$
    Gaussian, $\langle e^{j\Delta\phi}\rangle = e^{-\frac12\langle\Delta\phi^2\rangle}=e^{-D|\tau|}$,
    so $R_v(\tau) = \tfrac{A^2}{2}\cos(\omega_0\tau)\,e^{-D|\tau|}$, and its
    transform is a **Lorentzian**:

    $$\boxed{\;S_v(\omega) = \frac{A^2}{2}\,\frac{D/\pi}{(\omega-\omega_0)^2 + D^2}\;}\qquad\text{[output spectrum]}$$

    a line of **FWHM $= 2D$**, finite at the carrier, with total power
    $\int S_v\,d\omega = A^2/2$ *conserved*.

    ### 13.3 The reconciliation

    The two pictures are one physics in two regimes:

    - **Far from carrier ($\Delta\omega\gg D$):** the Lorentzian skirt is
      $S_v\approx \tfrac{A^2}{2}\tfrac{D/\pi}{\Delta\omega^2}\propto1/\Delta\omega^2$
      — exactly the §9 / Leeson region. $S_\phi$ is correct here.
    - **Close to carrier ($\Delta\omega\lesssim D$):** the Lorentzian
      *saturates* to a finite peak $\sim A^2/(2\pi D)$, while
      $S_\phi\propto1/\Delta\omega^2$ wrongly diverges. The divergence was an
      artifact of linearising $v\approx A\cos\omega_0 t - A\sin(\omega_0 t)\,\phi$
      (treating $\phi$ as small); the exact $\cos(\omega_0 t+\phi)$ keeps the
      power bounded.

    So the diffusion constant $D$ is the **cutoff** that regularises Leeson:
    the carrier is not an infinite spike but a finite-width line,
    $\Delta\omega_{\text{FWHM}}=2D$. This closes the §3 gap — Leeson's
    $1/\Delta\omega^2$ is the *tail* of a Lorentzian whose width is set,
    through $D$, by the same ISF projection §8 derived from Floquet theory.
    """)
    return


@app.cell
def _(go, mo, np):
    # Lorentzian output spectrum vs the 1/Δω² phase-PSD asymptote (D = 1).
    _D = 1.0
    _dw = np.logspace(-2.2, 2.0, 500)
    _lor = (_D / np.pi) / (_dw ** 2 + _D ** 2)        # exact output spectrum
    _asym = (_D / np.pi) / _dw ** 2                    # 1/Δω² (diverges)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_dw, y=_lor, mode="lines",
                              name="Lorentzian S_v (exact)",
                              line=dict(color="#00CC96", width=2.5)))
    _fig.add_trace(go.Scatter(x=_dw, y=_asym, mode="lines",
                              name="1/Δω² phase PSD (diverges)",
                              line=dict(color="#EF553B", width=1.8, dash="dash")))
    _fig.add_vline(x=_D, line=dict(color="#AB63FA", dash="dot"),
                   annotation_text="Δω = D (half-linewidth)",
                   annotation_position="top")
    _fig.update_layout(template="plotly_dark", height=400,
                       title="Output spectrum: Lorentzian flattens where the "
                             "1/Δω² picture diverges",
                       xaxis=dict(title="Δω / D", type="log"),
                       yaxis=dict(title="S(Δω)  (arb.)", type="log"),
                       legend=dict(orientation="h", y=-0.25))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Synchronization: isochrons, injection locking, and basins

    The phase-noise theory of §8 and the injection-locking of §19 look
    unrelated. **Phase reduction** shows they are the same construction.

    ### 14.1 Isochrons: phase off the cycle

    The ISF needed a phase defined *on* the orbit; synchronization needs phase
    defined *everywhere* in the basin. Extend it by **asymptotic phase**:
    assign to each point $\mathbf{x}$ the phase of the orbit it eventually
    merges with. Level sets of equal asymptotic phase are the **isochrons** —
    codimension-1 manifolds foliating the basin, chosen so that
    $\dot\phi=\omega_0$ *everywhere*, not just on the cycle. The ISF is exactly
    this geometry evaluated on the cycle:

    $$\boxed{\;\Gamma(\theta)\;\propto\;\nabla_{\mathbf{x}}\phi\big|_{\mathbf{x}_s(\theta)}\;}\qquad\text{[the ISF is the phase gradient]}$$

    — the §8 adjoint / left-Floquet eigenvector *is* $\nabla\phi$ on the orbit.
    This is **phase reduction** (Winfree–Kuramoto–Malkin): a weak perturbation
    $\mathbf{p}(t)$ collapses the full $n$-dimensional dynamics to one phase
    equation $\dot\phi = \omega_0 + \Gamma(\phi)\,p(t)$ — the common root of
    *both* the §8 phase-noise theory (random $\mathbf{p}$) and injection
    locking (periodic $\mathbf{p}$).

    ### 14.2 Injection locking is the Adler equation

    Inject a tone near $\omega_0$, keep the first ISF harmonic, set
    $\psi=\phi-\omega_{\text{inj}}t$, and average over a cycle:

    $$\boxed{\;\dot\psi = \Delta\omega - \omega_L\sin\psi,\qquad \Delta\omega=\omega_0-\omega_{\text{inj}},\;\; \omega_L\propto I_{\text{inj}}|c_1|\;}\qquad\text{[Adler equation]}$$

    A stable fixed point ($\sin\psi^*=\Delta\omega/\omega_L$) exists iff
    $|\Delta\omega| \le \omega_L$ (**the locking range**). At the edge the
    stable and unstable fixed points collide and annihilate — a **saddle-node
    on the invariant circle**. Inside, the oscillator locks; outside, $\psi$
    drifts and the output is a quasiperiodic beat. (This is the dynamics
    deferred from §19 to `interactive/vco_pulling.py`.)

    ### 14.3 Arnold tongues and the devil's staircase

    Sweep injection frequency *and* amplitude. In the
    $(\omega_{\text{inj}}/\omega_0,\,I_{\text{inj}})$ plane each rational ratio
    $p\!:\!q$ owns a wedge of locking — an **Arnold tongue** — anchored at the
    rational on the zero-amplitude axis and widening with drive. The $1\!:\!1$
    tongue is the Adler range; higher tongues are sub/superharmonic locking
    (the $2\!:\!1$ superharmonic ILFD of §19). On the critical line the locked
    ratio versus detuning is a **devil's staircase**, a plateau at every
    rational. Where tongues *overlap* at large drive, locking becomes
    multivalued and a route to chaos reopens — a second road to §12's strange
    attractor, distinct from period-doubling.

    ### 14.4 Basins and multistability

    Liénard (§4.5) gave the single planar oscillator a *globally unique* limit
    cycle. That uniqueness is special to $n=2$ with one active element; couple
    oscillators or raise $n$ and several attractors coexist, partitioning state
    space into **basins of attraction**. The basin a trajectory starts in — set
    by power-up transients and noise — decides the steady state. Three
    consequences a designer meets:

    - **Quadrature ambiguity.** Two coupled cores have *two* stable modes,
      $\pm90^\circ$. Which one the array wakes into is a basin question;
      breaking the symmetry (asymmetric coupling, a startup kick) is a real
      design task.
    - **Array and ring modes.** $N$ coupled oscillators (§19) or an $N$-stage
      ring support in-phase, anti-phase, and rotating modes at once; mode
      selection is again by basins.
    - **Cycle slips.** The locked state of §14.2 has a *finite* basin; a large
      enough perturbation ejects the oscillator from lock (a $2\pi$ phase
      slip), even though small perturbations are restored.

    The global certainty of Liénard is the exception, not the rule: dimension
    (§10) and coupling build the multistable landscape that basins organise.
    """)
    return


@app.cell
def _(go, mo, np):
    # Injection locking (Adler): average residual frequency vs detuning.
    _dw = np.linspace(-3.0, 3.0, 600)
    _fig = go.Figure()
    for _wL, _col in [(0.5, "#00CC96"), (1.0, "#636EFA"), (1.8, "#FFD700")]:
        _locked = np.abs(_dw) <= _wL
        _pull = np.where(_locked, 0.0,
                         np.sign(_dw) * np.sqrt(np.clip(_dw**2 - _wL**2, 0.0, None)))
        _fig.add_trace(go.Scatter(x=_dw, y=_pull, mode="lines",
                                  name=f"ω_L = {_wL}",
                                  line=dict(color=_col, width=2.5)))
    _fig.add_hline(y=0.0, line=dict(color="white", width=1, dash="dot"))
    _fig.update_layout(template="plotly_dark", height=400,
                       title="Injection locking (Adler): flat plateau = locked, "
                             "√-branch = beat note",
                       xaxis_title="detuning  Δω = ω₀ − ω_inj",
                       yaxis_title="⟨ψ̇⟩  (residual pull)",
                       legend=dict(orientation="h", y=-0.25))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part IV — mmWave oscillator considerations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 15. Tank Q at mmWave

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
    ## 16. The phase-noise figure of merit

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
    ## 17. Cross-coupled LC topology

    The standard mmWave VCO architecture is the differential cross-coupled
    pair (NMOS, PMOS, or complementary). The two transistors mutually
    drive each other's gates from the opposite drain, providing the
    negative resistance $-2/g_m$ across a differential tank.

    **Connection to §4.** This is the topology whose equation of motion §4
    derived from the square law: the pair's odd transconductance
    $\Delta I = g_m v - \beta v^3$ makes the differential tank obey van der
    Pol, with start-up at $g_m > 1/R_p$ and a Liénard-guaranteed unique
    limit cycle. The half-wave symmetry invoked below — $v(t+T/2)=-v(t)$,
    which zeros every even ISF coefficient — is the time-domain face of that
    same oddness of $\Delta I(V_{id})$.

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
    ## 18. Colpitts topology

    The Colpitts oscillator uses a single transistor and a capacitive
    voltage divider $C_1, C_2$ for feedback.

    **Negative resistance and start-up.** Looking into the divider, the
    transistor presents a series negative resistance $R_{\text{dev}} =
    -g_m/(\omega_0^2 C_1 C_2)$, which resonates the inductor against the
    series capacitance $C_s = C_1 C_2/(C_1+C_2)$ at $\omega_0^2 =
    (C_1+C_2)/(L C_1 C_2)$. The divider feeds back only a fraction
    $n = C_2/(C_1+C_2)$ of the tank voltage, so the loop gain is
    $g_m R_p\,n(1-n)$ and oscillation starts when

    $$ \boxed{\; g_m R_p > \frac{1}{n(1-n)} = \frac{(C_1+C_2)^2}{C_1 C_2}. \;} $$

    The minimum, $g_m R_p = 4$, occurs at equal capacitors ($n=\tfrac12$) —
    four times the cross-coupled requirement $g_m R_p > 1$. The penalty buys
    a single-device topology, useful when transistor count or DC headroom is
    constrained.

    **Why the waveform is asymmetric.** Unlike the differential pair, a
    single transistor is a *one-sided* nonlinearity: $i_D =
    \tfrac{k}{2}(V_{GS}-V_{TH})^2$ when $V_{GS}>V_{TH}$ and zero otherwise.
    In steady oscillation it conducts in a narrow pulse once per cycle
    (class-C), near the voltage extreme that forward-biases it. That
    conduction is *not* an odd function of the tank voltage — it carries a
    DC average and strong even harmonics — so the governing equation is
    Liénard-*like* but with a non-even $f$ and non-odd $F$: the clean van der
    Pol symmetry of §4 is broken. A limit cycle still exists
    (Poincaré–Bendixson on the pulsed system), but its asymmetry is exactly
    what produces the nonzero $c_0$ analysed next.

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
    | Startup $g_m$          | $g_m R_p > 1$         | $g_m R_p > (C_1{+}C_2)^2/(C_1 C_2)$ |
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
    ## 19. Coupled oscillator arrays

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

    # Part V — Wrap-up
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 20. Summary and bridge to notebook 07

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
    6. **Dimension and bifurcation.** The two-port state count $n$ caps the
       dynamics — Poincaré–Bendixson forbids chaos for the planar
       cross-coupled pair — and the limit cycle is born at a **Hopf
       bifurcation**, its $\sqrt{\mu-\mu_c}$ amplitude and phase/amplitude
       split following from the Stuart–Landau normal form.
    7. **Chaos and diffusion.** With $n\ge3$ the Colpitts period-doubles to
       chaos ($\lambda_1>0$); the same zero Lyapunov exponent that carries the
       phase makes it *diffuse*, turning Leeson's divergent $1/\Delta\omega^2$
       into a finite-width **Lorentzian** (FWHM $=2D$).
    8. **Synchronization.** Phase reduction unifies the ISF with injection
       locking (Adler equation, Arnold tongues); coupling and dimension create
       the multistability that **basins of attraction** organise.
    9. **mmWave application.** Tank $Q$ is the binding constraint
       ($\sqrt{f}$ skin effect, $f^2$ substrate eddy, varactor $1/(\omega
       C R_{\text{var}})$, SRF collapse). Cross-coupled differential
       topology zeros $c_0$ by half-wave symmetry; Colpitts trades 1/$f^3$
       for lower $\Gamma^2_{\text{rms}}$. Coupled arrays improve PN as
       $-10\log N$ at the cost of coupling-network complexity.

    **Bridge to notebook 07.** Notebook 04 handled receive-path noise — LNA
    parameters, cyclostationary processes (§3.8), and mixer/sampler noise
    (§4.9); notebook 05 handled passive interfaces and broadband matching;
    notebook 06 (this one) handled local oscillator phase noise. Notebook 07 will close the transmit
    chain: PA design, large-signal nonlinearity ($P_{1\text{dB}}$,
    $\text{IIP}_3$, AM-PM conversion), efficiency (Class A/B, Doherty,
    outphasing), and the full mmWave transceiver budget that combines
    LNA noise, mixer images, VCO phase noise, and PA distortion into a
    single error-vector-magnitude requirement.

    **Concept dependency map for notebook 07:**

    ```
    06 §5  P_sig·Q²  ──►  07 PA bias-Q tradeoff at mmWave
    06 §9  ISF c₀    ──►  07 AM-PM conversion (waveform asymmetry → flicker)
    06 §15 Tank Q    ──►  07 PA matching network Q vs. efficiency
    06 §18 Topology  ──►  07 Class-AB/E/F duty-cycle waveform shaping
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
