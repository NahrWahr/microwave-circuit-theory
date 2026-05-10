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
def _(mo):
    mo.md(r"""
    # 07 — Power Amplifiers, Linearity, and Volterra Series

    Once a low-noise amplifier and a clean local oscillator have produced a baseband signal, how do you transmit it back out at mmWave with enough output power and acceptable linearity? It develops the theory of power amplifier operation in three layers:
    
    1. **Bias-dependent waveform engineering** and conduction-angle classes.
    2. **Memoryless polynomial nonlinearity** yielding the canonical $P_{1\text{dB}}$ and IIP3 figures of merit.
    3. **Volterra series with memory**, derived rigorously by the harmonic-input probing method on a CMOS short-channel device.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part I — PA Fundamentals
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Operating point and load line

    In earlier notebooks, we analyzed microwave circuits through the lens of linear, small-signal approximations. We used two-port S-parameters (**Notebook 01**) to determine linear gain, defined available power and transducer power gain (**Notebook 02**), and designed matching networks to extract maximum small-signal power using conjugate matching (**Notebook 05**). We selected operating points on the Smith Chart based on stability and noise circles (**Notebook 03**), and we modeled the small-signal behavior of short-channel CMOS devices (**Notebook 04**). 

    Power amplifier (PA) design breaks this linear paradigm. The primary constraint is no longer small-signal matching, but **large-signal boundary conditions**. As we saw with the limit-cycle amplitude stabilization in oscillators (**Notebook 06**), a device cannot supply infinite voltage or current.

    ### First Principles of Large-Signal Operation

    Consider an idealized field-effect transistor (FET) driving a purely resistive load $R_L$. The physical boundaries of the device in the instantaneous $i_D-v_{DS}$ plane are:
    1. **Cutoff:** $i_D \ge 0$. The transistor cannot sink negative drain current.
    2. **Saturation Maximum:** $i_D \le I_{\max}$. The channel has a maximum carrier velocity and density.
    3. **Triode/Linear Region:** $v_{DS} \ge V_{\text{knee}}$. Below the knee voltage, the transistor drops out of saturation and acts as a voltage-controlled resistor, collapsing the transconductance.
    4. **Breakdown:** $v_{DS} \le V_{\text{BR}}$.

    The transistor is biased at a quiescent DC operating point $(V_{DQ}, I_{DQ})$, usually constrained by the supply voltage $V_{DQ} = V_{DD}$. When driven by an RF signal, the instantaneous state $(v_{DS}(t), i_D(t))$ traces an **AC load line** centered at the Q-point. The slope of this load line is exactly $-1/R_L$.

    ### The Optimum Load Resistance

    Unlike a small-signal LNA, where the load $Z_L$ is chosen as $S_{22}^*$ for conjugate matching, a power amplifier's load is chosen to maximize the area of the unclipped $v-i$ swing. 

    To extract maximum RF power without clipping, the AC load line must span from the maximum current point $(V_{\text{knee}}, I_{\max})$ to the maximum voltage point $(2V_{DD} - V_{\text{knee}}, 0)$. The load resistance that exactly achieves this trajectory is the **optimum large-signal load**, $R_{\text{opt}}$:

    $$
    R_{\text{opt}} = \frac{\Delta v_{DS}}{\Delta i_D} = \frac{(2V_{DD} - V_{\text{knee}}) - V_{\text{knee}}}{I_{\max} - 0} = \frac{V_{DD} - V_{\text{knee}}}{I_{\max} / 2}
    $$

    If $R_L > R_{\text{opt}}$, the voltage swing clips against $V_{\text{knee}}$ before the current reaches $I_{\max}$ (voltage-limited). If $R_L < R_{\text{opt}}$, the current swing clips against $I_{\max}$ before the voltage swing reaches its maximum (current-limited). In both cases, the output power is strictly less than what could be achieved with $R_{\text{opt}}$.

    ### Bias and Conduction Angle

    The quiescent gate bias $V_{GQ}$ relative to the threshold voltage $V_{\text{th}}$ determines the **conduction angle** $\theta$—the fraction of the RF cycle during which the transistor conducts current. Moving the bias point downward into cutoff clips the bottom of the current waveform, fundamentally trading gain and linearity for efficiency. We will derive this rigorously in Section 2.
    """)
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Static demonstration of load line and transfer curve clipping
    _vth = 0.4
    _vmax = 1.0
    _imax = 1.0
    
    # Grid for vgs
    _vgs_tc = np.linspace(0, 1.2, 200)
    _id_tc = np.maximum(0, (_vgs_tc - _vth) / (_vmax - _vth) * _imax)
    _id_tc = np.minimum(_id_tc, _imax)
    
    # Drive signal (Class AB example)
    _t = np.linspace(0, 4 * np.pi, 400)
    _vgq = 0.3
    _v_drive = 0.6
    _vgs_t = _vgq + _v_drive * np.cos(_t)
    _id_t = np.maximum(0, (_vgs_t - _vth) / (_vmax - _vth) * _imax)
    _id_t = np.minimum(_id_t, _imax)
    
    # Load line parameters
    _vdd = 1.0
    _vknee = 0.15
    _vds_ll = np.linspace(0, 2 * _vdd, 200)
    
    # Class A load line
    _id_ll_a = (_imax / 2) + (_vdd - _vds_ll) / ((_vdd - _vknee) / (_imax / 2))
    _id_ll_a = np.clip(_id_ll_a, 0, _imax)
    
    _fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Transfer Curve", "Drain Current Waveform", "AC Load Line"),
        column_widths=[0.33, 0.33, 0.33]
    )
    
    # Panel 1: Transfer curve
    _fig.add_trace(go.Scatter(x=_vgs_tc, y=_id_tc, mode="lines", name="i<sub>D</sub>(v<sub>GS</sub>)", line=dict(color="#636EFA", width=1.5)), row=1, col=1)
    _fig.add_vline(x=_vth, line=dict(color="#EF553B", dash="dot"), annotation_text="V<sub>th</sub>", annotation_position="top right", row=1, col=1)
    _fig.add_vline(x=_vgq, line=dict(color="#00CC96", dash="dot"), annotation_text="V<sub>GQ</sub>", annotation_position="top left", row=1, col=1)
    
    # Panel 2: Drain current vs time
    _fig.add_trace(go.Scatter(x=_t, y=_id_t, mode="lines", name="i<sub>D</sub>(t)", line=dict(color="#AB63FA", width=1.5)), row=1, col=2)
    _fig.add_hline(y=0, line=dict(color="#EF553B", dash="dot"), annotation_text="Cutoff", row=1, col=2)
    
    # Panel 3: Load line
    for _v_gs_step in np.linspace(_vth + 0.1, _vmax, 6):
        _i_ds_step = (_v_gs_step - _vth) / (_vmax - _vth) * _imax
        _v_ds_step = np.linspace(0, 2 * _vdd, 100)
        _i_ds_out = np.where(_v_ds_step < _vknee, _i_ds_step * (_v_ds_step / _vknee), _i_ds_step)
        _fig.add_trace(go.Scatter(x=_v_ds_step, y=_i_ds_out, mode="lines", showlegend=False, line=dict(color="#444444", width=1, dash="solid")), row=1, col=3)
    
    _fig.add_trace(go.Scatter(x=_vds_ll, y=_id_ll_a, mode="lines", name="Load Line", line=dict(color="#FFA15A", width=2.5)), row=1, col=3)
    _fig.add_scatter(x=[_vdd], y=[_imax / 2], mode="markers", marker=dict(size=12, color="#00CC96", symbol="x", line=dict(width=2)), name="Q-Point", row=1, col=3)
    
    _fig.update_layout(template="plotly_dark", height=420, showlegend=True,
                       legend=dict(orientation="h", y=-0.15))
    _fig.update_xaxes(title_text="Gate voltage v<sub>GS</sub> (V)", row=1, col=1)
    _fig.update_yaxes(title_text="Drain current i<sub>D</sub> (A)", row=1, col=1)
    _fig.update_xaxes(title_text="Time ωt (rad)", row=1, col=2)
    _fig.update_yaxes(title_text="Drain current i<sub>D</sub>(t) (A)", row=1, col=2)
    _fig.update_xaxes(title_text="Drain voltage v<sub>DS</sub> (V)", row=1, col=3)
    _fig.update_yaxes(title_text="Drain current i<sub>D</sub> (A)", row=1, col=3)
    
    fig_op_ll = mo.ui.plotly(_fig)
    fig_op_ll


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Conduction angle and Fourier decomposition

    **Problem Definition:** An active device is driven by a sinusoidal gate voltage $v_{GS}(t) = V_{GQ} + V_1 \cos(\omega t)$. The device conducts current only when $v_{GS}(t) > V_{\text{th}}$. We require the harmonic components of the resulting drain current $i_D(t)$ to analyze output power and efficiency.

    **Constraint:** Assume a piecewise-linear transfer characteristic $i_D \propto (v_{GS} - V_{\text{th}})$ for $i_D > 0$, clipped at $i_D = I_{\max}$. Assume $V_1$ is large enough to drive the device from cutoff to $I_{\max}$.

    The conduction interval is defined by the half-conduction angle $\theta$. The device turns on at $\omega t = -\theta$ and turns off at $\omega t = \theta$. At these boundaries, $v_{GS}(t) = V_{\text{th}}$, leading to the relation:
    
    $$
    \cos\theta = \frac{V_{\text{th}} - V_{GQ}}{V_1}
    $$

    The peak current $I_{\max}$ occurs at $\omega t = 0$. Normalizing the waveform to $I_{\max}$ yields the drain current expression:

    $$
    i_D(t) = 
    \begin{cases} 
    I_{\max} \frac{\cos(\omega t) - \cos\theta}{1 - \cos\theta}, & |\omega t| < \theta \\ 
    0, & \text{otherwise} 
    \end{cases}
    $$

    **Fourier Decomposition**

    Since $i_D(t)$ is an even function, its Fourier series contains only cosine terms:
    
    $$
    i_D(t) = I_0 + \sum_{n=1}^{\infty} I_n \cos(n\omega t)
    $$

    Evaluating the Fourier integrals provides the closed-form coefficients:

    $$
    \begin{aligned}
    I_0(\theta) &= \frac{I_{\max}}{\pi(1 - \cos\theta)} \left[ \sin\theta - \theta\cos\theta \right] \\
    I_1(\theta) &= \frac{I_{\max}}{\pi(1 - \cos\theta)} \left[ \theta - \sin\theta\cos\theta \right] \\
    I_n(\theta) &= \frac{I_{\max}}{\pi(1 - \cos\theta)} \left[ \frac{2\sin(n\theta)\cos\theta - 2n\sin\theta\cos(n\theta)}{n(n^2 - 1)} \right] \quad \text{for } n \ge 2
    \end{aligned}
    $$

    Evaluating the $n \ge 2$ expression explicitly for the second and third harmonics yields:

    $$
    \begin{aligned}
    I_2(\theta) &= \frac{I_{\max}}{\pi(1 - \cos\theta)} \left[ \frac{\sin(2\theta)\cos\theta - 2\sin\theta\cos(2\theta)}{3} \right] \\
    I_3(\theta) &= \frac{I_{\max}}{\pi(1 - \cos\theta)} \left[ \frac{\sin(3\theta)\cos\theta - 3\sin\theta\cos(3\theta)}{12} \right]
    \end{aligned}
    $$

    **Amplifier Classes**

    The choice of quiescent bias $V_{GQ}$ establishes the conduction angle $\theta$, categorizing the amplifier into canonical classes:

    - **Class A:** $\theta = \pi$. Full conduction. The device is biased midway between cutoff and $I_{\max}$. No clipping occurs; harmonics are theoretically zero.
    - **Class AB:** $\pi/2 < \theta < \pi$. The device conducts for more than half but less than the full cycle.
    - **Class B:** $\theta = \pi/2$. Half-wave conduction. Biased exactly at $V_{\text{th}}$. The fundamental $I_1$ is linear with drive, and all odd harmonics (except $n=1$) are exactly zero.
    - **Class C:** $\theta < \pi/2$. Under-biased. Conducts for less than half a cycle. Produces high efficiency but low fundamental power for a given $I_{\max}$.

    **Single-Pole Load Assumption**

    To analyze drain efficiency, we apply the constraint that the load network is a high-Q parallel LC tank resonant at $\omega$. The tank presents an impedance $R_{\text{opt}}$ at the fundamental frequency and approximates a short circuit ($0\,\Omega$) at all harmonic frequencies ($n \ge 2$). Under this condition, only the fundamental current $I_1$ produces a voltage swing across the load; harmonics circulate reactively without dissipating power.
    """)
    return


@app.cell
def _(go, make_subplots, mo, np):
    _theta = np.linspace(0.01, np.pi, 200)
    _imax = 1.0
    
    _i0 = (_imax / (np.pi * (1 - np.cos(_theta)))) * (np.sin(_theta) - _theta * np.cos(_theta))
    _i1 = (_imax / (np.pi * (1 - np.cos(_theta)))) * (_theta - np.sin(_theta) * np.cos(_theta))
    _i2 = (_imax / (np.pi * (1 - np.cos(_theta)))) * ((np.sin(2*_theta) * np.cos(_theta) - 2*np.sin(_theta) * np.cos(2*_theta)) / 3)
    _i3 = (_imax / (np.pi * (1 - np.cos(_theta)))) * ((np.sin(3*_theta) * np.cos(_theta) - 3*np.sin(_theta) * np.cos(3*_theta)) / 12)
    
    _fig_fourier = make_subplots(rows=1, cols=2, 
                                 subplot_titles=("Fourier Coefficients vs Conduction Angle", "Time-Domain Waveforms"),
                                 column_widths=[0.6, 0.4])
    
    _fig_fourier.add_trace(go.Scatter(x=_theta/np.pi, y=_i0, mode="lines", name="I<sub>0</sub> (DC)", line=dict(color="#00CC96", width=2)), row=1, col=1)
    _fig_fourier.add_trace(go.Scatter(x=_theta/np.pi, y=_i1, mode="lines", name="I<sub>1</sub> (Fund)", line=dict(color="#636EFA", width=2)), row=1, col=1)
    _fig_fourier.add_trace(go.Scatter(x=_theta/np.pi, y=_i2, mode="lines", name="I<sub>2</sub> (2nd Harm)", line=dict(color="#EF553B", width=2)), row=1, col=1)
    _fig_fourier.add_trace(go.Scatter(x=_theta/np.pi, y=_i3, mode="lines", name="I<sub>3</sub> (3rd Harm)", line=dict(color="#FFA15A", width=2)), row=1, col=1)
    
    _t_wave = np.linspace(-np.pi, np.pi, 200)
    
    def get_waveform(theta_val):
        _wform = _imax * (np.cos(_t_wave) - np.cos(theta_val)) / (1 - np.cos(theta_val))
        return np.maximum(0, _wform)
        
    _fig_fourier.add_trace(go.Scatter(x=_t_wave/np.pi, y=get_waveform(np.pi), mode="lines", name="Class A", line=dict(color="#00CC96")), row=1, col=2)
    _fig_fourier.add_trace(go.Scatter(x=_t_wave/np.pi, y=get_waveform(np.pi/2), mode="lines", name="Class B", line=dict(color="#636EFA")), row=1, col=2)
    _fig_fourier.add_trace(go.Scatter(x=_t_wave/np.pi, y=get_waveform(np.pi/4), mode="lines", name="Class C (π/4)", line=dict(color="#EF553B")), row=1, col=2)
    
    _fig_fourier.add_vline(x=1.0, line=dict(color="rgba(255,255,255,0.2)", dash="dot"), row=1, col=1)
    _fig_fourier.add_vline(x=0.5, line=dict(color="rgba(255,255,255,0.2)", dash="dot"), row=1, col=1)
    
    _fig_fourier.add_annotation(x=1.0, y=0.5, text="Class A", showarrow=False, xanchor="right", row=1, col=1)
    _fig_fourier.add_annotation(x=0.5, y=0.5, text="Class B", showarrow=False, xanchor="right", row=1, col=1)
    
    _fig_fourier.update_layout(template="plotly_dark", height=420,
                               legend=dict(orientation="h", y=-0.15))
    _fig_fourier.update_xaxes(title_text="Conduction angle θ/π", row=1, col=1)
    _fig_fourier.update_yaxes(title_text="Normalized current (i<sub>D</sub> / I<sub>max</sub>)", row=1, col=1)
    _fig_fourier.update_xaxes(title_text="Time ωt/π", row=1, col=2)
    _fig_fourier.update_yaxes(title_text="Normalized current", row=1, col=2)
    
    fig_fourier_decomp = mo.ui.plotly(_fig_fourier)
    fig_fourier_decomp


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Drain efficiency and output power

    **Problem Definition:** We must quantify the DC-to-RF energy conversion efficiency across the different conduction classes. The primary boundary condition is the DC supply voltage $V_{DD}$, which constrains the maximum voltage swing to $V_1 \le V_{DD}$ (assuming the load pulls the drain voltage down to $0\,\text{V}$ and up to $2V_{DD}$).

    The output RF power at the fundamental frequency is:
    $$
    P_{\text{out},1} = \frac{1}{2} V_1 I_1
    $$
    The DC power consumed from the supply is:
    $$
    P_{\text{DC}} = V_{DD} I_0
    $$

    **Drain Efficiency:**
    The drain efficiency $\eta_D$ is defined as the ratio of fundamental output power to DC power:
    $$
    \eta_D(\theta) = \frac{P_{\text{out},1}}{P_{\text{DC}}} = \frac{1}{2} \left( \frac{V_1}{V_{DD}} \right) \frac{I_1(\theta)}{I_0(\theta)}
    $$
    At maximum voltage swing (saturation), $V_1 = V_{DD}$, making the peak efficiency strictly a function of the conduction angle $\theta$:
    $$
    \eta_{D,\max}(\theta) = \frac{1}{2} \frac{I_1(\theta)}{I_0(\theta)} = \frac{1}{2} \frac{\theta - \sin\theta\cos\theta}{\sin\theta - \theta\cos\theta}
    $$

    Substituting the canonical conduction angles yields the theoretical limits:
    - **Class A ($\theta = \pi$):** $\eta_{D,\max} = \frac{1}{2} \frac{\pi}{\pi} = 50\%$
    - **Class B ($\theta = \pi/2$):** $\eta_{D,\max} = \frac{1}{2} \frac{\pi/2}{1} = \frac{\pi}{4} \approx 78.5\%$
    - **Class C ($\theta \to 0$):** Taking the limit as $\theta \to 0$ using Taylor expansions, $\eta_{D,\max} \to 100\%$.

    **Output Power and Back-off:**
    While decreasing $\theta$ improves efficiency, it severely reduces the fundamental current $I_1$ for a fixed peak current $I_{\max}$. Consequently, a Class C amplifier produces much lower maximum output power $P_{\text{out},1}$ from the same device area than a Class A amplifier.

    Furthermore, if the amplifier is driven below its saturation point (i.e., *power back-off*), the voltage swing $V_1$ drops linearly with the input drive, causing efficiency to plummet linearly with the voltage swing.

    **Interactive A & E — Class Explorer & Back-off Trajectories**
    Explore the time-domain waveforms, harmonic spectra, and back-off efficiency for varying bias conditions.
    """)
    return


@app.cell
def _(mo):
    vgq_slider = mo.ui.slider(-1.0, 1.0, step=0.05, value=0.0, label="Bias V_GQ (V)")
    vdrive_slider = mo.ui.slider(0.1, 2.0, step=0.05, value=1.0, label="Drive Amplitude V_1 (V)")
    ui_pa_classes_layout = mo.hstack([vgq_slider, vdrive_slider], wrap=True)
    return ui_pa_classes_layout, vdrive_slider, vgq_slider


@app.cell
def _(ui_pa_classes_layout):
    ui_pa_classes_layout
    return


@app.cell
def _(go, make_subplots, mo, np, vdrive_slider, vgq_slider):
    _vgq = vgq_slider.value
    _vdrive = vdrive_slider.value
    
    _vth = 0.0
    _imax = 1.0
    _vdd = 1.0
    _vknee = 0.1
    
    _t = np.linspace(0, 4*np.pi, 400)
    _vgs = _vgq + _vdrive * np.cos(_t)
    _id = np.maximum(0, _imax * (_vgs - _vth))
    _id = np.minimum(_id, _imax * 2) # Arbitrary max current cap
    
    _i0 = np.mean(_id)
    _i1 = 2 * np.mean(_id * np.cos(_t))
    _i2 = 2 * np.mean(_id * np.cos(2*_t))
    _i3 = 2 * np.mean(_id * np.cos(3*_t))
    
    _fig = make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5],
                         subplot_titles=("Drain Current i<sub>D</sub>(t)", "Harmonic Spectrum", 
                                         "Back-off Efficiency vs Class", "Operating Point"),
                         vertical_spacing=0.15)
                         
    _fig.add_trace(go.Scatter(x=_t, y=_id, mode="lines", name="i<sub>D</sub>(t)", line=dict(color="#636EFA", width=1.5)), row=1, col=1)
    _fig.add_trace(go.Bar(x=["DC", "f<sub>0</sub>", "2f<sub>0</sub>", "3f<sub>0</sub>"], y=[_i0, _i1, _i2, _i3], marker_color=["#00CC96", "#636EFA", "#EF553B", "#EF553B"], name="Harmonics"), row=1, col=2)
    
    # Back-off
    _backoff_db = np.linspace(-20, 0, 50)
    _v1_bo = _vdd * 10**(_backoff_db/20)
    _fig.add_trace(go.Scatter(x=_backoff_db, y=0.5 * (_v1_bo / _vdd) * 100, mode="lines", name="Class A (50%)", line=dict(color="#00CC96", width=1.5)), row=2, col=1)
    _fig.add_trace(go.Scatter(x=_backoff_db, y=(np.pi/4) * (_v1_bo / _vdd) * 100, mode="lines", name="Class B (78.5%)", line=dict(color="#AB63FA", width=1.5)), row=2, col=1)
    _fig.add_trace(go.Scatter(x=_backoff_db, y=0.9 * (_v1_bo / _vdd) * 100, mode="lines", name="Class C (90%)", line=dict(color="#FFA15A", width=1.5)), row=2, col=1)
    
    _vds_ll = np.linspace(0, 2*_vdd, 100)
    _r_l = (_vdd - _vknee) / (_imax / 2) if _i1 > 0 else 1.0
    _id_ll = _i0 + (_vdd - _vds_ll) / _r_l
    _fig.add_trace(go.Scatter(x=_vds_ll, y=_id_ll, mode="lines", name="Load Line", line=dict(color="#636EFA", width=2.0)), row=2, col=2)
    _fig.add_scatter(x=[_vdd], y=[_i0], mode="markers", marker=dict(size=12, color="#00CC96", symbol="x", line=dict(width=2)), name="Q-Point", row=2, col=2)
    _fig.add_annotation(x=_vknee, y=_imax/2, text="V<sub>knee</sub> Boundary", showarrow=True, arrowhead=2, ax=40, ay=-30, row=2, col=2)
    
    _fig.update_layout(template="plotly_dark", height=720, showlegend=True,
                       legend=dict(orientation="h", y=-0.08))
    _fig.update_xaxes(title_text="Time ωt (rad)", row=1, col=1)
    _fig.update_yaxes(title_text="Drain current i<sub>D</sub>(t)", row=1, col=1)
    _fig.update_xaxes(title_text="Harmonic component", row=1, col=2)
    _fig.update_yaxes(title_text="Amplitude", row=1, col=2)
    _fig.update_xaxes(title_text="Output power back-off (dB)", row=2, col=1)
    _fig.update_yaxes(title_text="Drain efficiency η<sub>D</sub> (%)", row=2, col=1)
    _fig.update_xaxes(title_text="Drain voltage v<sub>DS</sub> (V)", row=2, col=2)
    _fig.update_yaxes(title_text="Drain current i<sub>D</sub> (A)", row=2, col=2)
    
    fig_class_exp = mo.ui.plotly(_fig)
    fig_class_exp


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part II — Memoryless Nonlinearity
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Polynomial expansion of $i_d(v_{gs})$

    **Problem Definition:** For small-signal analysis (as discussed in **Notebook 04** on LNA design), we linearize the drain current as $i_d = g_m v_{gs}$. As the drive amplitude increases, the physical transfer characteristic curves smoothly due to mobility degradation and velocity saturation in short-channel CMOS devices, producing harmonic distortion and intermodulation.

    **Constraint:** Assume a strictly memoryless, weakly nonlinear transfer function around a fixed bias point.

    We model the AC drain current $i_d$ as a Taylor series expansion of the small-signal gate voltage $v_{gs}$:

    $$
    i_d = a_1 v_{gs} + a_2 v_{gs}^2 + a_3 v_{gs}^3 + \dots
    $$

    where the coefficients are the derivatives of the DC transfer characteristic evaluated at the quiescent point:
    $$
    a_1 = g_m, \quad a_2 = \frac{1}{2}\frac{\partial g_m}{\partial V_{GS}}, \quad a_3 = \frac{1}{6}\frac{\partial^2 g_m}{\partial V_{GS}^2}
    $$

    **Harmonic Distortion:**
    Injecting a single-tone input $v_{gs}(t) = A \cos(\omega t)$ and expanding the powers using trigonometric identities yields:

    $$
    i_d(t) = \left( \frac{a_2 A^2}{2} \right) + \left( a_1 A + \frac{3a_3 A^3}{4} \right) \cos(\omega t) + \left( \frac{a_2 A^2}{2} \right) \cos(2\omega t) + \left( \frac{a_3 A^3}{4} \right) \cos(3\omega t)
    $$

    The fractional harmonic distortion terms HD2 and HD3 are the ratios of the harmonic amplitudes to the fundamental amplitude (assuming $a_1 A \gg \frac{3}{4}a_3 A^3$):

    $$
    \text{HD2} = \left| \frac{a_2 A}{2 a_1} \right|, \quad \text{HD3} = \left| \frac{a_3 A^2}{4 a_1} \right|
    $$

    Crucially, HD2 grows linearly with drive amplitude $A$, while HD3 grows quadratically ($A^2$). This polynomial formulation is mathematically convenient but suffers a profound physical limitation: it assumes the coefficients $a_n$ are purely real and independent of frequency. As we will prove in Part III, reactive embedding makes memoryless assumptions fail drastically at mmWave frequencies.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Two-tone test

    **Problem Definition:** A single tone produces harmonics at $2\omega, 3\omega$, which are easily filtered by a tuned LC load (like the tanks used in **Notebook 06** for oscillators). In a real system, the signal contains multiple frequency components (e.g., OFDM subcarriers). We must analyze how nonlinearity causes these independent components to mix and generate in-band interference due to the nonlinear transconductance profile $g_m(v_{GS})$.

    **Constraint:** Inject a two-tone signal $v_{gs}(t) = A(\cos\omega_1 t + \cos\omega_2 t)$ where $\omega_1$ and $\omega_2$ are closely spaced ($\Delta\omega \ll \omega_1$).

    Expanding the memoryless polynomial up to the third order:

    - **Second-order products ($a_2 v_{gs}^2$):** Yields DC terms, second harmonics ($2\omega_1, 2\omega_2$), and second-order intermodulation (IM2) products at $\omega_1 + \omega_2$ and $\omega_1 - \omega_2$.
    - **Third-order products ($a_3 v_{gs}^3$):** Yields fundamental components ($\omega_1, \omega_2$), third harmonics ($3\omega_1, 3\omega_2$), and third-order intermodulation (IM3) products at $2\omega_1 - \omega_2$ and $2\omega_2 - \omega_1$.

    The fundamental amplitude for the $\omega_1$ tone becomes $a_1 A + \frac{9}{4}a_3 A^3$, and the IM3 amplitude at $2\omega_1 - \omega_2$ is $\frac{3}{4} a_3 A^3$.

    **Why IM3 Dominates:**
    The physical threat of IM3 lies in its frequency placement. The $2\omega_1 - \omega_2$ and $2\omega_2 - \omega_1$ IM3 products fall immediately adjacent to the fundamental tones inside the amplifier's passband. Unlike $2\omega$ or $\omega_1 + \omega_2$, they cannot be removed by the tank or matching network.

    **Interactive D — Two-tone spectrum simulator**
    Explore the spectrum generation of a memoryless nonlinearity driven by a two-tone signal. Notice how the slopes of IM2 and IM3 change with drive amplitude.
    """)
    return


@app.cell
def _(mo):
    df_slider = mo.ui.slider(1, 10, step=1, value=2, label="Tone Spacing Δf (MHz)")
    A_dbV_slider = mo.ui.slider(-40, 10, step=1, value=-10, label="Drive A (dBV)")
    a2_twotone_slider = mo.ui.slider(-0.5, 0.5, step=0.01, value=0.1, label="a_2 coefficient")
    a3_twotone_slider = mo.ui.slider(-0.5, 0.5, step=0.01, value=-0.1, label="a_3 coefficient")
    ui_twotone_layout = mo.hstack([df_slider, A_dbV_slider, a2_twotone_slider, a3_twotone_slider], wrap=True)
    return A_dbV_slider, a2_twotone_slider, a3_twotone_slider, df_slider, ui_twotone_layout


@app.cell
def _(ui_twotone_layout):
    ui_twotone_layout
    return


@app.cell
def _(go, mo, np, A_dbV_slider, a2_twotone_slider, a3_twotone_slider, df_slider):
    _df = df_slider.value
    _A = 10**(A_dbV_slider.value/20)
    _a1 = 1.0
    _a2 = a2_twotone_slider.value
    _a3 = a3_twotone_slider.value
    
    _f1 = 100
    _f2 = _f1 + _df
    
    _freqs = []
    _amps = []
    
    # Fundamentals
    _fund = np.abs(_a1*_A + 2.25*_a3*_A**3)
    _freqs.extend([_f1, _f2])
    _amps.extend([_fund, _fund])
    
    # IM2
    _im2_low = np.abs(_a2*_A**2)
    _freqs.extend([_df])
    _amps.extend([_im2_low])
    
    # IM3
    _im3 = np.abs(0.75*_a3*_A**3)
    _freqs.extend([2*_f1 - _f2, 2*_f2 - _f1])
    _amps.extend([_im3, _im3])
    
    # HD2
    _hd2 = np.abs(0.5*_a2*_A**2)
    _freqs.extend([2*_f1, 2*_f2])
    _amps.extend([_hd2, _hd2])
    
    # HD3
    _hd3 = np.abs(0.25*_a3*_A**3)
    _freqs.extend([3*_f1, 3*_f2])
    _amps.extend([_hd3, _hd3])
    
    _amps_db = 20 * np.log10(np.maximum(np.array(_amps), 1e-10))
    # Plot spectrum
    # Colors: f1/f2 (Blue), IM3 (Red), BB (Green), Harmonics (Purple)
    _colors = ["#636EFA", "#636EFA", "#00CC96", "#EF553B", "#EF553B", "#AB63FA", "#AB63FA", "#AB63FA", "#AB63FA"]
    _fig2 = go.Figure(data=[go.Bar(x=_freqs, y=_amps_db, width=1.0, marker_color=_colors, name="Spectral Tones")])
    _fig2.update_layout(template="plotly_dark", height=420, title="Two-Tone Intermodulation Spectrum",
                        xaxis_title="Frequency (MHz)", yaxis_title="Power (dBV)", yaxis_range=[-100, 20],
                        legend=dict(orientation="h", y=-0.15))
    
    # Add annotations
    _fig2.add_annotation(x=_f1, y=_amps_db[0], text="f<sub>1</sub>", showarrow=True, arrowhead=2, ax=0, ay=-30)
    _fig2.add_annotation(x=_f2, y=_amps_db[1], text="f<sub>2</sub>", showarrow=True, arrowhead=2, ax=0, ay=-30)
    _fig2.add_annotation(x=2*_f1 - _f2, y=_amps_db[2], text="IM<sub>3</sub>", showarrow=True, arrowhead=2, ax=-20, ay=-30)
    _fig2.add_annotation(x=2*_f2 - _f1, y=_amps_db[3], text="IM<sub>3</sub>", showarrow=True, arrowhead=2, ax=20, ay=-30)
    _fig2.add_annotation(x=_df, y=_amps_db[4], text="IM<sub>2</sub> (BB)", showarrow=True, arrowhead=2, ax=0, ay=-30)
    _fig2.add_annotation(x=2*_f1, y=_amps_db[5], text="HD<sub>2</sub>", showarrow=True, arrowhead=2, ax=0, ay=-30)
    _fig2.add_annotation(x=3*_f1, y=_amps_db[7], text="HD<sub>3</sub>", showarrow=True, arrowhead=2, ax=0, ay=-30)
    
    fig_twotone = mo.ui.plotly(_fig2)
    fig_twotone


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. $P_{1\text{dB}}$ and IIP3

    **Problem Definition:** We need canonical figures of merit to quantify the large-signal linearity of the PA. The two most common metrics are the 1-dB compression point ($P_{1\text{dB}}$) and the third-order input intercept point (IIP3). The physical origin of compression is the large-signal boundary conditions we explored in Part I (and **Notebook 06**)—as the voltage swing hits the $V_{\text{knee}}$ boundary or the current hits $I_{\max}$, the effective large-signal gain drops.

    **1-dB Compression Point ($P_{1\text{dB}}$)**
    At high drive levels, the $a_3 v_{gs}^3$ term (assuming $a_1$ and $a_3$ have opposite signs, reflecting a flattening transfer curve) subtracts from the fundamental amplitude. The input $1\text{dB}$ compression point is the drive amplitude $A_{1\text{dB}}$ where the fundamental gain drops by $1\,\text{dB}$ (a factor of $10^{-1/20} \approx 0.891$) relative to the ideal linear gain:

    $$
    20 \log_{10} \left( \frac{a_1 A + \frac{3}{4}a_3 A^3}{a_1 A} \right) = -1\,\text{dB}
    $$
    Solving this yields the closed-form expression:
    $$
    A_{1\text{dB}} = \sqrt{ \frac{4}{3} \left(1 - 10^{-1/20}\right) \left| \frac{a_1}{a_3} \right| } \approx 0.145 \sqrt{\left| \frac{a_1}{a_3} \right|}
    $$

    **Third-Order Intercept Point (IIP3)**
    IIP3 is a mathematical extrapolation point. On a logarithmic (dB) scale, the fundamental output power grows with a slope of 1 (1 dB/dB) with respect to input power, while the IM3 product grows with a slope of 3 (3 dB/dB). The IIP3 is the input amplitude $A_{\text{IIP3}}$ where these two asymptotic lines intersect:

    $$
    a_1 A_{\text{IIP3}} = \frac{3}{4} \left| a_3 \right| A_{\text{IIP3}}^3 \implies A_{\text{IIP3}} = \sqrt{ \frac{4}{3} \left| \frac{a_1}{a_3} \right| }
    $$

    **The 9.6 dB Rule of Thumb**
    For a strictly memoryless, third-order polynomial system, the ratio of IIP3 to $P_{1\text{dB}}$ is a constant:
    $$
    \frac{A_{\text{IIP3}}}{A_{1\text{dB}}} = \frac{\sqrt{4/3}}{\sqrt{0.145^2}} \approx 3.02
    $$
    Converting to power in dB: $20 \log_{10}(3.02) \approx 9.6\,\text{dB}$.
    Therefore, $\text{IIP3} - P_{1\text{dB}} \approx 9.6\,\text{dB}$. Deviations from this rule in physical hardware indicate the presence of significant higher-order nonlinearities ($a_5, a_7$) or frequency-dependent memory effects.

    **Interactive B — $P_{1\text{dB}}$ / IIP3 explorer**
    """)
    return


@app.cell
def _(mo):
    a1_iip3_slider = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="a_1 (Linear Gain)")
    a3_iip3_slider = mo.ui.slider(-0.5, -0.01, step=0.01, value=-0.1, label="a_3 (3rd order compression)")
    a5_iip3_slider = mo.ui.slider(-0.05, 0.05, step=0.005, value=0.0, label="a_5 (5th order)")
    ui_iip3_layout = mo.hstack([a1_iip3_slider, a3_iip3_slider, a5_iip3_slider], wrap=True)
    return a1_iip3_slider, a3_iip3_slider, a5_iip3_slider, ui_iip3_layout


@app.cell
def _(ui_iip3_layout):
    ui_iip3_layout
    return


@app.cell
def _(go, mo, np, a1_iip3_slider, a3_iip3_slider, a5_iip3_slider):
    _a1b = a1_iip3_slider.value
    _a3b = a3_iip3_slider.value
    _a5b = a5_iip3_slider.value
    
    _pin_dbm = np.linspace(-20, 20, 100)
    _vin = np.sqrt(100 * 10**(_pin_dbm/10) * 1e-3) 
    
    _pout_fund = 10 * np.log10( np.maximum(np.abs(_a1b*_vin + 0.75*_a3b*_vin**3 + 0.625*_a5b*_vin**5)**2 / 100 / 1e-3, 1e-10) )
    _pout_im3 = 10 * np.log10( np.maximum(np.abs(0.75*_a3b*_vin**3 + 1.25*_a5b*_vin**5)**2 / 100 / 1e-3, 1e-10) )
    
    _pout_fund_ideal = 10 * np.log10( np.maximum(np.abs(_a1b*_vin)**2 / 100 / 1e-3, 1e-10) )
    _pout_im3_ideal = 10 * np.log10( np.maximum(np.abs(0.75*_a3b*_vin**3)**2 / 100 / 1e-3, 1e-10) )
    
    _fig3 = go.Figure()
    _fig3.add_trace(go.Scatter(x=_pin_dbm, y=_pout_fund, name="Fundamental (compressed)", mode="lines", line=dict(color="#636EFA", width=2)))
    _fig3.add_trace(go.Scatter(x=_pin_dbm, y=_pout_im3, name="IM3 (compressed)", mode="lines", line=dict(color="#EF553B", width=2)))
    _fig3.add_trace(go.Scatter(x=_pin_dbm, y=_pout_fund_ideal, name="Ideal Linear (Slope 1)", mode="lines", line=dict(color="#636EFA", dash="dot", width=1.5)))
    _fig3.add_trace(go.Scatter(x=_pin_dbm, y=_pout_im3_ideal, name="Ideal IM3 (Slope 3)", mode="lines", line=dict(color="#EF553B", dash="dot", width=1.5)))
    
    # Calculate IIP3 Intersection
    _vin_iip3 = np.sqrt(np.abs(_a1b) / (0.75 * np.abs(_a3b)))
    _iip3_dbm = 10 * np.log10(_vin_iip3**2 / 100 / 1e-3)
    _oip3_dbm = 10 * np.log10((_a1b * _vin_iip3)**2 / 100 / 1e-3)
    _fig3.add_scatter(x=[_iip3_dbm], y=[_oip3_dbm], mode="markers", marker=dict(size=10, color="#FFA15A", symbol="x", line=dict(width=2)), name="Intercept Point")
    _fig3.add_annotation(x=_iip3_dbm, y=_oip3_dbm, text="Theoretical IIP3", showarrow=True, arrowhead=2, ax=-40, ay=-40)
    
    _fig3.update_layout(template="plotly_dark", height=420, title="P<sub>1dB</sub> and IIP3 Extrapolation",
                        xaxis_title="Input Power (dBm)", yaxis_title="Output Power (dBm)",
                        yaxis_range=[-80, 30], xaxis_range=[-20, 20],
                        legend=dict(orientation="h", y=-0.15))
    fig_iip3 = mo.ui.plotly(_fig3)
    fig_iip3


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. AM-AM and AM-PM (memoryless)

    **Problem Definition:** We wish to characterize the complex large-signal transfer function $H(A)$ of the amplifier as a function of the input amplitude $A$.

    **AM-AM Distortion:**
    AM-AM (Amplitude Modulation to Amplitude Modulation) represents the variation of the output magnitude with respect to the input magnitude. It is exactly the gain compression behavior $|H(A)|$ analyzed in Section 6.

    **AM-PM Distortion:**
    AM-PM (Amplitude Modulation to Phase Modulation) measures the shift in the phase of the fundamental output as the input amplitude changes, $\arg(H(A))$. In multi-level modulation schemes (like 64-QAM or 256-QAM), phase rotation at higher constellation points severely degrades the error vector magnitude (EVM).

    **The Memoryless Contradiction:**
    In our memoryless polynomial model:
    $$
    i_d(t) = a_1 v_{gs} + a_2 v_{gs}^2 + a_3 v_{gs}^3
    $$
    Assuming purely resistive source and load, the coefficients $a_n$ are strictly real scalars. The fundamental output current is $\left(a_1 A + \frac{3}{4}a_3 A^3\right)\cos(\omega t)$. 
    Because the fundamental term is a purely real scaling of the input cosine, the phase is clamped strictly to $0^\circ$ (or $180^\circ$ if it compresses past zero). 
    
    Therefore, **a strictly memoryless polynomial system exhibits exactly zero AM-PM distortion.**

    **The Physical Reality:**
    Empirical measurements of real CMOS and III-V power amplifiers consistently show non-zero AM-PM distortion (typically rotating by several degrees per dB near compression). The memoryless polynomial model is physically incomplete. 

    The physical origin of AM-PM lies in **nonlinear reactive elements**. As we analyzed in **Notebook 03** (stability and tuning) and **Notebook 04** (device noise models), the intrinsic device capacitances ($C_{gs}, C_{gd}$) are strongly voltage-dependent. As the large-signal drive amplitude increases, the time-averaged effective capacitance shifts. This dynamic change in capacitance detunes the conjugate matching networks dynamically, shifting the phase of the amplifier. The true nonlinear transfer function depends heavily on this reactive embedding and the frequency spectrum. This contradiction motivates the rigorous formulation of Volterra Series with memory in Part III.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part III — Volterra Series with Memory
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Why memoryless polynomial fails

    **Problem Definition:** In Part II, we assumed the drain current $i_d(t)$ responded instantaneously to the gate voltage $v_{gs}(t)$. At low frequencies, this approximation holds. However, at microwave and mmWave frequencies, device parasitic reactances (e.g., $C_{gs}, C_{gd}, L_s$) have impedances comparable to the system characteristic impedance.

    **The Role of Reactance:**
    A capacitor's impedance is $1/(j\omega C)$. It inherently integrates current over time to produce a voltage: $v(t) = \frac{1}{C} \int i(t) dt$. This means the voltage at time $t$ depends on the *history* of the current. The system possesses **memory**.

    **The Mixing Mechanism:**
    Consider a nonlinear transconductance driving a parallel RC load. A two-tone input generates a low-frequency envelope current at $\Delta\omega = \omega_2 - \omega_1$. 
    In a memoryless resistor, this current creates a voltage directly proportional to the current, regardless of $\Delta\omega$.
    But in an RC load, the impedance at $\Delta\omega$ is heavily frequency-dependent. The voltage generated at $\Delta\omega$ feeds back across the nonlinear $C_{gd}$ or mixes again with the fundamental tones to generate IM3.

    Because the intermediate mixing products ($2\omega, \Delta\omega$) see frequency-dependent reactive loads, the overall IM3 distortion becomes strongly dependent on the tone spacing $\Delta\omega$ and the absolute carrier frequency $\omega$. A Taylor series cannot model this. We must generalize to a **Volterra Series**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Harmonic-input probing method

    **Problem Definition:** The Volterra series is a generalized Taylor series for systems with memory. The output $y(t)$ is expressed as a sum of multidimensional convolution integrals of the input $x(t)$ with Volterra kernels $h_n(t_1, \dots, t_n)$. In the frequency domain, we need to find the nonlinear transfer functions $H_n(\omega_1, \dots, \omega_n)$.

    The **Harmonic-Input Probing Method** is a rigorous analytical technique to solve for these kernels by systematically extracting them order by order.

    **Step 1: First-Order Kernel $H_1(\omega)$**
    Inject a single exponential input $v_{\text{in}} = e^{j\omega t}$.
    Assume the output is purely linear: $y_1(t) = H_1(\omega)e^{j\omega t}$.
    Substitute into the circuit's differential equations (ignoring all nonlinear terms) and solve algebraically for $H_1(\omega)$. This is exactly identical to standard small-signal AC analysis.

    **Step 2: Second-Order Kernel $H_2(\omega_1, \omega_2)$**
    Inject a two-tone input $v_{\text{in}} = e^{j\omega_1 t} + e^{j\omega_2 t}$.
    The system generates second-order currents through its nonlinear components (e.g., $i_{NL2} = a_2 v_1^2$). We calculate these nonlinear currents using the previously solved first-order voltages $H_1(\omega_{1,2})$.
    We then treat these nonlinear currents as *independent forcing functions* injected into the *linearized* circuit. Solving the linear circuit at the sum frequency $(\omega_1 + \omega_2)$ yields $H_2(\omega_1, \omega_2)$.

    **Step 3: Third-Order Kernel $H_3(\omega_1, \omega_2, \omega_3)$**
    Inject three tones. The third-order nonlinear current $i_{NL3}$ arises from two distinct physical mechanisms:
    1. **Direct Generation:** The third-order nonlinearity ($a_3$) acting directly on the first-order voltages.
    2. **Second-Order Mixing:** The second-order nonlinearity ($a_2$) mixing a first-order voltage $H_1$ with a second-order voltage $H_2$.

    This reveals the core insight of Volterra analysis: **Third-order distortion is highly sensitive to second-order matching**, because second-order products (like the baseband envelope $\Delta\omega$ and second harmonic $2\omega$) re-mix with the fundamental to create IM3.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. CMOS short-channel device kernels

    **Problem Definition:** We apply the probing method to a simplified short-channel CMOS equivalent circuit containing a nonlinear transconductance $i_d(v_{gs}) = g_1 v_{gs} + g_2 v_{gs}^2 + g_3 v_{gs}^3$ and a nonlinear gate-source capacitance $q_{gs}(v_{gs}) = C_1 v_{gs} + C_2 v_{gs}^2 + C_3 v_{gs}^3$.

    Let the source impedance be $Z_S(\omega)$.

    **First-Order Kernel:**
    Solving the linear node equations:
    $$
    H_1(\omega) = \frac{v_{gs}}{v_{\text{in}}} = \frac{1}{1 + j\omega C_1 Z_S(\omega)}
    $$

    **Second-Order Kernel:**
    The second-order nonlinear current originates from both $g_2$ and $C_2$. The capacitance generates a nonlinear current proportional to the time derivative of the charge: $i_{C2} = j(\omega_1 + \omega_2) C_2 v_{gs}^2$.
    The nonlinear current acting as a forcing function on the linear circuit yields:
    $$
    H_2(\omega_1, \omega_2) = -H_1(\omega_1)H_1(\omega_2) H_1(\omega_1+\omega_2) Z_S(\omega_1+\omega_2) \left[ g_2 + j(\omega_1+\omega_2)C_2 \right]
    $$

    **Third-Order Kernel:**
    The third-order kernel evaluates the IM3 product at $2\omega_1 - \omega_2$. The total nonlinear current contains direct $g_3, C_3$ terms, and the critical second-order feedback terms:
    $$
    \begin{aligned}
    H_3(\omega_1, \omega_1, -\omega_2) &= \text{Direct} + \text{Mixing} \\
    \text{Direct} &= g_3 + j(2\omega_1 - \omega_2)C_3 \\
    \text{Mixing} &= \frac{2}{3} \left[ g_2 + j(2\omega_1 - \omega_2)C_2 \right] \left[ H_2(\omega_1, -\omega_2) + H_2(\omega_1, \omega_1) \right]
    \end{aligned}
    $$
    
    The terms $H_2(\omega_1, -\omega_2)$ and $H_2(\omega_1, \omega_1)$ represent the **baseband impedance** $Z_S(\Delta\omega)$ and the **second harmonic impedance** $Z_S(2\omega_1)$, respectively. 
    By engineering the termination impedances at $\Delta\omega$ and $2\omega$, the Mixing term can be intentionally phased to *cancel* the Direct term, drastically improving IIP3 without changing the DC bias. This is known as **Sweet-Spot Optimization**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Kernel embedding and IIP3 vs. tone spacing

    **Problem Definition:** Memory effects dictate that IM3 is not a static scalar. It is a dynamic function of the envelope frequency $\Delta\omega = \omega_2 - \omega_1$.

    **IM3 Asymmetry:**
    Because the Volterra kernels involve complex transfer functions evaluated at $2\omega_1 - \omega_2$ and $2\omega_2 - \omega_1$, the magnitude and phase of the lower and upper IM3 sidebands are generally unequal. This asymmetry is a tell-tale signature of memory effects in a measured spectrum.

    **IIP3 Sweet Spots:**
    As derived in Section 10, the baseband termination impedance $Z_S(\Delta\omega)$ strongly influences the second-order mixing term. If the bias network contains large decoupling capacitors and inductors, it creates an LC resonance at a specific low frequency. When $\Delta\omega$ sweeps across this resonance, $H_2(\Delta\omega)$ shifts rapidly in phase and magnitude, causing a sharp dip or peak in IIP3—a "memory sweet spot" or "memory notch".

    **Interactive C — Volterra memory effect visualizer**
    Explore how the tone spacing and baseband termination interact to create frequency-dependent IIP3 asymmetry.
    """)
    return


@app.cell
def _(mo):
    z_bb_slider = mo.ui.slider(10, 500, step=10, value=50, label="Baseband Impedance Resonance Z_BB (Ω)")
    f_res_slider = mo.ui.slider(1, 50, step=1, value=10, label="Baseband Resonance Freq (MHz)")
    q_bb_slider = mo.ui.slider(0.5, 10, step=0.5, value=2.0, label="Baseband Q factor")
    g3_sign_drop = mo.ui.dropdown(["Positive (+)", "Negative (-)"], value="Negative (-)", label="g_3 Sign (Direct term)")
    ui_volterra_layout = mo.hstack([z_bb_slider, f_res_slider, q_bb_slider, g3_sign_drop], wrap=True)
    return f_res_slider, g3_sign_drop, q_bb_slider, ui_volterra_layout, z_bb_slider


@app.cell
def _(ui_volterra_layout):
    ui_volterra_layout
    return


@app.cell
def _(go, mo, np, f_res_slider, g3_sign_drop, q_bb_slider, z_bb_slider):
    _z_bb = z_bb_slider.value
    _f_res = f_res_slider.value
    _q_bb = q_bb_slider.value
    _g3_sign = 1 if "Positive" in g3_sign_drop.value else -1
    
    _df = np.logspace(-1, 2, 200) # 0.1 MHz to 100 MHz
    
    # Simple behavioral model of Volterra mixing
    # Baseband impedance profile (parallel RLC)
    _s = 1j * (_df / _f_res)
    _z_env = _z_bb / (1 + _q_bb * (_s + 1/_s))
    
    # Direct 3rd order term (constant magnitude)
    _direct = _g3_sign * 10.0 + 1j*0.0 
    
    # Second-order mixing term is proportional to envelope impedance
    # Assuming phase rotation due to capacitive delay
    _mixing = 0.5 * _z_env * np.exp(-1j * np.pi/4) 
    
    # Total H3 kernels for lower and upper IM3
    _h3_lower = _direct + _mixing
    # Upper IM3 effectively sees the conjugate mixing due to delta_omega sign flip
    _h3_upper = _direct + np.conj(_mixing)
    
    # IIP3 is inversely proportional to sqrt(|H3|) -> in dBm: Const - 10*log10(|H3|^2)
    _const_iip3 = 30
    _iip3_lower = _const_iip3 - 10*np.log10(np.abs(_h3_lower)**2 + 1e-3)
    _iip3_upper = _const_iip3 - 10*np.log10(np.abs(_h3_upper)**2 + 1e-3)
    
    _fig_volt = go.Figure()
    _fig_volt.add_trace(go.Scatter(x=_df, y=_iip3_lower, mode="lines", name="Lower IM3 IIP3", line=dict(color="#636EFA", width=2)))
    _fig_volt.add_trace(go.Scatter(x=_df, y=_iip3_upper, mode="lines", name="Upper IM3 IIP3", line=dict(color="#EF553B", width=2, dash="dash")))
    
    _fig_volt.add_vline(x=_f_res, line=dict(color="#00CC96", dash="dot"), annotation_text="BB Resonance", annotation_position="top right")
    
    # Annotate peak asymmetry
    _idx_res = np.argmin(np.abs(_df - _f_res))
    _y_annot = max(_iip3_upper[_idx_res], _iip3_lower[_idx_res])
    _fig_volt.add_annotation(x=np.log10(_f_res), y=_y_annot + 2, text="Peak IM3 Asymmetry", showarrow=True, arrowhead=2, ax=40, ay=-30)
    
    _fig_volt.update_layout(template="plotly_dark", height=420, title="IIP3 vs Tone Spacing (Memory Effects)",
                            xaxis_title="Tone spacing Δf (MHz)", yaxis_title="IIP3 (dBm)",
                            xaxis_type="log",
                            legend=dict(orientation="h", y=-0.15))
                           
    fig_volterra = mo.ui.plotly(_fig_volt)
    fig_volterra


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part IV — Process Technology and CMOS Limits
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Tech comparison and CMOS limitations
    
    To contextualize CMOS power amplifiers, we compare typical high-frequency semiconductor processes:

    | Technology | $f_T$ (GHz) | $f_{\max}$ (GHz) | $V_{BR}$ (V) | $P_{\text{sat}}$ density (W/mm) | 1/f corner | Integration |
    |---|---|---|---|---|---|---|
    | **Bulk CMOS 28 nm** | 280 | 320 | ~1.0 | 0.05 | High | Excellent |
    | **SiGe BiCMOS** | 300 | 500 | ~2 (BV<sub>CEO</sub>) | 0.3 | Low | Very good |
    | **GaAs pHEMT** | 150 | 250 | ~15 | 1.0 | Medium | Poor |
    | **GaN HEMT** | 100 | 200 | ~80 | 5.0 | High | Poor |

    Since output power scales proportionally to $V_{\text{knee}} \cdot I_{\max}$ and strongly with breakdown voltage $V_{BR}$, CMOS is fundamentally handicapped. Its saturated power density is two orders of magnitude lower than GaN. However, CMOS remains ubiquitous at mmWave due to its unparalleled integration capabilities (combining the PA, LNA, LO, mixer, and DSP on a single die).

    **CMOS Limitations at mmWave:**
    1. **Low Breakdown Voltage:** The low $V_{BR}$ caps single-device $P_{\text{sat}}$. To achieve higher output power, designers use *stacking* (cascoded series transistors) or complex power-combining networks.
    2. **Hot-Carrier Degradation:** Operating near the breakdown limit causes hot-carrier injection, shifting $V_{\text{th}}$ and degrading $g_m$ over the device's lifetime.
    3. **Low Passive Q:** On-chip inductors and transformers suffer from substrate loss and thin-metal resistive loss, severely limiting the efficiency of matching networks.
    4. **AM-PM Distortion:** As rigorously derived in Section 10, the non-linear voltage dependence of $C_{gs}$ creates dynamic phase modulation (AM-PM), a unique penalty of CMOS devices compared to III-V technologies.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Part V — Wrap and bridge to notebook 08
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Recap and bridge
    
    This notebook developed the theory of power amplifiers across three levels of abstraction:
    
    1. **Bias and Waveform Engineering (Part I):** We established the fundamental boundaries of large-signal operation. By modulating the conduction angle $\theta$, we defined Classes A, AB, B, and C, identifying the fundamental trade-off between linearity and drain efficiency.
    2. **Memoryless Nonlinearity (Part II):** We introduced the Taylor polynomial expansion to derive canonical figures of merit ($P_{1\text{dB}}$ and IIP3), analyzed the spectrum of a two-tone test, and demonstrated the theoretical 9.6 dB rule.
    3. **Volterra Series with Memory (Part III):** We revealed the physical inadequacy of memoryless polynomials at mmWave. Using the harmonic-input probing method, we derived frequency-dependent Volterra kernels for a short-channel CMOS device, rigorously explaining AM-PM distortion and the tone-spacing dependence of IIP3.

    ### Bridge to Notebook 08

    This notebook focused on the transmit-path active device. **Notebook 08** returns to the local-oscillator side and asks the question that notebook 06 (phase noise) deferred: how *high in frequency and output power* can a CMOS oscillator actually go in a given process? It develops the Momeni–Afshari activity-condition design methodology — extending notebook 02's Mason-$U$ / $f_{\max}$ analysis with an explicit $(A, \varphi)$ parameterization — and shows why three-stage rings reach $f_{\max}$ while cross-coupled topologies stop short, then covers the triple-push technique for output above $f_{\max}$ (the 256 GHz / 482 GHz CMOS regime).

    <br>

    [← Notebook 06: Oscillators and VCOs](./06_oscillators_vco.py) | [Notebook 08: Systematic Oscillator Design →](./08_systematic_oscillator_design.py)
    """)
    return


if __name__ == "__main__":
    app.run()
