# Transmission-line bounce-diagram interactive — design spec

Date: 2026-05-06
Path: `interactive/transmission_line_bounce.py`

## Purpose

Browser-based interactive that teaches transient behavior on a lossless
transmission line and the bounce (lattice) diagram, paired tightly with an
animated V(z,t) / I(z,t) view. Follows Pozar Ch. 2 conventions and is the
first in a planned series of EM-wave interactives that will continue with
TE/TM/TEM modes in waveguides.

## Scope

- Single lossless line of length ℓ, characteristic impedance Z₀, phase
  velocity v_p (real Z₀, no loss).
- Real source impedance Z_s, Thevenin source v_s(t).
- Load Z_L: pure-resistive R, series R+L, or parallel R∥C. (Reactive loads
  are required so the Gaussian pulse case visibly distorts on reflection.)
- Excitations: smoothed step, Gaussian pulse, steady-state sinusoid.
- Cascaded line sections are out of scope for v1.
- Frequency-dependent or complex Z_s out of scope for v1.

## Architecture

Single-file Python launcher matching the existing pattern in
`interactive/vco_pulling.py` and siblings: stdlib `http.server` serves an
inlined HTML/JS document. All numerics and rendering run in the browser; the
Python side has no compute role.

Logical modules inside the inlined JS:

- **physics** — pure functions: `gammaLoad(omega, kind, params)`,
  `assembleSpectra(params)`, `vAndIAt(z, t, spectra, params)`. No DOM.
- **plots** — three Plotly panes (bounce diagram, V(z,t), I(z,t)) plus a
  shared time-cursor controller. No physics.
- **controls** — slider/preset/signal UI, emits a `params` object,
  subscribes to redraws.
- **app** — wiring only.

Theme: `plotly_dark`-style palette consistent with the rest of `interactive/`
(`--orange` for forward waves, `--cyan` for reverse, `--text` for total).

## Physics & numerics

### Symbolic definitions

Reflection coefficients

    Γ_S(ω) = (Z_s − Z₀) / (Z_s + Z₀)      (real, frequency-independent)
    Γ_L(ω) = (Z_L(ω) − Z₀) / (Z_L(ω) + Z₀)

with

    Z_L^{R}(ω)   = R
    Z_L^{R+L}(ω) = R + jωL
    Z_L^{R∥C}(ω) = 1 / (1/R + jωC)

Forward / reverse waves on a lossless line

    V⁺(z,t) = f(t − z/v_p)        I⁺ = +V⁺/Z₀
    V⁻(z,t) = g(t + z/v_p)        I⁻ = −V⁻/Z₀

Round-trip time τ = 2ℓ / v_p.

### Algorithm (FFT-based superposition of bounces)

1. Sample v_s(t) on a uniform grid of N_t = 4096 points over a window
   T_win ≥ 8τ. Compute V_s(ω) = FFT[v_s].
2. Initial launched-wave spectrum after the source-side voltage divider:

       A₀(ω) = V_s(ω) · Z₀ / (Z_s + Z₀)

3. Bounce n contributes a forward and reverse component with coefficient
   [Γ_S(ω)·Γ_L(ω)]^n and one extra Γ_L for the reverse leg, each carrying
   its own delay e^{−jωτ_n}. Sum until |Γ_S Γ_L|^n < 10⁻⁴ or n = 50.
4. For each animation frame at time t_k, evaluate V⁺(z,t_k), V⁻(z,t_k) on
   N_z = 256 position samples by IFFT; derive I⁺, I⁻ from Z₀ with the
   sign flip on I⁻.

### Steady-state sinusoid (separate code path)

Pick off Γ_L at ω₀, build the standing-wave envelope analytically

    |V(z)| = |V⁺| · |1 + Γ_L · e^{−j2β(ℓ−z)}|

and animate V(z,t) = Re{V(z) · e^{jω₀t}}. The bounce-diagram pane swaps to
a static envelope view with VSWR and λ/4-marker readouts.

### Step-input handling

Use v_s(t) = ½(1 + tanh(t/t_rise)) with default t_rise = τ/50 (clamped to
≥ 2·dt) to avoid the 1/(jω) singularity and Gibbs ringing.

### Defaults

- N_t = 4096, N_z = 256
- Animation 30 fps with frame skipping if behind
- Z₀ ∈ [10, 200] Ω (default 50)
- ℓ ∈ [0.1, 10] m (default 1.0)
- v_p ∈ [0.3c, c] (default c/2)
- Z_s ∈ [0, 500] Ω, real (default 50)
- L_L ∈ [0, 1] µH, C_L ∈ [0, 1] nF, R_L ∈ [0, 500] Ω

## UI layout

Vertical three-row layout:

    ┌────────────────────────────────────────────────────────────────────┐
    │  Header: title + topology schematic (Vs─Zs──[Z0,ℓ,vp]──ZL)         │
    ├──────────────────────┬─────────────────────────────────────────────┤
    │                      │   V(z,t)    [forward / reverse / total]     │
    │   Bounce diagram     │                                             │
    │   z horizontal,      ├─────────────────────────────────────────────┤
    │   t vertical-down    │   I(z,t)    [forward / reverse / total]     │
    │                      │                                             │
    ├──────────────────────┴─────────────────────────────────────────────┤
    │  Controls: signal type | Z0, Zs, ZL form | ℓ, vp                   │
    │            presets row | play/pause | time scrub | speed           │
    └────────────────────────────────────────────────────────────────────┘

### Synced time cursor

A horizontal line on the bounce diagram at t = t_now moves with the V/I
animation. Scrubbing the time slider drags the cursor and snaps the V/I
view to that instant. Polish (second pass): hovering on a bounce-diagram
segment highlights the corresponding wave packet in the V/I plots.

### Bounce-diagram pane

- Forward bounces in `--orange`, reverse in `--cyan`, opacity scales with
  |Γ_S Γ_L|^n (so weak late bounces fade visibly).
- Each bounce labeled with its amplitude coefficient (e.g. `Γ_L·V⁺`,
  `Γ_S·Γ_L·V⁺`).
- ~6 bounces shown by default; "show all" toggle reveals up to 50.

### V(z,t) and I(z,t) panes

Three traces each: forward (orange), reverse (cyan), total (white). Y-axis
fixed by the initial launched amplitude × (1 + |Γ_L|) headroom to prevent
jumpy autoscaling. Vertical marker at z = ℓ for the load.

### Controls

- Signal type dropdown: {Step, Gaussian pulse, Steady-state sinusoid};
  pulse width / rise time / frequency shown contextually.
- Line params: Z₀, ℓ, v_p sliders.
- Source: Z_s slider (real only in v1).
- Load: kind radio {R, R+L, R∥C}; sliders for the active components only.
- Presets (Pozar canonical): matched (Z_L=Z₀), open (Z_L=∞), short (Z_L=0),
  resistive mismatch (Z_L=2Z₀), reactive (R+L with τ_RL = τ),
  source-mismatched (Z_s≠Z₀ to show multi-bounce decay).
- Play/pause, time scrub (0 to ~5τ), speed (×0.25 – ×4).

### Sinusoid-mode swap

When the sinusoid signal is selected, the bounce-diagram pane swaps to a
static |V(z)| / |I(z)| envelope plot with VSWR and λ/4 markers; V/I panes
animate the time-harmonic standing wave. Same color scheme throughout.

## Edge cases

- Z_L = 0, Z_s = 0: Γ formula safe; resistive components clamped to ≥ 0 by
  slider.
- Z_s = Z₀ (matched source): Γ_S = 0, single load bounce. Iteration cap
  via amplitude threshold handles this naturally.
- Reactive load with R = 0: |Γ_L| = 1 forever; n_max = 50 caps the
  iteration without misrepresenting the physics.
- ω = 0 in Γ_L for R+L load: L → 0 impedance, formula handles directly.
- t_rise vs dt: clamp t_rise ≥ 2·dt.
- FFT wrap-around: T_win ≥ 8τ with zero padding; console warning if energy
  reaches the window edge.

## Error handling

- Python launcher: fall back to next free port if 8000 is taken (matches
  existing `interactive/vco_pulling.py` behavior).
- No backend round-trips during simulation: all numerics are in the
  browser, so there are no network failure paths.

## Testing

- **JS unit tests**, gated behind `?test=1`, results printed to console:
  - Γ_L matches hand-computed values at ω = 0, ω = 1/√(LC), ω → ∞.
  - Matched load → reflected wave is numerically zero (within FFT noise).
  - Open-load step → V_load → 2·V⁺ as t → ∞.
  - Energy on lossless line conserved within 1% over 5τ for a Gaussian
    pulse and lossless reactive load.
- **Manual smoke test** for each preset.
- **Launcher smoke test**:
  `uv run python interactive/transmission_line_bounce.py` starts and
  serves the page.

## Out of scope (future work)

- Lossy lines (α > 0).
- Cascaded line sections with intermediate Z₀ junctions.
- Complex / frequency-dependent Z_s.
- Dispersive lines.
- TE/TM/TEM mode visualizations in waveguides — separate apps in this
  series.
