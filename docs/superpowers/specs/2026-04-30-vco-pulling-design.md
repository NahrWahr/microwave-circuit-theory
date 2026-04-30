# VCO Pulling Interactive App — Design Spec

**Date:** 2026-04-30
**File:** `interactive/vco_pulling.py`

---

## Overview

A standalone interactive desktop tool that visualises the dynamics of VCO pulling using the Adler equation. Intended for use during circuit design — not part of the marimo notebook series. A second app (injection locking) will be a separate file built later.

---

## Architecture

Single Python file. When executed (`uv run python interactive/vco_pulling.py`):

1. Starts a custom `http.server` handler on a randomly chosen free port; the handler serves the inlined HTML/CSS/JS string directly from memory (no temp file).
2. Opens the browser via `webbrowser.open()`.
3. Serves until the user closes the terminal (KeyboardInterrupt).

No external Python dependencies. All physics, simulation, and rendering run in JavaScript inside the served HTML page. The Python file is a launcher and an inlined HTML/CSS/JS template.

---

## Physics Model

### Adler equation

```
dφ/dt = Δω − ω_L · sin(φ)
```

Where:
- `φ` = phase difference between VCO and injector (in the rotating frame at ω_inj)
- `Δω = 2π · Δf` = angular frequency detuning
- `ω_L = (π · f₀ / Q) · (V_inj / V_osc)` = locking bandwidth

Normalized form (τ = ω_L · t):

```
dφ/dτ = η − sin(φ)      η = Δω / ω_L
```

**Lock condition:**
- `|η| < 1`: stable fixed point at φ* = arcsin(η) — injection locked
- `|η| > 1`: no fixed point — VCO pulled, beat frequency ω_beat = ω_L · √(η² − 1)

### Integrator

RK4 at a fixed normalized timestep Δτ. The physical parameters (f₀, Δf, Q, V_inj/V_osc) determine η and ω_L each frame before integration. Sim speed multiplier scales Δτ per animation frame.

---

## User Controls

All controls live in the right sidebar above the spectrum panel.

| Parameter | Widget | Range | Default |
|---|---|---|---|
| f₀ (free-running freq) | Number spinner (GHz) | 1–100 GHz | 5 GHz |
| Δf (detuning) | Slider + number input (MHz) | −500 to +500 MHz | 150 MHz |
| Q (loaded Q factor) | Slider + number input | 5–200 | 20 |
| V_inj/V_osc (injection ratio) | Slider (dBc) | −40 to 0 dBc | −10 dBc |
| Sim speed | Slider | 0.1× to 10× | 1× |
| Run / Pause / Reset | Buttons | — | Running on load |

**Derived read-outs** (below sliders, not interactive):
- ω_L in MHz
- η (dimensionless)
- Lock state badge: green **LOCKED** when |η| < 1, red **PULLING** when |η| > 1

---

## Layout

Three-column layout. Phasor spans the full left column height. Phase portrait and frequency deviation stack vertically in the centre column. Controls (sliders, read-outs, buttons) and spectrum share the right column.

```
┌──────────────────┬───────────┬──────────────┐
│                  │ Phase     │              │
│  Phasor diagram  │ portrait  │  Controls    │
│  (left ~40%)     ├───────────┤  (sliders,   │
│                  │ Freq dev  │  read-outs,  │
│                  │ (time)    │  spectrum)   │
└──────────────────┴───────────┴──────────────┘
```

---

## Visualization Panels

### 1. Phasor diagram (left, large)

- Canvas 2D, rotating frame (co-rotating at ω_inj)
- **Injector phasor**: fixed, pointing right (red)
- **VCO phasor**: at angle φ(t), rotating at dφ/dt (blue)
- **Phase arc**: arc between the two phasors showing φ
- **Phase trail**: fading arc of the last ~2π of VCO tip trajectory
- Axes: Re / Im labels
- 60 fps via `requestAnimationFrame`

### 2. Phase portrait (top-right)

- Canvas 2D
- x-axis: φ ∈ [−π, π]
- y-axis: dφ/dτ = η − sin(φ)
- Static curve redrawn when parameters change
- Horizontal zero line
- Fixed points marked: stable (filled dot, orange), unstable (open dot, orange)
- Moving dot tracking current (φ mod 2π, dφ/dτ)
- When |η| > 1: no fixed points, dot traverses the curve periodically

### 3. Frequency deviation (middle-right)

- Scrolling time trace of dφ/dt in MHz (un-normalized via ω_L)
- Horizontal zero line
- x-axis: time window of ~10 beat periods (auto-scaled)
- For pulling: periodic oscillation at ω_beat
- For locking: exponential decay toward zero

### 4. Spectrum (bottom of right sidebar)

- Rolling FFT of the accumulated VCO output signal: `cos(ω_inj·t + φ(t))`
- x-axis: frequency offset from f_inj (MHz)
- y-axis: power (dBc), range −80 to 0
- Updates every ~0.5 s to avoid flicker
- Shows carrier spike and FM sidebands at multiples of ω_beat

---

## Out of scope

- Saving / exporting simulation data
- Injection locking app (separate file, separate spec)
- Noise / phase noise modelling
- Large-signal or nonlinear VCO model (Adler is the model)
