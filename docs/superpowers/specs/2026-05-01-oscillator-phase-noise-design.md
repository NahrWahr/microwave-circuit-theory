# Limit Cycles and Oscillator Phase Noise Interactive App - Design Spec

**Date:** 2026-05-01
**File:** `interactive/oscillator_phase_noise.py`

---

## Overview

A standalone browser-based teaching and design tool for nonlinear oscillator limit cycles, phase perturbations, and RF-style phase-noise intuition. The app uses a Van der Pol oscillator to show how amplitude errors are restored by nonlinear dynamics while phase errors persist as timing uncertainty.

This is separate from the VCO pulling app and is not a marimo notebook export.

---

## Architecture

Single Python file. When executed:

```bash
uv run python interactive/oscillator_phase_noise.py
```

The launcher:

1. Finds a free local port.
2. Serves one in-memory HTML/CSS/JavaScript page with `http.server`.
3. Opens the page with `webbrowser.open()`.
4. Runs until interrupted from the terminal.

No new Python dependencies are required. Simulation, plotting, controls, and readout updates run in browser JavaScript.

---

## Core Model

The nonlinear oscillator is modeled as:

```text
x'' - mu(1 - x^2)x' + omega0^2 x = noise + perturbation
```

Implemented as a first-order system:

```text
dx/dt = v
dv/dt = mu(1 - x^2)v - omega0^2 x
```

The deterministic dynamics use a fixed-step RK4 integrator. Stochastic phase and amplitude perturbations are applied in polar oscillator coordinates after each deterministic integration step so the UI can distinguish:

- phase diffusion from white phase noise
- amplitude noise that is restored by the limit cycle
- AM-to-PM conversion that turns amplitude fluctuation into phase error

The simulation is educational rather than a metrology-grade phase-noise engine.

---

## User Interface

Dense dark dashboard layout, matching the style of `interactive/vco_pulling.py`.

Main panels:

- Phase-plane view: trajectory, estimated limit cycle, current state, perturbation vector
- Time-domain waveform view
- Phase error and zero-crossing jitter trace
- Spectrum and phase-noise style offset display

Right sidebar controls:

| Parameter | Widget | Range | Default |
|---|---|---:|---:|
| Natural frequency `f0` | slider + number | 0.5-20 GHz | 5 GHz |
| Nonlinear gain/compression `mu` | slider + number | 0.1-6 | 1.6 |
| Noise strength | slider | 0-1 | 0.18 |
| AM-to-PM coupling | slider | 0-2 rad/A | 0.35 |
| Perturbation mode | segmented buttons | radial, tangential, random | radial |
| Run / pause / reset / kick | buttons | - | running |

The visual simulation advances in normalized oscillator-cycle time at a fixed
observable rate. Carrier frequency and phase-noise offset settings affect
readouts and spectra, not the browser animation speed.

Derived readouts:

- oscillation amplitude
- instantaneous frequency offset
- RMS phase error
- RMS jitter
- estimated phase noise at 10 kHz, 100 kHz, and 1 MHz offsets

---

## Visualization Behavior

### Phase Plane

Shows `x` versus `v/omega0`, the recent trajectory, the current state, and a reference limit-cycle radius. Radial perturbations move the state inward or outward from the cycle and then visibly decay. Tangential perturbations move the state along the cycle and remain as phase offset.

### Waveform

Shows a rolling oscillator waveform with an ideal reference overlay. Amplitude recovery and phase drift are visible in the same trace.

### Phase Error / Jitter

Shows unwrapped phase error in radians and zero-crossing timing error in picoseconds. Readouts use recent RMS values.

### Spectrum / Phase Noise

Shows an educational dBc/Hz-style display around the carrier. The curve combines a measured low-rate phase-error spectrum estimate with a white phase-diffusion floor and AM-to-PM contribution. Increasing noise broadens the carrier and raises the offset-noise readouts.

---

## Validation

Run:

```bash
python -m py_compile interactive/oscillator_phase_noise.py
uv run python interactive/oscillator_phase_noise.py
```

Manual checks:

- App opens on a local random port.
- Controls update plots without reload.
- Reset returns to defaults and clears traces.
- Radial perturbation decays back toward the limit cycle.
- Tangential perturbation persists mostly as phase error.
- Increasing noise broadens the spectrum and increases jitter.
- Narrow viewport stacks panels without text overlap.
