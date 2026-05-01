# Notebook 06 — Oscillators, VCOs, and mmWave Phase Noise

**Status:** Design spec (2026-05-01). Not yet implemented.
**Scope:** Source file at `marimo/notebooks/06_oscillators_vco.py`.
**Target length:** ~1400–1600 lines.

---

## 1. Purpose and position in the course

Notebook 06 asks: once you have a low-noise amplifier and a matched output stage, where does the local oscillator signal come from — and how clean can it be? It develops the complete theory of oscillator phase noise, from the historical Leeson model through the full Hajimiri-Lee ISF framework (Floquet theory, adjoint eigenvectors), then applies the results to mmWave-specific design constraints.

The Leeson model is introduced first as the established baseline, then its empirical parameters are exposed as limitations, and the ISF framework is developed as the principled resolution. Part III applies the ISF results directly to tank Q degradation, topology selection, and coupled oscillator arrays at mmWave frequencies.

### Position in the series

- **Notebooks 01–03** — two-port analysis, gain definitions, S-parameters, stability.
- **Notebook 04** — noise figure, four noise parameters, 28 GHz LNA design.
- **Notebook 05** — matching networks, Bode-Fano, cyclostationary noise in mixers.
- **Notebook 06** — oscillator phase noise theory and mmWave VCO design considerations (this notebook).
- **Notebook 07** — PA design, linearity (P1dB, IP3), frontend integration (deferred from this notebook).

### Bridge to notebook 07

Topics explicitly deferred to 07:
- Nonlinear device modeling (Volterra series, harmonic balance)
- PA load-pull and efficiency (Doherty, outphasing)
- Full phased-array / frontend integration (LNA + VCO + PA + mixer chain)

---

## 2. Backward dependencies

- **Notebook 01** — ABCD cascade, two-port state equations (used in §6 state-space formulation).
- **Notebook 02** — Available power, power gain (Leeson's P_sig is available signal power at the tank).
- **Notebook 03** — Closed-loop poles on the jω axis (Barkhausen); stability circles provide geometric intuition for the negative-resistance condition.
- **Notebook 04** — Noise figure F, cyclostationary noise sources (ISF upconversion of device noise in §9 directly parallels the cyclostationary mixer noise of notebook 05 §18–20).
- **Notebook 05** — Cyclostationary noise framework (notebook 06 §9 is a second application of the same LTV noise analysis).
- **VCO pulling interactive app** (`interactive/vco_pulling.py`) — Adler equation and injection locking dynamics are already treated there; §14 references that app rather than re-derives them.

---

## 3. Structure

```
Part I     Linear phase noise theory          §1–3
Part II    Nonlinear oscillator theory        §4–9
Part III   mmWave oscillator considerations   §10–14
Part IV    Wrap-up                            §15
```

Total: 15 top-level sections, four interactive visualisations.

---

## 4. Part I — Linear Phase Noise Theory (§1–3)

### §1. Oscillator fundamentals

Barkhausen criterion: loop gain = 1, loop phase = 0 (or 2πn).
Negative-resistance interpretation: −R_dev + R_tank = 0.
Open-loop poles of the linearised circuit sit on the jω axis at ±jω₀.
Why a real oscillator must be nonlinear: linear poles on the jω axis give undamped growth; amplitude stabilisation requires gain compression or an explicit limiter.

### §2. Leeson's model

Closed-loop noise analysis of a resonator (loaded Q_L) cascaded with an amplifier (noise figure F, signal power P_sig at the tank input).

Derivation sketch:
1. Model the tank as a bandpass filter with noise referred to its input.
2. Add the amplifier's input-referred noise: S_n = 2FkT.
3. Close the loop; compute the output phase noise spectrum.

Result:

```
L(Δω) = 10 log[ (2FkT / P_sig) · (1 + ω₀²/(4Q_L²Δω²)) · (1 + Δω_{1/f³}/|Δω|) ]
```

Identify the three spectral regions:
- 1/f³ region (close-in): device flicker noise upconverted to the carrier.
- 1/f² region: white noise filtered by the tank's Lorentzian response.
- Noise floor: far-from-carrier white noise, L → 2FkT/P_sig.

### §3. Leeson's limitations

Three empirical parameters with no first-principles values:
- F: called the "noise factor" but is not the same F as in Friis — it absorbs all nonlinear upconversion effects not modeled by the linear analysis.
- Q_L: which Q? Loaded tank Q, but loss mechanisms at mmWave are not captured by a single number.
- Δω_{1/f³}: the 1/f corner of the oscillator, which differs from the device 1/f corner due to cyclostationary weighting.

Motivation for Part II: the ISF framework gives first-principles, computable values for all three.

---

## 5. Part II — Nonlinear Oscillator Theory (§4–9)

### §4. Limit cycles

Van der Pol equation as the canonical nonlinear oscillator:

```
ẍ − ε(1 − x²)ẋ + ω₀²x = 0
```

Phase-space portrait for ε > 0: stable limit cycle, basin of attraction.
Perturbation decomposition: a noise kick displaces the state off the limit cycle.
- Amplitude perturbation: component perpendicular to the orbit trajectory; decays back exponentially (amplitude restoring force exists).
- Phase perturbation: component tangent to the orbit trajectory; does not decay (no phase restoring force).
- Physical consequence: amplitude noise is filtered; phase noise accumulates as a random walk → Lorentzian spectrum.

**Interactive I — Limit cycle visualiser**
Phase portrait of the Van der Pol oscillator. User controls ε and the magnitude/direction of a noise kick. Display shows: the limit cycle, the perturbed trajectory, and the decomposition of the kick into amplitude and phase components. A second panel shows the resulting φ(t) and A(t) time series to illustrate amplitude decay vs. phase diffusion.

### §5. Phase-voltage noise equipartition

Thermodynamic argument: a lossless LC tank in thermal equilibrium holds kT of total energy, equally split between electric and magnetic fields.

```
<v²_n> = kT/C    (electric field energy ½C<v²> = kT/2)
<i²_n> = kT/L    (magnetic field energy ½L<i²> = kT/2)
```

Consequence for oscillator design:
- The noise energy in the tank is fixed at kT, independent of L and C individually. Improving phase noise requires increasing the signal energy stored in the tank: E_sig = P_sig · Q / ω₀.
- The phase noise floor scales as kT / E_sig = kT ω₀ / (P_sig · Q).
- This is the physical origin of the P_sig · Q² figure of merit: doubling Q halves the noise-to-signal energy ratio at fixed power.

### §6. State-space formulation

Write the noisy oscillator as:

```
ẋ = f(x) + b(x) u(t)
```

where x ∈ ℝⁿ is the state vector, f(x) is the autonomous (noise-free) vector field, b(x) is the noise coupling vector (may depend on state for cyclostationary sources), and u(t) is white noise with spectral density S_u.

The noise-free system has a T-periodic solution x_s(t) = x_s(t+T) — the unperturbed limit cycle orbit.

### §7. Floquet theory

The linearisation of ẋ = f(x) about x_s(t) gives the T-periodic linear system:

```
δẋ = A(t) δx,    A(t) = ∂f/∂x|_{x_s(t)}
```

State transition matrix Φ(t, 0) satisfies Φ̇ = A(t)Φ.
**Monodromy matrix** M = Φ(T, 0).

Floquet multipliers: eigenvalues μᵢ of M.
- One multiplier μ₁ = 1 exactly (Floquet's theorem; corresponds to phase translation along the orbit).
- All other multipliers |μᵢ| < 1 for a stable limit cycle (amplitude modes decay).

The unit multiplier is the algebraic reason phase perturbations neither grow nor decay — they persist indefinitely, producing the 1/f² phase noise floor.

### §8. Adjoint eigenvectors and the ISF projection

Define the **adjoint** (transpose-conjugate) system:

```
ẏ = −A(t)ᵀ y
```

Its state transition matrix is [Φ(t,0)⁻¹]ᵀ. The left eigenvector v₁(t) of M corresponding to μ₁ = 1, normalised as v₁(t)ᵀ · (dx_s/dt) = 1, is the T-periodic phase sensitivity vector.

When a noise kick b(x_s(τ))δu(τ) arrives at time τ, the resulting phase perturbation is:

```
δφ(τ) = v₁(τ)ᵀ · b(x_s(τ)) δu(τ)
```

The **Impulse Sensitivity Function** is defined as:

```
Γ(ω₀τ) ≡ v₁(τ)ᵀ · b(x_s(τ)) / q_max
```

where q_max is the maximum charge displacement on the tank capacitor (normalisation).
Γ is dimensionless and T-periodic; it encodes how sensitive the oscillator's phase is to a noise impulse as a function of when in the cycle it arrives.

### §9. ISF to phase noise

Fourier decompose the ISF:

```
Γ(ω₀t) = c₀/2 + Σ_{n=1}^{∞} cₙ cos(nω₀t + θₙ)
```

A noise source with PSD i²_n/Δf (A²/Hz) coupled through Γ produces phase noise:

```
S_φ(Δω) = (i²_n/Δf) · Γ²_rms / (2 q²_max · Δω²)
```

where Γ²_rms = c₀²/4 + Σcₙ²/2.

Noise at frequency nω₀ + Δω (n = 0, 1, 2, …) is downconverted to Δω with weight cₙ²/2.
- c₀ term: DC noise (1/f device noise) upconverted to the carrier → 1/f³ region.
- c₁ term: noise at ω₀ downconverted → 1/f² region (dominant in white-noise region).
- cₙ terms (n ≥ 2): harmonics; usually smaller.

Reduction to Leeson: the ISF gives an explicit, computable noise factor:

```
F_eff = Γ²_rms · q²_max^{-1} · (kT R_p)^{-1} · (noise source sum)
```

and the 1/f³ corner is directly computable from c₀ and the device 1/f noise corner.

**Interactive II — ISF explorer**
User controls the oscillator waveform shape via Fourier coefficient sliders (fundamental + harmonics of the voltage waveform across the tank). The notebook computes Γ(t) numerically via the adjoint sensitivity formula (simplified to the single-state LC approximation for the interactive). Displays: Γ(t) waveform, cₙ bar spectrum, and L(Δω) phase noise curve. A toggle switches between a symmetric (cross-coupled-like) and asymmetric (Colpitts-like) waveform to preview the topology comparison of §12–13.

---

## 6. Part III — mmWave Oscillator Considerations (§10–14)

### §10. Tank Q at mmWave

**Inductor loss:**
- Series resistance R_s(f) ∝ √f (skin effect); Q_L = ωL/R_s ∝ √f, so Q_L rises with frequency in the skin-effect regime. Near the self-resonant frequency, parasitic shunt capacitance C_par reduces effective inductance and increases loss, causing Q_L to peak and then collapse.
- Substrate eddy currents in CMOS: induced currents in the lossy silicon substrate create an image inductance and additional resistance. Patterned ground shield (PGS) partially mitigates this.
- Self-resonant frequency (SRF): parasitic capacitance C_par shunts the inductor; above SRF the element is capacitive. Useful frequency range limited to f < SRF/2 to 2/3.

**Varactor (capacitor) loss:**
- Series resistance R_var; Q_C = 1/(ω C R_var).
- MOS varactor Q degrades rapidly above 30 GHz.

**Combined tank Q:**

```
1/Q_tank = 1/Q_L + 1/Q_C
```

The weakest element dominates. At mmWave, varactor Q is often the binding constraint.

**Interactive III — Tank Q breakdown**
Frequency sweep 1–100 GHz. Sliders: metal conductivity σ (proxy for process node), oxide thickness t_ox, substrate resistivity ρ_sub, varactor series resistance R_var, inductance L and capacitance C (sets f₀). Output: Q_L(f), Q_C(f), Q_tank(f) on a single plot, with a stacked bar at the user-selected operating frequency showing fractional loss contribution of each mechanism.

### §11. Phase noise FOM and the Q ceiling

Standard oscillator figure of merit:

```
FOM = L(Δω) − 20 log(f₀/Δω) + 10 log(P_DC / 1 mW)
```

Higher FOM is better (less negative). For fixed FOM, halving Q requires quadrupling P_DC to maintain the same phase noise — the power penalty of low-Q mmWave tanks.
Survey of published mmWave VCOs (28–60 GHz range): empirical FOM ceiling as a function of frequency, showing the Q-limited degradation trend.

### §12. Cross-coupled LC topology

Circuit: two cross-coupled transistors regenerate energy lost in R_p; differential pair ensures startup for gm > 1/R_p.

ISF analysis of the cross-coupled waveform:
- Differential symmetry: the voltage waveform has half-wave symmetry v(t + T/2) = −v(t).
- Half-wave symmetry zeros all even Fourier coefficients of Γ: c₀ = 0, c₂ = 0, c₄ = 0, …
- c₀ = 0 means 1/f device noise is NOT upconverted to 1/f³ phase noise (to first order).
- Dominant term is c₁ (noise at ω₀); the 1/f³ region is suppressed.
- Tail-current noise: the tail transistor couples noise at 2ω₀ via c₂ — but c₂ = 0 for a symmetric waveform. Tail filter (LC at 2ω₀) removes residual asymmetry-induced coupling.

### §13. Colpitts topology

Circuit: single transistor, capacitive voltage divider (C₁, C₂) for feedback. Startup condition: gm > (C₁+C₂)² / (ω₀² C₁² C₂ R_p).

ISF analysis of the Colpitts waveform:
- Asymmetric (impulsive) current waveform: transistor conducts in a short pulse near the voltage minimum.
- Non-zero c₀: 1/f noise upconverts to 1/f³ region — worse than cross-coupled.
- But: the pulsed conduction concentrates noise injection at the moment of minimum ISF magnitude (near the voltage peak), reducing Γ²_rms.
- Net result: worse 1/f³ but potentially better 1/f² phase noise (lower noise floor) at high offsets for the same P_DC.

**Topology comparison table:**

| Criterion               | Cross-coupled LC  | Colpitts          |
|-------------------------|-------------------|-------------------|
| Startup margin          | Moderate          | Requires gm boost |
| 1/f upconversion        | Suppressed (c₀=0) | Present (c₀≠0)    |
| High-offset PN          | Moderate          | Can be lower      |
| Tuning range            | Wide (varactor)   | Moderate          |
| mmWave suitability      | Standard choice   | Used at V-band+   |
| Tail noise sensitivity  | Needs tail filter | N/A               |

**Interactive IV — Phase noise budget tool**
Side-by-side Leeson vs. ISF model comparison. Sliders: tank Q, oscillation frequency f₀, DC power P_DC, device noise factor F (Leeson) / ISF Γ_rms and c₀ (ISF model), 1/f corner of device. Topology toggle: cross-coupled (c₀ = 0) vs. Colpitts (c₀ ≠ 0). Displays: L(Δω) curves for both models on the same axes; annotated budget breakdown showing contributions of 1/f³, 1/f², and noise floor regions; FOM readout.

### §14. Coupled oscillator arrays

**Superharmonic injection locking:**
An oscillator at 2f₀ injecting a signal into an f₀ oscillator. The f₀ oscillator locks its phase to half the injected phase. Locking bandwidth from the Adler equation (see `interactive/vco_pulling.py` for dynamics); phase noise within the locking bandwidth tracks the reference divided by N² (N = 2 here).

**N-element coupled arrays:**
N identical oscillators coupled symmetrically (resistive or injection). In-phase mode: output power scales as N², noise contributions from independent tanks add as N → phase noise improves as 1/N (10 log N dB) relative to a single oscillator.
Coupling path introduces additional noise; optimal coupling strength minimises the sum of tank noise and coupling noise.
Application: 28/60 GHz phased-array LO distribution using injection-locked divide-by-2 chains from a centralised 56/120 GHz VCO.

**Reference:** For Adler equation phase dynamics and injection locking transients, see `interactive/vco_pulling.py`.

---

## 7. Part IV — Wrap-up (§15)

### §15. Summary and bridge to notebook 07

Recap of the logical chain:
1. Leeson provides the spectral shape but not the coefficients (F, Q, 1/f corner are empirical).
2. Limit cycle analysis shows why phase noise accumulates while amplitude noise decays.
3. Equipartition gives the fundamental noise floor: kT energy split between E and H fields.
4. Floquet theory: the unit multiplier pins the origin of phase noise accumulation to the topology of the periodic orbit.
5. ISF (adjoint eigenvector projection) gives computable F, computable 1/f corner, and a design handle: shape the waveform to reduce Γ_rms and zero c₀.
6. At mmWave: Q collapse is the dominant constraint; cross-coupled LC with tail filtering is the standard architecture; Colpitts trades 1/f upconversion for lower noise floor at high offset.

Bridge to notebook 07: the LNA (notebook 04), matching network (notebook 05), and VCO (notebook 06) are the three receive-path blocks. Notebook 07 completes the transmit path: PA nonlinearity (P1dB, IP3, AM-PM), efficiency (class AB, Doherty), and frontend integration (full mmWave transceiver noise and linearity budget).

Navigation footer: Previous → `05_matching_networks.py` | Next → `07_pa_and_linearity.py`

---

## 8. Interactives summary

| # | Name                     | Section | Controls                                        | Outputs                              |
|---|--------------------------|---------|------------------------------------------------|--------------------------------------|
| I | Limit cycle visualiser   | §4      | ε, kick magnitude, kick angle                  | Phase portrait, φ(t), A(t)           |
| II| ISF explorer             | §9      | Waveform Fourier coefficients, topology toggle | Γ(t), cₙ spectrum, L(Δω)            |
|III| Tank Q breakdown         | §10     | σ, t_ox, ρ_sub, R_var, L, C, f_op             | Q_L/Q_C/Q_tank vs f, loss bar chart  |
|IV | Phase noise budget tool  | §13     | Q, f₀, P_DC, F/Γ_rms/c₀, topology toggle      | L(Δω) curves, FOM, budget breakdown  |

---

## 9. Verification checklist

Before marking notebook 06 complete:

1. `uv run marimo check marimo/notebooks/06_oscillators_vco.py` — only `markdown-indentation` warnings allowed.
2. `python -c "import ast; ast.parse(open('marimo/notebooks/06_oscillators_vco.py').read())"` — must pass.
3. `uv run python marimo/notebooks/06_oscillators_vco.py` — non-interactive smoke-test must succeed.
4. Manual smoke-test: `uv run marimo edit marimo/notebooks/06_oscillators_vco.py` — each of the four interactives renders and responds to input.

---

## 10. Out-of-scope (explicitly deferred)

- PA design, load-pull, Doherty/outphasing efficiency → notebook 07
- Linearity metrics (P1dB, IP3, AM-PM) → notebook 07
- Full frontend integration (LNA + VCO + PA + mixer chain budget) → notebook 07
- Injection locking transient dynamics (Adler equation) → `interactive/vco_pulling.py`
- EM-simulated inductor S-parameters (PDK-accurate Q models) → out of course scope
- Oscillator pulling from supply/substrate (covered in VCO pulling app) → `interactive/vco_pulling.py`
- Ring oscillator phase noise → out of course scope
- Oscillator-based ADCs / DCOs → out of course scope
