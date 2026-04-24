# Notebook 05 — Matching Networks, Broadband Design, and Cyclostationary Noise

**Status:** Design spec, approved outline (2026-04-23). Not yet implemented.
**Scope:** Source file at `marimo/notebooks/05_matching_networks.py`.
**Target length:** ~1200-1500 lines (main notebook); ~250 lines across two helper libraries.

---

## 1. Purpose and position in the course

Notebook 05 opens with the question notebook 04 left unanswered: **once you know where Γ_S and Γ_L must be, how do you build the matching network that gets you there?** It then broadens to ask how much bandwidth a lossless matching network can theoretically achieve (Bode-Fano), and concludes by treating the noise in the next block downstream of the LNA — the mixer — where cyclostationary analysis replaces the stationary framework of notebooks 04.

Notebooks 01-04 covered analysis and first-stage amplifier design. Notebook 05 covers synthesis: given a target impedance, design the network.

### Bridge forward

- **Notebook 06** will cover linearity (P1dB, IP3, AM-PM), PA design and efficiency, and integrate the noise + matching + PA pieces into a full mmWave frontend.

Topics explicitly deferred to 06:
- Nonlinear device modeling (Volterra series, harmonic balance)
- PA load-pull and efficiency
- Full phased-array / frontend integration

---

## 2. Backward dependencies

- **Notebook 01** — Z/Y/ABCD parameters (matching networks are two-ports; ABCD cascades directly)
- **Notebook 02** — Available gain G_A and transducer gain G_T (mismatch loss = G_T/G_A)
- **Notebook 03** — Γ-plane geometry, Smith chart, stability circles (stub design on Smith chart; gain circles used to verify output match quality)
- **Notebook 04** — Four noise parameters {F_min, R_n, Γ_opt, G_opt}, LNA §17.6 output match (tapped-cap topology revisited with distributed alternatives); Friis cascade (extended here to include mixer noise)

---

## 3. Structure

```
Part I     Impedance matching fundamentals        §1-3
Part II    Lumped matching networks               §4-7
Part III   Broadband matching                     §8-10
Part IV    Distributed and transformer matching   §11-14
Part V     mmWave output matching revisited       §15
Part VI    Cyclostationary noise                  §16-19
Part VII   Wrap-up                               §20
```

Total: 20 top-level sections, four interactive visualisations.

---

## 4. Part I — Impedance matching fundamentals (§1-3)

### §1. Motivation: mismatch loss and the power transfer theorem

Maximum power transfer: load receives P_avail only when Z_L = Z_S*.
Mismatch loss:

```
ML = 1 − |Γ_L|² = 4 R_S R_L / |Z_S + Z_L|²
```

Quality factor Q: series resonator Q = X/R, shunt resonator Q = B/G.
Bandwidth-Q tradeoff preview: for a single-resonator match, BW ≈ f₀/Q_loaded.
Motivation for going beyond the simple L-network when bandwidth or impedance ratio is extreme.

### §2. Network representations for matching

ABCD-matrix cascade rule for matching networks (from notebook 01).
Three representations equally valid: Z, Y, ABCD; matching problems are usually easiest in ABCD form because networks cascade by matrix multiplication.
Lossless constraint on ABCD: AD − BC = 1, A real, D real, B imaginary, C imaginary (for LC networks).
Reciprocal constraint: AD − BC = 1.

### §3. Smith chart review and matching trajectories

Series element moves along a constant-resistance circle. Shunt element moves along a constant-conductance circle.
Series-L: moves clockwise on constant-R circle. Series-C: counterclockwise. Shunt-L: counterclockwise on constant-G circle. Shunt-C: clockwise.
The matching procedure on the Smith chart is a sequence of arc moves from Z_L/Z₀ to the centre.
Purpose of §3: establish the visual grammar for Parts II and IV.

---

## 5. Part II — Lumped matching networks (§4-7)

### §4. L-network synthesis

Two topologies for transforming R_L to R_S (both real):

**Type 1 (step-up, R_S < R_L):** shunt reactive B_p at input, series reactive X_s in series arm.
```
Q = sqrt(R_H / R_L − 1)
X_s = Q · R_L       (series element)
B_p = Q / R_H       (shunt element, = 1/(Q · R_L))
```
Series element is inductive if stepping up from low-side; shunt element sign determines whether it's high-pass or low-pass.

**Type 2 (step-down, R_S > R_L):** mirror topology.

Frequency response: 2nd-order bandpass or low-pass depending on element choice.
3 dB bandwidth ≈ f₀ / Q.

### §5. π-network synthesis

Two back-to-back L-sections sharing a virtual resistor R_virt:
```
R_virt = min(R_S, R_L) / (1 + Q²)   →   Q > Q_min = sqrt(R_H / R_L − 1)
```
Design equations (high-pass / low-pass version; only low-pass shown):
```
C_1  = Q / (ω₀ R_S)
C_2  = Q / (ω₀ R_L)
L    = 1 / (ω₀² C_series)    where C_series = C_1·C_2/(C_1+C_2) for fixed topology
```
More precise: derive from two L-sections with Q_1, Q_2 such that Q_1 + Q_2 = Q_total.
Key property: π has two shunt elements and one series element — lower insertion loss for small R_L when the shunt branches can be physical capacitors.

### §6. T-network synthesis

Dual of π: two series elements and one shunt element.
Tapped-capacitor network as a degenerate T (capacitive voltage divider; the topology used in notebook 04 §17.6).
Design equations:
```
C_tap = C_total / n_tap²        (total capacitance transformed by tap ratio n_tap)
n_tap = sqrt(R_L / R_eff)
```
Full T-network element values from Q and R_S, R_L analogous to π.

### §7. Interactive I — L/π/T network explorer

Three-tab display (one tab per topology). Inputs:
- R_S (10-500 Ω), R_L (10-500 Ω)
- f₀ (1-100 GHz)
- Q_target (for π and T; greyed out for L)
- Element type selector: capacitor/inductor assignment for high-pass vs low-pass version

Outputs:
- **Tab 1:** Element values table (C₁, L₁, C₂ / L₂ depending on topology and type)
- **Tab 2:** Plotly dark — |S₁₁| (dB) and |S₂₁| (dB) vs frequency over [f₀/10, 10f₀]
- **Tab 3:** Smith chart (Γ_L trajectory from R_L/Z₀ to centre as elements are added)

---

## 6. Part III — Broadband matching (§8-10)

### §8. Bode-Fano criterion

For a parallel RC load:
```
∫₀^∞ ln(1 / |Γ(ω)|) dω  ≤  π / (R_L C)  =  π ω₀ / Q_load
```
Interpretation: the "area" under ln(1/|Γ|) over all frequencies is bounded by the load's RC time constant. Narrowing the band lets you push |Γ| → 0 there, at the cost of |Γ| → 1 everywhere else.

Chebyshev equal-ripple fill: achieves the bound by keeping |Γ| = |Γ_max| constant across the passband and |Γ| = 1 in the stopband.

Practical result for equal-ripple bandwidth BW:
```
BW · ln(1 / |Γ_max|) ≤ π f₀ / Q_load
```
→ trading reflection depth for bandwidth, bounded by the unloaded Q of the load.

For a series RL load (dual):
```
∫₀^∞ ln(1 / |Γ(ω)|) dω / ω²  ≤  π L / R_L  =  π / (ω₀ Q_load)
```

Qualitative: low-Q loads (small RC or small L/R) are easy to match broadly; high-Q loads (like a narrowband antenna) have a fundamental BW ceiling.

### §9. Chebyshev (maximally-flat ripple) matching network synthesis

Low-pass prototype element values g_k (from table; Pozar Table 5.5):
- Butterworth: g_k = 2 sin((2k−1)π / (2N))
- Chebyshev (ripple ε): computed from hyperbolic cotangent formulas (Pozar eq. 5.37-5.40)

Impedance and frequency scaling:
```
L_k = g_k · Z₀ / ω_c    (for series inductors in prototype)
C_k = g_k / (Z₀ ω_c)    (for shunt capacitors)
```

Bandpass transformation (series arm: series-LC; shunt arm: parallel-LC):
```
L_series = L_k · Z₀ / BW
C_series = BW / (ω₀² L_series)
L_shunt  = Z₀ · BW / (ω₀² C_k · BW)
C_shunt  = C_k / (Z₀ · BW)
```

Worked example: N=3 Chebyshev, 0.1 dB ripple, 50Ω → 200Ω, f₀ = 28 GHz, BW = 3 GHz.

### §10. Interactive II — Broadband Chebyshev explorer

Inputs:
- Prototype type (Butterworth / Chebyshev)
- N (2-6)
- Ripple ε² in dB (0.01-3 dB; only for Chebyshev)
- R_S, R_L (10-500 Ω)
- f₀ (1-100 GHz), BW (fraction of f₀, 1%-50%)

Outputs:
- Element values table (N elements, labelled C₁, L₁, C₂ …)
- Plotly dark: |S₁₁| (dB) and |S₂₁| (dB) vs frequency over [f₀ − 3BW, f₀ + 3BW]
- Bode-Fano bound overlay: theoretical minimum |Γ_max| for this BW and Q_load (computed from §8 equation, shown as horizontal dashed line on |S₁₁| plot)

---

## 7. Part IV — Distributed and transformer matching (§11-14)

### §11. Quarter-wave transformer

Matching R_S to R_L at frequency f₀:
```
Z₁ = sqrt(R_S · R_L),   ℓ = λ/4
```
Reflection coefficient vs electrical length θ:
```
|Γ(θ)| = |Γ_0| / sqrt(1 + (2 Γ_0 / sin 2θ)²)   (approximate; Pozar eq. 5.47)
```
Bandwidth to |Γ_max|:
```
θ_m = arccos(sqrt(Γ_max² / (1 − Γ_max²)) · |2 Z₁ / (Z₁² / R_S − R_S)|)
BW/f₀ = 2 − 4θ_m/π
```
Multi-section Chebyshev transformer: N λ/4 sections, impedances Z_k from Chebyshev design (Pozar §5.7).

### §12. Single-stub tuner (parallel stub)

Given complex load Z_L at frequency f₀, find d (distance from load to stub) and ℓ (stub length) such that Y_in = 1/Z₀.

Two solutions (Pozar §5.2):
```
t = tan(βd)   solves:   t² + t·(R_L/Z₀) · (R_L/Z₀ − 1) + X_L²/Z₀² − X_L/Z₀ = 0
(two roots per period)
```
For each d, stub length:
```
Y_stub = −B_in(d)   →   ℓ = arctan(B_stub / Y₀) / β   (short-circuit stub)
                         ℓ = −arctan(Y₀ / B_stub) / β  (open-circuit stub)
```
Choose the solution with shorter total line length (d + ℓ).
Bandwidth: sensitivity to B(ω) variation limits useful BW to ≈ 10-15% for single stub.

### §13. Double-stub tuner

Two parallel stubs at positions 0 and d = λ/8 (or λ/4).
Forbidden load conductance region: loads with G_L > 2Y₀ cannot be matched with d = λ/8 spacing (Pozar §5.3).
Design equations (two steps — stub 1 cancels susceptance; stub 2 sets conductance):
```
B_1 chosen such that G(d) = Y₀ after transforming Y_L + jB_1 through length d.
B_2 = −B(d) to cancel remaining susceptance.
```
Practical note: double stub is used in adjustable bench tuners; for IC design, single stub or lumped networks preferred.

### §14. On-chip transformer coupling

Mutual inductance model: k = M / sqrt(L_1 L_2), coupling coefficient.
Turns ratio: n = sqrt(L_1 / L_2) ≈ N_turns,1 / N_turns,2 for solenoidal; approximate for planar spiral.
Impedance transformation: looking into primary with load Z_L on secondary:
```
Z_in = jωL_1(1 − k²) + k² n² Z_L   (ideal transformer limit k → 1: Z_in = n² Z_L)
```
Winding resistance model: r₁ = ωL_1/Q₁, r₂ = ωL_2/Q₂.
NF penalty from lossy matching network (Friis for two-port with resistive loss):
```
ΔNF ≈ 10 log(1 + (1 − k²) / (k² · G_S · n² · R_2))   [dB]
```
At 28 GHz in 65 nm CMOS: Q_spiral ≈ 10-15; k ≈ 0.7-0.8 for 2-turn / 2-turn proximity coupling.
Differential / balun configuration: 1:2 turns ratio converts single-ended to differential — ΔNF ≈ 0.5-1 dB extra from loss.

### §15. Interactive III — Distributed matching explorer (Smith chart)

Inputs:
- Z_L (magnitude + phase of Γ_L)
- Z₀ (default 50 Ω)
- f₀ (1-100 GHz)
- Matching method (quarter-wave TL / single stub / double stub)

On the Smith chart (Plotly dark, `xaxis=dict(scaleanchor="y", scaleratio=1)`):
- Z_L plotted as initial point
- Matching trajectory (arcs for TL rotation, jumps for stub additions)
- Final matched point (should land at centre)
- For double stub: forbidden load region shaded in pale red

Sidebar: element values (d, ℓ in mm and as fraction of λ at f₀), |S₁₁| at f₀ confirmed to be < −30 dB.

---

## 8. Part V — mmWave output matching revisited (§15)

### §16. Revisiting the 28 GHz LNA §17.6 output match

Three output matching topologies applied to the same design point from notebook 04 §17.7:
- **Tapped capacitor** (notebook 04 §17.6): narrowband, no inductors in signal path, low NF penalty
- **λ/4 TL transformer**: broadband, 50 Ω microstrip, insertion loss ≈ 0.3 dB at 28 GHz → ΔNF ≈ 0.3 dB (referred to output, negligible for LNA)
- **On-chip spiral transformer** (1:1, k=0.75, Q=12): ΔNF ≈ 0.5 dB; enables differential output for mixer drive

Comparison table:

| Topology | BW (3dB) | ΔNF (dB) | Area (μm²) | Suitable for |
|---|---|---|---|---|
| Tapped cap | ~10% | <0.1 | Small | single-ended narrowband |
| λ/4 TL | ~20% | ~0.3 | Medium | single-ended broadband |
| Transformer | ~15% | ~0.5-1.0 | Large | differential, balun |

Γ_S-plane overlay: gain circle at notebook-04 MAG, then show how each topology's loss shifts the effective load circle.

---

## 9. Part VI — Cyclostationary noise (§17-20)

### §17. Cyclostationary processes

Periodic stationarity: R_x(t, τ) = E[x(t)x(t+τ)] satisfies R_x(t,τ) = R_x(t+T, τ) for T = 1/f_LO.
Fourier expansion:
```
R_x(t, τ) = Σ_{n=−∞}^{∞} R_n(τ) e^{jnω_LO t}
```
Cyclostationary PSD (double frequency representation):
```
S_x(f, f') = Σ_n S_n(f) δ(f' − f + n f_LO)
```
where S_n(f) = ∫ R_n(τ) e^{−j2πfτ} dτ are the cyclic spectral densities.

Why stationary analysis fails in a mixer: noise at f_s ± n f_LO all fold to the same IF frequency; treating each component independently misses the coherent combination.

Conversion matrix C: a periodically time-varying system with impulse response h(t,τ) = h(t+T,τ) maps input spectral vector a = [A(f−f_LO), A(f), A(f+f_LO), …] to output vector b = C · a. For noise, C maps the input noise spectral matrix S_in to output S_out = C S_in C†.

### §18. Mixer noise theory

MOSFET switch model: g_ds(t) ≈ g_on during LO+ half-cycle, ≈ g_off during LO− (square-wave drive approximation).
Time-varying conductance: g(t) = Σ_n g_n e^{jnω_LO t}.
Dominant Fourier coefficients: g_0 (DC), g_1 (fundamental), g_−1 = g_1* for real g(t).

Channel thermal noise current power: S_i(f) = 4kT · g_ds(t) → cyclostationary, with PSD proportional to g(t).

Noise figure derivations:
- **DSB NF:** both sidebands (f_IF and −f_IF after downconversion) carry equal noise → DSB NF = NF of conversion + thermal noise from g_on
- **SSB NF:** only upper sideband is signal; image noise adds → SSB NF = DSB NF + 3 dB (for ideal balanced mixer)
- **Image noise:** if image band is not rejected (no pre-filter), image thermal noise folds into IF → SNR penalty

Friis with mixer (extending notebook 04 §11):
```
F_total = F_LNA + (F_mixer − 1) / G_A,LNA
```
For typical LNA: G_A,LNA ≈ 15 dB, so F_mixer − 1 is suppressed by 32× → mixer NF matters less than the LNA.

Noise temperature extension: T_e,mix = T_conversion + T_image_fold.

### §19. Switched-capacitor (sampler) noise

Track-and-hold circuit: switch resistance R_sw, hold capacitor C_h.
During track: system is an RC low-pass filter; noise BW = 1/(4R_sw C_h).
Total integrated noise on capacitor:
```
<v_n²> = kT / C_h
```
This is independent of R_sw — the "kT/C noise" result.

Noise folding from under-sampling: if signal BW = B and sample rate f_s = 1/T_s, then N = B/f_s frequency bins fold into baseband.
Effective noise floor rises by 10 log(N) dB relative to continuous-time receiver.

Settling requirement: for N-bit accuracy, τ_settle = R_sw C_h ≪ T_s / (2N ln 2).

Practical implication: at 28 GHz LO and GHz-range IF bandwidth, direct RF sampling requires extreme settling → motivates superheterodyne architectures.

### §20. Interactive IV — Mixer noise explorer

Inputs:
- LO frequency f_LO (1-100 GHz)
- IF bandwidth BW_IF (10 MHz - 1 GHz)
- Conversion loss L_c (dB, 3-15 dB)
- Image rejection ratio IRR (dB, 0-60 dB)
- LNA gain G_A (dB) and NF (dB) from notebook 04 (preset to 28 GHz design values)
- N cascade stages total (1-5)

Outputs:
- **DSB NF** and **SSB NF** of the mixer alone (dB)
- **Friis cascade** table: each stage's NF, gain, contribution to total NF (analogous to Interactive II.1 from notebook 04)
- **Noise folding penalty** (dB) for direct sampling at f_s = f_LO (sampler mode)
- Plotly dark bar chart: per-stage noise contribution to total system NF

---

## 10. Part VII — Wrap-up (§20 → §21)

### §21. Summary and bridge to notebook 06

The complete receiver noise analysis chain is now assembled:
1. LNA: Γ_S placed near Γ_opt via inductive degeneration (notebook 04)
2. Matching network delivers specified Γ_L with bounded bandwidth (Bode-Fano)
3. Mixer converts RF to IF with SSB NF = DSB NF + 3 dB (this notebook)
4. Friis shows LNA dominates; broadband matching adds ΔNF ≤ 0.5-1 dB

Open problems for notebook 06:
- Nonlinearity: 1 dB compression, IP3, AM-PM
- PA design: load-pull, drain efficiency, back-off
- Full mmWave frontend: LNA + matching + mixer + PA integrated

Concept-dependency SVG diagram (matching → notebook 06 PA output matching; cyclostationary → notebook 06 PA nonlinear noise).

---

## 11. Interactives inventory

| # | Interactive | Section | Purpose |
|---|---|---|---|
| I.1 | L/π/T network explorer | §7 | Match element values to frequency response |
| II.1 | Broadband Chebyshev explorer | §10 | Bode-Fano bound + filter-table synthesis |
| III.1 | Distributed matching explorer | §15 | Smith-chart stub/TL design |
| IV.1 | Mixer noise explorer | §20 | Cyclostationary NF, image folding, Friis |

All follow repo conventions (`template="plotly_dark"`, `mo.ui.plotly(fig)`, Smith-chart panels use `xaxis=dict(scaleanchor="y", scaleratio=1)`, SVG markers with unique id suffixes).

---

## 12. Name-collision map

Variables that appear across interactives, with context suffixes:

- **Interactive I.1** (L/π/T): `f0_m`, `Q_m`, `Rs_m`, `Rl_m`, `elements_m`
- **Interactive II.1** (Chebyshev): `f0_bw`, `Q_load_bw`, `N_bw`, `ripple_bw`, `elements_bw`
- **Interactive III.1** (stub/TL): `Gamma_L_s`, `f0_s`, `d_stub`, `ell_stub`
- **Interactive IV.1** (mixer): `f_LO_mx`, `NF_DSB_mx`, `NF_SSB_mx`, `G_A_mx`, `F_total_mx`

---

## 13. Navigation footer

Per CLAUDE.md convention, notebook 05 ends with a `mo.md` cell:
- Previous → `04_noise_and_lna_design.py`
- Next → `06_linearity_and_pa_design.py` (placeholder until 06 is named)

When 05 is merged, update the footer of 04 to point Next → `05_matching_networks.py`.

---

## 14. Verification checklist

Before marking notebook 05 complete:

1. `uv run marimo check marimo/notebooks/05_matching_networks.py` — only `markdown-indentation` warnings allowed.
2. `python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"` — must pass.
3. `uv run python marimo/notebooks/05_matching_networks.py` — non-interactive smoke-test must succeed.
4. Manual smoke-test: `uv run marimo edit marimo/notebooks/05_matching_networks.py` — each of the four interactives renders and responds to input.

---

## 15. Out-of-scope (explicitly deferred)

- Nonlinear device modeling (Volterra, harmonic balance) → 06
- PA load-pull and Doherty efficiency → 06
- Full phased-array frontend integration → 06
- Image-rejection architectures (Hartley, Weaver) beyond a mention → 06
- PDK-accurate inductor models (EM-simulated S-parameters) → out of course scope
- Noise analysis of Σ-Δ oversampled ADC → out of course scope
