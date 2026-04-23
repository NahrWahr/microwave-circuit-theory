# Notebook 04 — Noise Analysis and Low-Noise Amplifier Design

**Status:** Design spec, approved outline (2026-04-22). Not yet implemented.
**Scope:** Source file at `marimo/notebooks/04_noise_and_lna_design.py`.
**Target length:** ~1400-1600 lines (longest of the six notebooks, driven by the level-iii stochastic-foundations intro and the worked 28 GHz design).

---

## 1. Purpose and position in the course

Notebook 04 opens the course's **design-oriented** phase. Notebooks 01-03 built the small-signal, gain, and stability machinery for passive and active two-ports as *analysis* tools. Notebook 04 reuses that machinery to make the first design-level quantitative choice in the course: **given a device and a frequency, where should we place Γ_S to minimise noise subject to gain and stability constraints?**

The notebook is the direct successor of the gain–stability arc in notebooks 02 and 03. Noise circles overlay on the same Γ_S-plane as the gain and stability circles, and simultaneous noise-plus-power match is the next optimisation problem after MAG. The notebook ends with a concrete 28 GHz, 65 nm CMOS LNA walkthrough — the first artifact in the course that looks like a real mmWave IC.

### Bridge forward

- **Notebook 05** will cover matching networks (L/π/T synthesis, broadband techniques, transformer coupling), the noise in mixers/samplers (cyclostationary deep dive), and revisit the §17.6 output match using real distributed-matching tools.
- **Notebook 06** will cover linearity (P1dB, IP3, AM-PM), PA design, and integrate the noise + matching + PA pieces into a full mmWave frontend.

Topics that are briefly mentioned in notebook 04 but explicitly deferred to 05 or 06:

- Cyclostationary noise (mixers, samplers, switched-cap) → 05 / 06
- Distributed and transformer-based matching networks → 05
- PA-specific nonlinearities and efficiency → 06
- Full phased-array / frontend integration → 06

---

## 2. Backward dependencies

The notebook assumes the reader has worked through:

- **Notebook 01** — Z/Y/H/ABCD two-port parameters (used implicitly in the noise correlation-matrix transformations), N-port power formulation (the basis for `<v²>` / `<i²>` / power-spectral arguments).
- **Notebook 02** — Gain definitions with emphasis on **available gain G_A** (used in Friis cascade and in overlaying available-gain circles with noise circles), signal flow graphs (for the Rothe-Dahlke derivation).
- **Notebook 03** — Γ-plane geometry, stability circles, gain circles, MAG, load-pull. The "noise / gain / stability circle explorer" in §15 is the direct successor of the notebook 03 stability-circle explorer.

No new prerequisites beyond 01-03. The stochastic-process foundations are built from scratch in Part I.

---

## 3. Structure

```
Part I     Stochastic foundations of noise         §1-6
Part II    Noise in two-ports                       §7-11
Part III   Noise matching                           §12-15
Part IV    LNA design — 28 GHz CMOS                 §16-19
Part V     Wrap-up                                  §20
```

Total: 20 top-level sections, four interactive visualisations.

---

## 4. Part I — Stochastic foundations of noise (§1-6)

Purpose: give the reader the stochastic-process vocabulary needed to read the advanced two-port noise literature (Rothe-Dahlke, Twiss, Haus-Adler) without hand-waving. Level-iii depth per user selection.

### §1. Motivation — why noise matters at mmWave

Receiver sensitivity; link budget dominated by noise figure F at low signal power; thermal noise floor kT₀B. Preview of F_min as the ultimate figure of merit for a device and the headline bridge from physics → two-port parameters → design.

### §2. Random processes, stationarity, ergodicity

Random process definition as an indexed ensemble of sample paths. Strict-sense vs wide-sense stationarity. Ergodicity — the empirical bridge that lets an experimenter replace ensemble averages with time averages on a single realisation. Worked examples: white Gaussian process (ergodic), random DC bias (non-ergodic).

### §3. Autocorrelation and Wiener-Khinchin

Autocorrelation R(τ) = E[x(t)x(t+τ)], basic properties (R(0) ≥ 0, Hermitian symmetry, decay for mixing processes). Power spectral density S(f) = ℱ{R(τ)}. Wiener-Khinchin theorem — proof sketch via the limiting argument on windowed periodograms. Closing paragraph on **cyclostationary noise**: one paragraph explaining the time-periodic autocorrelation idea, stating where it shows up (mixers, samplers, switched-capacitor circuits), and forward-referencing notebook 05/06 for the full treatment.

### §4. Vector processes and matrix-valued PSDs

Cross-correlation R_xy, cross-PSD S_xy. Matrix-valued PSD **S**(f) for vector random processes. Correlation coefficient / magnitude-squared coherence. This section exists specifically to set up the **noise correlation matrix C_A** in Part II — the reader should finish §4 comfortable with the idea that two correlated noise sources are described by a 2×2 Hermitian matrix of PSDs, not two scalars plus a "correlation coefficient" floating separately.

### §5. Physical noise sources

Four subsections:
- **§5.1 Thermal (Nyquist).** Derivation from equipartition of energy on a transmission-line cavity. Result `<v²> = 4kTRΔf`. One paragraph on the Johnson-Nyquist experiment.
- **§5.2 Shot noise.** Poisson statistics of carrier transit. `<i²> = 2qIΔf`. Note on when it applies (barriers: diodes, bipolar junctions) vs. when it doesn't (ohmic resistors — only thermal).
- **§5.3 Flicker (1/f) noise.** Phenomenological 1/f PSD. Surface-trap model (McWhorter). Corner frequency f_c. Practical scaling with device area in MOS (K_f/(WLC_ox f)).
- **§5.4 Generation-recombination noise.** Brief — Lorentzian-shaped spectrum from a single trap, sum over traps recovers 1/f.

### §6. Equivalent circuit-level noise sources

Noisy resistor = noiseless R in series with `v_n` (or parallel with `i_n`) where `S_v = 4kTR`, `S_i = 4kT/R`. Extension: noisy two-port = noiseless two-port + input-referred (v_n, i_n), generally correlated. Sum rules: S_total = Σ S_i for uncorrelated sources. Noise bandwidth vs. equivalent noise bandwidth (distinction matters for integrating detectors and oscillators).

### Interactive I.1 — Time-domain / autocorrelation / PSD triptych

Three-panel figure, `template="plotly_dark"`:
- **Left:** time-domain realisation x(t)
- **Centre:** estimated autocorrelation R̂(τ)
- **Right:** estimated power spectral density Ŝ(f)

User selects noise type: `white | flicker (1/f) | band-limited thermal | shot`. Also adjustable: sample length N, sample rate f_s. Purpose: make §2-3-5 concrete by showing the three representations of the same signal simultaneously.

---

## 5. Part II — Noise in two-ports (§7-11)

Purpose: reduce noisy linear two-ports to four real numbers {F_min, R_n, Γ_opt (complex)} and establish Friis for cascades.

### §7. The Rothe-Dahlke noise two-port

Theorem: any noisy linear two-port ≡ noiseless two-port + two correlated noise sources (v_n, i_n) at the input. Derivation starts from noisy Y-parameters (currents have internal noise sources) and pushes all noise to the input via the network's own transfer function. Introduce the **noise correlation matrix C_A** (admittance-form C_Y and impedance-form C_Z are stated as alternatives; all derivations stay in C_A).

### §8. Noise figure F

Classical definition F = SNR_in / SNR_out at standard-source temperature T₀ = 290 K. Equivalent noise temperature T_e = (F-1)T₀. When the receiver community prefers T_e over F (radio astronomy, deep-space links — sources well below 290 K make F notation counterintuitive).

### §9. F as a function of source admittance — the four noise parameters

Centrepiece derivation of Part II. From the C_A representation:

```
F = F_min + (R_n / G_s) · |Y_s − Y_opt|²
```

Γ-plane equivalent:

```
F = F_min + (4 R_n / Z_0) · |Γ_s − Γ_opt|² / ((1 − |Γ_s|²) · |1 + Γ_opt|²)
```

Conclusion: {F_min, R_n, Γ_opt} (four real numbers) completely characterise the noise of a two-port. This is the parameterisation the rest of the notebook uses.

### §10. Noise correlation matrix transformations

How C_A, C_Y, C_Z transform into each other via the same T-matrices that transform small-signal two-port parameters (since noise sources are simply additional current/voltage sources driven into the network). Why this matters for CAD: cascade and feedback analysis become matrix operations. One worked example — the noisy resistor's C_Z — establishes the pattern without a full CAD-style walkthrough. Deeper matrix machinery deferred to the reader's reference text (Hillbrand-Russer or Pozar).

### §11. Cascaded noise — Friis formula

Present the derivation twice:
1. **First via noise temperature** (simpler intuition): T_e,tot = T_e1 + T_e2/G_A1 + T_e3/(G_A1 G_A2) + …
2. **Then translate to F form:** F_tot = F_1 + (F_2 − 1)/G_A1 + (F_3 − 1)/(G_A1 G_A2) + …

Emphasise **why G_A** (available gain) and not G_T or G — cascade bookkeeping is cleanest when each stage is described by its available output noise power, which is source-match-independent. Practical consequence stated plainly: **the first stage dominates**, which is the reason the whole notebook is centred on LNA noise matching.

### Interactive II.1 — Cascade explorer

User sets N = 3 stages, each with (F_i in dB, G_Ai in dB). Live readout of F_tot (dB), per-stage noise-contribution breakdown (what fraction of F_tot - 1 each stage contributes), and a stage-reorder button that swaps stage positions to show how non-commutative the cascade is.

---

## 6. Part III — Noise matching (§12-15)

Purpose: place noise, gain, and stability circles on a single Γ_S-plane and let the reader see the design tradeoff geometrically.

### §12. Noise circles

From §9's F-expression, setting F = F_target and solving for the Γ_s locus gives a circle. Derive centre C_F and radius r_F:

```
N     = ((F − F_min) / (4 R_n / Z_0)) · |1 + Γ_opt|²
C_F   = Γ_opt / (1 + N)
r_F   = sqrt(N² + N(1 − |Γ_opt|²)) / (1 + N)
```

Small F_target → tight circle around Γ_opt; large F_target → circle eventually encloses the whole unit disk.

### §13. The gain-noise tradeoff

The fundamental design tension: **Γ_opt ≠ S₁₁\*** in general, so noise-matching costs gain and power-matching costs noise figure. Overlay available-gain circles (from notebook 03) on noise circles in Γ_S-plane — the geometric picture of the compromise. Introduce the **noise measure** M (Haus-Adler) as the invariant that collapses the tradeoff into a single number. State without full derivation: infinite cascades of identical stages approach F = 1 + M as G → ∞. M is the right figure of merit when you care about "how good can a noisy amplifier ever be."

### §14. Simultaneous noise + gain match

When is it achievable? The Haus condition — roughly, Γ_opt must be reachable from S₁₁\* by a lossless transformation. At mmWave on CMOS, it usually isn't exactly achievable, so the designer picks Γ_S on a locus minimising a weighted cost α·F + β·(1/G_A). Inductive source degeneration is introduced as **the** classic feedback trick that moves Γ_opt toward S₁₁\* — forward-reference to §16-17 where the reader actually applies it.

### §15. Interactive — Noise / Gain / Stability circle explorer

The centrepiece visualisation of Part III. Inputs:

- S-parameters (S₁₁, S₁₂, S₂₁, S₂₂) at design frequency (complex-number input via magnitude + phase sliders)
- Noise parameters: F_min (dB), Γ_opt (magnitude + phase), R_n/Z₀

On the Γ_S-plane (unit disk, `xaxis=dict(scaleanchor="y", scaleratio=1)`):

- Source stability circle with the stable region shaded
- Family of **available-gain circles** at user-chosen levels (default: MAG, MAG − 1 dB, MAG − 2 dB, MAG − 3 dB)
- Family of **noise circles** at user-chosen levels (default: F_min, F_min + 0.5 dB, F_min + 1 dB, F_min + 2 dB)
- Draggable Γ_S marker with live readouts: F (dB), G_A (dB), |Γ_in|, K

The reader uses this tool to see "I can give up 1 dB of gain to drop 0.3 dB of NF" directly on the plane.

---

## 7. Part IV — LNA design case study, 28 GHz CMOS (§16-19)

Purpose: turn the preceding theory into a concrete artifact. Technology: 65 nm CMOS, textbook Shaeffer-Lee device model (not a PDK BSIM extraction). The idealised physics is kept readable; deviations from PDK reality get a dedicated section (§19).

### §16. The inductively-degenerated common-source LNA

Four subsections.

#### §16.1 Topology and rationale

Common-source device with source inductor L_s, gate inductor L_g, load network. Motivation: this is the only simple configuration that creates a **real** input impedance without adding a lossy resistor — a resistor would contribute thermal noise and ruin NF. Source inductive degeneration gives Re(Z_in) = ω_T L_s without resistive loss.

#### §16.2 Small-signal model at mmWave

Standard π-hybrid FET: g_m, C_gs, C_gd, r_g, r_o. Parasitics that matter above ~20 GHz: gate resistance r_g (multi-finger layout), substrate resistance, pad capacitance. Textbook Shaeffer-Lee model used throughout (§19 handles the deltas vs. PDK).

#### §16.3 Input impedance derivation

To first order (neglecting C_gd):

```
Z_in ≈ jω(L_g + L_s) + 1/(jω C_gs) + ω_T L_s
```

Real part `ω_T L_s` set equal to 50 Ω by choice of L_s. Resonant imaginary part zeroed at operating frequency by choice of L_g. Crucial property: **no resistor in the input path**, so no thermal-noise penalty.

#### §16.4 Noise analysis

Two MOSFET noise sources:
- **Channel thermal noise** (parameter γ; ≈ 2/3 long-channel, 1-2 for short-channel at mmWave)
- **Induced gate noise** (parameter δ; correlated with channel via c ≈ j0.395)

Apply the C_A machinery from §10 to the inductively-degenerated CS stage. Recover the classical Shaeffer-Lee result:

```
F_min ≈ 1 + (2.4 γ / α) · (ω / ω_T)
```

Key structural result: the matched Γ_opt is approximately at S₁₁\*. That is, **inductive source degeneration is the geometrical trick** that pushes Γ_opt toward the power-match point, making simultaneous noise-plus-gain match nearly achievable in practice.

### §17. Worked design — 28 GHz, 65 nm CMOS LNA

Seven-step concrete design. Target spec:

- NF < 2.5 dB
- S₂₁ > 15 dB
- |S₁₁| < −10 dB
- P_DC < 10 mW
- 50 Ω terminations, operating frequency 28 GHz

Steps:

1. **§17.1 Device sizing.** Choose current density J_opt (minimum-NF point, ≈ 0.15 mA/μm in 65 nm). Scale W to hit P_DC budget. Extract g_m, C_gs, f_T from model equations — numbers the reader can reproduce.
2. **§17.2 Choose L_s for Re(Z_in) = 50 Ω.** L_s = 50 / ω_T.
3. **§17.3 Choose L_g for resonance at 28 GHz.** L_g = 1/(ω² C_gs) − L_s.
4. **§17.4 Verify noise match.** Compute Γ_opt from the C_A result; confirm it sits near S₁₁\*.
5. **§17.5 Cascode stage.** Add common-gate cascode for reverse isolation and stability. Noise cost ≈ 0.1-0.3 dB; gain and K-factor benefit.
6. **§17.6 Output match.** Tapped-capacitor network transforming 50 Ω to the load Γ_L that maximises G_L (using notebook 03's gain-circle result). Deliberately simple — broadband and transformer matching are deferred to notebook 05.
7. **§17.7 Verification plots.** S-parameters and NF vs frequency over 20-40 GHz; Rollett K vs frequency; noise/gain/stability circles at 28 GHz with final Γ_S overlaid. Every number traceable to the preceding steps.

### §18. Interactive — 28 GHz LNA design studio

The culminating interactive. Sliders:

- W (device width, 20-200 μm)
- L_s (0-100 pH)
- L_g (0-500 pH)
- J (bias current density, 0.05-0.3 mA/μm)
- Cascode enable (toggle)
- Operating frequency (20-40 GHz)

Displays:

- **Top panel:** |S₁₁|, |S₂₁|, |S₂₂|, NF vs frequency 20-40 GHz
- **Middle panel:** K, |Δ| vs frequency
- **Bottom panel:** Γ_S-plane at centre frequency with stability + gain + noise circles, Γ_opt and S₁₁\* marked, current Γ_S marker
- **Sidebar readout:** g_m, f_T, I_D, P_DC, per-source noise contribution breakdown (channel / induced-gate / cascode / L_g loss), total NF

Reader should be able to land on a slider configuration that reproduces the §17 design, then explore sensitivities.

### §19. Practical mmWave realities

What separates the idealised notebook design from silicon.

- **Inductor Q.** On-chip spiral / transmission-line inductors have Q ≈ 10-20 at 28 GHz. Loss couples directly to an NF penalty — one worked subsection computes ΔNF from finite-Q L_g and L_s for a representative Q = 12 at 28 GHz (numerical result produced in the notebook, not asserted here). Mitigation: elevated-inductor / transformer techniques (notebook 05).
- **Gate resistance layout.** r_g ∝ R_□ · W_f / (12 · N_f²) for multi-finger layout with contacts both ends. NF-vs-finger-count plot showing diminishing returns beyond N_f ≈ 20-40.
- **Substrate coupling.** Pattern-ground-shield; deep-nwell well-isolation — one paragraph with forward-reference.
- **PDK vs textbook model.** What Shaeffer-Lee gets wrong: short-channel velocity saturation (g_m-vs-V_GS plateaus earlier), non-quasi-static effects above f_T / 3 (C_gs becomes frequency-dependent), γ's frequency dependence.
- **Beyond 28 GHz.** One paragraph each on 60, 77, 140 GHz: inductor Q drops, r_g dominates NF, pad/interconnect parasitics raise the noise floor. Forward-reference to notebook 05 (matching tricks) and 06 (PA design).

---

## 8. Part V — Wrap-up (§20)

### §20. Summary and bridge to notebooks 05 and 06

- The unified geometric picture: gain, noise, and stability all as circles on the Γ_S-plane.
- Friis → first-stage dominance → LNA as the defining block of any receiver.
- Open problems for notebook 05: (i) broadband matching, (ii) transformer and distributed techniques, (iii) mixer and sampler noise (cyclostationary).
- Open problems for notebook 06: linearity, PA, full frontend integration.
- Concept-dependency diagram (SVG) showing which sections of notebook 04 feed which topics in 05 and 06.

---

## 9. Interactives inventory

| # | Interactive | Section | Purpose |
|---|---|---|---|
| I.1 | Time / R(τ) / PSD triptych | §6 (end of Part I) | Make stochastic-process concepts concrete |
| II.1 | Friis cascade explorer | §11 (end of Part II) | Show first-stage dominance + stage-order sensitivity |
| III.1 | Noise / gain / stability circle explorer | §15 (end of Part III) | Centrepiece tradeoff visualisation |
| IV.1 | 28 GHz LNA design studio | §18 | Culminating design-space explorer |

All four follow the repo conventions (per `CLAUDE.md`):

- `template="plotly_dark"`
- Displayed via `mo.ui.plotly(fig)`, not bare `fig`
- Smith-chart / Γ-plane panels use `xaxis=dict(scaleanchor="y", scaleratio=1)`
- SVG markers (arrow-heads etc.) use unique id suffixes per instance
- Name collisions between Interactive III.1 and IV.1 (both compute Γ_opt, F, G_A) resolved with context suffixes on the conflicting cell

---

## 10. Name-collision map

Variables that appear in multiple interactives get suffixes per repo convention:

- **Interactive III.1** (noise/gain/stability explorer): `Gamma_opt_c`, `F_c`, `GA_c`, `K_c`, `Delta_c`
- **Interactive IV.1** (LNA studio): `Gamma_opt_l`, `F_l`, `GA_l`, `K_l`, `Delta_l`

---

## 11. Navigation footer

Per CLAUDE.md convention, notebook 04 ends with a `mo.md` cell with Previous/Next links:

- Previous → `03_s_parameters_stability.py`
- Next → `05_<matching>.py` (placeholder until 05 is named)

When 04 is merged, update the footer of 03 to point Next → 04.

---

## 12. Verification checklist

Before marking notebook 04 complete:

1. `uv run marimo check marimo/notebooks/04_noise_and_lna_design.py` — only `markdown-indentation` warnings allowed.
2. `python -c "import ast; ast.parse(open('marimo/notebooks/04_noise_and_lna_design.py').read())"` — must pass.
3. `uv run python marimo/notebooks/04_noise_and_lna_design.py` — non-interactive smoke-test must succeed.
4. Manual smoke-test: `uv run marimo edit marimo/notebooks/04_noise_and_lna_design.py` and verify each of the four interactives renders and responds to input.

---

## 13. Out-of-scope (explicitly deferred)

Listed here so scope creep into notebook 04 during implementation is caught at review:

- Cyclostationary noise deep dive → 05 / 06
- L/π/T matching-network synthesis → 05
- Transformer and distributed matching → 05
- Broadband LNA design → 05
- Mixer noise, image-rejection architectures → 05 / 06
- PA nonlinearity, P1dB, IP3 → 06
- Full phased-array frontend integration → 06
- PDK-accurate BSIM device modeling → out of course scope (reference only)
