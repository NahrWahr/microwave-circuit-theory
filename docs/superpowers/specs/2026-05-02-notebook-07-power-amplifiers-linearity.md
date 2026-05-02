# Notebook 07 — Power Amplifiers, Linearity, and Volterra Series

**Status:** Design spec (2026-05-02). Not yet implemented.
**Scope:** Source file at `marimo/notebooks/07_power_amplifiers_linearity.py`.
**Target length:** ~2400–2600 lines.

---

## 1. Purpose and position in the course

Notebook 07 asks: once a low-noise amplifier and a clean local oscillator have produced a baseband signal, how do you transmit it back out at mmWave with enough output power and acceptable linearity? It develops the theory of power amplifier operation in three layers — (i) bias-dependent waveform engineering and conduction-angle classes, (ii) memoryless polynomial nonlinearity yielding the canonical $P_{1\text{dB}}$ and IIP3 figures of merit, and (iii) Volterra series with memory, derived rigorously by the harmonic-input probing method on a CMOS short-channel device.

The notebook is theory-driven rather than design-example-driven. CMOS limitations at mmWave (low breakdown, hot-carrier stress, $C_{gs}(V_{gs})$ AM-PM) motivate the analysis throughout, and a compact technology comparison places CMOS in context against SiGe, GaAs, and GaN.

### Position in the series

- **Notebooks 01–03** — two-port analysis, gain definitions, S-parameters, stability.
- **Notebook 04** — noise figure, CMOS short-channel device model, 28 GHz LNA.
- **Notebook 05** — matching networks, Bode-Fano, cyclostationary noise in mixers.
- **Notebook 06** — oscillator phase noise theory, ISF/Floquet framework, mmWave VCOs.
- **Notebook 07** — PA fundamentals, large-signal nonlinearity, Volterra series with memory (this notebook).
- **Notebook 08** — switching-mode and digital PA techniques (Class E/F, Doherty, outphasing, switched-cap), full transceiver/EVM budget (deferred from this notebook).

### Bridge to notebook 08

Topics explicitly deferred to 08:
- Switching-mode single-transistor classes (E, F)
- Doherty PA and back-off efficiency restoration
- Outphasing (LINC) and digital PA architectures
- Switched-cap PAs and other digital techniques used at mmWave
- Full LNA + mixer + VCO + PA transceiver chain and EVM budget

---

## 2. Backward dependencies

- **Notebook 01** — ABCD cascade, two-port matrices (used in §11 to embed device kernels into source/load matching networks).
- **Notebook 02** — Available power, output-power conventions (drain efficiency definitions in §3 use available-power language consistently with 02).
- **Notebook 03** — Load-line and operating-point geometry, stability framework (operating-point linearization in §1 references the load-line construction).
- **Notebook 04** — CMOS short-channel device model with $V_{ds}$-dependent $g_m$ and $g_{ds}$ (the same model is reused in §10 for kernel derivation; the two-tone test concept first introduced in noise context here gets generalized to large-signal use).
- **Notebook 05** — Matching networks, Bode-Fano (kernel embedding in §11; back-off-vs-bandwidth note in §13).
- **Notebook 06** — No theorem-level reuse, but the parallel rigor structure (intuitive theory → rigorous theory → mmWave specifics) mirrors 06's organization.

---

## 3. Structure

```
Part I     PA fundamentals                       §1–3
Part II    Memoryless nonlinearity               §4–7
Part III   Volterra series with memory           §8–11
Part IV    Process technology and CMOS limits    §12
Part V     Wrap and bridge to notebook 08        §13
```

Total: 13 sections, ~2400–2600 lines.

---

## 4. Section-by-section content

### Part I — PA Fundamentals

#### §1 Operating point and load line (~150 lines)

- $i_D(v_{GS}, v_{DS})$ for an idealized FET, parameterized by quiescent $V_{GQ}$.
- Load-line geometry in the $i_D$–$v_{DS}$ plane: DC load line vs. AC load line, optimum load $R_L = (V_{DD} - V_{\text{knee}})/I_{\text{max}}$.
- Why the operating point sets conduction angle: as $V_{GQ}$ moves below threshold, the drive sinusoid clips earlier per period. Visual: bias point on the transfer curve, drive amplitude swing, resulting clipped output.
- Linearization (small-signal $g_m$) deferred to §4.

No interactive in this section.

#### §2 Conduction angle and Fourier decomposition (~200 lines)

- Half-conduction angle $\theta$ defined; clipped-sinusoid drain current
  $$i_D(t) = \begin{cases} I_{\max}(\cos\omega t - \cos\theta)/(1-\cos\theta), & |\omega t| < \theta \\ 0, & \text{otherwise} \end{cases}$$
- Fourier coefficients $I_n(\theta)$ in closed form. State $I_0(\theta), I_1(\theta), I_2(\theta), I_3(\theta)$ explicitly; show the harmonic-suppression behavior at the canonical class boundaries.
- Class definitions as bias regimes:
  - **Class A:** $\theta = \pi$ (full conduction, no clipping).
  - **Class AB:** $\pi/2 < \theta < \pi$.
  - **Class B:** $\theta = \pi/2$ (half-wave conduction).
  - **Class C:** $\theta < \pi/2$ (under-biased, heavily clipped).
- Tank-tuned single-pole load assumption: only the fundamental reaches the antenna; harmonics circulate in the resonator. This is the assumption that makes drain efficiency a meaningful number in §3.

No interactive in this section.

#### §3 Drain efficiency and output power (~250 lines)

- Drain efficiency $\eta_D = P_{\text{out},1}/P_{DC} = (1/2)V_1 I_1/(V_{DD} I_0)$.
- Closed-form $\eta_D(\theta)$:
  $$\eta_D(\theta) = \frac{1}{2}\,\frac{V_1}{V_{DD}}\,\frac{I_1(\theta)}{I_0(\theta)}$$
  with $V_1/V_{DD} \to 1$ at saturation. Derive the canonical numbers: 50% (A), 78.5% (B), and 100% theoretical limit as $\theta \to 0$ (C).
- Output power $P_{\text{out}} = (1/2) V_1 I_1$ at saturation, in absolute terms and per-area for CMOS.
- Saturation vs. breakdown: $V_1 \le V_{DD} - V_{\text{knee}}$ on the upper rail, $V_1 \le V_{DD}$ on the lower rail at clipping.
- Efficiency–power tradeoff stated qualitatively: lower $\theta$ → higher $\eta$ but lower $P_{\text{out}}$ for the same $V_{DD}$.
- Back-off behavior introduced: at $P_{\text{back-off}}$ below saturation, $\eta_D$ falls — preview of why notebook 08 covers Doherty and outphasing.

**Interactive A — Class explorer** lives here.
- Sliders: $V_{GQ}$ (sets $\theta$), $V_{\text{drive}}$ amplitude.
- Outputs: drain current waveform vs. time, harmonic spectrum (DC, fundamental, 2nd, 3rd), $\eta_D$, $P_{\text{out},1}$, the operating point on the $i_D$–$v_{DS}$ load line.

**Interactive E — Back-off vs. efficiency** also lives here.
- Class selector (A / AB / B / C).
- Outputs: $\eta_D$ vs. $P_{\text{back-off}}$ in dB for each class overlaid on the same axes; observation that all classes degrade below saturation.

### Part II — Memoryless Nonlinearity

#### §4 Polynomial expansion of $i_d(v_{gs})$ (~150 lines)

- Taylor expansion around bias: $i_d = a_1 v_{gs} + a_2 v_{gs}^2 + a_3 v_{gs}^3 + \cdots$
- Coefficient mapping: $a_1 = g_m$, $a_2 = (1/2)\partial g_m/\partial V_{GS}$, $a_3 = (1/6)\partial^2 g_m/\partial V_{GS}^2$.
- Single-tone input $v_{gs}(t) = A\cos\omega t$: expand $v_{gs}^2, v_{gs}^3$, identify DC, fundamental, 2nd, 3rd harmonic components.
- Harmonic distortion definitions:
  $$\text{HD2} = \left|\frac{a_2 A}{2 a_1}\right|, \quad \text{HD3} = \left|\frac{a_3 A^2}{4 a_1}\right|$$
- Slope-vs-drive observation: HD2 grows as $A^1$, HD3 as $A^2$.
- Note the limitation foreshadowed for §8: this analysis assumes the polynomial coefficients $a_n$ are independent of frequency, which is exactly what fails at mmWave.

No interactive in this section.

#### §5 Two-tone test (~150 lines)

- Two-tone input $v_{gs}(t) = A(\cos\omega_1 t + \cos\omega_2 t)$ with $\omega_1, \omega_2$ closely spaced.
- Expand $v_{gs}^2$: produces DC, $2\omega_1, 2\omega_2, \omega_1+\omega_2, \omega_1-\omega_2$ (the IM2 products).
- Expand $v_{gs}^3$: produces fundamentals, $3\omega_1, 3\omega_2$, and crucially $2\omega_1-\omega_2, 2\omega_2-\omega_1$ (the IM3 products).
- Why IM3 is the dominant practical concern: $2\omega_1-\omega_2$ falls inside the channel (close to $\omega_1$) and cannot be filtered.
- Spectrum visualization: stem plot of products on $\omega$ axis, scaled by $A^n$.

**Interactive D — Two-tone spectrum simulator** lives here.
- Sliders: tone spacing $\Delta f = f_2 - f_1$, drive level $A$ in dBV, polynomial coefficients $a_2, a_3$.
- Outputs: spectrum plot showing fundamentals, IM2, IM3, IM5 with linear-vs-cubic-vs-quintic slope clearly visible as $A$ changes.

#### §6 $P_{1\text{dB}}$ and IIP3 (~250 lines, includes Interactive B)

- $P_{1\text{dB}}$: amplitude at which $a_3 A^3$ has compressed the fundamental by 1 dB; closed-form $A_{1\text{dB}} = 0.145\sqrt{|a_1/a_3|}$ for $\text{sgn}(a_3) = -\text{sgn}(a_1)$.
- IIP3 construction: extrapolated input at which fundamental and IM3 power lines intersect on a log-log plot. Closed form: $A_{\text{IIP3}} = \sqrt{(4/3)|a_1/a_3|}$.
- Derive $\text{IIP3} - P_{1\text{dB}} \approx 9.6$ dB result and explain why this value is canonical.
- Slope-of-3 construction: IM3 grows as $A^3$, so on a dBV-vs-dBV plot it has slope 3 while the fundamental has slope 1 — they meet at IIP3.
- Observation: the construction assumes memoryless polynomial. Foreshadow §11: real measurements show IIP3 depends on tone spacing.

**Interactive B — $P_{1\text{dB}}$ / IIP3 explorer** lives here.
- Sliders: $a_1, a_3, a_5$ coefficients, drive amplitude sweep range.
- Outputs: fundamental output power vs. input (showing 1-dB compression), two-tone IIP3 extrapolation construction with the slope-1/slope-3 lines, $P_{1\text{dB}}$ and IIP3 markers.

#### §7 AM-AM and AM-PM (memoryless) (~150 lines)

- Define AM-AM: magnitude of fundamental output as a function of input amplitude, $|H(A)|$.
- Define AM-PM: phase of fundamental output as a function of input amplitude, $\arg(H(A))$ in degrees-per-dB or radians.
- In a strictly memoryless polynomial model, AM-PM is zero — the output is a real polynomial of a real input, so the fundamental phase cannot rotate with amplitude.
- Yet measurements show nonzero AM-PM in real CMOS PAs. Phenomenological observation here; rigorous answer deferred to §10.
- This section is the bridge into Part III: it defines what the Volterra theory must explain.

No interactive in this section.

### Part III — Volterra Series with Memory

#### §8 Why memoryless polynomial fails (~200 lines)

- Counter-example construction: same nonlinear device $i_d = a_1 v + a_3 v^3$, but two different source/load impedance environments. Show that the IIP3 measured at the system terminals differs between the two configurations, despite identical $a_n$.
- Conclusion: the memoryless polynomial cannot represent reality when the surrounding network has frequency-dependent impedance. We need kernels that depend on frequency.
- Volterra series:
  $$y(t) = \sum_{n=1}^\infty \int\cdots\int h_n(\tau_1,\ldots,\tau_n)\prod_{i=1}^n x(t-\tau_i)\,d\tau_i$$
- Multi-dimensional Fourier transform:
  $$H_n(\omega_1,\ldots,\omega_n) = \int\cdots\int h_n(\tau_1,\ldots,\tau_n)\,e^{-j\sum_i \omega_i \tau_i}\,d\tau_1\cdots d\tau_n$$
- Symmetrization: $H_n$ is taken symmetric in its arguments by convention.
- Output product mapping: an input with tones at $\omega_a, \omega_b, \omega_c, \ldots$ produces an output component at $\omega_a + \omega_b + \omega_c$ with amplitude $H_3(\omega_a, \omega_b, \omega_c)\cdot$(amplitude product).
- Memoryless polynomial recovered as $H_n(\omega_1,\ldots,\omega_n) = a_n$ (constant), confirming the limit.

No interactive in this section.

#### §9 Harmonic-input probing method (~250 lines)

- Goal: extract $H_n$ recursively from a system equation.
- Method: input $x(t) = A_1 e^{j\omega_1 t} + A_2 e^{j\omega_2 t} + \cdots$, expand $y(t)$ to order $n$, identify the unique frequency $\omega_1 + \omega_2 + \cdots + \omega_n$ that appears only at order $n$, equate coefficients.
- Worked example: nonlinear admittance $i = g_1 v + g_2 v^2 + g_3 v^3$ embedded behind a passive source impedance $Z_s(\omega)$.
- Step 1 ($n=1$): single-tone input, derive $H_1(\omega) = g_1/(1 + g_1 Z_s(\omega))$.
- Step 2 ($n=2$): two-tone input, derive
  $$H_2(\omega_1,\omega_2) = -\frac{g_2 H_1(\omega_1) H_1(\omega_2)}{1 + g_1 Z_s(\omega_1+\omega_2)}$$
- Step 3 ($n=3$): three-tone input, derive $H_3$ in closed form (will involve both $g_3$ direct and $g_2$-cascade contributions).
- Key insight: $H_n$ depends on $Z_s$ evaluated at intermodulation frequencies, not just at the input tones — this is where memory enters.
- Cross-reference: the matrix structure of the recursion will be reused in §11 with ABCD embedding.

No interactive in this section.

#### §10 CMOS short-channel device kernels (~300 lines)

- Reuse the CMOS short-channel model from notebook 04 §13: $i_D(v_{GS}, v_{DS}) = $ short-channel current expression, plus parasitic capacitances $C_{gs}(v_{GS}), C_{gd}, C_{db}$.
- Linearize around bias: extract small-signal $g_m, g_{ds}, C_{gs}, C_{gd}$ at the operating point.
- Apply probing method to derive:
  - $H_1(\omega)$: small-signal frequency response, including $C_{gs}$ pole and $C_{gd}$ Miller.
  - $H_2(\omega_1, \omega_2)$: dominant contributions from $\partial g_m/\partial V_{GS}$ (transconductance second derivative) and $\partial g_{ds}/\partial V_{DS}$ (output-conductance second derivative).
  - $H_3(\omega_1, \omega_2, \omega_3)$: dominant contributions from $\partial^2 g_m/\partial V_{GS}^2$, plus the $g_2$-cascade term $\propto H_2 \cdot g_2$.
- **Rigorous AM-PM derivation:** show that $\arg(H_1(\omega; A))$ — the phase of the fundamental output as a function of drive amplitude $A$ — picks up amplitude dependence through the $C_{gs}(V_{GS})$ voltage coefficient. Specifically: at large $A$, the time-averaged $C_{gs}$ shifts because $C_{gs}$ is voltage-dependent, the input pole moves, and the small-signal fundamental phase rotates. This effect is invisible to memoryless polynomial.
- Numerical example: plug in nominal CMOS 28-nm short-channel parameters at 28 GHz, compute $|H_1|, |H_3|, \arg(H_1)$ vs. drive amplitude. Show the AM-PM curve in degrees-per-dB.
- This section is the substantive answer to the §7 phenomenology question.

No interactive in this section (Interactive C in §11 visualizes these results).

#### §11 Kernel embedding and IIP3 vs. tone spacing (~250 lines, includes Interactive C)

- System view: device kernels $H_n^{\text{dev}}$ embedded between input matching network ABCD$_{\text{in}}$ and output matching network ABCD$_{\text{out}}$.
- Derive system kernels $H_n^{\text{sys}}$ as compositions:
  - $H_1^{\text{sys}}(\omega) = T_{\text{in}}(\omega)\,H_1^{\text{dev}}(\omega)\,T_{\text{out}}(\omega)$ where $T$ are voltage/current transfer ratios from the ABCD matrices.
  - $H_2^{\text{sys}}, H_3^{\text{sys}}$: each device-kernel argument carries its own $T_{\text{in}}$ factor, the output sum frequency carries a single $T_{\text{out}}$, and the recursion picks up additional $T(\omega_1+\omega_2)$, $T(\omega_1+\omega_2+\omega_3)$ factors at the intermodulation frequencies.
- IIP3 as a function of tone spacing $\Delta f = f_2 - f_1$:
  - At $\Delta f \to 0$ (tones coincident): kernel evaluated at degenerate point, recover the memoryless result.
  - At larger $\Delta f$: kernel arguments separate, $H_3(\omega_1, \omega_1, -\omega_2)$ probes off-diagonal kernel structure. IIP3 changes — usually degrades — visibly.
- Why mmWave is harder: at 28 GHz, parasitic poles are close to the operating frequency, so a 100 MHz tone spacing already shifts the kernel evaluation point by a significant fraction of a pole. At 2.4 GHz, the same 100 MHz spacing is a much smaller fractional perturbation of distant poles — memory effects appear at much wider spacings.
- On-chip vs. off-chip matching: off-chip matching has lower self-resonance and tighter Q, on-chip matching has lower Q and broader bandwidth. Trade-off shows up in the IIP3-vs-spacing curve shape.

**Interactive C — Volterra kernel explorer** lives here.
- Sliders: bias $V_{GQ}$, two-tone spacing $\Delta f$, source impedance $Z_s$ pole frequency (representing matching network selectivity).
- Outputs: $|H_1(\omega)|$ vs. $\omega$ (showing the small-signal response with $C_{gs}$ pole and Miller pole), $|H_3(\omega_1, \omega_1, -\omega_2)|$ vs. $\Delta f$, $\arg(H_1)$ AM-PM curve vs. drive amplitude, derived IIP3 vs. $\Delta f$ on a separate panel.

### Part IV — Process Technology and CMOS Limits

#### §12 Tech comparison and CMOS limitations (~150 lines)

- Compact comparison table:

| Tech | $f_T$ (GHz) | $f_{\max}$ (GHz) | $V_{BR}$ (V) | $P_{\text{sat}}$ density (W/mm) | 1/f corner | Integration |
|---|---|---|---|---|---|---|
| Bulk CMOS 28 nm | 280 | 320 | ~1.0 | 0.05 | high | excellent |
| SiGe BiCMOS | 300 | 500 | ~2 (BV<sub>CEO</sub>) | 0.3 | low | very good |
| GaAs pHEMT | 150 | 250 | ~15 | 1.0 | medium | poor |
| GaN HEMT | 100 | 200 | ~80 | 5.0 | high | poor |

- Reading the table: $P_{\text{sat}}$ density scales with $V_{BR}^2$ in the squared-sinusoid limit; CMOS is ~100× behind GaN.
- CMOS-specific limitations at mmWave PA:
  1. **Low breakdown** caps single-device $P_{\text{sat}}$. Stacking (cascoded series transistors) is the standard workaround — full treatment defers to 08.
  2. **Hot-carrier degradation** under sustained $V_{DS}$ stress shifts $V_{TH}$ and $g_m$ over the device lifetime; reliability constraint is real and design-binding.
  3. **Low passive Q** at mmWave (notebook 06 §10 already covered this for tank Q; same physics here for matching networks).
  4. **AM-PM from $C_{gs}(V_{gs})$** as derived rigorously in §10 — this is the unique CMOS contribution to large-signal phase distortion.
  5. **Substrate loss** and inductor self-resonance at mmWave further squeeze matching-network Q.
- Why CMOS is used anyway: integration with the rest of the transceiver (LNA, mixer, baseband, DSP) outweighs the per-device $P_{\text{sat}}$ disadvantage in many sub-10-W applications.

No interactive in this section.

### Part V — Wrap and Bridge to Notebook 08

#### §13 Recap and bridge (~100 lines)

- Three-pillar recap:
  1. **Bias and waveform engineering** (§1–3): conduction angle, classes A/AB/B/C, drain efficiency.
  2. **Memoryless nonlinearity** (§4–7): polynomial expansion, $P_{1\text{dB}}$, IIP3, two-tone test, AM-AM phenomenology.
  3. **Volterra with memory** (§8–11): kernels, probing method, CMOS device kernels, AM-PM, IIP3 vs. tone spacing.
- Explicit deferral list to notebook 08:
  - Switching-mode classes (E with closed-form ZVS, F with harmonic shaping).
  - Doherty PA — back-off efficiency restoration via auxiliary amplifier.
  - Outphasing (LINC) — amplitude modulation by combining two constant-envelope PAs.
  - Switched-cap PAs and digital PA architectures used at mmWave.
  - Stacking for breakdown enhancement in CMOS.
  - Full LNA + mixer + VCO + PA transceiver chain → EVM budget.
- Concept dependency map for notebook 08:

  ```
  07 §3   Class A/AB/B/C  ──►  08 Class E/F as switched-mode extension
  07 §6   IIP3 / P_1dB    ──►  08 EVM budget (tx-side distortion contribution)
  07 §10  CMOS AM-PM      ──►  08 Pre-distortion / DPD techniques
  07 §11  Kernel embedding ──►  08 Doherty load-modulation kernels
  07 §12  CMOS V_BR limit ──►  08 Stacking / cascoded PA topology
  ```

- Updated nav footer: previous → notebook 06, next → notebook 08 (in preparation).

---

## 5. Interactive demos summary

| ID | Title | Section | Sliders | Outputs |
|---|---|---|---|---|
| A | Class explorer | §3 | $V_{GQ}$, $V_{\text{drive}}$ | drain waveform, harmonic spectrum, $\eta_D$, $P_{\text{out}}$, load-line trajectory |
| B | $P_{1\text{dB}}$ / IIP3 explorer | §6 | $a_1, a_3, a_5$, drive sweep | compression curve with 1-dB marker, two-tone IIP3 extrapolation with slope-1/slope-3 construction |
| C | Volterra kernel explorer | §11 | $V_{GQ}$, $\Delta f$, $Z_s$ pole frequency | $\|H_1(\omega)\|$, $\|H_3\|$ vs. $\Delta f$, $\arg(H_1)$ AM-PM, IIP3 vs. $\Delta f$ |
| D | Two-tone spectrum simulator | §5 | $\Delta f$, drive level, $a_2, a_3$ | spectrum stem plot showing fundamentals, IM2, IM3, IM5 with slope vs. drive |
| E | Back-off vs. efficiency | §3 | class selector (A/AB/B/C) | $\eta_D$ vs. $P_{\text{back-off}}$ overlay across classes |

All figures use `template="plotly_dark"` and `mo.ui.plotly(fig)` per repo convention. Interactives that compute the same quantity (e.g. both B and D involve polynomial coefficients) follow the name-collision rule from CLAUDE.md: if any cell-level variable collides, append a context suffix per interactive (e.g. `a3_b`, `a3_d`).

---

## 6. Pedagogical arc and rigor budget

The notebook is a three-layer build:

- **Layer 1 (Part I):** one parameter (conduction angle) explains four classes and a canonical efficiency curve. Easy entry, immediate visual payoff via Interactive A.
- **Layer 2 (Part II):** add a polynomial nonlinearity at fixed bias, derive the canonical figures of merit. Standard textbook content; Interactive B and D land here.
- **Layer 3 (Part III):** generalize the polynomial to kernels with frequency dependence, derive rigorously by probing, apply to CMOS short-channel, observe IIP3-vs-spacing as the experimental signature. Most of the rigor budget; Interactive C is the visualization.

Part IV (tech comparison) and Part V (wrap) are short and serve as connectors — Part IV grounds the abstraction in physical process limits, Part V sets up notebook 08.

---

## 7. Forward bridge

Notebook 08 is the system-integration capstone. It will treat:

- Switching-mode and digital PA architectures (Class E/F, Doherty, outphasing, switched-cap, digital PA).
- The full mmWave transceiver: LNA noise floor (notebook 04) + mixer image and cyclostationary noise (notebook 05) + VCO phase noise (notebook 06) + PA distortion (notebook 07) → EVM budget.
- Pre-distortion (DPD) using the Volterra kernels derived here as the inverse model.
- Phased-array considerations: per-element PA back-off, beam-direction-dependent linearity, calibration.

The deferral list in §13 above is the canonical reference for what 08 will cover.
