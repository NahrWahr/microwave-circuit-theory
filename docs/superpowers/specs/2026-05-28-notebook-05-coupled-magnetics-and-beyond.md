# Notebook 05 Expansion — Coupled Magnetics, T-coils, and Beyond Passive LTI

**Status:** Design spec, approved outline (2026-05-28). Implementation pending.
**Scope:** In-place expansion of `marimo/notebooks/05_matching_networks.py`.
**Target length:** ~3500 lines after expansion (currently ~2170).
**Predecessor spec:** `2026-04-23-notebook-05-matching-networks.md` (original design).

---

## 1. Motivation

The existing notebook 05 leaves three pedagogical gaps:

1. **Mutual inductance is never derived.** §15 jumps from the coupled-inductor Z-matrix to the strong-coupling limit $Z_{\text{in}}\to n^2 Z_L$ in ~25 lines. No Faraday-law foundation, no T-model derivation, no answer to the natural student question "do the impedances on one side *copy* to the other?"
2. **T-coils are quoted, not derived.** §6.2 states the constant-$R$ result and the $k=1/3$ / $k=1/2$ design points but offers no s-domain derivation, no pole-zero picture, no comparison vs shunt/series peaking.
3. **Nothing on "beyond Bode-Fano."** Modern matching has long since moved past passive LTI: non-Foster active matching, distributed amplifiers, N-path / time-modulated networks, reconfigurable matching. None of this is touched.

This expansion fills all three gaps in one notebook (single-file decision: keeps the Bode-Fano → coupled-magnetics → beyond-Bode-Fano narrative continuous).

---

## 2. Restructure

| Current | After expansion |
|---|---|
| Part I — Fundamentals (§1–3) | unchanged |
| Part II — Lumped (§4–7), including §6.2 T-coil | §6.2 reduced to a 4-line teaser pointing forward; full T-coil treatment relocated to Part V §19 |
| Part III — Broadband / Bode-Fano (§8–11) | unchanged |
| Part IV — Distributed (§12–14) | unchanged |
| Part V — mmWave revisited (current §17) | renumbered → Part VII; comparison table gains rows for T-coil and distributed-amp output match |
| Part VI — Summary (current §18) | renumbered → Part VIII; cross-references updated to §15–§26 |
| — | **Part V — Coupled Magnetics & Transformers (NEW, §15–§21)** |
| — | **Part VI — Beyond Passive LTI Matching (NEW, §22–§26)** |

---

## 3. Part V — Coupled Magnetics & Transformers (NEW)

### §15. Mutual Inductance from First Principles

- **15.1 Faraday's law and flux linkage.** $\Psi_k = \sum_j L_{kj} I_j$ derived from $\oint \vec E \cdot d\vec\ell = -d\Phi/dt$ applied to each loop.
- **15.2 Neumann's formula.** Geometric definition $M_{12} = \tfrac{\mu_0}{4\pi}\oint\!\oint \tfrac{d\vec\ell_1 \cdot d\vec\ell_2}{|\vec r_1 - \vec r_2|}$ for filamentary loops.
- **15.3 Reciprocity.** $M_{12}=M_{21}$ — proven two ways: Neumann symmetry under index exchange, and energy-conservation argument (Maxwell stress tensor / virtual work).
- **15.4 Coupling coefficient.** $k = M/\sqrt{L_1 L_2}$. Proof $|k| \le 1$ from non-negative magnetic energy $W = \tfrac12 L_1 I_1^2 + \tfrac12 L_2 I_2^2 + M I_1 I_2 \ge 0\ \forall (I_1,I_2)$ → LMI / Schur condition.
- **15.5 Dot convention.** Energy/flux-sign convention; statement of when $M$ is taken positive vs negative.
- **15.6 Energy storage.** Full quadratic form, link to Foster's reactance theorem from §22.

### §16. Equivalent Circuit Models for Coupled Inductors

- **16.1 The T-model.** $(L_1 - M,\ L_2 - M,\ M)$ derived from Z-matrix by inspection; show element values can go *negative* without violating passivity.
- **16.2 The π-model.** Dual; explicit Z↔Y conversion.
- **16.3 Ideal-transformer + leakage.** $L_1, L_2, M \to (L_\sigma, L_m, n)$ where $L_m = M\sqrt{L_2/L_1}$, $L_\sigma = L_1(1-k^2)$.
- **16.4 Impedance reflection mechanism.** Derive $Z_{\text{in}}$ explicitly: $Z_{\text{in}} = j\omega L_1 + (\omega M)^2 / (Z_L + j\omega L_2)$. Settle the "do impedances copy?" question — they do not; the secondary current $I_2$ induces a back-EMF $j\omega M I_2$ in the primary, which the source sees as an added impedance.
- **16.5 Strong-coupling limit.** $k\to 1$, $\omega L_2 \gg |Z_L|$: recover $Z_{\text{in}} \to n^2 Z_L$.
- **16.6 Weak-coupling limit.** $k\to 0$: decouples to two isolated inductors.
- **16.7 Interactive.** Z-matrix vs T-model overlay, parametric $k$ sweep, phasor animation of $I_1, I_2, V_1, V_2$.

### §17. Losses in Coupled Magnetics

- **17.1 Winding resistance.** DC resistance, series-Q model.
- **17.2 Skin effect.** Skin depth $\delta = \sqrt{2/(\omega\mu\sigma)}$ at 1–100 GHz; hollow-conductor approximation; ${R}_{\text{ac}}/R_{\text{dc}}$ scaling.
- **17.3 Proximity effect.** Current crowding from neighbouring-conductor magnetic fields; quantitative form for parallel-conductor case.
- **17.4 Substrate eddy currents.** Si substrate as a lossy secondary loop; image-current model; ground-shielding (patterned vs solid).
- **17.5 Magnetic-core losses.** Hysteresis + eddy current in ferrite cores (relevant <1 GHz, off-chip transformers); briefly noted for completeness.
- **17.6 Total Q-degradation and noise impact.** $Q_{\text{tot}}^{-1} = Q_{\text{wind}}^{-1} + Q_{\text{sub}}^{-1} + \ldots$; Friis-referred NF penalty derived from §15 of notebook 04.
- **17.7 Typical Q(f) curves.** Indicative on-chip-spiral $Q(f)$ trends for 65 nm CMOS at 1–100 GHz.

### §18. Transformer-Based Impedance Matching

- **18.1 Ideal 1:n revisited.** Why bandwidth is infinite for the textbook ideal transformer (zero leakage, infinite $L_m$).
- **18.2 Leakage-inductance bandwidth limit.** Real $k<1$: $L_\sigma$ creates a low-pass at $f_{HP} \sim R/(2\pi L_\sigma)$.
- **18.3 Tuned-primary / tuned-secondary.** Parallel resonance to absorb $L_m$ or $L_\sigma$.
- **18.4 Single-tuned vs double-tuned.** Under-coupled (single peak), critically-coupled ($k = k_{\text{crit}} = 1/\sqrt{Q_1 Q_2}$, maximally flat), over-coupled (double peak, broader 3-dB bandwidth).
- **18.5 Design equations.** Closed-form for double-tuned matching given $R_S, R_L, Q_1, Q_2, f_0$, BW target.
- **18.6 Interactive.** Critical-coupling explorer: sweep $k$, show $|S_{21}|$, peaking, 3-dB BW; highlight $k_{\text{crit}}$ point.

### §19. The Bridged T-Coil — Full Rigor

- **19.1 Why a T-coil.** The Bode-Fano dodge: load is *fixed parasitic $C$*, not an $R$ to transform; bridged-T absorbs $C$ into a constant-$R$ all-pass.
- **19.2 Network equations.** KCL/KVL on the bridged-T with two coupled half-coils, bridge cap $C_B$, parasitic $C$ on tap.
- **19.3 Constant-$R$ derivation.** Symbolic — demand $Z_{\text{in}}(s) = R\ \forall s$, solve simultaneously for $L_{\text{half}}, M, C_B$. Recover $L_T \equiv 2L_{\text{half}}(1+k) = R^2 C$ as a $k$-independent total inductance.
- **19.4 Tap-node transfer function.** $H(s) = V_{\text{tap}}/V_{\text{src}}$ as a rational function in $s$ parametrized by $k$ alone (after the constant-$R$ constraint).
- **19.5 $k=1/3$ Butterworth.** Pole positions on a circle, magnitude flatness, $-3\,\text{dB}$ BW = $2\sqrt 2/(RC) \approx 2.83\times$ bare-$RC$, step response.
- **19.6 $k=1/2$ Bessel.** Maximally flat group delay, eye-closure derivation, why SerDes prefers this.
- **19.7 Comparison.** Shunt-peaking ($1.7\times$), series-peaking ($1.4$–$1.6\times$), T-coil ($2.83\times$) — derive each from peaking-element analysis.
- **19.8 Interactive.** Sweep $k$ ∈ [0, 1]: show pole-zero map, $|S_{11}|$, tap-node magnitude, step response, eye diagram (synthetic PRBS).
- **19.9 Practical layout.** Symmetric vs asymmetric T-coils, layout for SerDes TX/RX, mutual-coupling tolerance.

### §20. Marchand Balun and Coupled-Line Distributed Transformers

- **20.1 Coupled-line two-port.** Even/odd mode impedances $Z_{0e}, Z_{0o}$; four-port to two-port reduction.
- **20.2 Single-section Marchand balun.** Two $\lambda/4$ coupled-line sections back-to-back; derive the 2:1 differential output condition and balun action from even/odd-mode S-parameters.
- **20.3 Frequency response.** Amplitude/phase balance vs frequency; broadband behaviour (~50% BW for 1-dB amplitude balance, 5° phase balance).
- **20.4 Compact realizations.** Slow-wave Marchand (lumped capacitors at coupled-line ends), multi-section / coupled-resonator balun.
- **20.5 mmWave use case.** Single-ended LNA driving a differential mixer in a 28/60 GHz receiver.

### §21. On-Chip Geometry & Practical Considerations

- **21.1 Spiral shape.** Square/octagonal/hexagonal/circular spirals; corner-current discontinuity argument for going octagonal.
- **21.2 $k$ vs geometry.** Stacked spirals (0.85–0.95), interleaved/intertwined (0.7–0.85), concentric (0.4–0.7); trade vs self-resonance and substrate coupling.
- **21.3 EM-simulation vs lumped model.** When the lumped $L, M, k, R, C_{\text{ox}}$ model breaks down; momentum / HFSS / EMX comparison.
- **21.4 Metal-stack.** Thick top metal for Q, patterned ground shield (PGS) for substrate isolation.

---

## 4. Part VI — Beyond Passive LTI Matching (NEW)

### §22. Non-Foster (Active) Matching

- **22.1 Foster's reactance theorem.** $dX/d\omega \ge 0$ for any passive lossless one-port; consequence: a passive $-C$ or $-L$ is impossible.
- **22.2 Negative impedance converters.** Linvill 1953 transistor-pair NIC, op-amp NIC; derivation of $Z_{\text{in}} = -Z_L$ from feedback equations.
- **22.3 Defeating Bode-Fano in principle.** A non-Foster $-C$ cancels the load $C$ at all frequencies, freeing the integral bound to be $\infty$.
- **22.4 Stability.** The NIC sees a closed-loop RHP pole — terminating-network sensitivity, Nyquist stability across the operating band.
- **22.5 Modern applications.** Electrically-small antennas (Sussman-Fort, 2009), broadband mmWave pre-match, active impedance tuners.
- **22.6 Interactive.** NIC impedance vs frequency + stability envelope (Nyquist plot).

### §23. Distributed Amplifier / Artificial Transmission Line

- **23.1 The $g_m / C_{gs}$ ceiling.** Single-stage gain-bandwidth product is bounded; cascade doesn't help (each stage loads the previous).
- **23.2 Distributed-amp idea.** Cascade $N$ transistors along gate-line and drain-line; outputs sum coherently along drain TL.
- **23.3 Artificial TL.** $L$-$C$ ladder with $C = C_{gs}$ (input) or $C_{ds}$ (output) + added series $L$; equivalent characteristic impedance $Z_0 = \sqrt{L/C}$.
- **23.4 Gain budget.** $|S_{21}| \approx N\,g_m Z_0 / 2$; bandwidth = LC-ladder cutoff $f_c = 1/(\pi\sqrt{LC})$.
- **23.5 Loss limits $N$.** Gate-line attenuation and drain-line attenuation set an optimum $N \sim 4$–$8$ for mmWave CMOS.
- **23.6 Modern use.** Ultra-wideband mmWave PAs and LNAs (>50 GHz BW).
- **23.7 Interactive.** $N$-section gain-BW explorer, gate/drain line attenuation curves.

### §24. N-Path & Time-Modulated Matching

- **24.1 The N-path filter.** $N$ cyclically-switched capacitors with LO duty cycle $1/N$; creates a tunable narrowband filter centred at $f_{\text{LO}}$.
- **24.2 Impedance translation theorem.** A baseband impedance $Z_{BB}$ across the switches appears at the RF port as $Z_{BB}$ shifted to $f_{\text{LO}}$ with a $\text{sinc}^2$-weighted folding from harmonics.
- **24.3 N-path matching network.** Pure tunable narrowband match, no LC tank.
- **24.4 LTV escape from Bode-Fano.** The network is *not* LTI — Bode-Fano's BR-function machinery does not apply.
- **24.5 Applications.** SDR front-end, blocker-tolerant receiver, reconfigurable interferer rejection.
- **24.6 Parametric matching.** Pumped-varactor reactance as a time-varying $C(t)$; brief mention of Manley-Rowe relations.

### §25. Reconfigurable / Tunable Matching

- **25.1 Tuning elements.** Varactor diodes (Q vs tuning range), switched-capacitor arrays (binary-weighted), RF-MEMS, BST (barium strontium titanate) caps.
- **25.2 ATU topologies.** Tunable $\pi$ and T networks; closed-form synthesis for impedance-impedance mapping.
- **25.3 Closed-loop adaptive.** VSWR sensor + impedance-search algorithm (gradient descent, simplex).
- **25.4 Massive-MIMO / phased-array.** Per-element ATU to handle scan-angle-dependent active impedance.
- **25.5 Interactive.** Adaptive tuner: sweep load impedance, show tuner converging.

### §26. Other Frontiers (brief)

- **26.1 Metamaterial/metasurface matching.** Sub-wavelength impedance synthesis.
- **26.2 Reconfigurable intelligent surfaces (RIS).** 6G-era impedance-shaping surfaces.
- **26.3 Coupled-resonator filter as broadband match.** Multi-pole synthesis (Cameron / Cohn approach) for combined filter + match.
- **26.4 Optical analog.** Anti-reflection coatings as multi-section quarter-wave Chebyshev matching transformers.

---

## 5. Helpers cell additions

Add to the existing pure-Python helpers cell:

- `coupled_inductors_Zmatrix(L1, L2, k, w) -> (2,2) complex` — Z-matrix evaluator
- `coupled_inductors_T_model(L1, L2, k) -> (L1m, L2m, M)` — T-model element values
- `transformer_input_Z(L1, L2, k, ZL, w) -> complex` — closed-form $Z_{\text{in}}$
- `tcoil_constant_R(R, C, k) -> dict` — returns $L_{\text{half}}, M, C_B$, tap-transfer function coefficients
- `tcoil_tap_response(R, C, k, freqs) -> complex` — $H(s)$ on the imaginary axis
- `marchand_balun_S(Z0e, Z0o, f0, freqs) -> (N,4,4)` — coupled-line S-parameters
- `nic_impedance(Z_ref, gm, R_load, w) -> complex` — Linvill NIC small-signal $Z_{\text{in}}$
- `npath_translated_Z(Zbb, N, f_LO, freqs) -> complex` — N-path impedance-translation result
- `distributed_amp_gain(N, gm, Z0, R_loss, freqs) -> complex` — gain along finite-loss ladder

---

## 6. Implementation phases

1. **Phase 1 — Helpers** (lowest risk): add the new helper functions. Smoke-test in isolation.
2. **Phase 2 — Part V §15–§16** (foundation cells). Stop here, let user review style/depth before continuing.
3. **Phase 3 — Part V §17–§19** (losses, transformer matching, full T-coil).
4. **Phase 4 — Part V §20–§21** (Marchand balun, on-chip geometry).
5. **Phase 5 — Part VI §22–§26** (beyond passive LTI).
6. **Phase 6 — Renumber Part VII / VIII, update nav, fix concept-dependency map.**
7. **Phase 7 — Validation:**
   - `uv run marimo check marimo/notebooks/05_matching_networks.py` (only `markdown-indentation` warnings allowed)
   - `python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"`
   - `uv run python marimo/notebooks/05_matching_networks.py` (smoke test)

---

## 7. Conventions

Per project `CLAUDE.md`:
- All figures use `template="plotly_dark"`.
- All figures displayed via `mo.ui.plotly(fig)`.
- Smith chart aspect ratio: `xaxis=dict(scaleanchor="y", scaleratio=1)`.
- File version bumped at top (currently `v1.0` → `v2.0`).
- SVG markers: unique `id` per instance.
- Name-collision rule: if §16 and §18 both compute $Z_{\text{in}}$, use suffixes (e.g. `Zin_T` for T-model section, `Zin_dt` for double-tuned section).

Per user global `CLAUDE.md`:
- First-principles derivations; symbolic results before numeric.
- Tag major results as [Definition] / [Theorem] / [Axiom] / [Corollary] / [Result].
- No emojis. Concise text. Precise language.

---

## 8. Open implementation questions

These will be asked as the relevant section is reached, not up front:

- §19 step / eye response: render as static SVG or interactive PRBS sweep?
- §22 NIC schematic: mermaid diagram of a Linvill transistor-pair NIC, or stay at the block-diagram abstraction?
- §24 N-path: include the LPTV harmonic-balance derivation of the translation theorem, or quote the result and cite Mirzaei 2010?
