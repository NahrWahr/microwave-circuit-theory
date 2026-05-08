# Notebook 08 — Systematic mmWave Oscillator Design via the Activity Condition

**Status:** Design spec (2026-05-08). Not yet implemented.
**Source:** Momeni & Afshari, "High Power Terahertz and Millimeter-Wave Oscillator Design: A Systematic Approach," IEEE JSSC vol. 46, no. 3, March 2011 (`docs/ref/High_Power_Terahertz_and_Millimeter-Wave_Oscillator_Design_A_Systematic_Approach.pdf`).
**File:** `marimo/notebooks/08_systematic_oscillator_design.py`.
**Target length:** ~1900–2200 lines (slightly larger than 06/07 due to triple-push content).

---

## 1. Purpose and position in the course

Notebook 06 asked *how clean* a mmWave oscillator can be (phase noise). Notebook 08 asks the orthogonal question: *how high in frequency, and how much output power*, can a given technology actually deliver — and how does the topology determine the answer?

The central observation is that Mason's invariant `U` and `f_max` (notebook 02 §2) describe the **upper bound** of what a transistor can do under any lossless reciprocal embedding, but they do not by themselves tell you whether a *specific* topology — cross-coupled, three-stage ring, Colpitts, triple-push — can reach that bound. Notebook 08 fills this gap by extending the activity condition with an explicit `(A, φ)` parameterization (the amplitude ratio and phase shift between the device's two ports) and showing that every topology imposes a particular `(A', φ')` on the device. The maximum oscillation frequency `f_m` of that topology is the highest frequency at which the constrained `(A', φ')` still satisfies the activity condition. Three-stage rings come close to `f_max`; cross-coupled topologies do not. Triple-push extracts power above `f_max` by collecting the third harmonic.

### Position in the series

- **Notebooks 01–03** — two-port analysis, gain definitions, S-parameters, stability.
- **Notebook 02 §2–3** — Mason's `U`, activity condition, `f_max` as `U=1`. **Direct prerequisite.**
- **Notebook 04** — noise figure, four noise parameters, 28 GHz LNA design.
- **Notebook 05** — matching networks, Bode-Fano, cyclostationary mixers.
- **Notebook 06** — oscillator phase noise (Leeson, ISF), mmWave tank Q, cross-coupled vs. Colpitts in the *phase-noise* context. **Complementary.**
- **Notebook 07** — PA design, linearity, Volterra series.
- **Notebook 08** — activity-based fundamental design near `f_max`, triple-push for `f_osc > f_max` (this notebook).

### What is *new* relative to notebook 02

Notebook 02 derives `U` by considering the activity of the **unilateralized** device under arbitrary excitation (so `(A, φ)` are free). Notebook 08 considers the activity of the **original** device under a **constrained** excitation `V₂/V₁ = A e^{jφ}` imposed by the surrounding circuit. The optimum `(A_opt, φ_opt)` is the unconstrained extremum; topologies pin `(A', φ')` to a different value, and the gap between `(A', φ')` and `(A_opt, φ_opt)` causes `f_m < f_max`.

### What is *new* relative to notebook 06

Notebook 06 §12 introduces cross-coupled and Colpitts topologies as carriers for the phase-noise discussion (half-wave symmetry zeroing `c₀`, etc.). It does not compute `f_m` for those topologies, does not discuss ring oscillators, and does not address harmonic extraction. Notebook 08 covers exactly those gaps. Cross-references in both directions.

---

## 2. Backward dependencies

| Notebook | Section | Used in 08 § |
|----------|---------|--------------|
| 01 | ABCD cascade, Y-matrix port conventions | §2 (power flow) |
| 02 | §2 Mason's `U`, activity condition `4G₁₁G₂₂ < |Y₁₂+Y₂₁*|²` | §1, §3 |
| 02 | §3 invariance proof (used to motivate why `f_max` is intrinsic) | §1 |
| 03 | MAG, MSG, stability circles | §1 (recap) |
| 06 | §10 mmWave inductor Q, skin effect, SRF | §13 (cross-ref only) |
| 06 | §11 phase-noise FOM | §17 (triple-push PN penalty) |

No new derivations from earlier notebooks are required; everything is by reference.

---

## 3. Structure

```
Part I    Activity, (A, φ), and the topology-constrained optimum   §1–§4
Part II   Topology-constrained f_m                                  §5–§8
Part III  Large-signal swing dynamics                               §9–§10
Part IV   Systematic 7-step design + 121/104 GHz example            §11–§13
Part V    Beyond f_max: triple-push                                 §14–§17
Part VI   Wrap-up                                                   §18
```

Five interactives, summarized in §6 of this spec.

---

## 4. Part I — Activity, (A, φ), and the topology-constrained optimum (§1–§4)

### §1. Recap from notebook 02

One paragraph plus pointer. Mason's `U` is invariant under any lossless reciprocal embedding (notebook 02 §3); `f_max` is the frequency at which `U = 1`, the boundary above which **no** lossless reciprocal embedding can render the device active. Above `f_max` a fundamental oscillator is impossible by any topology; below `f_max` it is *possible* but not *automatic* — a real topology must also satisfy the constrained activity condition derived next.

### §2. Power-flow with constrained (A, φ)

Starting from the time-averaged real power flowing **into** the two-port:

$$
P_{in} = \mathrm{Re}(V_1^* I_1 + V_2^* I_2)
$$

Substituting `I_i = Y_{ij} V_j` and writing `V_1*V_2 = |V_1||V_2| e^{jφ}` with `A = |V_2|/|V_1|`, `φ = ∠(V_2/V_1)`:

$$
\frac{P_{in}}{|V_1||V_2|} = A^{-1} G_{11} + A\, G_{22} + \mathrm{Re}\!\left[ (Y_{12} + Y_{21}^*)\, e^{j\varphi} \right]
$$

Real power flowing **out** of the device, available for the embedding network to absorb:

$$
\frac{P_R}{|V_1||V_2|} = -\bigl(A^{-1} G_{11} + A\, G_{22}\bigr) - \bigl|Y_{12} + Y_{21}^*\bigr|\, \cos\!\bigl(\angle(Y_{12} + Y_{21}^*) + \varphi\bigr).
$$

This is paper eq. (6), with the algebra spelled out. Side derivation expanded inline showing the trig identity `Re((Y_{12}+Y_{21}^*) e^{jφ}) = (G_{12}+G_{21}) cos φ + (B_{21}-B_{12}) sin φ`.

### §3. Optimum amplitude and phase

For `G_{11}, G_{22} > 0` (the physical CMOS regime), AM-GM gives:

$$
A^{-1} G_{11} + A\, G_{22} \;\geq\; 2\sqrt{G_{11} G_{22}}, \quad \text{equality at } A_{opt} = \sqrt{G_{11}/G_{22}}.
$$

The cosine term is bounded above by `|Y_{12}+Y_{21}^*|`, achieved at `φ_opt = (2k+1)π − ∠(Y_{12}+Y_{21}^*)`. Combining:

$$
\max_{A, \varphi} \frac{P_R}{|V_1||V_2|} \;=\; -2\sqrt{G_{11} G_{22}} \;+\; |Y_{12} + Y_{21}^*|.
$$

Activity (`max P_R > 0`) is therefore equivalent to `4 G_{11} G_{22} < |Y_{12} + Y_{21}^*|²`, which is `U > 1` (notebook 02 §2). The two derivations of the activity condition — Sylvester's criterion on `Y_H` (nb 02) and the (A, φ) optimization (here) — give the *same* boundary; they differ in interpretation. The Sylvester argument bounds activity under arbitrary excitation; the (A, φ) argument identifies the **unique** `(A_opt, φ_opt)` that achieves the bound.

### §4. Negative-conductance vs. transfer activity

Two distinct regimes for the device to be active:
- **Negative-conductance activity** (`G_{11} < 0` or `G_{22} < 0`): rare in semiconductor transistors at normal bias.
- **Transfer activity** (`G_{11}, G_{22} > 0`, `4G_{11}G_{22} < |Y_{12}+Y_{21}^*|²`): the universal CMOS case.

The notebook works exclusively in the transfer-activity regime.

### Interactive I — Activity power-flow explorer

Hybrid-π MOSFET with sliders: `gm`, `Cgs`, `Cgd`, `Cds`, `Rg`, `ro`. The notebook computes `Y(f)` analytically from the model. Frequency slider `f`.

Two panels:
1. **Power-flow heatmap** of `P_R(A, φ) / (|V_1||V_2|)` at the chosen `f`, with the optimum marked. Color shows positive (active) vs. negative (passive) regions; contour at `P_R = 0` is the activity boundary in `(A, φ)`-space.
2. **Optimum vs. frequency** — `A_opt(f)` and `φ_opt(f)` curves over 1–300 GHz. Reproduces paper Fig. 11 / Fig. 25 qualitatively.

---

## 5. Part II — Topology-constrained `f_m` (§5–§8)

### §5. The constraint a topology imposes

Every connected oscillator topology forces the section's `(A', φ')` to take **fixed values** determined by Kirchhoff's laws and the cyclic structure of the loop. The device cannot freely choose its `(A, φ)`. The maximum oscillation frequency of the topology, `f_m`, is the highest `f` at which `P_R(A', φ', f) > 0` even after subtracting the loss of the passive embedding.

### §6. N-stage ring oscillator with inductive loading

Geometry: N identical sections each terminated by an inductor with parallel conductance `G_d`. Loop phase constraint `Σφ_i = 2πk` and amplitude constraint (steady state) give `A' = 1`, `φ' = k · 2π/N` per section.

Embedded `Y'` for one section (transistor in parallel with inductor) has `G_{22}' = G_{22} + G_d`. Substituting `A'=1`, `φ'=k·2π/N` into the constrained `P_R` expression:

$$
G_m \;\equiv\; \frac{P_R'}{|V_1'||V_2'|} \;=\; -\bigl(G_{11} + G_{22} + G_d\bigr) - |Y_{12} + Y_{21}^*|\, \cos\!\bigl(\angle(Y_{12}+Y_{21}^*) + k\cdot 2\pi/N\bigr).
$$

`G_m` is interpreted as the **largest extra parallel conductance** (beyond `G_d`) the section can tolerate while remaining active. `G_m > 0` ⇒ oscillation possible; `G_m = 0` ⇒ activity boundary; `G_m < 0` ⇒ impossible. For a given inductor Q, `G_d = ω L / Q_L · |Y_L|²` is fixed; the highest `f` at which `G_m(f) > 0` is `f_{m-N}`.

### §7. Cross-coupled (N=2) and three-stage ring (N=3)

- **Cross-coupled** (`N=2`, `k=1`, `φ' = π`): the cosine argument is `∠(Y_{12}+Y_{21}^*) + π`. For a typical hybrid-π, `∠(Y_{12}+Y_{21}^*)` is small and positive at mmWave, so the cosine sits well away from `−1`. Numerical example using the default hybrid-π (gm = 50 mS, Cgs = 30 fF, Cgd = 8 fF, Cds = 8 fF, Rg = 15 Ω, ro = 2 kΩ, target f_max ≈ 150–170 GHz) gives `f_{m-2} ≈ 110–130 GHz`, well below `f_max`. Reproduces paper Fig. 5.
- **Three-stage ring** (`N=3`, `k=1`, `φ' = 2π/3 = 120°`): much closer to typical `φ_opt(f) ≈ 110°–140°` at mmWave for CMOS. The cosine sits near `−1`, making the bound nearly tight. `f_{m-3}` lands within a few GHz of `f_max`. Reproduces paper Fig. 6.

The two cases are presented side-by-side with `G_m(f)` overlaid. The cross-coupled curve crosses zero at `f_{m-2}`; the three-stage curve crosses zero at `f_{m-3} ≈ f_max`.

### §8. `f_m` as a function of forced phase shift

Continuous sweep of `f_m(φ')` for `φ' ∈ [-180°, 180°]`, computed by finding the highest `f` at which `G_m(f, φ') > 0`. The curve has a broad plateau near `φ_opt` and collapses to zero where `cos(∠(Y_{12}+Y_{21}^*) + φ') > 0`. Discrete N-stage points (`N=2, 3, 4, 5`) are annotated. Reproduces paper Fig. 7.

### Interactive II — Ring-oscillator `f_m` explorer

Sliders: hybrid-π parameters (shared with Interactive I), `G_d` value, target `f_max` setpoint.

Two panels:
1. **`G_m(f)` overlay** for selectable `N ∈ {2, 3, 4, 5}` plus the `G_d` line. Highlights `f_{m-N}` at the zero crossing.
2. **`f_m(φ')` continuous sweep** with discrete N-stage markers annotated.

---

## 6. Part III — Large-signal swing dynamics (§9–§10)

### §9. Small-signal start-up to steady-state

Small-signal `G_m > 0` is the start-up condition. As the oscillation amplitude grows, the transistor's effective Y-parameters (especially `gm`) compress; `G_m(|V|, f)` decreases monotonically with `|V|`. Steady-state oscillation occurs at the swing `|V|_{ss}` where `G_m(|V|_{ss}, f_{osc}) = 0` (with `G_d` already absorbed into the expression). In practical units this means the transistor's regenerated power equals the inductor's dissipation. Output power scales as `P_{out} ∝ |V|_{ss}² · G_d`.

### §10. Large-signal Y from the hybrid-π

Quasi-static charge model with a tanh-saturating `gm(V_{gs})` characterizing the device. The transconductance compresses with peak `V_{gs}` swing as `gm_eff(|V|) = gm_0 · tanh(α |V|) / (α |V|)` (or equivalent simple saturation). Terminal capacitances are taken as bias-independent in this notebook (a simplification noted explicitly in scope). Y-parameters are computed at each `|V|` in this approximation.

`G_m(f)` plotted as a family for `|V| ∈ {0.5, 1.0, 1.2, 1.5}` V. Reproduces paper Fig. 8 qualitatively. The horizontal `G_d` line picks off `f_osc` at each `|V|`; following the family of curves down to small-signal at the top traces the approach to steady state.

### Interactive III — Swing-dependent `G_m(f, |V|)`

Sliders: hybrid-π parameters, `G_d`, swing-saturation parameter `α`.

Two panels:
1. **`G_m(f)` family** parameterized by `|V|`.
2. **Steady-state readout** — for a chosen `f_osc`, traces `|V|_{ss}` (intersection of family with `G_d` at that `f`) and reports `P_{out}` estimate.

---

## 7. Part IV — Systematic 7-step design + 121/104 GHz example (§11–§13)

### §11. The 7-step methodology

Reproduces paper Fig. 10 as a numbered list with the *why* of each step and the *failure-mode* that returns control to a previous step:

1. **Determine `f_max` of the process.** From Y-parameter measurement or simulation. Layout-dependent. Note that extrapolated `f_max` from `U(f)` slopes is typically optimistic.
2. **Choose `f_osc < f_max`.** Initial pick — often 50–80% of `f_max`. Iterated later if passives can't meet the constraint.
3. **Compute `(A_opt, φ_opt)` at `f_osc`** from §3. These set the topology target.
4. **Pick a topology with `(A', φ')` near `(A_opt, φ_opt)`.** N-stage ring chooses N to match `φ_opt` modulo `2π/N`. Colpitts trades capacitor-divider ratios. Cross-coupled enforces `(1, π)` and is usually a poor match at mmWave.
5. **Size transistor and pick passive component values.** Ideal passives at this stage. Choose W, finger count, biasing.
6. **Compute `G_d^{max} = G_m(f_osc; A', φ')`.** The largest parallel conductance the embedding can have while sustaining oscillation with margin.
7. **Realize the passives meeting `G_d^{max}`.** Inductor Q, varactor Q, parasitic shunts. If `G_d^{realized} > G_d^{max}`, return to step 5 (resize) or step 2 (lower `f_osc`).

Loop structure spelled out: each "fail" branch returns to a specific earlier step.

### §12. Worked example — 121 GHz / 104 GHz three-stage ring (0.13 µm-class hybrid-π)

Walk through with concrete numbers, defaults chosen so that the model approximates the paper's `f_max ≈ 174 GHz` regime. Step-by-step:

1. `f_max` ≈ 170 GHz (model).
2. `f_osc` = 121 GHz (initial).
3. `A_opt(121 GHz) ≈ 1.03`, `φ_opt(121 GHz) ≈ 129°`.
4. Three-stage ring (`A'=1`, `φ'=120°`) — close match.
5. `W = 10 µm × 10 fingers`, bias chosen for paper-comparable `gm`, `L_d ≈ 70 pH`.
6. `G_d^{max} ≈ 1.5–2.5 mS` (extracted from `G_m(f)` curve).
7. Inductor Q requirement: `Q ≥ ω L / R_p^{−1}` ⇒ `Q ≥ ~8` for the chosen `L`. Achievable with shielded coplanar transmission line (cross-ref nb 06 §10).

Output: simulated swing `|V|_{ss}`, output power, matched to paper Table I (within model tolerance).

The same flow is repeated for 104 GHz (different `L_d`).

### §13. Inductor Q at mmWave — pointer to nb 06

Brief paragraph cross-referencing nb 06 §10 for the skin-effect / substrate / SRF mechanisms. Notebook 08 takes Q as a design constraint, not a derived quantity.

### Interactive IV — Design-flow calculator

Form-style UI with seven sequential sections, one per methodology step. Each section reads inputs from the previous section's outputs.

Inputs: `f_osc` target, transistor parameters (or default profile), inductor Q at `f_osc`.

Outputs per step:
- Step 1: `f_max` of the model (computed).
- Step 2: pass/fail `f_osc < f_max`.
- Step 3: `A_opt`, `φ_opt`.
- Step 4: ranking of N-stage rings by `|φ' − φ_opt|`; recommended N.
- Step 5: transistor sizing (taken as input).
- Step 6: `G_d^{max}` (computed).
- Step 7: pass/fail check `G_d^{realized} = ω L / Q < G_d^{max}` with margin.

Final readout: pass/fail summary, recommended swing, output power estimate. No layout, no EM.

---

## 8. Part V — Beyond `f_max`: triple-push (§14–§17)

### §14. Why fundamental oscillation above `f_max` is impossible

`U(f) < 1` for `f > f_max` ⇒ no lossless reciprocal embedding makes the device active at `f`. Any single-frequency oscillator above `f_max` cannot exist, regardless of topology. The only way to extract energy at `f > f_max` is to oscillate at a fundamental `f_{fund} < f_max` and collect a harmonic `n · f_{fund}`.

### §15. Triple-push topology — third-harmonic phase coherence

Three-stage ring with sections separated by `φ' = 120°` at the fundamental. The third harmonic of section `i` has phase `3 · (i · 120°) = i · 360°`, so all three sections' third harmonics are **in phase** at any common-node summing point. The fundamental and second harmonic, by contrast, sum to zero. Output node collects `3 · f_{fund}`.

Three sections × in-phase summing ⇒ `3×` voltage amplitude at `3f_{fund}` relative to a single section, i.e., `9×` power. Fundamental and 2nd-harmonic rejection is set by section-mismatch (typically `−40` to `−50` dBc in simulation).

### §16. Gate-inductor `L_g` optimization

With the bare three-stage ring, each section sits at `(A'=1, φ'=120°)`. At the fundamental, the optimum is closer to `(A_opt ≈ 0.84, φ_opt ≈ 144°)` (paper Fig. 11 at 85 GHz fundamental for 256 GHz triple-push).

Adding a gate inductor `L_g` between the input pad and the gate accomplishes two things simultaneously:
1. **Shifts the per-section `(A', φ')` toward `(A_opt, φ_opt)`** by adding gate-side phase delay and reducing gate-side voltage swing. `G_m` at the fundamental rises (paper Fig. 21).
2. **Provides impedance matching at `3 · f_{fund}`** between the drain (where the harmonic is generated) and the gate (where it would otherwise be lost). Drain reflection coefficient at `3 · f_{fund}` is minimized at the optimum `L_g` (paper Fig. 22).

These two goals do not coincide perfectly. The notebook plots `G_m` at the fundamental and `|Γ|` at `3·f_{fund}` on a single `L_g` axis to expose the trade-off. The optimum `L_g` typically sits between the two individual optima, biased toward the harmonic match because output power at `3·f_{fund}` is the figure of merit.

### §17. Worked numbers — 256 GHz and 482 GHz

256 GHz example (0.13 µm-class hybrid-π, `f_{fund} ≈ 85 GHz`): with optimum `L_g ≈ 30 pH`, output power `~−3 dBm` simulated at `255 GHz` (paper). 482 GHz example (65 nm-class, `f_{fund} ≈ 150 GHz`, `L_g ≈ 17 pH`, `L_d ≈ 26 pH`): `~−3 dBm` simulated, `~−7.9 dBm` measured.

**Phase noise penalty.** Harmonic extraction multiplies the phase noise by `n²`, so triple-push degrades `L(Δω)` by `+20 log 3 ≈ +9.5` dB at the same offset. Cross-reference notebook 06 §11 for the FOM consequence: a triple-push 256 GHz oscillator with the same fundamental swing as a hypothetical fundamental 256 GHz oscillator has 9.5 dB worse phase noise. Whether this matters depends on the application.

### Interactive V — Triple-push gate-inductor optimization

Sliders: `L_g`, `L_d`, transistor `W`, `f_{fund}`.

Three panels:
1. **`G_m(f)` family parameterized by `L_g`** — paper Fig. 21 reproduction.
2. **Drain `|Γ|(f)` near `3·f_{fund}`** for selected `L_g` values — paper Fig. 22 reproduction.
3. **Time-domain `V_G(t)` and `V_{out}(t)`** at the chosen `L_g` — paper Fig. 23 reproduction.

Readouts: steady-state swing at the gate, output power at `3·f_{fund}`, fundamental rejection at `f_{fund}`.

---

## 9. Part VI — Wrap-up (§18)

Recap of the logical chain, in order:

1. Mason's `U` (nb 02) bounds what a device can do; `U=1` defines `f_max`.
2. The (A, φ) parameterization of activity (this nb §2–3) identifies the unconstrained optimum `(A_opt, φ_opt)` and reproduces the activity condition.
3. Real topologies impose a constrained `(A', φ')`. The gap `(A', φ') ↔ (A_opt, φ_opt)` causes `f_m < f_max`.
4. Three-stage rings happen to land near the optimum (`120° ≈ φ_opt` for typical CMOS), reaching nearly `f_max`. Cross-coupled topologies do not.
5. Large-signal swing settles at `G_m(|V|) = 0` after `G_d` absorption; output power follows.
6. The 7-step methodology mechanizes the above into a repeatable design flow.
7. Above `f_max`, fundamental oscillation is impossible; triple-push extracts the third harmonic of a sub-`f_max` ring. Gate-inductor `L_g` tunes the section toward `(A_opt, φ_opt)` and matches at `3·f_{fund}`. Phase noise penalty is `+20 log n` dB.

Cross-references:
- Phase noise of these topologies — notebook 06.
- mmWave inductor Q mechanisms — notebook 06 §10.
- Linearity / efficiency — notebook 07.

Navigation footer: Previous → `07_power_amplifiers_linearity.py` | Next → *TBD*.

---

## 10. Interactives summary

| # | Name                                  | Section | Sliders / inputs                                  | Outputs                                                               |
|---|---------------------------------------|---------|---------------------------------------------------|-----------------------------------------------------------------------|
| I | Activity power-flow explorer          | §4      | hybrid-π params, f                                | P_R(A,φ) heatmap; A_opt(f), φ_opt(f) curves                           |
| II| Ring-oscillator `f_m` explorer        | §8      | hybrid-π params, G_d, N                           | G_m(f) overlay for N; f_m(φ') continuous sweep                        |
|III| Swing-dependent G_m(f, |V|)           | §10     | hybrid-π params, G_d, α                           | G_m(f) family by |V|; |V|_ss readout at chosen f_osc; P_out estimate |
|IV | Design-flow calculator                | §11–13  | f_osc, transistor profile, inductor Q             | step-by-step pass/fail, recommended N, G_d^max, output power estimate |
| V | Triple-push gate-inductor optimization| §17     | L_g, L_d, W, f_fund                               | G_m(f, L_g); drain |Γ| near 3f_fund; time-domain V_G, V_out          |

---

## 11. Hybrid-π model specification

To keep the notebook self-contained, a common hybrid-π module is defined once and used by every interactive.

Topology:
- `C_gs` from intrinsic gate `g'` to source.
- `C_gd` from intrinsic gate `g'` to drain.
- `C_ds` from drain to source.
- `g_m · v_{gs}` controlled current, drain to source.
- `r_o` between drain and source.
- Series `R_g` between external gate (port 1) and intrinsic gate `g'`.

Intrinsic Y-matrix (port `g'`, port `d`, source common):

```
Y_{11}' = jω(C_gs + C_gd)
Y_{12}' = -jω C_gd
Y_{21}' = g_m - jω C_gd
Y_{22}' = 1/r_o + jω(C_ds + C_gd)
```

External Y-matrix accounting for series `R_g` at port 1 (standard T-network port reduction):

```
Y_{11} = Y_{11}' / (1 + Y_{11}' R_g)
Y_{12} = Y_{12}' / (1 + Y_{11}' R_g)
Y_{21} = Y_{21}' / (1 + Y_{11}' R_g)
Y_{22} = Y_{22}' − Y_{21}' R_g Y_{12}' / (1 + Y_{11}' R_g)
```

Without `R_g` the model has `G_{11} = 0` and infinite `f_max`; the gate resistance is the dominant `f_max`-limiting mechanism in this idealization.

Default values targeting `f_max ≈ 150–170 GHz` (representative of a 0.13 µm-class device):
`g_m = 50 mS`, `C_gs = 30 fF`, `C_gd = 8 fF`, `C_ds = 8 fF`, `R_g = 15 Ω`, `r_o = 2 kΩ`.

Large-signal extension (Part III only): `g_m_eff(|V|) = g_m · tanh(α |V|) / (α |V|)` with `α` user-tunable; capacitances held bias-independent.

The model is **not** intended to match a specific PDK quantitatively. It is intended to reproduce the *qualitative* behavior of the paper's Fig. 5 / 6 / 7 / 8 / 11 / 21 with paper-comparable numbers (within ~20% on `f_max`, `f_m`, and `(A_opt, φ_opt)`).

---

## 12. Verification checklist

Before marking notebook 08 complete:

1. `uv run marimo check marimo/notebooks/08_systematic_oscillator_design.py` — only `markdown-indentation` warnings allowed.
2. `python -c "import ast; ast.parse(open('marimo/notebooks/08_systematic_oscillator_design.py').read())"` — must pass.
3. `uv run python marimo/notebooks/08_systematic_oscillator_design.py` — non-interactive smoke-test must succeed.
4. Manual smoke-test: `uv run marimo edit marimo/notebooks/08_systematic_oscillator_design.py` — each of the five interactives renders and responds to input.
5. Numerical sanity:
   - At default hybrid-π, `f_max` is within 140–180 GHz.
   - `f_{m-2}` (cross-coupled) is below `f_{m-3}` (three-stage ring) by at least 30 GHz.
   - `f_{m-3}` is within 10 GHz of `f_max`.
   - `A_opt(100 GHz)` and `φ_opt(100 GHz)` lie in the ranges suggested by paper Fig. 11 (A_opt ~ 1.0, φ_opt ~ 120°–135°).

---

## 13. Out of scope (explicitly deferred)

- Phase-noise theory (already in nb 06; cross-referenced only).
- Push-push (×2) topology details and trade-offs.
- EM-simulated inductor S-parameters or PDK-accurate transistor models.
- Layout extraction, parasitic-aware layout iteration.
- Oscillator pulling, injection locking dynamics (in `interactive/vco_pulling.py`).
- Bias-dependent `C_gs`, `C_gd` (large-signal Part III uses bias-independent capacitances).
- Antenna integration / on-chip output matching beyond the bias-tee level.
- Class-E / Class-F PA-like efficiency optimization (in nb 07).

---

## 14. Updates to neighboring notebooks

- **Notebook 06 §12:** add a forward reference noting that the cross-coupled `f_{m-2}` ceiling is derived in nb 08 §7.
- **Notebook 06 §10:** confirm the inductor Q discussion is the canonical source (nb 08 cites it).
- **Notebook 07 navigation footer:** update Next to `08_systematic_oscillator_design.py`.
- **Notebook 08 navigation footer:** Previous = `07_power_amplifiers_linearity.py`, Next = TBD.
