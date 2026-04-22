# Microwave Circuit Theory — Notebook Rewrite Guide

Authoritative spec for the six-notebook series. Each section lists what the notebook
**introduces** (new to the reader), what it **uses** (from prior notebooks), the
required derivations, and the interactive figures to include.

Tone: symbolic expressions before numerical values; first principles throughout;
concise prose, no hand-waving. Every final result should be boxed or displayed.

---

## Dependency chain

```
01 → 02 → 03 → 04
               ↓
              05 (uses 04's U)
               ↓
              06 (uses 03's stability circles + gain circles)
```

Each notebook may only reference concepts from earlier notebooks. When editing, if a
definition is used before it is introduced, move the derivation earlier rather than
adding a forward reference.

---

## 01 · Two-Port Fundamentals

**File:** `notebooks/01_two_port_fundamentals.py`

**Introduces:** Everything needed to describe a linear, time-invariant two-port in
matrix form. No frequency-domain scattering yet — that is notebook 02.

**Uses:** Phasor conventions only (assumed known).

### Required sections

1. **Port conventions and the N-port framework**
   - Definition of port voltage $V_n$ and port current $I_n$; reference directions.
   - Linear superposition; the general N-port admittance relation $\mathbf{I} = \mathbf{Y}\mathbf{V}$.

2. **Impedance matrix Z**
   - Definition: $\mathbf{V} = \mathbf{Z}\mathbf{I}$, open-circuit measurement.
   - Physical meaning of $Z_{ij}$.

3. **Admittance matrix Y**
   - Definition: $\mathbf{I} = \mathbf{Y}\mathbf{V}$, short-circuit measurement.
   - Relation $\mathbf{Y} = \mathbf{Z}^{-1}$ (when non-singular).

4. **Hybrid matrix H**
   - Mixed excitation: $\begin{pmatrix}V_1\\I_2\end{pmatrix} = \mathbf{H}\begin{pmatrix}I_1\\V_2\end{pmatrix}$.
   - Relevance: transistor small-signal models are often specified in H.

5. **ABCD (chain/transmission) matrix**
   - Definition for cascading: $\begin{pmatrix}V_1\\I_1\end{pmatrix} = \mathbf{T}\begin{pmatrix}V_2\\-I_2\end{pmatrix}$.
   - Cascade property: $\mathbf{T}_\text{total} = \mathbf{T}_1 \mathbf{T}_2 \cdots$.

6. **Transformation table**
   - Step-by-step algebraic derivations of Z↔Y, Z↔ABCD, Y↔H. Show the algebra
     explicitly; don't just quote results.

7. **Matrix properties from physics**
   - **Reciprocity:** $Z_{12} = Z_{21}$, $Y_{12} = Y_{21}$; proof from Green's reciprocity.
   - **Losslessness:** $\operatorname{Re}(\mathbf{Z}_H) = 0$ (skew-Hermitian real part).
   - **Passivity:** $\mathbf{Y}_H = \frac{\mathbf{Y}+\mathbf{Y}^\dagger}{2} \succeq 0$
     (positive semi-definite); connection to Sylvester's criterion (preview — full
     development in notebook 04).

8. **Interactive figure: Smith-chart cascade**
   - Sliders for two-section ABCD network; plot input reflection coefficient on
     Smith chart as sections change.

9. **Navigation footer** pointing to NB02.

---

## 02 · Power, Waves, and Network Representations

**File:** `notebooks/02_power_gain_definitions.py`

**Introduces:** Complex power, available power, mismatch, Kurokawa power waves,
S-matrix, all inter-parameter conversions (S↔Z↔Y↔ABCD), signal flow graphs, Mason's
non-touching loop rule, Γ_in and Γ_out. Nothing about gain definitions or stability —
those are NB03.

**Uses:** NB01 (Z, Y, ABCD, reciprocity).

### Required sections

1. **Complex power from first principles**
   - Instantaneous power $p(t) = v(t)i(t)$; phasor form; active power $P$, reactive
     power $Q$, complex power $S = P + jQ = \frac{1}{2}V I^*$.
   - Power conservation across lossless networks.

2. **Available power of a source and mismatch factor**
   - Source $V_s$, $Z_s = R_s + jX_s$; available power $P_\text{avs} = |V_s|^2/(8R_s)$.
   - Delivered power as a function of load $Z_L$; mismatch factor
     $M = 1 - |\Gamma_L|^2$ derivation.
   - **Interactive:** mismatch factor vs. $\Gamma_L$ on Smith chart (slider for $Z_s$,
     drag $Z_L$).

3. **Kurokawa power waves**
   - Motivation: voltage/current split depends on reference plane; power waves do not.
   - Definitions: $a = (V + Z_0 I)/(2\sqrt{R_0})$, $b = (V - Z_0^* I)/(2\sqrt{R_0})$.
   - Proof: $P = |a|^2 - |b|^2$.
   - Connection to reflection coefficient $\Gamma = b/a$.

4. **The scattering matrix**
   - Definition $\mathbf{b} = \mathbf{S}\mathbf{a}$; port excitation protocol.
   - Physical meaning of $S_{11}$, $S_{21}$, $S_{12}$, $S_{22}$.
   - **Losslessness:** $\mathbf{S}^\dagger \mathbf{S} = \mathbf{I}$.
   - **Reciprocity:** $\mathbf{S} = \mathbf{S}^T$.

5. **Conversion between representations**
   - Derive $\mathbf{S}$ from $\mathbf{Z}$: $\mathbf{S} = (\mathbf{Z} - Z_0\mathbf{I})(\mathbf{Z} + Z_0\mathbf{I})^{-1}$ and inverse.
   - S↔Y, S↔ABCD for two-port (explicit 2×2 formulas with algebra shown).

6. **Signal flow graphs**
   - Node = wave variable; branch = S-parameter; direction = causality.
   - Build the SFG for a two-port with source and load terminations.
   - Topological rules: series, parallel, self-loop elimination.

7. **Mason's non-touching loop rule**
   - Statement: $T = \frac{\sum_k P_k \Delta_k}{\Delta}$.
   - $\Delta = 1 - \sum L_1 + \sum L_2 - \cdots$ (loop gain sums).
   - Worked example: closed-form expression for any branch in the two-port SFG.

8. **Γ_in and Γ_out via Mason**
   - Derive $\Gamma_\text{in} = S_{11} + S_{12}S_{21}\Gamma_L/(1-S_{22}\Gamma_L)$ from
     the SFG using Mason's rule; same for $\Gamma_\text{out}$.

9. **Summary and navigation** to NB03.

---

## 03 · Power Gains and Stability

**File:** `notebooks/03_s_parameters_stability.py`

**Introduces:** The three power gain definitions, unilateral factorisation, stability
criteria (K, μ), stability circles, MAG/MSG, gain circles, load-pull, unilateral error
sweep. First notebook where transistor design trade-offs appear.

**Uses:** NB01 (Z, Y), NB02 (S-params, Γ_in/Γ_out, Mason's rule).

### Required sections

1. **Three gain definitions in Y-parameters**
   - Operating gain $G = P_L / P_\text{in}$; derivation in terms of $Y_{ij}$ and $Y_L$.
   - Available gain $G_A = P_\text{avout} / P_\text{avs}$; depends only on $Y_S$.
   - Transducer gain $G_T = P_L / P_\text{avs}$; depends on both terminations.
   - Inequality $G_A \geq G_T$ (with equality only at conjugate output match).

2. **Three gain definitions in S-parameters**
   - Re-derive $G$, $G_A$, $G_T$ in terms of $S_{ij}$, $\Gamma_S$, $\Gamma_L$ using
     $\Gamma_\text{in}$ / $\Gamma_\text{out}$ from NB02 §8.
   - Compact forms: $G_T = |S_{21}|^2 (1-|\Gamma_S|^2)(1-|\Gamma_L|^2) / |D|^2$
     where $D = (1-S_{11}\Gamma_S)(1-S_{22}\Gamma_L) - S_{12}S_{21}\Gamma_S\Gamma_L$.

3. **Unilateral factorisation**
   - Set $S_{12} = 0$; factorised form $G_T^{(U)} = G_S \cdot |S_{21}|^2 \cdot G_L$.
   - Unilateral figure of merit $U_m$: bound on error $1/(1+U_m)^2 \leq G_T/G_T^{(U)} \leq 1/(1-U_m)^2$.
   - Note: $U_m$ is distinct from Mason's invariant $U$ (introduced in NB04).

4. **Stability setup**
   - Condition for stability: $|\Gamma_\text{in}| < 1$ and $|\Gamma_\text{out}| < 1$ for
     all passive terminations.
   - Geometric view: stability boundaries are circles in the $\Gamma$ plane.

5. **Rollett K derivation**
   - Input stability circle: centre $C_S$, radius $R_S$ (full algebra).
   - Output stability circle: centre $C_L$, radius $R_L$.
   - Unconditional stability condition from geometry: $|C_S| - R_S > 1$ and
     $|C_L| - R_L > 1$.
   - Reduce to Rollett's scalar criteria:
     $$K = \frac{1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2}{2|S_{12}S_{21}|} > 1$$
     $$|\Delta| = |S_{11}S_{22} - S_{12}S_{21}| < 1$$

6. **μ-test (single-parameter unconditional stability)**
   - Derive $\mu = (1 - |S_{11}|^2) / (|S_{22} - \Delta S_{11}^*| + |S_{12}S_{21}|) > 1$.
   - Proof that $\mu > 1$ is equivalent to $K > 1$ and $|\Delta| < 1$.

7. **Stability circles: full derivation**
   - Input stability circle in $\Gamma_S$-plane; output in $\Gamma_L$-plane.
   - Sign convention: which region is stable (requires checking a test point).
   - **Interactive:** Smith chart with input/output stability circles; sliders for
     $S_{11}$, $S_{22}$, $S_{12}$, $S_{21}$.

8. **MAG and MSG**
   - MAG (maximum available gain): achieved at simultaneous conjugate match when $K > 1$.
     $$\text{MAG} = \left|\frac{S_{21}}{S_{12}}\right|(K - \sqrt{K^2 - 1})$$
   - MSG (maximum stable gain): $K = 1$ boundary value $|S_{21}/S_{12}|$.
   - Derive from the gain expression with $\Gamma_S = \Gamma_\text{in}^*$,
     $\Gamma_L = \Gamma_\text{out}^*$.

9. **Gain circles and simultaneous conjugate match loci**
   - Constant-$G_T$ circles in $\Gamma_L$-plane (derivation).
   - Constant-$G_A$ circles in $\Gamma_S$-plane.
   - **Interactive:** overlay stability and gain circles on Smith chart; show the
     feasible design region.

10. **Load-pull analysis**
    - Linear load-pull circles (from $G_T^{(U)}$):
      centre $c_L = g_L S_{22}^* / (1 - (1-g_L)|S_{22}|^2)$,
      radius $r_L = \sqrt{1-g_L}(1-|S_{22}|^2) / (1-(1-g_L)|S_{22}|^2)$.
    - Bilateral correction when $S_{12} \neq 0$.
    - **Interactive:** constant-gain contours sweeping $g_L$.

11. **Unilateral error sweep**
    - Plot $G_T / G_T^{(U)}$ vs. frequency for a realistic device; show where the
      unilateral approximation breaks down.

12. **Design strategies summary**
    - Flowchart: check stability → choose terminations on gain/stability circles → verify.

13. **Navigation footer** (Previous: NB02, Next: NB04).

---

## 04 · Mason's Unilateral Power Gain U

**File:** `notebooks/04_unilateral_power_gain.py`

**Introduces:** Hermitian/skew-Hermitian decomposition, Sylvester's activity criterion,
Mason's invariant $U$, invariance proof, $U$ in S-parameters, −20 dB/decade rolloff,
$f_\text{max}$.

**Uses:** NB01 (Y-matrix, passivity preview), NB03 (K, $\Delta$, MAG/MSG ordering).

### Required sections

1. **Motivation — why a new gain metric?**
   - MAG changes when you embed the transistor in a lossless network.
   - Mason's question (1954): is there an embedding-invariant gain? Answer: $U$.

2. **Derivation of U from the activity criterion**
   - Hermitian decomposition: $\mathbf{Y}_H = (\mathbf{Y}+\mathbf{Y}^\dagger)/2$.
   - Real power $P_\text{diss} = \mathbf{v}^\dagger \mathbf{Y}_H \mathbf{v}$; passivity iff $\mathbf{Y}_H \succeq 0$.
   - Sylvester's criterion for 2×2: conditions A ($G_{11} \geq 0$), B ($G_{22} \geq 0$),
     C ($\det(\mathbf{Y}_H) \geq 0$).
   - In a physical transistor, A and B hold; activity requires $\neg$C:
     $$G_{11}G_{22} - |Y_{12}+Y_{21}^*|^2/4 < 0$$
   - Unilateralization sets $Y'_{12} = 0$, $Y'_{21} = Y_{21} - Y_{12}$; apply $\neg$C:
     $$\boxed{U = \frac{|Y_{21}-Y_{12}|^2}{4G_{11}G_{22}} > 1}$$
   - $U = 1$ defines $f_\text{max}$.

3. **Proof of invariance**
   - Invariance under lossless reciprocal parallel feedback (shunt susceptance $jB_f$):
     numerator $|Y_{21}-Y_{12}|^2$ unchanged; $G_{ij}$ unchanged → $U$ unchanged.
   - Invariance under series reactances (Z-parameter form of $U$, identical structure).
   - Invariance under ideal transformers (scaling cancels).
   - Conclusion: $U$ is a fundamental topological invariant of the device.

4. **Gain ordering**
   $$U \geq \text{MAG} \geq \text{MSG} = |S_{21}/S_{12}|$$
   - Proof that unilateralized MAG equals $U$.

5. **U in S-parameters**
   $$U = \frac{|S_{21}/S_{12} - 1|^2}{2K|S_{21}/S_{12}| - 2\operatorname{Re}(S_{21}/S_{12})}$$
   - Derive by substituting Y↔S conversion; use K from NB03 §5.
   - Limiting case $S_{12} \to 0$: $U \to \infty$ (lossless feedback).

6. **Interactive: U vs. frequency**
   - Load realistic GaN/MOSFET S-parameter data (or parametric model); plot $U$(dB)
     and MAG(dB) vs. frequency on same axes; mark $f_\text{max}$ at $U = 0$ dB.
   - Slider for $R_g$ to show sensitivity.

7. **The −20 dB/decade law**
   - For an intrinsic transistor at high frequency, $U \propto 1/f^2$ →
     20 dB/decade rolloff. Derivation from MOSFET Y-parameters (preview of NB05).

8. **Navigation footer** (Previous: NB03, Next: NB05).

---

## 05 · MOSFET Small-Signal Model

**File:** `notebooks/05_mosfet_small_signal.py`

**Introduces:** Physical small-signal equivalent circuit, intrinsic Y-parameter
expressions, $f_T$ from $h_{21}$, $f_\text{max}$ from Mason's $U$, GF 22FDX numbers.
First notebook with device-level physics.

**Uses:** NB02 (Y-parameters, power waves), NB04 ($U$, $f_\text{max}$).

### Required sections

1. **Intrinsic small-signal equivalent circuit**
   - Element table: $C_{gs}$, $C_{gd}$, $g_m$, $g_{ds}$, $R_g$ (gate loss), $R_s$, $R_d$.
   - Distinction between intrinsic (no parasitics) and extrinsic model.
   - Schematic diagram (inline SVG or table description).

2. **Y-parameters of the intrinsic MOSFET ($R_g = R_s = R_d = 0$)**
   - Derive by nodal analysis:
     $$Y_{11} = j\omega(C_{gs}+C_{gd}), \quad Y_{12} = -j\omega C_{gd}$$
     $$Y_{21} = g_m - j\omega C_{gd}, \quad Y_{22} = g_{ds} + j\omega C_{gd}$$
   - Note: $Y_{21} - Y_{12} = g_m$ (real, frequency-independent) — this drives $U$.
   - Non-reciprocity measure: $|Y_{21}| \gg |Y_{12}|$ for $\omega \ll g_m/C_{gd}$.

3. **Derivation of f_T**
   - Short-circuit current gain $h_{21} = Y_{21}/Y_{11}$.
   - For $\omega \ll g_m/C_{gd}$: $|h_{21}| \approx g_m/[\omega(C_{gs}+C_{gd})]$.
   - Setting $|h_{21}| = 1$:
     $$\boxed{f_T = \frac{g_m}{2\pi(C_{gs}+C_{gd})}}$$
   - Effect of $R_s$ degeneration: $g_{m,\text{eff}} = g_m/(1+g_mR_s)$.

4. **Derivation of f_max from U**
   - For the intrinsic model (no $R_g$): $\operatorname{Re}(Y_{11}) = 0$ → $U \to \infty$.
   - Gate resistance $R_g$ provides the finite loss:
     $$\operatorname{Re}(Y'_{11}) \approx \omega^2 R_g(C_{gs}+C_{gd})^2 \quad (\omega R_g C_{gs} \ll 1)$$
   - Substitute into $U$ formula:
     $$f_\text{max} = \frac{f_T}{2\sqrt{R_g(g_{ds} + \omega^2 C_{gd}^2/g_m^2 \cdot g_{ds}/g_m)}}$$
     (simplified leading-order result; show the full algebra).
   - Approximation for $g_{ds} \to 0$: $f_\text{max} \approx f_T / (2\sqrt{R_g g_{ds}})$.

5. **GF 22FDX device numbers**
   - Table of extracted $g_m$, $C_{gs}$, $C_{gd}$, $R_g$, $g_{ds}$ at a representative
     bias point.
   - Compute $f_T$ and $f_\text{max}$ numerically; compare to published values.

6. **Interactive: f_T and f_max vs. bias**
   - Sliders for $g_m$ and $C_{gd}$; plot $|h_{21}|$(dB) and $U$(dB) vs. frequency;
     mark $f_T$ and $f_\text{max}$.

7. **Navigation footer** (Previous: NB04, Next: NB06).

---

## 06 · Load-Pull and PA Design

**File:** `notebooks/06_loadpull_stability.py`

**Introduces:** Nonlinear load-pull framework, source-pull, combined stability + gain
contours, efficiency metrics (η, PAE), practical PA design flow. Extension of NB03's
linear gain circles into the power-amplifier regime.

**Uses:** NB02 (S-params), NB03 (stability circles, gain circles, linear load-pull).

### Required sections

1. **Scope of this notebook**
   - NB03 §10 derived *linear* load-pull circles (small-signal, $S_{12}$ correction).
   - Here: large-signal load-pull and practical design methodology.
   - Topics: nonlinear load-pull (numerical), source-pull, combined stability + gain
     contours, simultaneous input/output stability.

2. **Linear load-pull circles (recap + bilateral correction)**
   - Restate centre/radius formulas from NB03 §10.
   - Bilateral correction via $\Gamma_\text{in}(\Gamma_L)$ iteration.
   - **Interactive:** draggable $\Gamma_L$ on Smith chart; show $G_T$ vs. position.

3. **Nonlinear load-pull — large-signal model**
   - Breakdown of small-signal $S_{21}$ in compression.
   - Simplified quasi-static nonlinear model; harmonic-balance motivation.
   - Constant-$P_\text{out}$ contours vs. constant-$G_T$ contours: shape comparison.

4. **Source-pull**
   - Same formalism applied to $\Gamma_S$; constant-$G_A$ contours.
   - Interaction between source-pull and load-pull when $S_{12} \neq 0$.

5. **Efficiency metrics**
   - Drain efficiency $\eta = P_\text{out}/P_{DC}$.
   - Power-added efficiency $\text{PAE} = (P_\text{out} - P_\text{in})/P_{DC}$.
   - Constant-PAE contours on Smith chart; relationship to gain circles.

6. **Combined stability + gain contour design space**
   - Overlay stability circles (from NB03 §7), gain circles, and PAE contours.
   - Identify feasible region: stable, sufficient gain, target efficiency.
   - **Interactive:** full Smith chart overlay; all three circle families; parameter
     sliders.

7. **PA design flow summary**
   1. Check unconditional stability (K > 1, μ > 1) at all frequencies.
   2. If conditionally stable, choose $\Gamma_L$ inside stable region.
   3. Identify load-pull contour for target $P_\text{out}$.
   4. Check PAE; iterate source-pull for input match.
   5. Verify across temperature/process corners.

8. **Navigation footer** (Previous: NB05).

---

## Cross-notebook navigation links

Each notebook ends with a `mo.md` cell containing Previous / Next links:

```python
mo.md(r"""
**Previous:** [NN-1 — Title](NN-1_file.py)

**Next:** [NN+1 — Title](NN+1_file.py)
""")
```

When renaming a notebook, update the link text in both neighbours and in the summary
table in NB06 §1.

---

## Interactive figure conventions

- All Plotly figures: `template="plotly_dark"`.
- Wrap in `mo.ui.plotly(fig)` for interactive zoom/pan.
- Sliders: `mo.ui.slider(start, stop, step, value=default, label="...")`.
- Layout: `mo.vstack([slider_row, fig])` or `mo.hstack([fig, controls])`.
- Smith chart: draw unit circle + real/imaginary grid in Plotly `go.Scatter` with
  `mode="lines"`, then overlay data traces.
- Cell-local variables (not shared downstream): prefix with `_`.
- If two interactives in the same notebook share a quantity name (e.g. $K$), append a
  context suffix: `K_c` for the stability-circle explorer, `K` for the gain explorer.

---

## Marimo DAG rules (summary)

- Each `@app.cell` function's parameters are its upstream dependencies.
- All top-level assignments are outputs returned as a tuple.
- Two cells assigning the same name is a DAG collision; `marimo check` flags it.
- Private scratch variables: prefix with `_`.
- Before declaring done: `uv run marimo check notebooks/NN_xxx.py` (only
  `markdown-indentation` warnings are acceptable) and
  `python -c "import ast; ast.parse(open('notebooks/NN_xxx.py').read())"`.
