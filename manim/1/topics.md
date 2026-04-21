# Video 1: Microwave Circuit Theory — From Telegrapher's Equation to Unilateral Power Gain

This video builds the conceptual and mathematical backbone of microwave circuit theory.
Starting from first principles on a transmission line, we develop the traveling-wave description
of signals, introduce S-parameters as the natural language of high-frequency networks, and
culminate in the unilateral power gain — a fundamental, frequency-independent figure of merit
for any active two-port.

---

## Topic Outline

| # | Topic | Key Concepts / Equations | Purpose in Narrative |
|---|-------|--------------------------|----------------------|
| 1 | Transmission Line Model | Distributed RLCG ladder; lumped-element limit breaks down as λ → line length | Motivates why we must treat wires as wave-guiding structures at microwave frequencies |
| 2 | Telegrapher's Equations | ∂V/∂z = −(R+jωL)I · · ∂I/∂z = −(G+jωC)V | Derives the coupled PDE governing voltage and current along the line |
| 3 | Wave Solutions & Propagation Constant | V(z) = V⁺e^{−γz} + V⁻e^{+γz} · · γ = α + jβ = √((R+jωL)(G+jωC)) | Shows V and I decompose into forward and backward traveling waves |
| 4 | Characteristic Impedance | Z₀ = √((R+jωL)/(G+jωC)) · · lossless: Z₀ = √(L/C) | Intrinsic impedance of the line; ratio of V⁺ to I⁺ |
| 5 | Reflection Coefficient | Γ_L = (Z_L − Z₀)/(Z_L + Z₀) | Quantifies how much of the incident wave is reflected at a load |
| 6 | Input Impedance & Standing Waves | Z_in = Z₀ · (Z_L + jZ₀ tan βl)/(Z₀ + jZ_L tan βl) · · VSWR = (1+\|Γ\|)/(1−\|Γ\|) | Impedance transformation along the line; standing wave pattern |
| 7 | Power Flow on a Transmission Line | P_avg = ½\|V⁺\|²/Z₀ · (1 − \|Γ\|²) | Separates available power from mismatch loss; foundation for gain definitions |
| 8 | Two-Port Network Basics | Port voltages/currents; Z, Y, ABCD parameter matrices | General framework before specializing to the S-parameter formalism |
| 9 | Why S-Parameters? | Z/Y require open/short terminations — impractical at GHz; S-params use matched loads | Motivates the switch to a traveling-wave, VNA-compatible description |
| 10 | S-Parameter Definitions | a_i = V_i⁺/√Z₀ · · b_i = V_i⁻/√Z₀ · · b_i = Σ_j S_ij a_j | Core definition in terms of normalized incident and reflected wave amplitudes |
| 11 | S-Matrix Equations | [b] = [S][a] · · S_11: input reflection · S_21: forward gain · S_12: reverse · S_22: output reflection | Physical meaning of each S-parameter; how to read a data sheet |
| 12 | Properties of the S-Matrix | Reciprocal: S = Sᵀ · · Lossless: S†S = I · · Passive: \|S_ij\| ≤ 1 | Constraints that physics (time-reversal, energy conservation) places on S |
| 13 | Signal Flow Graphs | Nodes (wave variables), directed branches (S_ij), Mason's gain rule | Systematic tool for cascading and analyzing multi-port S-parameter networks |
| 14 | Power Gain Definitions | Transducer gain G_T · Available gain G_A · Operating gain G_P | Distinguishes contributions of source mismatch, device, and load mismatch |
| 15 | Power Gain Expressions | G_T = \|S_21\|²(1−\|Γ_S\|²)(1−\|Γ_L\|²) / \|1−S_11Γ_S\|²\|1−S_22'Γ_L\|² | Explicit formulas in terms of S-params and source/load reflection coefficients |
| 16 | Stability Conditions | Rollett K-factor: K = (1 − \|S_11\|² − \|S_22\|² + \|Δ\|²)/(2\|S_12 S_21\|) · Δ = S_11S_22 − S_12S_21 · · Unconditional stability: K > 1 and \|Δ\| < 1 | Necessary prerequisite before maximizing gain; defines the safe design space |
| 17 | Unilateral Approximation | S_12 ≈ 0 · · Unilateral figure of merit: U_f = \|S_12 S_21 S_11 S_22\| / (1−\|S_11\|²)(1−\|S_22\|²) | Simplifies gain expressions when reverse transmission is negligible |
| 18 | Maximum Unilateral Transducer Gain | G_TU,max = \|S_21\|² / ((1−\|S_11\|²)(1−\|S_22\|²)) · achieved at Γ_S = S_11*, Γ_L = S_22* | Peak achievable gain under the unilateral assumption with conjugate matching |
| 19 | Unilateral Power Gain (Mason's U) | U = \|Y_21/Y_12 − 1\|² / (2K\|Y_21/Y_12\| − 2 Re{Y_21/Y_12}) · · equivalent S-param form | Invariant under lossless reciprocal embedding; the only gain that is a true device property |
| 20 | Physical Interpretation of U | U rolls off as 1/f² · · U = 1 defines f_max (maximum oscillation frequency) · · extrapolate measured U to find f_max | Ties the entire narrative back to a single number that characterizes an amplifier's ultimate speed limit |

---

## Manim Scene Breakpoints

| Scene | Topics | Description |
|-------|--------|-------------|
| `TransmissionLine` | 1–4 | Animate the RLCG ladder, derive telegrapher's equations, show traveling waves |
| `ReflectionAndImpedance` | 5–7 | Reflection coefficient, Smith chart intro, standing waves, power flow |
| `TwoPortAndSParams` | 8–12 | Two-port setup, S-parameter definition, S-matrix properties |
| `SignalFlowAndGain` | 13–15 | Signal flow graphs, three gain definitions side by side |
| `BilinearTransformations` (4.5) | — | Interlude: Möbius geometry behind Γ → Γ' maps; circles-to-circles; prepares the viewer for stability and gain circles |
| `StabilityAndUnilateral` | 16–18 | K-factor stability, unilateral approximation, max unilateral gain |
| `MasonsU` | 19–20 | Derive Mason's U, show frequency rolloff, define f_max |
| `NonlinearAndNoise` | beyond | Compression, harmonics, IP3, AM-PM, noise figure, Friis |
