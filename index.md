# Microwave Circuit Theory: Implementation Index

This document outlines the systematic implementation plan for the microwave circuit theory series. The modules must be developed in the exact order listed below to ensure a rigorous, monotonic dependency chain where each concept is derived from first principles established in preceding modules.

## Implementation Order & Detail Outline

### 01. Two-Port Fundamentals
*Objective: Establish foundational network representations and terminal conditions.*

**Implementation Details:**
- **Port Conditions:** Define N-port networks, linearity, and current flow conditions.
- **Z (Impedance) Parameters:** Derive from open-circuit conditions.
- **Y (Admittance) Parameters:** Derive from short-circuit conditions.
- **ABCD (Transmission) Parameters:** Define cascade matrices and prove their utility in chain networks.
- **Reciprocity:** Provide the mathematical conditions for reciprocal vs. active/non-reciprocal networks.

---

### 02. Power, Waves, and S-Parameters
*Objective: Transition from voltage/current to high-frequency power waves.*

**Implementation Details:**
- **Complex Power:** Define time-domain instantaneous power and phasor representations.
- **Available Power:** Derive the conjugate match condition ($Z_L = Z_S^*$) mathematically.
- **Power Waves:** Define incident ($a$) and reflected ($b$) variables in terms of voltage, current, and reference impedance.
- **Scattering Matrix (S-Parameters):** Define $S_{11}, S_{21}, S_{12}, S_{22}$. Prove properties for unitary (lossless) matrices.
- **Signal Flow Graphs (SFG):** Introduce nodes, branches, and the topological approach.
- **Mason's Non-Touching Loop Rule:** Implement the formal analytical engine for solving $\Gamma_{in}$, $\Gamma_{out}$, and complex network transfer functions.

---

### 03. Power Gains and Stability
*Objective: Analyze bilateral gain and define the boundaries of oscillation.*

**Implementation Details:**
- **Gain Trio:** Formally derive the three fundamental gains using S-parameters and SFG:
  1. Operating Power Gain ($G$)
  2. Available Power Gain ($G_A$)
  3. Transducer Power Gain ($G_T$)
- **Stability Criteria:** Set up the input/output reflection coefficient boundary conditions ($|\Gamma_{in}| < 1, |\Gamma_{out}| < 1$).
- **Rollett's $K$ Factor:** Derive the K-factor and the auxiliary condition ($\Delta < 1$) for unconditional stability.
- **The $\mu$-test:** Implement the single-parameter Edwards-Sinsky stability metric.
- **Stability Circles:** Map unstable regions geometrically onto the Smith Chart (centers and radii for input/output planes).
- **Maximum Gain:** Derive Maximum Available Gain (MAG) and Maximum Stable Gain (MSG).

---

### 04. Mason's Unilateral Power Gain U
*Objective: Define invariant figures of merit for active devices.*

**Implementation Details:**
- **U-Parameter Derivation:** Define Mason's $U$ based on device activity/passivity boundaries.
- **Invariance Proof:** Demonstrate that $U$ remains invariant under any lossless, reciprocal embedding network.
- **Speed Limits:** Relate the $U$ parameter to the maximum frequency of oscillation ($f_{\max}$).
- **Roll-off Dynamics:** Implement the $-20\text{ dB/decade}$ high-frequency roll-off law.

---

### 05. MOSFET Small-Signal Model
*Objective: Connect abstract two-port theory to high-frequency transistor physics.*

**Implementation Details:**
- **Equivalent Circuit:** Define the intrinsic and extrinsic MOSFET parameters ($g_m$, $g_{ds}$, $C_{gs}$, $C_{gd}$, $R_g$, etc.).
- **$f_T$ Derivation:** Analytically derive the unity current-gain frequency ($f_T$).
- **$f_{\max}$ Derivation:** Analytically derive the unity power-gain frequency ($f_{\max}$) and contrast it with $f_T$.
- **Technology Context:** Apply these metrics to real-world scenarios, specifically targeting GF 22FDX (FDSOI) performance data.

---

### 06. Load-Pull and Stability In-Depth
*Objective: Apply the theoretical framework to practical power amplifier design.*

**Implementation Details:**
- **Load-Pull & Source-Pull:** Implement the logic for mapping constant gain and constant power contours.
- **Efficiency Metrics:** Define and calculate Power-Added Efficiency (PAE) in both linear and compressed operating regimes.
- **Design Flow:** Formalize the 5-step procedure for RF amplifier design:
  1. Device characterization
  2. Stability verification
  3. Source/Load pull optimization
  4. Matching network synthesis
  5. Large-signal validation

---
*Note: All future code, interactive figures (e.g., Plotly Smith Charts), and mathematical derivations should strictly adhere to this roadmap.*
