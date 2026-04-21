# Microwave Circuit Theory: Advanced Implementation Index

This document outlines the systematic implementation plan for the microwave circuit theory series. The modules are structured monotonically, emphasizing mathematical rigor, matrix theory, and deep first-principles derivations.

## Implementation Order & Detail Outline

### 01. Two-Port Networks & Formal Parameter Transformations
*Objective: Establish a rigorous mathematical foundation for network parameters and their matrix properties.*

**Implementation Details:**
- **Formal Theory of N-Port Networks:** Define linear networks, terminal voltage/current conditions, and port conventions.
- **Parameter Sets:** Rigorous definitions of Impedance ($Z$), Admittance ($Y$), Hybrid ($H$), and Chain/Transmission ($ABCD$) matrices.
- **Transformations:** Step-by-step mathematical derivations of transformations between parameter sets (e.g., $Z \leftrightarrow Y \leftrightarrow ABCD$).
- **Matrix Properties & Physics:** 
  - **Reciprocity:** Derived as symmetry in $Z/Y$ matrices ($Z = Z^T$).
  - **Losslessness:** The emergence of skew-Hermitian matrices ($Z = -Z^\dagger$).
  - **Passivity:** Introduction to the Hermitian part of the admittance matrix and positive semi-definiteness.

---

### 02. Power Waves & Scattering Parameters
*Objective: Transition from abstract voltage/current matrices to Kurokawa power waves and network topology.*

**Implementation Details:**
- **Complex Power Theory:** Formulate instantaneous and phasor power, separating active (real) and reactive (imaginary) components.
- **Power Waves:** Formal derivation of Kurokawa's incident ($a$) and reflected ($b$) power waves, distinct from traveling voltage/current waves.
- **The Scattering Matrix ($S$):** Definition and physical meaning. Normalization to arbitrary reference impedances.
- **S-Parameter Transformations:** Deriving the relationship between $S$, $Z$, and $Y$ matrices. Proof of the Unitary property for lossless networks ($S^\dagger S = I$).
- **Signal Flow Graphs (SFG):** Topological representation of the S-matrix.
- **Mason's Non-Touching Loop Rule:** The formal analytical engine for solving complex networks (e.g., evaluating $\Gamma_{in}$ and $\Gamma_{out}$).

---

### 03. Power Gains, Circles, and Stability
*Objective: Geometric and analytical derivation of performance boundaries and the onset of oscillation.*

**Implementation Details:**
- **The Gain Trio:** Derivation of Operating ($G$), Available ($G_A$), and Transducer ($G_T$) power gains using SFG and S-parameters.
- **Geometric Stability Criteria:** 
  - Derivation of the input and output stability circles on the $\Gamma$ plane.
  - Formally defining unconditional stability using the geometric distance between the center of the Smith Chart and the center of the stability circle ($|C| - R > 1$).
- **Analytical Stability Metrics:** Deriving Rollett's $K$, $\Delta$, and the $\mu$-test strictly from the geometric distance criteria.
- **Gain Circles:** Geometric derivations for constant-gain loci and simultaneous conjugate match boundaries.
- **Maximum Limits:** Derivations of Maximum Available Gain (MAG) and Maximum Stable Gain (MSG).

---

### 04. Mason's Unilateral Gain ($U$) & Matrix Algebra
*Objective: Deep dive into invariant figures of merit and the limits of active devices.*

**Implementation Details:**
- **Network Embedding:** The concept of neutralizing an active device by embedding it in a lossless, reciprocal external network.
- **Hermitian & Skew-Hermitian Decomposition:** Decomposing the overall network matrix (e.g., $Y = Y_H + Y_{SH}$) to isolate the active/dissipative terms from the reactive terms.
- **Sylvester's Criterion:** Applying Sylvester's criterion for positive semi-definiteness to the Hermitian part of the matrix to determine the absolute boundary of activity.
- **Mason's $U$ Derivation:** Formally deriving the invariant $U$ parameter from Sylvester's criterion.
- **Invariance Proof:** Mathematical proof that $U$ remains invariant under any reciprocal transformation or lossless embedding.
- **Speed Limits ($f_{\max}$):** Linking $U$ to the maximum frequency of oscillation and deriving the theoretical $-20\text{ dB/dec}$ roll-off.

---

### 05. Transistor Small-Signal Models & High-Frequency Physics
*Objective: Connect abstract matrix theory to solid-state device physics.*

**Implementation Details:**
- **MOSFET Equivalent Circuit:** Formulating the intrinsic and extrinsic small-signal model ($g_m, g_{ds}, C_{gs}, C_{gd}, R_g$).
- **Parameter Extraction:** Converting the physical circuit into a canonical $Y$-matrix.
- **Current Gain Limit ($f_T$):** Analytical derivation of the unity short-circuit current-gain frequency using $h_{21}$.
- **Power Gain Limit ($f_{\max}$):** Applying Mason's $U$ (from Module 04) to the extracted Y-parameters to evaluate $f_{\max}$, explicitly showing the impact of parasitic gate resistance ($R_g$) and feedback capacitance ($C_{gd}$).

---

### 06. Large-Signal Operation & Load-Pull
*Objective: Expand beyond linear S-parameters into power amplifier design.*

**Implementation Details:**
- **Linear vs. Non-Linear:** The breakdown of small-signal S-parameters as the device enters compression.
- **Load-Pull Theory:** Mapping constant power and constant efficiency contours, contrasting their shapes and physical origins with linear gain circles.
- **Efficiency Metrics:** Defining Drain Efficiency ($\eta$) and Power-Added Efficiency (PAE).
- **Design Flow Application:** Synthesizing the concepts—from linear stability checks to large-signal contour optimization—to formalize a practical RF PA design flow.
