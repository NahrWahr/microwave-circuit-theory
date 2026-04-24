# Notebook 05 — Matching Networks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `marimo/notebooks/05_matching_networks.py` — a 20-section marimo notebook covering lumped/distributed matching network synthesis, broadband (Bode-Fano / Chebyshev) design, transformer coupling, and cyclostationary noise in mixers and samplers, per the spec at `docs/superpowers/specs/2026-04-23-notebook-05-matching-networks.md`.

**Architecture:** One primary notebook (`05_matching_networks.py`) plus two new shared helper modules: `_lib_matching.py` (L/π/T synthesis, stub tuners, Chebyshev prototypes, λ/4 transformer, Bode-Fano bound) and `_lib_cyclo.py` (cyclostationary noise generation, mixer conversion-matrix NF, kT/C sampler noise). The existing `_lib_circles.py` (gain/stability circles) and `_lib_noise.py` (Friis) are reused via import — never modified. The notebook follows the same `@app.cell` + marimo boilerplate as notebooks 01–04.

**Tech Stack:** Python 3.14+, marimo 0.23.0, numpy, plotly (dark template, `mo.ui.plotly`), uv.

---

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `marimo/notebooks/05_matching_networks.py` | create | Main notebook — 20 sections across 7 parts |
| `marimo/notebooks/_lib_matching.py` | create | L/π/T synthesis, stub tuners, Chebyshev prototype, λ/4 TL, Bode-Fano |
| `marimo/notebooks/_lib_cyclo.py` | create | Cyclostationary noise simulation, mixer NF, kT/C sampler |
| `marimo/notebooks/04_noise_and_lna_design.py` | modify (last task only) | Update navigation footer Next link → `05_matching_networks.py` |

---

## Verification loop (runs after every task)

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

- Command 1: no output (syntactic validity).
- Command 2: only `markdown-indentation` warnings acceptable.
- Command 3: exits 0.

After any task that touches `_lib_*.py`:

```bash
python -c "from marimo.notebooks._lib_matching import *; print('lib_matching ok')"
python -c "from marimo.notebooks._lib_cyclo import *;   print('lib_cyclo ok')"
```

---

## Task index

```
Task 0    Scaffold notebook file
Task 1    §1  Motivation: mismatch loss and power transfer theorem
Task 2    §2  Network representations (ABCD cascade)
Task 3    §3  Smith chart review and matching trajectories
Task 4    _lib_matching.py — L/π/T synthesis helpers
Task 5    §4  L-network synthesis
Task 6    §5  π-network synthesis
Task 7    §6  T-network synthesis
Task 8    Interactive I.1 — L/π/T network explorer
Task 9    §7  Bode-Fano criterion
Task 10   §8  Chebyshev matching network synthesis + _lib_matching.py extension
Task 11   §9  Real-frequency technique (brief)
Task 12   Interactive II.1 — Broadband Chebyshev explorer
Task 13   §10 Quarter-wave transformer + _lib_matching.py extension
Task 14   §11 Single-stub tuner + _lib_matching.py extension
Task 15   §12 Double-stub tuner + _lib_matching.py extension
Task 16   §13 On-chip transformer coupling
Task 17   Interactive III.1 — Distributed matching explorer (Smith chart)
Task 18   §14 Revisiting the 28 GHz LNA output match
Task 19   _lib_cyclo.py — cyclostationary helpers
Task 20   §15 Cyclostationary processes
Task 21   §16 Mixer noise theory
Task 22   §17 Switched-capacitor (sampler) noise
Task 23   Interactive IV.1 — Mixer noise explorer
Task 24   §18 Summary and bridge to notebook 06
Task 25   Final cross-notebook wiring (update 04's footer)
```

---

## Task 0: Scaffold the notebook file

**Files:**
- Create: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Create the file with marimo boilerplate matching notebooks 01-04**

```python
# v1.0
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # 05 — Matching Networks, Broadband Design, and Cyclostationary Noise

    Builds on notebooks 01–04. Lumped and distributed impedance matching,
    Bode-Fano bandwidth limits, transformer coupling, and mixer/sampler noise.
    """)
    return


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 2: Run the three verification commands**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

Expected: command 1 no output; command 2 no errors; command 3 exits 0.

---

## Task 1: §1 Motivation — mismatch loss and the power transfer theorem

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §1 cell after the title cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ## Part I — Impedance Matching Fundamentals

    ### §1. Motivation: Mismatch Loss and the Power Transfer Theorem

    Maximum power transfer from source $Z_S = R_S + jX_S$ to load $Z_L$ occurs when
    $Z_L = Z_S^*$, delivering the **available power**:

    $$P_{\text{avail}} = \frac{|V_S|^2}{8 R_S}$$

    For any other load the fraction actually delivered is:

    $$\text{ML} = 1 - |\Gamma_L|^2 = \frac{4 R_S R_L}{|Z_S + Z_L|^2}$$

    where $\Gamma_L = (Z_L - Z_S^*)/(Z_L + Z_S)$. In dB: $\text{ML}_{\text{dB}} = 10\log_{10}(1 - |\Gamma_L|^2)$.

    **Quality factor.** For a series resonator: $Q = X/R$; for a shunt resonator: $Q = B/G$.
    A single-resonator match has approximate 3 dB bandwidth:

    $$\text{BW}_{3\,\text{dB}} \approx \frac{f_0}{Q_{\text{loaded}}}$$

    Large impedance transformation ratios ($R_H/R_L \gg 1$) force high $Q$, narrow bandwidth.
    Parts II–IV provide the tools to escape this constraint.
    """)
    return
```

- [ ] **Step 2: Run the three verification commands**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 2: §2 Network representations and §3 Smith chart review

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §2 cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §2. Network Representations for Matching

    Matching networks are two-ports between $Z_S$ and $Z_L$.
    ABCD matrices cascade by multiplication, making them the most
    convenient representation for ladder networks:

    $$\begin{bmatrix}V_1\\I_1\end{bmatrix} = \begin{bmatrix}A & B\\C & D\end{bmatrix}
    \begin{bmatrix}V_2\\-I_2\end{bmatrix}$$

    For a **series impedance** $Z$: $A=D=1,\; B=Z,\; C=0$.
    For a **shunt admittance** $Y$: $A=D=1,\; B=0,\; C=Y$.

    A lossless LC network satisfies $AD - BC = 1$, $A,D \in \mathbb{R}$, $B,C \in j\mathbb{R}$.
    Reciprocity requires $AD - BC = 1$ for passive networks.
    """)
    return
```

- [ ] **Step 2: Add §3 cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §3. Smith Chart Review and Matching Trajectories

    The Smith chart maps $Z/Z_0$ to $\Gamma = (Z - Z_0)/(Z + Z_0)$.
    Adding a **series element** moves along a constant-resistance circle;
    adding a **shunt element** moves along a constant-conductance circle.

    | Element | Type | Direction on Smith chart |
    |---------|------|--------------------------|
    | Series $L$ | +jX | Clockwise on constant-$R$ circle |
    | Series $C$ | −jX | Counter-clockwise on constant-$R$ circle |
    | Shunt $C$ | +jB | Clockwise on constant-$G$ circle |
    | Shunt $L$ | −jB | Counter-clockwise on constant-$G$ circle |

    A matching procedure is a sequence of arc moves from $Z_L/Z_0$
    to the chart centre ($\Gamma = 0$). Parts II and IV make this concrete
    for L/π/T networks and stub tuners respectively.
    """)
    return
```

- [ ] **Step 3: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 3: §4 L-network synthesis (theory cell)

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §4 theory cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ## Part II — Lumped Matching Networks

    ### §4. L-Network Synthesis

    The L-network transforms between two real impedances $R_S$ and $R_L$ ($R_H = \max$, $R_L = \min$)
    using two reactive elements. The design Q is fixed by the impedance ratio:

    $$Q = \sqrt{\frac{R_H}{R_L} - 1}$$

    **Low-pass topology** (shunt $C$ at high-impedance port, series $L$ in series arm):

    $$B_p = \frac{Q}{R_H} \quad X_s = Q \cdot R_L$$

    giving element values at $f_0$:

    $$C_p = \frac{Q}{\omega_0 R_H}, \qquad L_s = \frac{Q \cdot R_L}{\omega_0}$$

    **High-pass topology**: swap $C \leftrightarrow L$ in each position (series $C$, shunt $L$).

    The 3 dB bandwidth is $\text{BW} \approx f_0/Q$. For a given ratio $R_H/R_L$, Q is
    fixed — the L-network has no bandwidth knob. This motivates π and T networks.
    """)
    return
```

- [ ] **Step 2: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 4: `_lib_matching.py` — L/π/T synthesis helpers

**Files:**
- Create: `marimo/notebooks/_lib_matching.py`

- [ ] **Step 1: Create `_lib_matching.py` with L-network, π-network, T-network, and two-port S-parameter helper**

```python
# v1.0
"""Pure-Python helpers for impedance matching network synthesis.

Used by notebook 05 (Matching Networks). No marimo imports.
Conventions: Pozar, *Microwave Engineering* 4th ed., Chapters 5 and 6.
All frequencies in Hz, impedances in Ω, elements in H or F.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# L-network
# ---------------------------------------------------------------------------

def l_network(R_S: float, R_L: float, f0: float,
              topology: str = "lowpass") -> dict:
    """Design an L-network matching R_S to R_L at f0.

    topology: 'lowpass'  → shunt-C at high-Z port, series-L in series arm
              'highpass' → shunt-L at high-Z port, series-C in series arm

    Returns dict with keys: Q, Cp (F), Ls (H) for lowpass;
    Lp (H), Cs (F) for highpass.
    """
    assert R_S > 0 and R_L > 0 and f0 > 0
    R_H, R_Lo = max(R_S, R_L), min(R_S, R_L)
    Q = np.sqrt(R_H / R_Lo - 1.0)
    w0 = 2.0 * np.pi * f0
    if topology == "lowpass":
        Cp = Q / (w0 * R_H)
        Ls = Q * R_Lo / w0
        return {"Q": Q, "Cp_F": Cp, "Ls_H": Ls}
    else:  # highpass
        Lp = R_H / (w0 * Q)
        Cs = 1.0 / (w0 * Q * R_Lo)
        return {"Q": Q, "Lp_H": Lp, "Cs_F": Cs}


# ---------------------------------------------------------------------------
# π-network
# ---------------------------------------------------------------------------

def pi_network(R_S: float, R_L: float, f0: float, Q: float) -> dict:
    """Design a low-pass π-network (shunt-C, series-L, shunt-C).

    Q must satisfy Q > sqrt(max(R_S,R_L)/min(R_S,R_L) - 1).
    Returns dict with keys: C1_F, L_H, C2_F, Q_min.
    """
    R_H, R_Lo = max(R_S, R_L), min(R_S, R_L)
    Q_min = np.sqrt(R_H / R_Lo - 1.0)
    assert Q > Q_min, f"Q={Q:.3f} must exceed Q_min={Q_min:.3f}"
    w0 = 2.0 * np.pi * f0
    # Virtual resistor R_virt = R_H / (1 + Q^2) <= R_Lo
    # Two L-sections: left Q_1 from R_S to R_virt, right Q_2 from R_virt to R_L
    R_virt = R_H / (1.0 + Q ** 2)
    Q1 = np.sqrt(R_S / R_virt - 1.0)
    Q2 = np.sqrt(R_L / R_virt - 1.0)
    C1 = Q1 / (w0 * R_S)
    C2 = Q2 / (w0 * R_L)
    # Series element: two inductors in series (one from each L-section) sum
    L = (Q1 * R_virt + Q2 * R_virt) / w0
    return {"Q_min": Q_min, "C1_F": C1, "L_H": L, "C2_F": C2}


# ---------------------------------------------------------------------------
# T-network
# ---------------------------------------------------------------------------

def t_network(R_S: float, R_L: float, f0: float, Q: float) -> dict:
    """Design a low-pass T-network (series-L, shunt-C, series-L).

    Q must exceed Q_min. Returns dict: L1_H, C_F, L2_H, Q_min.
    """
    R_H, R_Lo = max(R_S, R_L), min(R_S, R_L)
    Q_min = np.sqrt(R_H / R_Lo - 1.0)
    assert Q > Q_min, f"Q={Q:.3f} must exceed Q_min={Q_min:.3f}"
    w0 = 2.0 * np.pi * f0
    R_virt = R_Lo * (1.0 + Q ** 2)
    Q1 = np.sqrt(R_virt / R_S - 1.0)
    Q2 = np.sqrt(R_virt / R_L - 1.0)
    L1 = Q1 * R_S / w0
    L2 = Q2 * R_L / w0
    C = (Q1 / R_S + Q2 / R_L) / w0   # two shunt caps in parallel
    return {"Q_min": Q_min, "L1_H": L1, "C_F": C, "L2_H": L2}


# ---------------------------------------------------------------------------
# Frequency-domain S-parameter evaluation for 2-element ladder networks
# ---------------------------------------------------------------------------

def abcd_series_Z(Z):
    """ABCD matrix for series impedance Z (numpy array or scalar)."""
    one = np.ones_like(Z)
    zero = np.zeros_like(Z)
    # shape (..., 2, 2)
    return np.array([[one, Z], [zero, one]]).transpose(2, 0, 1) if Z.ndim > 0 else np.array([[1, Z], [0, 1]])


def abcd_shunt_Y(Y):
    """ABCD matrix for shunt admittance Y."""
    one = np.ones_like(Y)
    zero = np.zeros_like(Y)
    return np.array([[one, zero], [Y, one]]).transpose(2, 0, 1) if Y.ndim > 0 else np.array([[1, 0], [Y, 1]])


def abcd_to_S(ABCD, Z0: float = 50.0):
    """Convert ABCD matrices (shape N×2×2) to S-parameters (shape N×2×2)."""
    A = ABCD[:, 0, 0]
    B = ABCD[:, 0, 1]
    C = ABCD[:, 1, 0]
    D = ABCD[:, 1, 1]
    denom = A + B / Z0 + C * Z0 + D
    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S21 = 2.0 / denom
    S22 = (-A + B / Z0 - C * Z0 + D) / denom
    N = len(A)
    S = np.zeros((N, 2, 2), dtype=complex)
    S[:, 0, 0] = S11
    S[:, 0, 1] = S12
    S[:, 1, 0] = S21
    S[:, 1, 1] = S22
    return S


def l_network_S(R_S: float, R_L: float, f0: float,
                topology: str, freqs: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Return S-matrix (N×2×2) of an L-network over frequency array freqs (Hz).

    The network is designed at f0. S-params computed from ABCD chain.
    Source and load terminations are R_S, R_L; Z0 used for ABCD→S conversion.
    """
    params = l_network(R_S, R_L, f0, topology)
    w = 2.0 * np.pi * freqs
    if topology == "lowpass":
        # shunt-C at input (high-Z side), series-L in series arm
        # Assuming R_S >= R_L (high-Z at port 1)
        Cp, Ls = params["Cp_F"], params["Ls_H"]
        Z_series = 1j * w * Ls
        Y_shunt = 1j * w * Cp
        M1 = np.stack([np.array([[1, 0], [yy, 1]]) for yy in Y_shunt])
        M2 = np.stack([np.array([[1, zz], [0, 1]]) for zz in Z_series])
    else:
        Lp, Cs = params["Lp_H"], params["Cs_F"]
        Z_series = 1.0 / (1j * w * Cs)
        Y_shunt = 1.0 / (1j * w * Lp)
        M1 = np.stack([np.array([[1, 0], [yy, 1]]) for yy in Y_shunt])
        M2 = np.stack([np.array([[1, zz], [0, 1]]) for zz in Z_series])
    ABCD = np.einsum('nij,njk->nik', M1, M2)
    return abcd_to_S(ABCD, Z0)


def pi_network_S(R_S: float, R_L: float, f0: float, Q: float,
                 freqs: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Return S-matrix (N×2×2) of a low-pass π-network over freqs."""
    params = pi_network(R_S, R_L, f0, Q)
    w = 2.0 * np.pi * freqs
    Y1 = 1j * w * params["C1_F"]
    Y2 = 1j * w * params["C2_F"]
    ZL = 1j * w * params["L_H"]
    M1 = np.stack([np.array([[1, 0], [y, 1]]) for y in Y1])
    M2 = np.stack([np.array([[1, z], [0, 1]]) for z in ZL])
    M3 = np.stack([np.array([[1, 0], [y, 1]]) for y in Y2])
    ABCD = np.einsum('nij,njk->nik', np.einsum('nij,njk->nik', M1, M2), M3)
    return abcd_to_S(ABCD, Z0)


def t_network_S(R_S: float, R_L: float, f0: float, Q: float,
                freqs: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Return S-matrix (N×2×2) of a low-pass T-network over freqs."""
    params = t_network(R_S, R_L, f0, Q)
    w = 2.0 * np.pi * freqs
    Z1 = 1j * w * params["L1_H"]
    YC = 1j * w * params["C_F"]
    Z2 = 1j * w * params["L2_H"]
    M1 = np.stack([np.array([[1, z], [0, 1]]) for z in Z1])
    M2 = np.stack([np.array([[1, 0], [y, 1]]) for y in YC])
    M3 = np.stack([np.array([[1, z], [0, 1]]) for z in Z2])
    ABCD = np.einsum('nij,njk->nik', np.einsum('nij,njk->nik', M1, M2), M3)
    return abcd_to_S(ABCD, Z0)


# ---------------------------------------------------------------------------
# Bode-Fano bound (parallel RC load)
# ---------------------------------------------------------------------------

def bode_fano_gamma_max(BW_Hz: float, R_L: float, C_L: float) -> float:
    """Maximum achievable |Γ_max| over BW_Hz for a parallel-RC load (Pozar §5.8).

    Returns |Γ_max| (linear). If the bound implies |Γ_max| ≥ 1, returns 1.0.
    """
    # ∫ ln(1/|Γ|) dω ≤ π/(R_L C_L)
    # For equal-ripple fill over BW (rad/s):
    # BW * ln(1/|Γ_max|) ≤ π / (R_L * C_L)
    BW_rad = 2.0 * np.pi * BW_Hz
    exponent = np.pi / (R_L * C_L * BW_rad)
    return float(np.exp(-exponent))


# ---------------------------------------------------------------------------
# Chebyshev low-pass prototype element values
# ---------------------------------------------------------------------------

def chebyshev_prototype(N: int, ripple_dB: float) -> list[float]:
    """Return normalised element values g_0, g_1, ..., g_N, g_{N+1} (Pozar Table 5.5).

    Ladder network: g_0 = source conductance, g_1..g_N are L/C values,
    g_{N+1} = load conductance/resistance.
    ripple_dB: passband ripple in dB (0.01 to 3 dB typical).
    """
    epsilon = np.sqrt(10.0 ** (ripple_dB / 10.0) - 1.0)
    beta = np.log(1.0 / np.tanh(ripple_dB / 17.37))
    gamma = np.sinh(beta / (2.0 * N))
    g = [1.0]  # g_0
    for k in range(1, N + 1):
        a_k = np.sin((2 * k - 1) * np.pi / (2 * N))
        b_k = gamma ** 2 + np.sin(k * np.pi / N) ** 2
        if k == 1:
            g.append(2.0 * a_k / gamma)
        else:
            g.append(4.0 * a_k * g[k - 1] / (b_k * g[k - 1]))
    # g_{N+1}
    if N % 2 == 0:
        g.append(1.0 / np.tanh(beta / 4.0) ** 2)
    else:
        g.append(1.0)
    return g


def chebyshev_bandpass_elements(N: int, ripple_dB: float,
                                R_S: float, R_L: float,
                                f0: float, BW: float) -> list[dict]:
    """Scale Chebyshev prototype to a bandpass network between R_S and R_L.

    Returns list of N dicts, each with keys: type ('series_LC' or 'shunt_LC'),
    L_H, C_F.
    BW: 3-dB bandwidth in Hz.
    """
    g = chebyshev_prototype(N, ripple_dB)
    w0 = 2.0 * np.pi * f0
    bw = 2.0 * np.pi * BW
    elements = []
    for k in range(1, N + 1):
        if k % 2 == 1:  # series arm
            L = g[k] * R_S / bw
            C = bw / (w0 ** 2 * L * bw)
            elements.append({"type": "series_LC", "L_H": L, "C_F": C})
        else:  # shunt arm
            C = g[k] / (bw * R_S)
            L = bw / (w0 ** 2 * C * bw)
            elements.append({"type": "shunt_LC", "L_H": L, "C_F": C})
    return elements


# ---------------------------------------------------------------------------
# Quarter-wave transformer
# ---------------------------------------------------------------------------

def qw_transformer(R_S: float, R_L: float) -> dict:
    """Design a single-section quarter-wave transformer.

    Returns Z1 (Ω) — the transformer line impedance.
    """
    return {"Z1_ohm": float(np.sqrt(R_S * R_L))}


def qw_transformer_S11(R_S: float, R_L: float, f0: float,
                       freqs: np.ndarray) -> np.ndarray:
    """Return |S11| of a single λ/4 transformer vs frequency.

    Uses the exact transmission-line two-port (Pozar eq. 5.47).
    """
    Z1 = np.sqrt(R_S * R_L)
    theta = np.pi / 2.0 * freqs / f0  # electrical length (π/2 at f0)
    # Input impedance: Z_in = Z1 * (R_L + j Z1 tan θ) / (Z1 + j R_L tan θ)
    t = np.tan(theta)
    Z_in = Z1 * (R_L + 1j * Z1 * t) / (Z1 + 1j * R_L * t)
    Gamma = (Z_in - R_S) / (Z_in + R_S)
    return np.abs(Gamma)


# ---------------------------------------------------------------------------
# Single-stub tuner
# ---------------------------------------------------------------------------

def single_stub_tuner(Z_L: complex, Z0: float, f0: float,
                      stub_type: str = "short") -> list[dict]:
    """Design a parallel single-stub tuner (Pozar §5.2).

    Returns up to 2 solutions: each dict has d_lambda (position as fraction of λ)
    and ell_lambda (stub length as fraction of λ).
    stub_type: 'short' or 'open'.
    """
    Y0 = 1.0 / Z0
    YL = 1.0 / Z_L
    GL = YL.real
    BL = YL.imag
    solutions = []
    # Solve for t = tan(βd) such that Re(Y_in) = Y0
    # G_in(d) = Y0 when t satisfies quadratic (Pozar eq. 5.8)
    # G_L(1 + t^2) / ((G_L + Y0 t^2 * ... )) = Y0  →  derived below:
    # Using admittance transformation: Y_in = Y0 * (YL + jY0 t) / (Y0 + j YL t)
    # Re(Y_in) = Y0 =>
    # GL*(1 + t^2) / (1 + 2*BL/Y0*t + (GL^2 + BL^2)/Y0^2 * t^2) = Y0
    a = GL / Y0
    b = BL / Y0
    # Equation: GL*(1+t²) = Y0 * |Y0 + j*YL*t|² / Y0²  ... standard result:
    # t² (GL² + BL² - GL*Y0) + t * 2*BL*Y0 + GL*Y0 - Y0² = 0  ...
    # Simplified Pozar (5.8): t = [BL ± sqrt(GL*(Y0 - GL) + BL²)] / (GL - Y0)
    discriminant = GL * (Y0 - GL) + BL ** 2
    if discriminant < 0:
        return []  # load cannot be matched (forbidden region)
    for sign in [+1.0, -1.0]:
        t = (BL + sign * np.sqrt(discriminant)) / (GL - Y0) if abs(GL - Y0) > 1e-15 * Y0 else np.inf
        d_lambda = np.arctan(t) / (2.0 * np.pi) % 0.5
        # Susceptance at position d
        Y_in = Y0 * (YL + 1j * Y0 * t) / (Y0 + 1j * YL * t)
        B_in = Y_in.imag
        # Stub susceptance must cancel B_in
        if stub_type == "short":
            # B_stub = -Y0 * cot(beta*ell)  => cot(x) = Y0 / B_stub_need
            # Y0 * cot(x) = -B_in  => cot(x) = -B_in/Y0  => tan(x) = -Y0/B_in
            ell_lambda = np.arctan(-Y0 / B_in) / (2.0 * np.pi) % 0.5
        else:  # open
            # B_stub = Y0 * tan(beta*ell)  => tan = -B_in / Y0
            ell_lambda = np.arctan(-B_in / Y0) / (2.0 * np.pi) % 0.5
        solutions.append({
            "d_lambda": float(d_lambda),
            "ell_lambda": float(ell_lambda),
            "d_mm": float(d_lambda * 3e8 / (f0 * np.sqrt(1.0)) * 1e3),  # in air
            "ell_mm": float(ell_lambda * 3e8 / (f0 * np.sqrt(1.0)) * 1e3),
        })
    return solutions
```

- [ ] **Step 2: Verify the library imports and runs basic assertions**

```bash
python -c "
from marimo.notebooks._lib_matching import l_network, pi_network, t_network, qw_transformer, bode_fano_gamma_max, chebyshev_prototype, single_stub_tuner
import numpy as np

# L-network
p = l_network(50, 200, 1e9, 'lowpass')
assert abs(p['Q'] - np.sqrt(200/50 - 1)) < 1e-9
print('l_network ok')

# pi-network: Q must exceed Q_min
p = pi_network(50, 200, 1e9, 2.0)
assert p['Q_min'] < 2.0
print('pi_network ok')

# T-network
p = t_network(50, 200, 1e9, 2.0)
assert p['Q_min'] < 2.0
print('t_network ok')

# QW transformer
d = qw_transformer(50, 200)
assert abs(d['Z1_ohm'] - 100.0) < 1e-6
print('qw_transformer ok')

# Bode-Fano
gmax = bode_fano_gamma_max(1e9, 200, 1e-12)
assert 0 < gmax < 1
print('bode_fano ok')

# Chebyshev prototype
g = chebyshev_prototype(3, 0.1)
assert len(g) == 5  # g0..g3, g4
print('chebyshev_prototype ok')

# Single-stub
sols = single_stub_tuner(100+50j, 50.0, 1e9)
assert len(sols) > 0
print('single_stub_tuner ok')

print('lib_matching ok')
"
```

Expected: all lines print, final line is `lib_matching ok`.

- [ ] **Step 3: Run the three notebook verification commands**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 5: §5 π-network and §6 T-network theory cells

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §5 π-network cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §5. π-Network Synthesis

    The π-network uses two shunt elements and one series element, giving
    a free Q parameter $Q > Q_{\min} = \sqrt{R_H/R_L - 1}$.
    Higher $Q$ → narrower bandwidth (useful for harmonic suppression);
    lower $Q$ → wider bandwidth (down to $Q_{\min}$, which recovers the L-network).

    A virtual resistance $R_v = R_H/(1 + Q^2)$ splits the network into
    two L-sections. For the low-pass version:

    $$C_1 = \frac{Q_1}{\omega_0 R_S}, \quad
      L   = \frac{(Q_1 + Q_2) R_v}{\omega_0}, \quad
      C_2 = \frac{Q_2}{\omega_0 R_L}$$

    where $Q_1 = \sqrt{R_S/R_v - 1}$, $Q_2 = \sqrt{R_L/R_v - 1}$, and
    $Q = Q_1 + Q_2$ approximately.
    """)
    return
```

- [ ] **Step 2: Add §6 T-network cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §6. T-Network Synthesis

    The T-network (two series arms, one shunt arm) is the dual of the π.
    Its virtual resistance is $R_v = R_L(1 + Q^2)$.

    The **tapped-capacitor** network used in notebook 04 §17.6 is
    a degenerate T: the two series arms are capacitors and the shunt
    arm is the device output capacitance $C_{ds}$. The tap ratio:

    $$n_{\text{tap}} = \sqrt{R_L / R_{\text{eff}}}, \quad
      C_{\text{tap}} = C_{\text{total}} / n_{\text{tap}}^2$$

    provides impedance transformation without any inductors, at the cost of
    narrower bandwidth (a pure-capacitor divider has no resonance shaping).

    For a full T with tunable Q:

    $$L_1 = \frac{Q_1 R_S}{\omega_0}, \quad
      C   = \frac{Q_1/R_S + Q_2/R_L}{\omega_0}, \quad
      L_2 = \frac{Q_2 R_L}{\omega_0}$$
    """)
    return
```

- [ ] **Step 3: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 6: Interactive I.1 — L/π/T network explorer

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add the interactive cell**

```python
@app.cell
def _(mo, np, go, make_subplots):
    import sys
    sys.path.insert(0, "marimo/notebooks")
    from _lib_matching import (l_network, pi_network, t_network,
                               l_network_S, pi_network_S, t_network_S)

    _topo_m = mo.ui.dropdown(
        options=["L (lowpass)", "L (highpass)", "pi", "T"],
        value="L (lowpass)", label="Topology"
    )
    _Rs_m = mo.ui.slider(10, 500, value=50, step=10, label="R_S (Ω)")
    _Rl_m = mo.ui.slider(10, 500, value=200, step=10, label="R_L (Ω)")
    _f0_m = mo.ui.slider(1e9, 100e9, value=28e9, step=1e9, label="f₀ (Hz)")
    _Q_m  = mo.ui.slider(1.1, 20.0, value=3.0, step=0.1, label="Q (π/T only)")

    mo.md("### §7. Interactive I — L/π/T Network Explorer")
    return (_topo_m, _Rs_m, _Rl_m, _f0_m, _Q_m,
            l_network, pi_network, t_network,
            l_network_S, pi_network_S, t_network_S)


@app.cell
def _(_topo_m, _Rs_m, _Rl_m, _f0_m, _Q_m,
      l_network, pi_network, t_network,
      l_network_S, pi_network_S, t_network_S,
      mo, np, go, make_subplots):
    _Rs = float(_Rs_m.value)
    _Rl = float(_Rl_m.value)
    _f0 = float(_f0_m.value)
    _Q  = float(_Q_m.value)
    _topo = _topo_m.value

    _freqs = np.linspace(_f0 / 10, _f0 * 10, 500)

    # Compute element values and S-params
    if _topo == "L (lowpass)":
        _params = l_network(_Rs, _Rl, _f0, "lowpass")
        _S = l_network_S(_Rs, _Rl, _f0, "lowpass", _freqs)
        _rows = [(k, f"{v:.4g}") for k, v in _params.items()]
    elif _topo == "L (highpass)":
        _params = l_network(_Rs, _Rl, _f0, "highpass")
        _S = l_network_S(_Rs, _Rl, _f0, "highpass", _freqs)
        _rows = [(k, f"{v:.4g}") for k, v in _params.items()]
    elif _topo == "pi":
        _Q_min = (_params_pi := pi_network(_Rs, _Rl, _f0, max(_Q, 1.001 * (_pqm := ((_Rs/_Rl if _Rs>_Rl else _Rl/_Rs) - 1)**0.5 + 0.01))))
        _params = pi_network(_Rs, _Rl, _f0, max(_Q, _params_pi["Q_min"] + 0.01))
        _S = pi_network_S(_Rs, _Rl, _f0, max(_Q, _params["Q_min"] + 0.01), _freqs)
        _rows = [(k, f"{v:.4g}") for k, v in _params.items()]
    else:  # T
        _params = t_network(_Rs, _Rl, _f0, max(_Q, t_network(_Rs, _Rl, _f0, 99)["Q_min"] + 0.01))
        _S = t_network_S(_Rs, _Rl, _f0, max(_Q, _params["Q_min"] + 0.01), _freqs)
        _rows = [(k, f"{v:.4g}") for k, v in _params.items()]

    _S11_dB = 20 * np.log10(np.abs(_S[:, 0, 0]) + 1e-20)
    _S21_dB = 20 * np.log10(np.abs(_S[:, 1, 0]) + 1e-20)

    _fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["S₁₁ and S₂₁ vs Frequency", "Element Values"])
    _fig.add_trace(go.Scatter(x=_freqs/1e9, y=_S11_dB, name="|S₁₁| dB",
                              line=dict(color="#EF553B")), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_freqs/1e9, y=_S21_dB, name="|S₂₁| dB",
                              line=dict(color="#00CC96")), row=1, col=1)
    _fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=1)
    _fig.update_yaxes(title_text="dB", row=1, col=1)

    # Element values as a table trace
    _fig.add_trace(go.Table(
        header=dict(values=["Parameter", "Value"], fill_color="#1e1e2e"),
        cells=dict(values=[[r[0] for r in _rows], [r[1] for r in _rows]],
                   fill_color="#2a2a3e")), row=1, col=2)

    _fig.update_layout(template="plotly_dark", height=450,
                       title=f"{_topo} matching: R_S={_Rs}Ω → R_L={_Rl}Ω at {_f0/1e9:.1f} GHz")

    mo.vstack([
        mo.hstack([_topo_m, _Rs_m, _Rl_m, _f0_m, _Q_m]),
        mo.ui.plotly(_fig)
    ])
```

- [ ] **Step 2: Run the three verification commands**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 7: §7 Bode-Fano and §8 Chebyshev synthesis (theory cells)

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §7 Bode-Fano cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ## Part III — Broadband Matching

    ### §7. Bode-Fano Criterion

    For a **parallel-RC load** ($R_L \| C_L$), the reflection coefficient
    of any lossless matching network satisfies:

    $$\int_0^\infty \ln\frac{1}{|\Gamma(\omega)|}\,d\omega \;\leq\; \frac{\pi}{R_L C_L}$$

    For an equal-ripple design that holds $|\Gamma| = |\Gamma_{\max}|$ over bandwidth $\Delta\omega$
    and $|\Gamma| = 1$ outside:

    $$\Delta\omega \cdot \ln\frac{1}{|\Gamma_{\max}|} \;\leq\; \frac{\pi}{R_L C_L}$$

    **Key insight:** the product (bandwidth) × (reflection depth) is bounded by the
    load's time constant $R_L C_L$. To double the bandwidth you must accept
    $|\Gamma_{\max}|$ closer to 1. A higher-order network (larger $N$) fills the
    integral more efficiently but cannot exceed the bound.

    Dual result for a **series-RL load**:
    $$\int_0^\infty \frac{\ln(1/|\Gamma(\omega)|)}{\omega^2}\,d\omega \;\leq\; \frac{\pi L}{R_L}$$
    """)
    return
```

- [ ] **Step 2: Add §8 Chebyshev synthesis cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §8. Chebyshev Matching Network Synthesis

    The low-pass prototype provides normalised element values $g_k$ for
    Butterworth (maximally flat) and Chebyshev (equiripple) responses.

    **Chebyshev element values** ($N$ elements, ripple $\varepsilon$):

    $$g_0 = 1, \quad
      g_k = \frac{4 a_{k-1} a_k}{b_{k-1} g_{k-1}}, \quad
      a_k = \sin\!\left(\frac{(2k-1)\pi}{2N}\right), \quad
      b_k = \gamma^2 + \sin^2\!\left(\frac{k\pi}{N}\right)$$

    where $\gamma = \sinh\!\bigl(\beta/(2N)\bigr)$ and
    $\beta = \ln\!\cot(\varepsilon_{\text{dB}}/17.37)$.

    **Bandpass transformation** scales and frequency-shifts the prototype:

    | Prototype arm | Bandpass realisation |
    |---|---|
    | Series inductor $g_k$ | Series LC: $L = g_k Z_0/\Delta\omega$, $C = \Delta\omega/(\omega_0^2 L)$ |
    | Shunt capacitor $g_k$ | Parallel LC: $C = g_k/(\Delta\omega Z_0)$, $L = \Delta\omega/(\omega_0^2 C)$ |

    where $\Delta\omega = 2\pi \cdot \text{BW}$ and $Z_0 = R_S$.

    **Worked example:** $N=3$, 0.1 dB Chebyshev, $50\,\Omega \to 200\,\Omega$,
    $f_0 = 28\,\text{GHz}$, $\text{BW} = 3\,\text{GHz}$:

    Prototype values: $g_1 \approx 1.032$, $g_2 \approx 1.147$, $g_3 \approx 1.032$,
    $g_4 = 1$.
    Scaled series arm: $L_1 = g_1 \cdot 50 / (2\pi \cdot 3\text{e}9) \approx 2.74\,\text{pH}$.
    """)
    return
```

- [ ] **Step 3: Add §9 real-frequency technique cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §9. Real-Frequency Technique (Overview)

    When the load impedance $Z_L(f)$ is known from VNA measurements
    rather than a simple $R\|C$ model, the Chebyshev prototype is
    no longer exact. Yarman's real-frequency technique solves for the
    lossless two-port $[T]$ that maximises:

    $$\int_{f_1}^{f_2} |S_{21}(f)|^2\,df$$

    subject to $[T]$ being lossless and of prescribed degree $N$.

    The solution is obtained numerically (gradient descent on the
    reflection polynomial coefficients). Practical use: match a
    transistor's measured $Y_{22}^*(f)$ to $50\,\Omega$ over a wide
    IF band without assuming a lumped-element model.

    This technique is beyond the scope of this notebook; the interested
    reader is referred to Yarman, *Design of Ultra Wideband Antenna Matching
    Networks* (Springer, 2008).
    """)
    return
```

- [ ] **Step 4: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 8: Interactive II.1 — Broadband Chebyshev explorer

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add the broadband Chebyshev interactive cell**

```python
@app.cell
def _(mo, np, go):
    import sys as _sys
    _sys.path.insert(0, "marimo/notebooks")
    from _lib_matching import chebyshev_prototype, bode_fano_gamma_max

    _N_bw      = mo.ui.slider(2, 6, value=3, step=1, label="N (order)")
    _ripple_bw = mo.ui.slider(0.01, 3.0, value=0.1, step=0.01, label="Ripple (dB)")
    _Rs_bw     = mo.ui.slider(10, 500, value=50, step=10, label="R_S (Ω)")
    _Rl_bw     = mo.ui.slider(10, 500, value=200, step=10, label="R_L (Ω)")
    _f0_bw     = mo.ui.slider(1e9, 100e9, value=28e9, step=1e9, label="f₀ (Hz)")
    _bw_frac   = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="BW / f₀")

    mo.md("### §10. Interactive II — Broadband Chebyshev Explorer")
    return (_N_bw, _ripple_bw, _Rs_bw, _Rl_bw, _f0_bw, _bw_frac,
            chebyshev_prototype, bode_fano_gamma_max)


@app.cell
def _(_N_bw, _ripple_bw, _Rs_bw, _Rl_bw, _f0_bw, _bw_frac,
      chebyshev_prototype, bode_fano_gamma_max, mo, np, go):
    _N       = int(_N_bw.value)
    _rip     = float(_ripple_bw.value)
    _Rs      = float(_Rs_bw.value)
    _Rl      = float(_Rl_bw.value)
    _f0      = float(_f0_bw.value)
    _BW      = float(_bw_frac.value) * _f0

    # Prototype values
    _g = chebyshev_prototype(_N, _rip)
    _rows_bw = [(f"g_{k}", f"{_g[k]:.4f}") for k in range(_N + 2)]

    # Simple frequency-domain simulation of the prototype (lowpass → bandpass scaling)
    _w0 = 2 * np.pi * _f0
    _dw = 2 * np.pi * _BW
    _freqs_bw = np.linspace(_f0 - 3 * _BW, _f0 + 3 * _BW, 800)
    _w = 2 * np.pi * _freqs_bw
    # Bandpass frequency variable for prototype: Ω = (w/w0 - w0/w) / (dw/w0)
    _Omega = (_w / _w0 - _w0 / _w) / (_dw / _w0)
    # Chebyshev response: |S21|² = 1 / (1 + ε² T_N²(Ω))
    _eps = np.sqrt(10 ** (_rip / 10) - 1)
    _TN = np.cosh(_N * np.arccosh(np.abs(_Omega) + 0j)).real
    _S21_sq = 1.0 / (1.0 + _eps ** 2 * _TN ** 2)
    _S11_sq = np.clip(1.0 - _S21_sq, 0, 1)
    _S21_dB_bw = 10 * np.log10(np.maximum(_S21_sq, 1e-20))
    _S11_dB_bw = 10 * np.log10(np.maximum(_S11_sq, 1e-20))

    # Bode-Fano bound
    _C_load = 1.0 / (_w0 * _Rl)  # estimate C from R_L and f0 (Q=1 assumption)
    _gamma_bf = bode_fano_gamma_max(_BW, _Rl, _C_load)
    _bf_line = 20 * np.log10(max(_gamma_bf, 1e-10))

    _fig_bw = go.Figure()
    _fig_bw.add_trace(go.Scatter(x=_freqs_bw/1e9, y=_S11_dB_bw,
                                  name="|S₁₁| dB", line=dict(color="#EF553B")))
    _fig_bw.add_trace(go.Scatter(x=_freqs_bw/1e9, y=_S21_dB_bw,
                                  name="|S₂₁| dB", line=dict(color="#00CC96")))
    _fig_bw.add_hline(y=_bf_line, line_dash="dash", line_color="#FFA15A",
                       annotation_text=f"Bode-Fano |Γ_max| = {_gamma_bf:.3f}",
                       annotation_position="bottom right")
    _fig_bw.update_layout(template="plotly_dark", height=420,
                           xaxis_title="Frequency (GHz)", yaxis_title="dB",
                           title=f"N={_N} Chebyshev, {_rip} dB ripple, BW={_BW/1e9:.1f} GHz")

    _tbl = go.Figure(go.Table(
        header=dict(values=["Prototype element", "Value"], fill_color="#1e1e2e"),
        cells=dict(values=[[r[0] for r in _rows_bw], [r[1] for r in _rows_bw]],
                   fill_color="#2a2a3e")))
    _tbl.update_layout(template="plotly_dark", height=300)

    mo.vstack([
        mo.hstack([_N_bw, _ripple_bw, _Rs_bw, _Rl_bw, _f0_bw, _bw_frac]),
        mo.ui.plotly(_fig_bw),
        mo.ui.plotly(_tbl),
    ])
```

- [ ] **Step 2: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 9: §10-§12 Distributed matching theory cells

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §10 quarter-wave transformer cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ## Part IV — Distributed and Transformer Matching

    ### §10. Quarter-Wave Transformer

    A transmission line section of length $\ell = \lambda/4$ with
    characteristic impedance $Z_1 = \sqrt{R_S R_L}$ transforms $R_L$ to $R_S$
    at $f_0$. Bandwidth to maximum reflection $|\Gamma_{\max}|$ (Pozar eq. 5.47):

    $$\frac{\Delta f}{f_0} = 2 - \frac{4}{\pi}\arccos\!\left(
      \frac{|\Gamma_{\max}|}{\sqrt{1 - |\Gamma_{\max}|^2}}
      \cdot \frac{2Z_1}{|Z_1^2/R_S - R_S|}\right)$$

    For $R_S = 50\,\Omega$, $R_L = 200\,\Omega$: $Z_1 = 100\,\Omega$,
    bandwidth to $|\Gamma_{\max}| = 0.2$ is approximately 30% of $f_0$.

    **Multi-section Chebyshev transformer:** $N$ cascaded $\lambda/4$ sections
    with impedances $Z_k$ chosen from the Chebyshev synthesis (Pozar §5.7) —
    achieves the Bode-Fano bound for a resistive load, and bandwidth scales as $N^{1/2}$.
    """)
    return
```

- [ ] **Step 2: Add §11 single-stub tuner cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §11. Single-Stub Tuner

    A parallel stub (open- or short-circuit) is placed at distance $d$ from
    the load. The design procedure (Pozar §5.2):

    1. Find $d$ such that $\text{Re}(Y_{\text{in}}(d)) = Y_0$.
    2. Find stub length $\ell$ such that $B_{\text{stub}} = -B_{\text{in}}(d)$.

    Two solutions exist per frequency. Choosing $d$ and $\ell$ as fractions of $\lambda$:

    $$d = \frac{1}{2\pi}\arctan(t), \quad
      t = \frac{B_L \pm \sqrt{G_L(Y_0 - G_L) + B_L^2}}{G_L - Y_0}$$

    For a **short-circuit** stub: $\ell = \frac{1}{2\pi}\arctan(-Y_0/B_{\text{in}})$.
    For an **open-circuit** stub: $\ell = \frac{1}{2\pi}\arctan(-B_{\text{in}}/Y_0)$.

    If $G_L > Y_0$: discriminant negative → no real solution at this frequency
    (the load is in the "forbidden conductance region" for single-stub matching).
    """)
    return
```

- [ ] **Step 3: Add §12 double-stub tuner cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §12. Double-Stub Tuner

    Two stubs at fixed positions $0$ and $d = \lambda/8$ apart (or $\lambda/4$)
    can match any load outside the **forbidden region** $G_L > 2Y_0$ (for $d = \lambda/8$).

    Design steps (Pozar §5.3):

    1. Choose $B_1$ (stub 1 susceptance) such that after transforming
       $Y_L + jB_1$ through length $d$, the conductance $G = Y_0$.
    2. Set $B_2 = -B_{\text{remaining}}$ to cancel residual susceptance.

    The forbidden region shifts with $d$: choosing $d = 3\lambda/16$ avoids
    the original forbidden zone but creates a new one. Double-stub tuners
    appear in bench test setups; single-stub (simpler, manufacturable) is
    preferred on-chip.
    """)
    return
```

- [ ] **Step 4: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 10: §13 On-chip transformer and Interactive III.1

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §13 on-chip transformer cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §13. On-Chip Transformer Coupling

    A coupled-inductor pair with mutual inductance $M = k\sqrt{L_1 L_2}$ and
    turns ratio $n = \sqrt{L_1/L_2}$ transforms impedances by $n^2$:

    $$Z_{\text{in}} = j\omega L_1(1 - k^2) + k^2 n^2 Z_L$$

    In the limit $k \to 1$: $Z_{\text{in}} = n^2 Z_L$ (ideal transformer).

    **Q penalty.** Winding resistance $r_1 = \omega L_1/Q_1$ adds series noise.
    The noise figure penalty (Friis referred back through the network):

    $$\Delta NF \approx 10\log_{10}\!\left(1 + \frac{r_1}{R_S}\right) \quad [\text{dB}]$$

    At 28 GHz in 65 nm CMOS: $Q_{\text{spiral}} \approx 10$–$15$, $k \approx 0.7$–$0.8$.
    For $Q = 12$, $L_1 = 100\,\text{pH}$, $R_S = 50\,\Omega$:
    $r_1 = 2\pi \cdot 28\text{e}9 \cdot 100\text{e-12}/12 \approx 1.5\,\Omega$,
    $\Delta NF \approx 0.13\,\text{dB}$ (referred to input — acceptable).

    **Balun application.** A 1:2 turns-ratio transformer converts single-ended
    to differential, enabling a differential mixer drive from a single-ended LNA.
    $\Delta NF$ rises to 0.5–1 dB including imbalance and layout parasitics.
    """)
    return
```

- [ ] **Step 2: Add Interactive III.1 — Distributed matching explorer (Smith chart)**

```python
@app.cell
def _(mo, np, go):
    import sys as _sys2
    _sys2.path.insert(0, "marimo/notebooks")
    from _lib_matching import single_stub_tuner, qw_transformer

    _GammaL_mag_s = mo.ui.slider(0.0, 0.99, value=0.5, step=0.01, label="|Γ_L|")
    _GammaL_ang_s = mo.ui.slider(-180, 180, value=45, step=5, label="∠Γ_L (°)")
    _Z0_s         = mo.ui.number(value=50.0, label="Z₀ (Ω)")
    _f0_s         = mo.ui.slider(1e9, 100e9, value=28e9, step=1e9, label="f₀ (Hz)")
    _method_s     = mo.ui.dropdown(
        options=["Quarter-wave TL", "Single stub (short)", "Single stub (open)"],
        value="Single stub (short)", label="Method"
    )

    mo.md("### §14. Interactive III — Distributed Matching Explorer")
    return (_GammaL_mag_s, _GammaL_ang_s, _Z0_s, _f0_s, _method_s,
            single_stub_tuner, qw_transformer)


@app.cell
def _(_GammaL_mag_s, _GammaL_ang_s, _Z0_s, _f0_s, _method_s,
      single_stub_tuner, qw_transformer, mo, np, go):
    _Gamma_L_s = _GammaL_mag_s.value * np.exp(1j * np.radians(_GammaL_ang_s.value))
    _Z0 = float(_Z0_s.value)
    _f0 = float(_f0_s.value)
    _Z_L = _Z0 * (1.0 + _Gamma_L_s) / (1.0 - _Gamma_L_s)

    # Smith chart background (unit circle + constant R/G circles)
    _theta_sc = np.linspace(0, 2 * np.pi, 300)
    _unit_x = np.cos(_theta_sc)
    _unit_y = np.sin(_theta_sc)

    _fig_sc = go.Figure()
    _fig_sc.add_trace(go.Scatter(x=_unit_x, y=_unit_y,
                                  mode="lines", line=dict(color="#444", width=1),
                                  showlegend=False))

    # Load point
    _fig_sc.add_trace(go.Scatter(x=[_Gamma_L_s.real], y=[_Gamma_L_s.imag],
                                  mode="markers+text", name="Z_L",
                                  marker=dict(size=12, color="#EF553B"),
                                  text=[f"Z_L={_Z_L.real:.1f}+j{_Z_L.imag:.1f}Ω"],
                                  textposition="top right"))

    _info_lines = [f"Z_L = {_Z_L.real:.2f} + j{_Z_L.imag:.2f} Ω"]

    if "Quarter-wave" in _method_s.value:
        _R_L_eff = abs(_Z_L)
        _d = qw_transformer(_Z0, _R_L_eff)
        _info_lines.append(f"λ/4 TL impedance: Z₁ = {_d['Z1_ohm']:.1f} Ω")
        _info_lines.append(f"Length: λ/4 at f₀ = {3e8/(4*_f0)*1e3:.2f} mm (in air)")
        _fig_sc.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Matched",
                                      marker=dict(size=12, color="#00CC96")))
    else:
        _stype = "short" if "short" in _method_s.value else "open"
        _sols = single_stub_tuner(_Z_L, _Z0, _f0, _stype)
        for _i, _sol in enumerate(_sols):
            _info_lines.append(
                f"Solution {_i+1}: d={_sol['d_lambda']:.3f}λ ({_sol['d_mm']:.2f} mm), "
                f"ℓ={_sol['ell_lambda']:.3f}λ ({_sol['ell_mm']:.2f} mm)"
            )
        _fig_sc.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Target (centre)",
                                      marker=dict(size=12, color="#00CC96")))

    _fig_sc.update_layout(
        template="plotly_dark", height=500, width=520,
        xaxis=dict(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1, title="Re(Γ)"),
        yaxis=dict(range=[-1.2, 1.2], title="Im(Γ)"),
        title="Smith Chart — Distributed Matching"
    )

    mo.vstack([
        mo.hstack([_GammaL_mag_s, _GammaL_ang_s, _Z0_s, _f0_s, _method_s]),
        mo.hstack([
            mo.ui.plotly(_fig_sc),
            mo.md("\n".join(f"- {l}" for l in _info_lines))
        ])
    ])
```

- [ ] **Step 3: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 11: §15 Revisiting the 28 GHz LNA output match

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §15 cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ## Part V — mmWave Output Matching Revisited

    ### §15. The 28 GHz LNA Output Match: Three Topologies

    Notebook 04 §17.6 used a **tapped-capacitor** to transform the optimal load
    impedance at 28 GHz. Here we compare it against two distributed alternatives
    for the same design point ($R_L = 200\,\Omega \to 50\,\Omega$, $f_0 = 28\,\text{GHz}$).

    | Topology | Realisation | BW (−3 dB) | ΔNF | On-chip area |
    |---|---|---|---|---|
    | Tapped capacitor | $C_1$, $C_2$ divider | ~10% (~2.8 GHz) | <0.1 dB | Very small |
    | λ/4 microstrip TL | $Z_1 = 100\,\Omega$, $\ell = 1.34\,\text{mm}$ (air) | ~30% | ~0.3 dB | Large |
    | On-chip transformer | $n=2$, $k=0.75$, $Q=12$ | ~15% | ~0.5–1 dB | Medium; enables differential |

    **ΔNF calculation for the λ/4 TL** (50 Ω microstrip with 0.3 dB/mm loss at 28 GHz;
    ℓ ≈ 0.9 mm in SiO₂ effective medium): insertion loss ≈ 0.27 dB.
    Friis: the output matching loss is divided by LNA gain (15 dB = 32×),
    contributing only $0.27\,\text{dB}/32 \approx 0.008\,\text{dB}$ to input-referred NF —
    negligible. The 0.3 dB figure refers to the degradation of the output signal
    power delivered to the next stage.

    **Practical choice at 28 GHz CMOS:** tapped-capacitor for narrowband;
    transformer coupling when a differential IF mixer is required downstream.
    """)
    return
```

- [ ] **Step 2: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 12: `_lib_cyclo.py` — cyclostationary noise helpers

**Files:**
- Create: `marimo/notebooks/_lib_cyclo.py`

- [ ] **Step 1: Create `_lib_cyclo.py`**

```python
# v1.0
"""Pure-Python helpers for cyclostationary noise analysis.

Used by notebook 05 (Matching Networks and Cyclostationary Noise).
No marimo imports. Covers mixer conversion-matrix NF and kT/C sampler noise.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mixer noise figure (conversion-matrix approximation)
# ---------------------------------------------------------------------------

def mixer_NF_DSB(L_c_dB: float, T0: float = 290.0) -> float:
    """DSB noise figure of an ideal resistive mixer (dB).

    For a purely resistive mixing element (switches between R_on and R_off),
    the DSB NF equals the conversion loss in dB.
    L_c_dB: conversion loss (dB), positive number.
    """
    assert L_c_dB >= 0
    return float(L_c_dB)


def mixer_NF_SSB(L_c_dB: float, IRR_dB: float = 0.0, T0: float = 290.0) -> float:
    """SSB noise figure of a mixer (dB), accounting for image noise.

    IRR_dB: image rejection ratio (dB). 0 dB = no image rejection (single balanced).
    For ideal double-balanced mixer with infinite IRR: NF_SSB = NF_DSB + 3 dB.
    """
    L_c = 10 ** (L_c_dB / 10)
    IRR = 10 ** (IRR_dB / 10)
    F_DSB = L_c  # linear
    # Image noise contribution: power folds in with factor 1/IRR
    F_SSB = F_DSB + (1.0 / IRR) * (T0 / T0)  # simplified: image adds kT_0/IRR
    # For zero IRR (no rejection): F_SSB = 2*F_DSB → NF_SSB = NF_DSB + 3 dB
    return float(10 * np.log10(F_SSB))


def friis_cascade(stages: list[dict]) -> dict:
    """Friis noise cascade for a list of stages.

    Each stage dict: {"F_dB": float, "G_dB": float}.
    Returns: {"F_total_dB": float, "contributions": list[float]}.
    The contribution list shows (F_i - 1)/prod(G_j, j<i) / (F_total - 1) as a fraction.
    """
    Fs = [10 ** (s["F_dB"] / 10) for s in stages]
    Gs = [10 ** (s["G_dB"] / 10) for s in stages]
    F_total = Fs[0]
    G_prod = Gs[0]
    for k in range(1, len(stages)):
        F_total += (Fs[k] - 1.0) / G_prod
        G_prod *= Gs[k]
    contributions = []
    G_prod = 1.0
    for k, (F, G) in enumerate(zip(Fs, Gs)):
        contrib = (F - 1.0) / G_prod / (F_total - 1.0) if F_total > 1.0 else 0.0
        contributions.append(float(contrib))
        G_prod *= G
    return {"F_total_dB": float(10 * np.log10(F_total)),
            "F_total_lin": float(F_total),
            "contributions": contributions}


# ---------------------------------------------------------------------------
# kT/C sampler noise
# ---------------------------------------------------------------------------

def ktc_noise_voltage(C_F: float, T: float = 290.0) -> float:
    """RMS noise voltage on a hold capacitor (V).

    V_n = sqrt(kT / C). Independent of switch resistance.
    """
    k = 1.380649e-23
    return float(np.sqrt(k * T / C_F))


def noise_folding_penalty_dB(N_aliases: int) -> float:
    """Noise figure penalty (dB) from N-fold aliasing in an under-sampled receiver.

    Assumes the noise is white over the full pre-sample bandwidth.
    """
    assert N_aliases >= 1
    return float(10 * np.log10(N_aliases))


def sampler_settling_requirement(f_s: float, N_bits: int) -> float:
    """Maximum RC settling time constant (s) for N-bit accuracy at sample rate f_s.

    Condition: exp(-T_s / (2*tau)) < 2^(-N) → tau < T_s / (2 * N * ln2).
    """
    T_s = 1.0 / f_s
    return float(T_s / (2.0 * N_bits * np.log(2.0)))


# ---------------------------------------------------------------------------
# Cyclostationary PSD helper
# ---------------------------------------------------------------------------

def cyclo_time_avg_psd(S_n: list[float], f_LO: float,
                       freqs: np.ndarray) -> np.ndarray:
    """Time-averaged PSD of a cyclostationary process.

    Given cyclic spectral components S_n (one-sided, n=0,1,...) at
    IF frequencies near f_IF = freqs, the time-averaged PSD is:

        S_avg(f) = Σ_n S_n * δ-like contribution at f ± n*f_LO

    Here we return a broadened representation: each S_n contributes
    a Gaussian of width σ=f_LO/50 centred at f_IF + n*f_LO (n=0 term only
    is at the IF; higher n appear at image-band offsets).

    This is for illustration only — the delta functions are convolved with
    a narrow Gaussian to be plottable.
    """
    psd = np.zeros_like(freqs, dtype=float)
    sigma = f_LO / 50.0
    for n, Sn in enumerate(S_n):
        f_center = n * f_LO
        psd += Sn * np.exp(-0.5 * ((freqs - f_center) / sigma) ** 2)
    return psd
```

- [ ] **Step 2: Verify the library**

```bash
python -c "
from marimo.notebooks._lib_cyclo import (mixer_NF_DSB, mixer_NF_SSB, friis_cascade,
                                          ktc_noise_voltage, noise_folding_penalty_dB,
                                          sampler_settling_requirement)
import numpy as np

# DSB NF equals conversion loss
assert abs(mixer_NF_DSB(6.0) - 6.0) < 1e-9
print('mixer_NF_DSB ok')

# SSB NF >= DSB NF
assert mixer_NF_SSB(6.0, 0.0) >= mixer_NF_DSB(6.0)
print('mixer_NF_SSB ok')

# Friis: two stages
r = friis_cascade([{'F_dB': 3.0, 'G_dB': 15.0}, {'F_dB': 10.0, 'G_dB': 0.0}])
assert r['F_total_dB'] > 3.0
print('friis_cascade ok')

# kT/C
v = ktc_noise_voltage(1e-12)
assert abs(v - np.sqrt(1.380649e-23 * 290 / 1e-12)) < 1e-20
print('ktc_noise_voltage ok')

# Folding
assert abs(noise_folding_penalty_dB(4) - 6.020) < 0.01
print('noise_folding ok')

print('lib_cyclo ok')
"
```

Expected: all lines print, final line `lib_cyclo ok`.

- [ ] **Step 3: Run notebook verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 13: §16-§18 Cyclostationary noise theory cells

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add §16 cyclostationary processes cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ## Part VI — Cyclostationary Noise in Mixers and Samplers

    ### §16. Cyclostationary Processes

    A process $x(t)$ is **wide-sense cyclostationary** with period $T = 1/f_{\text{LO}}$ if:

    $$R_x(t, \tau) \triangleq E[x(t)\,x(t+\tau)] = R_x(t+T, \tau)$$

    Fourier expanding in $t$:

    $$R_x(t, \tau) = \sum_{n=-\infty}^{\infty} R_n(\tau)\,e^{jn\omega_{\text{LO}} t}, \qquad
      R_n(\tau) = \frac{1}{T}\int_0^T R_x(t,\tau)\,e^{-jn\omega_{\text{LO}} t}\,dt$$

    The **cyclic spectral densities** $S_n(f) = \mathcal{F}\{R_n(\tau)\}$ describe
    spectral correlation between components separated by $nf_{\text{LO}}$.
    The $n=0$ term is the ordinary time-averaged PSD.

    **Why stationary analysis fails in a mixer.** Thermal noise at the RF input
    at frequency $f_{\text{RF}}$ and image frequency $f_{\text{RF}} - 2f_{\text{IF}}$
    both mix to the same $f_{\text{IF}}$. The **conversion matrix** $\mathbf{C}$ maps
    input noise spectral vector $\mathbf{a}$ (components at $f \pm nf_{\text{LO}}$)
    to output vector $\mathbf{b} = \mathbf{C}\mathbf{a}$, with
    $C_{mn} = g_{m-n}$ (Fourier coefficient of the time-varying conductance).
    Output noise matrix: $\mathbf{S}_{\text{out}} = \mathbf{C}\,\mathbf{S}_{\text{in}}\,\mathbf{C}^\dagger$.
    """)
    return
```

- [ ] **Step 2: Add §17 mixer noise cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §17. Mixer Noise Theory

    Model the MOSFET switch as a time-varying conductance $g(t)$ driven by a
    square-wave LO: $g(t) = g_{\text{on}}$ for $0 < t < T/2$, $g_{\text{off}}$ otherwise.
    Fourier coefficients:

    $$g_n = \frac{1}{T}\int_0^T g(t)\,e^{-jn\omega_{\text{LO}} t}\,dt = \frac{g_{\text{on}} - g_{\text{off}}}{j\pi n},\quad n \neq 0$$

    Channel thermal noise current PSD: $S_i(t,f) = 4kT\,g(t)$ — cyclostationary.

    **DSB noise figure.** The IF output collects noise from both the signal
    band ($f_{\text{LO}} + f_{\text{IF}}$) and the image band ($f_{\text{LO}} - f_{\text{IF}}$)
    with equal weight. For a resistive switch: $NF_{\text{DSB}} = L_c$ (dB).

    **SSB noise figure.** Only one sideband carries the signal; the image contributes
    noise without signal. For an ideal balanced mixer with **no** image rejection:

    $$NF_{\text{SSB}} = NF_{\text{DSB}} + 10\log_{10}(2) \approx NF_{\text{DSB}} + 3\,\text{dB}$$

    With image rejection ratio $\text{IRR}$ (dB), the image noise contribution is
    suppressed by $10^{-\text{IRR}/10}$, recovering $NF_{\text{SSB}} \to NF_{\text{DSB}}$
    as $\text{IRR} \to \infty$.

    **Friis with mixer** (extending notebook 04 §11):

    $$F_{\text{sys}} = F_{\text{LNA}} + \frac{F_{\text{mixer}} - 1}{G_{A,\text{LNA}}}$$

    For $G_{A,\text{LNA}} = 15\,\text{dB}$ (32×) and $F_{\text{mixer}} = 10$ (10 dB SSB):
    $(10 - 1)/32 = 0.28$ linear → negligible vs. $F_{\text{LNA}} - 1 \approx 0.78$ for 3.5 dB NF.
    """)
    return
```

- [ ] **Step 3: Add §18 sampler noise cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ### §18. Switched-Capacitor (Sampler) Noise

    A track-and-hold samples with switch resistance $R_{\text{sw}}$ and
    hold capacitor $C_h$. During track: RC low-pass with noise bandwidth
    $BW_n = 1/(4R_{\text{sw}} C_h)$. Total integrated noise charge:

    $$\langle v_n^2 \rangle = \int_0^\infty \frac{4kT R_{\text{sw}}}{1 + (\omega R_{\text{sw}} C_h)^2}\,\frac{d\omega}{2\pi} = \frac{kT}{C_h}$$

    The $R_{\text{sw}}$ cancels — the noise power on the capacitor is independent
    of the switch resistance. This is the **kT/C noise** result.

    **Noise folding from aliasing.** If the input noise BW is $B$ and the sample
    rate is $f_s$, then $N = \lceil B/f_s \rceil$ noise images fold into $[0, f_s/2]$.
    The effective noise PSD rises by $10\log_{10}(N)\,\text{dB}$:

    $$\text{Noise folding penalty} = 10\log_{10}(N)\,\text{dB}$$

    **Settling requirement** for $N_b$-bit resolution:
    $\tau = R_{\text{sw}} C_h \ll T_s / (2 N_b \ln 2)$.

    At 28 GHz direct RF sampling ($f_s = 56\,\text{GHz}$, $N_b = 8$):
    $\tau \ll 1.3\,\text{ps}$ → $R_{\text{sw}} C_h \ll 1.3\,\text{ps}$,
    requiring $R_{\text{sw}} < 10\,\Omega$ for $C_h = 100\,\text{fF}$.
    Sub-10 Ω switch on-resistance at 28 GHz is feasible but challenging.
    This motivates heterodyne (down-convert first, then sample at lower IF).
    """)
    return
```

- [ ] **Step 4: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 14: Interactive IV.1 — Mixer noise explorer

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add the mixer noise interactive cell**

```python
@app.cell
def _(mo, np, go):
    import sys as _sys3
    _sys3.path.insert(0, "marimo/notebooks")
    from _lib_cyclo import mixer_NF_DSB, mixer_NF_SSB, friis_cascade, noise_folding_penalty_dB

    _Lc_mx      = mo.ui.slider(3.0, 15.0, value=7.0, step=0.5, label="Conv. Loss (dB)")
    _IRR_mx     = mo.ui.slider(0.0, 60.0, value=20.0, step=1.0, label="Image Reject. (dB)")
    _fLO_mx     = mo.ui.slider(1e9, 100e9, value=28e9, step=1e9, label="f_LO (Hz)")
    _GA_LNA_mx  = mo.ui.slider(0.0, 30.0, value=15.0, step=0.5, label="LNA Gain (dB)")
    _NF_LNA_mx  = mo.ui.slider(0.5, 10.0, value=3.5, step=0.1, label="LNA NF (dB)")
    _N_aliases  = mo.ui.slider(1, 16, value=1, step=1, label="Alias folds N")

    mo.md("### §19. Interactive IV — Mixer Noise Explorer")
    return (_Lc_mx, _IRR_mx, _fLO_mx, _GA_LNA_mx, _NF_LNA_mx, _N_aliases,
            mixer_NF_DSB, mixer_NF_SSB, friis_cascade, noise_folding_penalty_dB)


@app.cell
def _(_Lc_mx, _IRR_mx, _fLO_mx, _GA_LNA_mx, _NF_LNA_mx, _N_aliases,
      mixer_NF_DSB, mixer_NF_SSB, friis_cascade, noise_folding_penalty_dB,
      mo, np, go):
    _Lc    = float(_Lc_mx.value)
    _IRR   = float(_IRR_mx.value)
    _GA    = float(_GA_LNA_mx.value)
    _NF_L  = float(_NF_LNA_mx.value)
    _N_al  = int(_N_aliases.value)

    _NF_DSB_mx = mixer_NF_DSB(_Lc)
    _NF_SSB_mx = mixer_NF_SSB(_Lc, _IRR)
    _fold_dB   = noise_folding_penalty_dB(_N_al)

    _stages = [
        {"F_dB": _NF_L, "G_dB": _GA},
        {"F_dB": _NF_SSB_mx, "G_dB": -_Lc},
    ]
    _res = friis_cascade(_stages)
    _F_total_dB = _res["F_total_dB"]
    _contrib = _res["contributions"]

    _fig_mx = go.Figure()
    _fig_mx.add_trace(go.Bar(
        x=["LNA", "Mixer (SSB)"],
        y=[_contrib[0] * 100, _contrib[1] * 100],
        marker_color=["#636EFA", "#EF553B"],
        text=[f"{c*100:.1f}%" for c in _contrib],
        textposition="outside"
    ))
    _fig_mx.update_layout(
        template="plotly_dark", height=400,
        yaxis_title="Contribution to F_total − 1 (%)",
        title=f"System NF = {_F_total_dB:.2f} dB | DSB NF = {_NF_DSB_mx:.1f} dB | SSB NF = {_NF_SSB_mx:.1f} dB"
    )

    _summary = mo.md(f"""
    | Quantity | Value |
    |---|---|
    | Mixer DSB NF | {_NF_DSB_mx:.2f} dB |
    | Mixer SSB NF | {_NF_SSB_mx:.2f} dB |
    | System NF (LNA + Mixer) | {_F_total_dB:.2f} dB |
    | Noise folding penalty ({_N_al}×) | {_fold_dB:.2f} dB |
    """)

    mo.vstack([
        mo.hstack([_Lc_mx, _IRR_mx, _fLO_mx, _GA_LNA_mx, _NF_LNA_mx, _N_aliases]),
        mo.ui.plotly(_fig_mx),
        _summary
    ])
```

- [ ] **Step 2: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 15: §19 Summary, navigation footer

**Files:**
- Modify: `marimo/notebooks/05_matching_networks.py`

- [ ] **Step 1: Add the §20 summary cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ## Part VII — Summary

    ### §20. Summary and Bridge to Notebook 06

    The complete receiver design chain is now assembled:

    1. **LNA** (notebook 04): place $\Gamma_S$ near $\Gamma_{\text{opt}}$ via inductive
       degeneration → $NF \approx F_{\min}$.
    2. **Input/output matching** (this notebook): Bode-Fano bounds the achievable
       bandwidth; L/π/T or stub/transformer networks realise the target impedance.
    3. **Mixer** (this notebook): SSB $NF_{\text{mixer}} = NF_{\text{DSB}} + 3\,\text{dB}$
       (absent image rejection); Friis shows it is suppressed by LNA gain.
    4. **Sampler** (this notebook): kT/C sets the noise floor; noise folding penalises
       direct-RF architectures.

    **Open problems for notebook 06:**

    - **Nonlinearity:** 1 dB compression point $P_{\text{1dB}}$, input-referred
      third-order intercept $\text{IIP}_3$, AM-PM conversion.
    - **PA design:** load-pull, drain efficiency, Doherty topology.
    - **Full mmWave frontend:** LNA + matching + mixer + PA as an integrated design.

    Concept-dependency map:

    ```
    05 §4–6  L/π/T synthesis  ──►  06 PA output matching
    05 §7–9  Bode-Fano / Chebyshev  ──►  06 broadband PA
    05 §16–18 Cyclostationary  ──►  06 AM-PM / PA nonlinear noise
    ```
    """)
    return
```

- [ ] **Step 2: Add the navigation footer cell**

```python
@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Previous:** [04 — Noise Analysis and LNA Design](04_noise_and_lna_design.py)  |
    **Next:** *06 — Linearity and PA Design (in preparation)*
    """)
    return
```

- [ ] **Step 3: Run verification**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
```

---

## Task 16: Final cross-notebook wiring — update 04's footer

**Files:**
- Modify: `marimo/notebooks/04_noise_and_lna_design.py`

- [ ] **Step 1: Locate the navigation footer cell in notebook 04**

The footer is near line 1432 of `04_noise_and_lna_design.py`. Find the cell containing:
```
**Next:** *05 — Matching Networks (in preparation)*
```

- [ ] **Step 2: Replace the placeholder Next link**

Old string:
```
    **Next:** *05 — Matching Networks (in preparation)*
```

New string:
```
    **Next:** [05 — Matching Networks](05_matching_networks.py)
```

- [ ] **Step 3: Verify notebook 04 still passes**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/04_noise_and_lna_design.py').read())"
uv run marimo check marimo/notebooks/04_noise_and_lna_design.py
uv run python marimo/notebooks/04_noise_and_lna_design.py
```

- [ ] **Step 4: Final verification of notebook 05**

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/05_matching_networks.py').read())"
uv run marimo check marimo/notebooks/05_matching_networks.py
uv run python marimo/notebooks/05_matching_networks.py
python -c "from marimo.notebooks._lib_matching import *; print('lib_matching ok')"
python -c "from marimo.notebooks._lib_cyclo import *;   print('lib_cyclo ok')"
```

All four commands must pass.

---

## Self-review against spec

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| §1 Mismatch loss, power transfer theorem | Task 1 |
| §2 ABCD cascade | Task 2 |
| §3 Smith chart arcs | Task 2 |
| §4 L-network synthesis | Task 3 |
| §5 π-network synthesis | Tasks 4, 5 |
| §6 T-network / tapped-cap | Tasks 4, 5 |
| Interactive I — L/π/T explorer | Task 6 |
| §7 Bode-Fano criterion | Task 7 |
| §8 Chebyshev synthesis | Tasks 7, 8 |
| §9 Real-frequency technique | Task 7 |
| Interactive II — Chebyshev explorer | Task 8 |
| §10 Quarter-wave transformer | Task 9 |
| §11 Single-stub tuner | Tasks 9, 4 (_lib_matching) |
| §12 Double-stub tuner | Tasks 9, 4 (_lib_matching) |
| §13 On-chip transformer | Task 10 |
| Interactive III — Distributed explorer | Task 10 |
| §15 28 GHz output match comparison | Task 11 |
| _lib_cyclo.py helpers | Task 12 |
| §16 Cyclostationary processes | Task 13 |
| §17 Mixer noise (DSB/SSB NF, Friis) | Task 13 |
| §18 Sampler (kT/C, noise folding) | Task 13 |
| Interactive IV — Mixer noise explorer | Task 14 |
| §20 Summary + navigation footer | Task 15 |
| Cross-notebook wiring (04 footer) | Task 16 |

No gaps identified.

**Placeholder scan:** No TBD, no "similar to Task N", no steps without code.

**Type consistency:** `l_network_S`, `pi_network_S`, `t_network_S` defined in Task 4 and called in Task 6. `single_stub_tuner`, `qw_transformer` defined in Task 4 and called in Task 10. `mixer_NF_DSB`, `mixer_NF_SSB`, `friis_cascade`, `noise_folding_penalty_dB` defined in Task 12 and called in Task 14. All consistent.
