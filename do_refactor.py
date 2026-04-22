import re

def get_cells(filename):
    with open(filename, 'r') as f:
        content = f.read()
    parts = content.split('@app.cell\n')
    header = parts[0]
    cells = parts[1:]
    cells = [c.rstrip('\n') for c in cells]
    return header, cells

def write_notebook(filename, header, cells):
    with open(filename, 'w') as f:
        f.write(header)
        for cell in cells:
            f.write('@app.cell\n')
            f.write(cell)
            f.write('\n\n\n')

_, cells1 = get_cells('marimo/notebooks/01_two_port_fundamentals.py')
_, cells2 = get_cells('marimo/notebooks/02_power_gain_definitions.py')
_, cells3 = get_cells('marimo/notebooks/03_s_parameters_stability.py')
_, cells4 = get_cells('marimo/notebooks/04_unilateral_power_gain.py')

# === Build Notebook 1 ===
nb1_header = """import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

"""
nb1_cells = []
nb1_cells.extend(cells1[0:11])  # 0 to 10

# N-port and power matrix addition
nport_cell = """def _(mo):
    mo.md(r\"\"\"
    ## 10. $N$-Port Networks and the Matrix Formulation of Power

    While we have extensively discussed two-ports, the concepts easily generalize to $N$-ports.
    An $N$-port network has $N$ voltage/current pairs. These are collected into column vectors:
    $$\mathbf{V} = [V_1, V_2, \dots, V_N]^T, \quad \mathbf{I} = [I_1, I_2, \dots, I_N]^T$$

    The impedance matrix $\mathbf{Z}$ and admittance matrix $\mathbf{Y}$ generalize to $N \\times N$ matrices:
    $$\mathbf{V} = \mathbf{Z}\mathbf{I}, \quad \mathbf{I} = \mathbf{Y}\mathbf{V}$$
    \"\"\")
    return"""
nb1_cells.append(nport_cell)

# Cell 2 from 02 is "Complex Power". We need to modify it to include matrix math.
complex_power_cell = cells2[1]
complex_power_cell = complex_power_cell.replace(
    "Equivalently, using the current phasor $\\tilde{I}$:\n\n    $$S = \\frac{1}{2}Z_L|\\tilde{I}|^2 \\implies P = \\frac{1}{2}R_L|\\tilde{I}|^2$$",
    "Equivalently, using the current phasor $\\tilde{I}$:\n\n    $$S = \\frac{1}{2}Z_L|\\tilde{I}|^2 \\implies P = \\frac{1}{2}R_L|\\tilde{I}|^2$$\n\n    ### 1.4 Algebraic Formulation of Power for $N$-Ports\n\n    Using matrices, the total complex power entering an $N$-port is the sum of the power at each port:\n\n    $$S = \\frac{1}{2} \sum_{i=1}^{N} \\tilde{V}_i \\tilde{I}_i^* = \\frac{1}{2} \\mathbf{I}^H \\mathbf{V} = \\frac{1}{2} \\mathbf{V}^T \\mathbf{I}^*$$\n\n    where $\\mathbf{I}^H = (\\mathbf{I}^*)^T$ is the Hermitian transpose. The real active power is:\n\n    $$P = \\operatorname{Re}(S) = \\frac{1}{2} \\operatorname{Re}(\\mathbf{I}^H \\mathbf{V}) = \\frac{1}{4} (\\mathbf{I}^H \\mathbf{V} + \\mathbf{V}^H \\mathbf{I})$$\n\n    Substituting $\\mathbf{V} = \\mathbf{Z}\\mathbf{I}$:\n\n    $$P = \\frac{1}{4} (\\mathbf{I}^H \\mathbf{Z} \\mathbf{I} + \\mathbf{I}^H \\mathbf{Z}^H \\mathbf{I}) = \\frac{1}{2} \\mathbf{I}^H \\left( \\frac{\\mathbf{Z} + \\mathbf{Z}^H}{2} \\right) \\mathbf{I}$$\n\n    The matrix $\\frac{1}{2}(\\mathbf{Z} + \\mathbf{Z}^H)$ is the **Hermitian part** of $\\mathbf{Z}$. Thus, power dissipation is governed strictly by the Hermitian part of the impedance (or admittance) matrix."
)
nb1_cells.append(complex_power_cell)

# Now add Available Power, Mismatch, Power waves, S-matrix, Conversions
nb1_cells.extend(cells2[2:9])

nb1_summary = """def _(mo):
    mo.md(r\"\"\"
    ## 16. Summary

    We have established the fundamentals of network parameters (Z, Y, H, ABCD) and generalized them to $N$-ports and Scattering parameters (S-matrix). Furthermore, the Hermitian matrix formulation elegantly captures power dissipation across an $N$-port.

    ---

    **Next:** [02 — Power Gains, Signal Flow Graphs, and Mason's Gain](02_power_gain_definitions.py)
    \"\"\")
    return"""
nb1_cells.append(nb1_summary)

write_notebook('marimo/notebooks/01_two_port_fundamentals.py', nb1_header, nb1_cells)

# === Build Notebook 2 ===
nb2_header = """import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

"""
nb2_cells = []
nb2_cells.append(cells2[0])  # imports

# Make a new intro for nb2
nb2_intro = """def _(mo):
    mo.md(r\"\"\"
    # Power Gains, Signal Flow Graphs, and Mason's Gain

    This notebook builds upon the fundamentals of power waves and scattering matrices to define the essential tools for amplifier design. We introduce Signal Flow Graphs (SFG) and Mason's Rule to solve for network reflections, then rigorously define the three canonical power gains ($G, G_A, G_T$). Finally, we explore Mason's Unilateral Gain $U$, an invariant figure of merit whose boundary defines $f_{\\max}$.
    \"\"\")
    return"""
nb2_cells.append(nb2_intro)

# From 02: SFG, Mason's Rule, Deriving Gamma
nb2_cells.extend(cells2[9:12])

# From 03: Gain definitions in Y, Gain in S, Unilateral Factorisation
nb2_cells.extend(cells3[1:4])

# From 04: Mason's U sections 1 to 5
nb2_cells.extend(cells4[1:6])

nb2_summary = """def _(mo):
    mo.md(r\"\"\"
    ---

    **Previous:** [01 — Two-Port Fundamentals](01_two_port_fundamentals.py)

    **Next:** [03 — S-Parameters and Stability Analysis](03_s_parameters_stability.py)
    \"\"\")
    return"""
nb2_cells.append(nb2_summary)

write_notebook('marimo/notebooks/02_power_gain_definitions.py', nb2_header, nb2_cells)


# === Build Notebook 3 ===
nb3_header = """import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

"""
nb3_cells = []
nb3_cells.append(cells3[0])  # imports

nb3_intro = """def _(mo):
    mo.md(r\"\"\"
    # S-Parameters and Stability Analysis

    With all power definitions and Mason's Gain established in Notebook 02, we now focus entirely on **stability**. This notebook covers the criteria for oscillation, derives Rollett's $K$ factor and the $\\mu$-test, plots stability and gain circles, and computes Maximum Available Gain (MAG).
    \"\"\")
    return"""
nb3_cells.append(nb3_intro)

# From 03: Stability sections (cells 4 to 24)
nb3_cells.extend(cells3[4:25])

nb3_summary = """def _(mo):
    mo.md(r\"\"\"
    ---

    **Previous:** [02 — Power Gains, Signal Flow Graphs, and Mason's Gain](02_power_gain_definitions.py)

    **Next:** [04 — (Deleted/Merged)](04_unilateral_power_gain.py)
    \"\"\")
    return"""
nb3_cells.append(nb3_summary)

write_notebook('marimo/notebooks/03_s_parameters_stability.py', nb3_header, nb3_cells)

print("Refactoring complete.")
