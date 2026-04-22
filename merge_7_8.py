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

header, cells = get_cells('marimo/notebooks/01_two_port_fundamentals.py')

# We want to replace cells 2, 3, 4, 5, 6, 7
new_cell_2 = """def _(mo):
    mo.md(r\"\"\"
    ## 7. Interactive Two-Port Explorer

    Select a basic topology and tune the component values to see how the network matrices and frequency response change. The matrices are evaluated at the specific frequency chosen, while the plot shows the broadband behavior.
    \"\"\")
    return"""

new_cell_3 = cells[3] # topology_selector
new_cell_4 = cells[4] # t = topology_selector.value, sliders...

new_cell_5 = """def _(c_slider, freq_slider, l_slider, mo, np, r_slider, t, go, make_subplots):
    f_Hz = freq_slider.value * 1e9
    w = 2 * np.pi * f_Hz

    R_val = float(r_slider.value)
    L_val = float(l_slider.value) * 1e-9
    C_val = float(c_slider.value) * 1e-15

    # Build the 2x2 Z-matrix for each topology
    if t == "series":
        Zs = R_val + 1j * w * L_val
        if C_val > 0:
            Zs = 1.0 / (1.0 / Zs + 1j * w * C_val)
        Z = np.array([[Zs, Zs], [Zs, Zs]])
    elif t == "shunt":
        Yp = 1.0 / R_val
        if C_val > 0:
            Yp += 1j * w * C_val
        if L_val > 0:
            Yp += 1.0 / (1j * w * L_val)
        Y_mat = np.array([[Yp, -Yp], [-Yp, Yp]])
        try:
            Z = np.linalg.inv(Y_mat)
        except np.linalg.LinAlgError:
            Z = np.full((2, 2), np.inf)
    elif t == "tee":
        Z1 = R_val + 1j * w * L_val
        Z2 = Z1  # symmetric T
        Z3 = float(c_slider.value)  # reuse c_slider as shunt resistor
        Z = np.array([[Z1 + Z3, Z3], [Z3, Z2 + Z3]])
    elif t == "pi":
        Y1 = 1.0 / R_val
        Zs = float(l_slider.value)  # series arm in ohms
        Y2 = 1.0 / float(c_slider.value)
        Ys = 1.0 / Zs if Zs != 0 else 1e12
        Y_mat = np.array([[Y1 + Ys, -Ys], [-Ys, Y2 + Ys]])
        try:
            Z = np.linalg.inv(Y_mat)
        except np.linalg.LinAlgError:
            Z = np.full((2, 2), np.inf)
    else:  # lsec
        Zser = R_val + 1j * w * L_val
        Ysh  = 1j * w * C_val
        Zsh = 1.0 / Ysh if abs(Ysh) > 1e-30 else 1e12
        Z = np.array([[Zser + Zsh, Zsh], [Zsh, Zsh]])

    # Derive Y, ABCD
    detZ = Z[0, 0] * Z[1, 1] - Z[0, 1] * Z[1, 0]
    Z11, Z12, Z21, Z22 = Z[0,0], Z[0,1], Z[1,0], Z[1,1]

    if abs(detZ) > 1e-30:
        Y_mat = np.linalg.inv(Z)
        y_valid = True
    else:
        Y_mat = np.full((2, 2), np.nan)
        y_valid = False

    if abs(Z21) > 1e-30:
        A = Z11 / Z21
        B = detZ / Z21
        C_abcd = 1.0 / Z21
        D = Z22 / Z21
        ABCD_mat = np.array([[A, B], [C_abcd, D]])
        abcd_valid = True
    else:
        ABCD_mat = np.full((2, 2), np.nan)
        abcd_valid = False

    def tex_matrix(mat, unit=""):
        if np.any(np.isinf(mat)) or np.any(np.isnan(mat)):
            return r"\\text{Undefined}"
        def fmt(z):
            r, i = np.real(z), np.imag(z)
            if abs(i) < 1e-5:
                return f"{r:.2f}"
            elif abs(r) < 1e-5:
                return f"{i:.2f}j"
            else:
                sign = "+" if i > 0 else "-"
                return f"{r:.2f} {sign} {abs(i):.2f}j"
        return f"\\begin{{bmatrix}} {fmt(mat[0,0])} & {fmt(mat[0,1])} \\\\\\\\ {fmt(mat[1,0])} & {fmt(mat[1,1])} \\end{{bmatrix}} \\,\\text{{{unit}}}"

    matrices_md = f\"\"\"
    ### Network Matrices at {freq_slider.value} GHz

    **Z-parameters:** $\\mathbf{{Z}} = {tex_matrix(Z, '\\Omega')}$
    **Y-parameters:** $\\mathbf{{Y}} = {tex_matrix(Y_mat, 'S') if y_valid else r'\\text{Singular}$'}$
    **ABCD-parameters:** $\\mathbf{{T}} = {tex_matrix(ABCD_mat, '') if abcd_valid else r'\\text{Undefined}$'}$
    \"\"\"

    # Plot frequency response
    freqs = np.logspace(8, 11, 200)
    ws = 2 * np.pi * freqs
    Z11_f = np.zeros_like(freqs, dtype=complex)
    Z21_f = np.zeros_like(freqs, dtype=complex)

    for i, _w in enumerate(ws):
        if t == "series":
            _Zs = R_val + 1j * _w * L_val
            if C_val > 0:
                _Zs = 1.0 / (1.0/_Zs + 1j*_w*C_val)
            Z11_f[i] = _Zs; Z21_f[i] = _Zs
        elif t == "shunt":
            _Yp = 1.0/R_val
            if C_val > 0: _Yp += 1j*_w*C_val
            if L_val > 0: _Yp += 1.0/(1j*_w*L_val)
            Z11_f[i] = 1.0/_Yp if abs(_Yp)>1e-30 else np.nan
            Z21_f[i] = 1.0/_Yp if abs(_Yp)>1e-30 else np.nan
        elif t == "tee":
            _Z1 = R_val + 1j*_w*L_val
            _Z3 = float(c_slider.value)
            Z11_f[i] = _Z1 + _Z3; Z21_f[i] = _Z3
        elif t == "pi":
            _Y1 = 1.0/R_val
            _Zs_pi = float(l_slider.value) if abs(l_slider.value) > 1e-15 else 1e-9
            _Y2 = 1.0/float(c_slider.value) if abs(c_slider.value) > 1e-15 else 0.0
            _Ys = 1.0/_Zs_pi
            _Ym = np.array([[_Y1+_Ys, -_Ys],[-_Ys, _Y2+_Ys]])
            try:
                _Z = np.linalg.inv(_Ym)
                Z11_f[i] = _Z[0,0]; Z21_f[i] = _Z[1,0]
            except Exception:
                Z11_f[i] = np.nan; Z21_f[i] = np.nan
        else:
            _Zser = R_val + 1j*_w*L_val
            _Ysh  = 1j*_w*C_val
            _Zsh  = 1.0/_Ysh if abs(_Ysh) > 1e-30 else 1e12
            Z11_f[i] = _Zser + _Zsh; Z21_f[i] = _Zsh

    fGHz = freqs / 1e9

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fGHz, y=np.abs(Z11_f), name="|Z₁₁| (Input)", line=dict(color="#636EFA", width=2.5)))
    fig.add_trace(go.Scatter(x=fGHz, y=np.abs(Z21_f), name="|Z₂₁| (Transfer)", line=dict(color="#00CC96", width=2.5)))
    
    fig.update_xaxes(type="log", title="Frequency (GHz)")
    fig.update_yaxes(type="log", title="Magnitude (Ω)")
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        title="Impedance Magnitude vs Frequency",
        legend=dict(orientation="h", y=-0.25)
    )

    return mo.vstack([
        mo.md(matrices_md),
        mo.ui.plotly(fig)
    ])"""

# Cells 0 to 1 are the same.
# Then new_cell_2, new_cell_3, new_cell_4, new_cell_5
# Then cells 8 to the end.

# Let's verify what cell 8 is.
# cells[8] should be '## 9. ABCD Parameters and Cascaded Two-Port Stages'
# Let's adjust the numbering in cell 8 to be '8' instead of '9'.
cell_8_mod = cells[8].replace("## 9. ABCD Parameters", "## 8. ABCD Parameters")
cell_10_mod = cells[10].replace("## 10. $N$-Port Networks", "## 9. $N$-Port Networks")

new_cells = [cells[0], cells[1], new_cell_2, new_cell_3, new_cell_4, new_cell_5, cell_8_mod, cells[9], cell_10_mod] + cells[11:]

write_notebook('marimo/notebooks/01_two_port_fundamentals.py', header, new_cells)
print("Merged 7 and 8 successfully.")
