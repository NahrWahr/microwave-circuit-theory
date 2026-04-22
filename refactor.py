import re

def get_cells(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by @app.cell
    parts = content.split('@app.cell\n')
    header = parts[0]
    cells = parts[1:]
    
    # Clean up trailing newlines
    cells = [c.rstrip('\n') for c in cells]
    return header, cells

def write_notebook(filename, header, cells):
    with open(filename, 'w') as f:
        f.write(header)
        for cell in cells:
            f.write('@app.cell\n')
            f.write(cell)
            f.write('\n\n\n')

header1, cells1 = get_cells('marimo/notebooks/01_two_port_fundamentals.py')
header2, cells2 = get_cells('marimo/notebooks/02_power_gain_definitions.py')
header3, cells3 = get_cells('marimo/notebooks/03_s_parameters_stability.py')
header4, cells4 = get_cells('marimo/notebooks/04_unilateral_power_gain.py')

# Print lengths to identify them
print(f"Cells in 01: {len(cells1)}")
print(f"Cells in 02: {len(cells2)}")
print(f"Cells in 03: {len(cells3)}")
print(f"Cells in 04: {len(cells4)}")
