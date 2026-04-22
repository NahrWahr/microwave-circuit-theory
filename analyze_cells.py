import re

def get_cells(filename):
    with open(filename, 'r') as f:
        content = f.read()
    parts = content.split('@app.cell\n')
    header = parts[0]
    cells = parts[1:]
    cells = [c.rstrip('\n') for c in cells]
    return header, cells

for name in ['01_two_port_fundamentals.py', '02_power_gain_definitions.py', '03_s_parameters_stability.py', '04_unilateral_power_gain.py']:
    print(f"\n--- {name} ---")
    header, cells = get_cells('marimo/notebooks/' + name)
    for i, cell in enumerate(cells):
        first_lines = [l.strip() for l in cell.split('\n') if l.strip() and not l.strip().startswith('def _(')]
        title = first_lines[0] if first_lines else ''
        if 'mo.md' in title:
            match = re.search(r'mo\.md\(r?\"\"\"\n\s*(#.*)', cell)
            if match:
                title = match.group(1)
        print(f"Cell {i}: {title[:60]}")
