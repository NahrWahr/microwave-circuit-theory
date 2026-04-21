# Microwave Circuit Theory

An interactive microwave / RF circuit theory course built from reactive
[marimo](https://marimo.io) notebooks and companion [Manim](https://www.manim.community/)
video scenes. Each topic is derived from first principles, visualised with adjustable
Smith-chart and frequency-domain plots, and cross-checked across representations.

### Entry Point

The primary entry point for the series is the **[Index Notebook](marimo/notebooks/00_index.py)**, which provides summaries and direct links to each module.

### Notebook series

| # | Topic | Covers |
|---|---|---|
| 01 | Two-port fundamentals | Z, Y, ABCD; cascading; reciprocity |
| 02 | Power, waves, and network representations | Complex and available power, mismatch, power waves, S-matrix, S↔Z↔Y↔ABCD conversions, signal flow graphs, Mason's rule, Γ_in / Γ_out |
| 03 | Power gains and stability | Operating / available / transducer gain in Y and S form, unilateral factorisation, Rollett K, μ-test, stability circles, MAG / MSG, gain circles, load-pull, error sweep |
| 04 | Mason's unilateral power gain U | Sylvester / Hermitian activity criterion, invariance proof, U in S-parameters, −20 dB/decade law, f_max |
| 05 | MOSFET small-signal | Intrinsic Y-parameters, f_T, f_max mapping to GF 22FDX (22 nm FDSOI) |
| 06 | Load-pull and stability in depth | Source-pull, combined stability + gain contours, practical PA design choices |

The dependency chain is strict and monotonic: notebook *N* uses only material introduced
in notebooks *1..N*. Moving between notebooks should never require flipping backward.

## Running the notebooks

Install [uv](https://github.com/astral-sh/uv), then:

```bash
uv sync

# Interactive editor (browser UI with live reactive updates)
uv run marimo edit marimo/notebooks/03_s_parameters_stability.py

# Read-only app mode
uv run marimo run marimo/notebooks/03_s_parameters_stability.py

# Quick static sanity check
uv run marimo check marimo/notebooks/*.py
```

## Manim scenes

The `manim/1/` directory holds narrated animated scenes that parallel the notebooks.
Render with:

```bash
cd manim/1
./render_all.sh
```

## Dependencies

Python ≥ 3.14, managed by `pyproject.toml`:

- `marimo` — reactive notebook runtime
- `plotly` — interactive Smith-chart and Bode plots
- `numpy`, `scipy` — numerics
- `manim` — animated video scenes

## License

Educational use.
