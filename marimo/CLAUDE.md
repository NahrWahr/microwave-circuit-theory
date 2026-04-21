# CLAUDE.md

Guidance for Claude Code when working in this directory.

## Project overview

A series of [marimo](https://marimo.io) notebooks covering microwave circuit theory as a
progressive, self-contained course. Each notebook builds on earlier ones; the trajectory
is monotonic — never bounce back and forth.

The sister directory `../manim/` contains corresponding animated video scenes (Manim
Community edition).

## Commands

Dependencies managed with [uv](https://github.com/astral-sh/uv):

```bash
# Install / sync
uv sync

# Edit a notebook interactively (browser UI)
uv run marimo edit notebooks/03_s_parameters_stability.py

# Run a notebook as a read-only app
uv run marimo run notebooks/03_s_parameters_stability.py

# Lint-check a notebook (surfaces DAG issues, indentation warnings)
uv run marimo check notebooks/*.py
```

## Notebook series

| # | File | Role |
|---|---|---|
| 01 | `01_two_port_fundamentals.py` | Z / Y / ABCD representations, cascading |
| 02 | `02_power_gain_definitions.py` | **Foundations**: complex power, available power, mismatch, power waves, S-matrix, network conversions, SFGs, Mason's rule, Γ_in / Γ_out |
| 03 | `03_s_parameters_stability.py` | **Gains + stability**: G / G_A / G_T (Y and S form), unilateral factorisation + U_m, Rollett K, μ-test, stability circles, MAG / MSG, gain circles, load-pull, unilateral error sweep |
| 04 | `04_unilateral_power_gain.py` | Mason's invariant U via Sylvester / Hermitian criterion, invariance proof, U in S-params, −20 dB/decade law, f_max |
| 05 | `05_mosfet_small_signal.py` | Intrinsic MOSFET Y-parameters, f_T, f_max mapping, GF 22FDX numbers |
| 06 | `06_loadpull_stability.py` | Load-pull and source-pull applied to PA design, combined stability + gain contours |

### Dependency rule

Each notebook may only use concepts established in earlier notebooks. When editing, if
you notice a definition being used before it is introduced, move the derivation earlier
rather than forward-referencing. This is a hard invariant of the series.

## Architecture

### Marimo reactive DAG

Cells are plain Python functions decorated with `@app.cell`. The reactive graph is built
by static analysis of top-level variable names:

- A cell's parameters declare its **dependencies** (names it reads from upstream cells).
- A cell's top-level assignments are its **outputs**, returned as a tuple.
- **Any name assigned at top level is a global across the notebook.** Two cells that
  both assign `K_num` cause a DAG-level collision — `marimo check` will flag it.

### Naming conventions used in this repo

- Cell-local variables (not meant to be shared) are prefixed with `_`, e.g.
  `_theta`, `_fig_tmp`, `_cL`. These are treated as private by marimo and do not
  participate in the reactive graph.
- When two interactives in the same notebook represent the same underlying quantity
  (e.g. an S-parameter slider set), append a context suffix: `S11` for the
  log-magnitude "gain explorer" and `S11_c` for the linear-magnitude "stability
  explorer". Same for `K` / `K_c`, `Delta` / `Delta_c`, etc.

### Markdown + LaTeX

- Theory is written as `mo.md(r"""...""")` raw strings.
- Equations use `$...$` inline and `$$...$$` display. `\boxed{}` and `\begin{aligned}`
  work as in standard LaTeX.
- Inline SVG diagrams (signal-flow graphs, circuits) are embedded directly in markdown
  using `<svg>` with `currentColor` so dark/light themes work.

### Interactive widgets

- `mo.ui.slider`, `mo.ui.radio`, `mo.ui.dropdown` — declared in their own cell, returned
  so downstream cells can read `.value`.
- Layout helpers: `mo.hstack`, `mo.vstack` with `gap="..."`.
- Plotly figures wrapped in `mo.ui.plotly(fig)` for interactive zoom/pan.
- Standard theme: `template="plotly_dark"`.

### Cross-notebook navigation

Each notebook ends with Previous / Next links pointing to adjacent files:

```python
mo.md(r"""
**Previous:** [02 — Power, Waves, and Network Representations](02_power_gain_definitions.py)

**Next:** [04 — Mason's Unilateral Power Gain U](04_unilateral_power_gain.py)
""")
```

When renaming a notebook, update the link text in **both** neighbouring notebooks and
any cross-reference summary tables (e.g. NB06 has one).

## Before declaring a notebook change done

1. `uv run marimo check notebooks/NN_xxx.py` — should only report `markdown-indentation`
   warnings (cosmetic). Any `duplicate-definition` error is a real DAG collision; fix
   by renaming or `_`-prefixing.
2. `python -c "import ast; ast.parse(open('notebooks/NN_xxx.py').read())"` catches
   syntax errors.
3. Grep for stale `notebook 0N §…` cross-references when moving sections between
   notebooks.

## See also

- Project-wide skill: `~/.claude/skills/marimo-notebooks.md` (general marimo patterns,
  interactive figures, inline diagrams).
