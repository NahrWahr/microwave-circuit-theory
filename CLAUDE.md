# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A microwave circuit theory course with two parallel output formats:
- **`marimo/notebooks/`** — interactive marimo notebooks, served as WASM at `public/01/`
- **`manim/1/`** — animated lecture scenes
- **`interactive/`** — standalone apps (separate from the notebooks)

## Commands

```bash
# Lint a notebook's DAG and syntax
uv run marimo check marimo/notebooks/01_two_port_fundamentals.py

# Syntax-only check (catches parse errors before marimo check)
python -c "import ast; ast.parse(open('marimo/notebooks/01_two_port_fundamentals.py').read())"

# Run a notebook non-interactively (smoke-test)
uv run python marimo/notebooks/01_two_port_fundamentals.py
```

## marimo notebook conventions

The global `~/.claude/skills/marimo-notebooks.md` skill has the full reference. Key points specific to this repo:

**Before marking any notebook work done:**
1. `uv run marimo check marimo/notebooks/NN_xxx.py` — only `markdown-indentation` warnings are acceptable; everything else must be fixed.
2. `python -c "import ast; ast.parse(open('marimo/notebooks/NN_xxx.py').read())"` — must pass.

**Plotly / visualization:**
- All figures use `template="plotly_dark"`.
- All figures displayed with `mo.ui.plotly(fig)` (not bare `fig`) to enable zoom/pan.
- Smith chart aspect ratio: `xaxis=dict(scaleanchor="y", scaleratio=1)` is mandatory.

**Name collision rule:** When two interactives in the same notebook compute the same quantity (e.g. `K`, `Delta`), append a context suffix to every output of the conflicting cell (e.g. `K_c`, `Delta_c` for the stability-circle explorer).

**Navigation footer:** Every notebook ends with a `mo.md` cell with Previous / Next links pointing to neighbouring `.py` files. When renaming notebooks, update both neighbours.

**SVG diagrams:** Every `<marker id="...">` must have a unique id suffix per SVG instance (e.g. `arrow1`, `arrow2`) — duplicate ids silently collide when marimo renders them on the same page.

## Architecture note

The `public/01/` directory is the deployed WASM export of the marimo notebooks. Do not edit files there directly — they are generated outputs. The source is always `marimo/notebooks/`.
