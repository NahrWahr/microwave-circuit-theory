# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A microwave circuit theory course with two parallel output formats:
- **`marimo/notebooks/`** — six interactive marimo notebooks (the primary deliverable), served as WASM at `public/01/`
- **`manim/1/`** — animated lecture scenes rendered into a single video (`microwave_circuit_theory.mp4`)
- **`interactive/`** — standalone PyQt5 desktop apps for Smith chart and gain analysis (separate from the notebooks)

The six notebooks follow a strict dependency chain: `01 → 02 → 03 → 04 → 05`, `04 → 05 → 06`. A concept may only be used in the notebook where it is introduced or in later ones. The authoritative spec for all six notebooks is **`marimo/index.md`** — consult it before adding or restructuring any section.

## Commands

```bash
# Edit a notebook interactively (hot-reloads on file save)
uv run marimo edit marimo/notebooks/01_two_port_fundamentals.py

# Lint a notebook's DAG and syntax
uv run marimo check marimo/notebooks/01_two_port_fundamentals.py

# Syntax-only check (catches parse errors before marimo check)
python -c "import ast; ast.parse(open('marimo/notebooks/01_two_port_fundamentals.py').read())"

# Run a notebook non-interactively (smoke-test)
uv run python marimo/notebooks/01_two_port_fundamentals.py

# Render Manim scenes into the combined video
cd manim/1 && ./render_all.sh          # default 1080p60
cd manim/1 && ./render_all.sh m        # 720p30 (faster dev renders)

# Run a single Manim scene
uv run manim -pqh manim/1/scene_01_transmission_line.py TransmissionLine

# Run a PyQt5 interactive tool
uv run python interactive/gain_analysis.py
uv run python interactive/smith_chart.py
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

## Manim scene conventions

Each scene file exports one `Scene` subclass. Shared drawing helpers (component boxes, ground symbols, arrow markers) are defined at module level and reused across scenes. `render_all.sh` renders every scene listed in its `SCENES` array and concatenates them via `ffmpeg`.

## Architecture note

The `public/01/` directory is the deployed WASM export of the marimo notebooks. Do not edit files there directly — they are generated outputs. The source is always `marimo/notebooks/`.
