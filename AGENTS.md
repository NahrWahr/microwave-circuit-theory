# Repository Guidelines

## Project Structure & Module Organization

This repository contains microwave circuit theory course material in several output formats:

- `marimo/notebooks/`: source interactive notebooks. Use numbered names such as `01_two_port_fundamentals.py`.
- `public/01/`: generated WASM export of the notebooks. Do not edit this directory directly.
- `manim/1/`: animated lecture scenes.
- `interactive/`: standalone Python apps, separate from marimo notebooks.
- `docs/superpowers/specs/`: design specs and implementation plans.
- `.github/workflows/`: CI and deployment automation.

## Build, Test, and Development Commands

Use `uv` for project commands.

```bash
uv run marimo check marimo/notebooks/01_two_port_fundamentals.py
```

Checks a marimo notebook DAG and syntax. Only `markdown-indentation` warnings are acceptable.

```bash
python -c "import ast; ast.parse(open('marimo/notebooks/01_two_port_fundamentals.py').read())"
```

Runs a syntax-only parse check before heavier notebook validation.

```bash
uv run python marimo/notebooks/01_two_port_fundamentals.py
uv run python interactive/vco_pulling.py
```

Smoke-tests a notebook or launches a standalone app.

## Coding Style & Naming Conventions

Python files use 4-space indentation and descriptive snake_case names. Keep notebook filenames numbered with a two-digit prefix and topic slug, for example `05_matching_networks.py`.

For marimo notebooks, avoid cell output name collisions by adding context suffixes, such as `K_c` or `Delta_c`. Plotly figures should use `template="plotly_dark"` and be displayed with `mo.ui.plotly(fig)`. Smith chart plots must preserve aspect ratio with `xaxis=dict(scaleanchor="y", scaleratio=1)`.

SVG marker IDs must be unique per SVG instance.

## Testing Guidelines

Before marking notebook work complete, run both:

```bash
uv run marimo check marimo/notebooks/NN_topic.py
python -c "import ast; ast.parse(open('marimo/notebooks/NN_topic.py').read())"
```

For standalone apps, run Python syntax checks and any relevant embedded JavaScript checks when applicable. Example:

```bash
python -m py_compile interactive/vco_pulling.py
```

## Commit & Pull Request Guidelines

Recent commits usually use concise conventional prefixes such as `docs:`, `feat:`, `fix:`, `refactor:`, and `chore:`. Prefer imperative, scoped messages, for example `docs: add VCO pulling interactive app design spec`.

Pull requests should describe the change, list validation commands run, and include screenshots or short recordings for visual notebook, Manim, or interactive app changes. Mention affected notebooks and update Previous / Next navigation footers when adding, renaming, or reordering notebooks.

## Agent-Specific Instructions

Do not edit generated `public/01/` assets directly. Change source files under `marimo/notebooks/` and regenerate exports through the project workflow. Preserve unrelated worktree changes, especially in notebooks or specs that are not part of the current task.
