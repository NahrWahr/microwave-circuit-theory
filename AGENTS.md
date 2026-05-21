<!-- v1.2 -->
# Repository Guidelines

## Project Structure & Module Organization

This repository contains microwave circuit theory course material in several output formats:

- `marimo/notebooks/`: source interactive notebooks. Use numbered names such as `01_two_port_fundamentals.py`.
- `public/01/`: generated WASM export of the notebooks. Do not edit this directory directly.
- `manim/1/`: animated lecture scenes.
- `interactive/`: standalone Python apps, separate from marimo notebooks.
- `docs/superpowers/specs/`: design specs and implementation plans.
- `.github/workflows/`: CI and deployment automation.

---

## Agent Development Workflow

Agents working in this repository must follow this sequential planning and verification workflow:

### 1. Research & Analysis
- Investigate requirements, mathematical derivations, and existing code structure.
- Do not make source code modifications or execute modifying commands during this stage.

### 2. Implementation Planning
- Create or update the `implementation_plan.md` artifact in the active conversation directory.
- Define the technical approach, files to change, and specific verification steps.
- Set `request_feedback: true` in the artifact metadata.
- Stop and wait for explicit user approval before executing any code changes.

### 3. Execution Tracking
- Create and update the `task.md` artifact to track development tasks:
  - `[ ]` for uncompleted tasks
  - `[/]` for in-progress tasks
  - `[x]` for completed tasks

### 4. Verification
- Verify changes using the repository test commands before concluding.
- Create or update the `walkthrough.md` artifact summarizing modifications, test commands executed, and results.

---

## Build, Test, and Development Commands

Use `uv` for project commands.

### Notebook Check
```bash
uv run marimo check marimo/notebooks/01_two_port_fundamentals.py
```
Checks the marimo notebook DAG and syntax. Only `markdown-indentation` warnings are acceptable.

### Syntax Check
```bash
python -c "import ast; ast.parse(open('marimo/notebooks/01_two_port_fundamentals.py').read())"
```
Runs a syntax-only parse check before heavier notebook validation.

### Execution Test
```bash
uv run python marimo/notebooks/01_two_port_fundamentals.py
uv run python interactive/vco_pulling.py
```
Smoke-tests a notebook or launches a standalone app.

### Standalone App Check
```bash
python -m py_compile interactive/vco_pulling.py
```
Verifies syntax of standalone scripts.

---

## Coding Style & Naming Conventions

- Python files use 4-space indentation and descriptive snake_case names.
- Keep notebook filenames numbered with a two-digit prefix and topic slug, for example `05_matching_networks.py`.
- Avoid cell output name collisions by adding context suffixes, such as `K_c` or `Delta_c`.
- Plotly figures must use `template="plotly_dark"` and be displayed with `mo.ui.plotly(fig)`.
- Smith chart plots must preserve aspect ratio with `xaxis=dict(scaleanchor="y", scaleratio=1)`.
- SVG marker IDs must be unique per SVG instance.

---

## Commit & Pull Request Guidelines

- Commits should use concise conventional prefixes: `docs:`, `feat:`, `fix:`, `refactor:`, and `chore:`. Use imperative, scoped messages (e.g., `docs: add VCO pulling interactive app design spec`).
- Pull requests must describe changes, list validation commands run, and include screenshots or short recordings for visual UI changes.
- Update Previous / Next navigation footers when adding, renaming, or reordering notebooks.

---

## Agent-Specific Constraints

- Do not edit generated `public/01/` assets directly. Change source files under `marimo/notebooks/` and regenerate exports through the project workflow.
- Preserve unrelated worktree changes, especially in notebooks or specs not part of the current task.
