# Antigravity / AI Agent Guidelines for Microwave Circuit Theory

## CRITICAL DIRECTIVE: NO DELETIONS
**NEVER use `rm` or delete any files from the file system under any circumstances.**
In a past session, an AI agent erroneously deleted the primary pedagogical notebooks and failed to maintain a backup. This resulted in significant data loss of mathematical derivations and interactive figures.

To prevent this:
1. **Never delete a file.** If a file needs to be removed from active use, move it to an `archive/` or `deprecated/` directory instead.
2. **Use Git:** A Git repository is now active. If you must restructure, use `git mv` and commit frequently. 
3. **Commit before major changes:** Before attempting any multi-file refactoring, run `git add . && git commit -m "pre-refactor backup"`.

## Project Context
- **Frameworks:** Python ≥ 3.14, `marimo`, `plotly`.
- **Implementation Guide:** Always refer to `index.md` at the project root for the roadmap of what details and derivations need to be included in each notebook.
- **Tone:** Mathematical rigor, objective, concise, grounded in first principles. Provide symbolic expressions before numerical values.
