# Rules

- Avoid over-engineering. Only make changes that are directly requested or clearly necessary.
- Do not add features, refactor code, or make "improvements" beyond what was asked.
- Do not create helpers, utilities, or abstractions for one-time operations.
- Do not add error handling or validation for scenarios that can't happen.
- Read existing code before suggesting modifications. Do not propose changes to code you haven't read.
- Prefer editing existing files over creating new ones.
- Do not commit or push unless explicitly asked.

## General Python rules

- When use `zip`, always set `strict=True`.

## Python scripting rules

This applies for Python scripts outside of the `lead` package:

- Simple, pipeline-style coding. Assume data is always complete and minimize exceptions handling.
- Assume running script from project root, so just access files accordingly, no complicated path handling.
