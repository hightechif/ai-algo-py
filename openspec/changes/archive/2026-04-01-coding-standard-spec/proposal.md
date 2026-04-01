# Proposal: Formalize AI Algorithm Coding Standard

## What:
We propose to create a formal OpenSpec specification file (`spec.md`) that codifies the project's coding standards for AI algorithm implementations. This includes the recently established rules for class naming, file structure, and tutorial script requirements.

## Why:
While `.clinerules` provides instructions for the agent, a formal project specification acts as a "source of truth" within the `openspec` directory. This ensures:
1.  **Durable Documentation**: Future developers (and agents) have a clear reference for "how things are done here".
2.  **Validation**: Clear rules allow for easier auditing of modules (01-05) and future additions.
3.  **Consistency**: Enforces the `SomethingFromScratch` / `SomethingFramework` pattern and the use of `tutorial.py` scripts across the entire curriculum.

## Scope:
- Create `openspec/specs/coding-standard/spec.md`.
- Detail Python typing requirements.
- Formalize README structure and mathematical rendering rules.
- Codify the new implementation standards (naming and `tutorial.py`).
