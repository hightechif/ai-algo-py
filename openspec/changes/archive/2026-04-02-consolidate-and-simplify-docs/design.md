# Design: Consolidate & Simplify Documentation (SSOT Architecture)

## Approach

We will designate **OpenSpec** (at `openspec/specs/`) as the **Single Source of Truth** for all technical and functional requirements. Every other document will then be refactored to serve as a high-level "View" or "Pointer" to those specs, effectively removing redundant technical instructions.

## Changes Plan

### 1. `openspec/specs/algorithm-curriculum/spec.md`
Merge the 32-algorithm "Master Roadmap" from `CURRICULUM.md` into this specification. This establishes a single location for both the curriculum's structure and the individual algorithm requirements.

### 2. `.clinerules`
Strip out the detailed technical requirements (like dataset choices, implementation standards, etc.) and replace them with high-level pointers to the specifications. This ensures the AI agent (Antigravity/Cline) always reads the latest spec before taking action.

### 3. `README.md`
Simplify this for human-level "Marketing."
*   **Remove**: The full algorithm list (point to `algorithm-curriculum/spec.md` instead).
*   **Remove**: Redundant technical standards (point to `coding-standard/spec.md` instead).
*   **Keep**: High-level vision, installation, and architectural "Map" of the project.

### 4. `CURRICULUM.md` & `algo-list.txt`
*   **Action**: Archive `CURRICULUM.md` after its data has been graduated into the spec.
*   **Action**: (If unused by scripts) Delete `algo-list.txt`. If used by shell scripts, replace with a command that extracts its content from the main spec.

## Delta Spec Strategy

A delta spec at `openspec/changes/consolidate-and-simplify-docs/specs/algorithm-curriculum/spec.md` will contain the full 32-algorithm roadmap to be merged into the main spec file.
A separate delta for `coding-standard/spec.md` may be created if general rules are refined.
