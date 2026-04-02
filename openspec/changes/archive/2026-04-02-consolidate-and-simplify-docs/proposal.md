# Proposal: Consolidate & Simplify Documentation (SSOT Architecture)

## What

Establish a "Single Source of Truth" (SSOT) for the project by consolidating fragmented documentation files into the OpenSpec system. This involves merging overlapping files and replacing technical duplication with pointers to official specifications.

## Why

1. **Avoid Knowledge Drift**: Currently, functional requirements and coding standards are duplicated across `.clinerules`, `CURRICULUM.md`, `README.md`, and the main OpenSpec files. When a standard changes, it must be updated in four places.
2. **Reduce Cognitive Load**: A cleaner root directory makes the project's architecture immediately apparent to new contributors.
3. **Atomic Updates**: Changing a project requirement (like switching a dataset) should be a single change in a single spec, reflecting immediately across the AI agent's behavior and the documentation.
4. **Agent Efficiency**: Pointers to specs in `.clinerules` prevent the agent from using stale instructions if the specs are updated but the rules are forgotten.

## Scope

- **Roadmap Consolidation**: Move the 32-algorithm list from `CURRICULUM.md` into `openspec/specs/algorithm-curriculum/spec.md`.
- **Rule Decoupling**: Update `.clinerules` to serve as a high-level "Pointer" to OpenSpec files instead of containing redundant technical requirements.
- **Root Simplification**: Streamline `README.md` to focus on high-level goals, installation, and architectural links.
- **Redundancy Cleanup**: Either automate or deprecate `algo-list.txt` based on project usage.
