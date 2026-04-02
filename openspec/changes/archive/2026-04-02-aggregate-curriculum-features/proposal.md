# Proposal: Aggregate Curriculum Features into a Central Specification

## What

This change aims to establish a central "Source of Truth" for the AI Algorithm Curriculum by aggregating the requirements and features from all 9 archived changes into a new main specification file: `openspec/specs/algorithm-curriculum/spec.md`. 

While the coding standards are already codified in a separate spec, the **functional requirements** of the algorithms themselves (which datasets to use, which optimization methods are mandatory, and specific architectural choices) currently only exist in the history of archived changes.

## Why

1. **Durable Knowledge**: Implementation details like "using the log-sum-exp trick for Naive Bayes" or "SMO for SVM" should be documented as ongoing requirements, not just one-off tasks.
2. **Standardization**: Ensures that all 32 planned algorithms follow a consistent "Product" vision, not just a consistent "Coding" style.
3. **Auditability**: New modules (Module 2+) can be validated against these synthesized requirements immediately upon proposal.
4. **Onboarding**: Future contributors (and agents) can understand the "Curriculum Features" at a glance without digging through the `archive/` directory.

## Scope

- Analyze the 8 foundational archived changes (Module 1, Algorithms 01-05).
- Synthesize "Capability Requirements" for each algorithm.
- Create a new main spec at `openspec/specs/algorithm-curriculum/spec.md`.
- Map past implementations to the new specification requirements.
