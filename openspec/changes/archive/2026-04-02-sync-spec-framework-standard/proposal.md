# Proposal: Sync Main Spec with Differentiable Framework Standard

## What

Update the main specification at `openspec/specs/coding-standard/spec.md` to accurately reflect the current state of the project. The spec is currently stale — it still describes `framework.py` as a "Scikit-learn wrapper class," but the project has already transitioned to a **TinyGrad + PyTorch** dual-framework standard for all differentiable algorithms (Modules 01, 02, 05).

## Why

The main spec is the **source of truth** for the project's standards. When it drifts from reality, it causes confusion and defeats its purpose. Specifically:

1. **Accuracy**: The spec says `framework.py` is a "Scikit-learn wrapper class," but in practice it now implements `fit_tinygrad()` and `fit_pytorch()` methods with explicit autograd optimization loops.
2. **Completeness**: The spec doesn't document the Kernel Trick pattern used in the SVM framework implementation, nor the hardware acceleration support (MPS/CUDA) in PyTorch implementations.
3. **Alignment**: `.clinerules` already reflects the new standard, but `spec.md` does not. They must be synchronized.

## Scope

- **MODIFY** Section 2.1 (File Structure): Update the `framework.py` description.
- **MODIFY** Section 2.2 (Class Naming): Expand the framework implementation details to cover the autograd-first approach with concrete examples.
- **ADD** Section 2.4 (Framework Implementation Standards): Document the dual-engine pattern, optimizer choices, and hardware acceleration.
- **No code changes** — this is a documentation-only change.
