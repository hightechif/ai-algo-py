# Design: Sync Main Spec with Differentiable Framework Standard

## Approach

This is a pure documentation change. We will edit `openspec/specs/coding-standard/spec.md` directly, guided by a delta spec that declares the exact modifications.

## Changes to Main Spec

### Section 2.1 (File Structure)

**Current** (stale):

> `framework.py`: Scikit-learn wrapper class.

**Updated**:

> `framework.py`: Framework implementation class (TinyGrad + PyTorch for differentiable algorithms, Scikit-learn for discrete algorithms).

### Section 2.2 (Class Naming Convention)

Expand the "Differentiable Algorithms" bullet to include concrete details:

- Must implement `fit_tinygrad()` and `fit_pytorch()` methods.
- Each method must contain an explicit optimization loop: Loss → Backward → Step.
- Use `Adam` or `SGD` optimizers from the respective framework.

### New Section 2.4 (Framework Implementation Standards)

Add a dedicated section documenting:

- **Autograd-First Priority**: TinyGrad (educational) > PyTorch (industry) > Scikit-learn (discrete fallback).
- **Optimization Loop Pattern**: The canonical `zero_grad → forward → loss → backward → step` loop.
- **Hardware Acceleration**: PyTorch implementations should detect and utilize MPS (Mac) or CUDA (NVIDIA) when available.
- **Numerical Stability**: Framework implementations must handle edge cases (e.g., numerically stable BCE loss, scalar tensor requirements for `.backward()`).

## Delta Spec Strategy

A delta spec file at `openspec/changes/sync-spec-framework-standard/specs/coding-standard/spec.md` will declare the MODIFIED requirements using the standard delta spec format. The `/opsx-sync` workflow will then merge these into the main spec.
