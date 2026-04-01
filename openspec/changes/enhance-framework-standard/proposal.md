# Proposal: Differentiable Framework Standard (TinyGrad + PyTorch)

## What:
We propose to redefine the core `framework.py` implementation strategy across the entire curriculum. Instead of solely relying on `scikit-learn` for industry solvers, we will prioritize an "Autograd-first" approach using:
1.  **TinyGrad**: As the primary educational framework for white-box autograd logic.
2.  **PyTorch**: As the industry-standard benchmark for differentiable programming.

`scikit-learn` will be relegated to a "Discrete Fallback" role for non-differentiable algorithms (e.g., Decision Trees, Random Forests, Naive Bayes).

## Why:
- **Educational Alignment**: Using TinyGrad and PyTorch forces the learner to explicitly define loss functions and optimization loops, bridging the gap between "Pure NumPy" implementation and high-level solvers.
- **Modern Proficiency**: Tensor-based programming is the dominant paradigm in ML/AI today. Teaching it from Module 1 (Linear Regression) better prepares students for later chapters on Deep Learning (Module 4) and Reinforcement Learning (Module 5).
- **Comparative Insight**: Providing both TinyGrad and PyTorch allows students to compare the internals of a minimal autograd engine (TinyGrad) against a production-grade library (PyTorch).

## Scope:
- **Rule Codification**: Update `.clinerules` and `openspec/specs/coding-standard/spec.md` to reflect this new standard.
- **Documentation**: Add "Autograd Logic" sections to Module READMEs.
- **Module Refactoring**: Synchronize Module 1 (01-05) to this new standard.
