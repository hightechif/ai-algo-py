# Design: Framework Implementation Logic (TinyGrad + PyTorch)

## Module Types for Frameworks:

### Type A: Differentiable Algorithms (Linear, Logistic, SVM, Neural Nets)
- **Hierarchy of Implementations**:
  - `fit_tinygrad(X, y)`: Implemented using `tinygrad.Tensor`. Must explicitly define the loss function and optimization loop.
  - `fit_pytorch(X, y)`: Implemented using `torch.nn.Module` or `torch.Tensor`. Must include a training loop mirroring the TinyGrad version as a benchmark.
- **Shared API**: A class structure named `[AlgorithmName]Framework` where the results are stored after calling the framework-specific `fit` methods.

### Type B: Discrete Algorithms (Trees, Forests, Naive Bayes)
- **Logic**: These stay on `scikit-learn` or other legacy libraries as they are not standard "gradient" optimization problems.

## Structural Consistency:
- Use consistent variable naming across all frameworks to match the mathematical symbols used in the `README.md`.
- Ensure all framework versions produce comparable outputs for cross-validation in `tutorial.py`.

## New Project Standards to Implement:
- Update **`.clinerules`** to mandate "TinyGrad Priority" for framework implementations.
- Update **`openspec/specs/coding-standard/spec.md`** to list `tinygrad` and `torch` as mandatory core library dependencies.
- Fix all Module 01-05 `README.md` and `framework.py` files.
