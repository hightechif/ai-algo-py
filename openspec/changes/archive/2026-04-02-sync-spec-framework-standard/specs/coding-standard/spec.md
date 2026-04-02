# Delta Spec: Coding Standard — Framework Standard Sync

## MODIFIED Requirements

### Section 2.1 File Structure

Each algorithm module must follow this structure:

- `README.md`: Technical documentation.
- `scratch.py`: Pure NumPy implementation.
- `framework.py`: Framework implementation class (TinyGrad + PyTorch for differentiable algorithms, Scikit-learn for discrete algorithms).
- `tutorial.py`: Demonstration script.

### Section 2.2 Class Naming Convention

To maintain architectural cohesion, a strict naming convention is enforced:

- **Scratch Implementation**: `[AlgorithmName]FromScratch` (e.g., `LinearRegressionFromScratch`).
- **Framework Wrapper**: `[AlgorithmName]Framework` (e.g., `LinearRegressionFramework`).
  - **Differentiable Algorithms**: MUST implement both `fit_tinygrad()` and `fit_pytorch()` methods. Each method must contain an explicit optimization loop following the canonical pattern: `zero_grad → forward → loss → backward → step`.
  - **Discrete Algorithms**: Use `scikit-learn` as a fallback (e.g., KNN, Naive Bayes, Decision Trees).

## ADDED Requirements

### Section 2.4 Framework Implementation Standards

Framework implementations for differentiable algorithms follow a strict **Autograd-First** approach:

#### 2.4.1 Engine Priority

The implementation priority for `framework.py` is:

1. **TinyGrad**: Primary educational autograd engine — makes internal mechanics (loss computation, gradient flow) explicit and transparent.
2. **PyTorch**: Industry-standard benchmark — demonstrates production-grade patterns and hardware acceleration.
3. **Scikit-learn**: Fallback for non-differentiable / discrete algorithms only (e.g., KNN, Naive Bayes, Decision Trees).

#### 2.4.2 Optimization Loop Pattern

Every `fit_tinygrad()` and `fit_pytorch()` method MUST follow this canonical loop structure:

```python
for _ in range(n_iters):
    optimizer.zero_grad()        # 1. Clear gradients
    preds = forward(X)           # 2. Forward pass
    loss = compute_loss(preds)   # 3. Compute loss (must be scalar)
    loss.backward()              # 4. Backpropagate gradients
    optimizer.step()             # 5. Update parameters
```

- The `loss` tensor MUST be a scalar (shape `()`) before calling `.backward()`. Use `.sum()` or `.mean()` to reduce.
- Optimizer choice: `Adam` (recommended for convergence) or `SGD` (for simplicity).

#### 2.4.3 Hardware Acceleration

PyTorch implementations SHOULD detect and utilize available hardware:

- **MPS** (Apple Silicon Mac): `torch.device("mps")` when `torch.backends.mps.is_available()`.
- **CUDA** (NVIDIA GPU): `torch.device("cuda")` when `torch.cuda.is_available()`.
- **CPU**: Default fallback.

#### 2.4.4 Numerical Stability

Framework implementations MUST handle numerical edge cases:

- **Binary Cross-Entropy**: Use the numerically stable formulation `max(x, 0) - x*y + log(1 + exp(-|x|))` instead of naive `y*log(p) + (1-y)*log(1-p)`.
- **Kernel Methods**: Precompute the kernel matrix when using the Kernel Trick to avoid redundant computation during training.
