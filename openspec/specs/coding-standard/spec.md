# AI Algorithm Coding Standard Specification

## 1. Python Coding Standards

### 1.1 Strict Static Typing

Every Python file MUST use complete and strict static type hinting to ensure full compliance with `mypy`.

- **Modules**: Use `Tuple`, `Optional`, `Union`, `List` from `typing`.
- **Data Shapes**: Explicitly type NumPy arrays as `np.ndarray`.
- **Signatures**: Function arguments and return values must always be hinted.

### 1.2 Environment & Dependencies

- **Target OS**: Mac (Development Environment).
- **Python Version**: 3.10.17.
- **Core Libraries**:
  - `numpy`: For mathematical operations and vectorization.
  - `matplotlib`: For visualization.
  - `tinygrad`: Primary educational autograd engine.
  - `torch` (PyTorch): Industry-standard differentiable benchmark.
  - `scikit-learn`: For dataset loading and discrete algorithm fallbacks.

## 2. Implementation Standards

### 2.1 File Structure

Each algorithm module must follow this structure:

- `README.md`: Technical documentation.
- `scratch.py`: Pure NumPy implementation.
- `framework.py`: Framework implementation class (TinyGrad + PyTorch for differentiable algorithms, Scikit-learn for discrete algorithms).
- `tutorial.py`: Demonstration script.

### 2.2 Class Naming Convention

To maintain architectural cohesion, a strict naming convention is enforced:

- **Scratch Implementation**: `[AlgorithmName]FromScratch` (e.g., `LinearRegressionFromScratch`).
- **Framework Wrapper**: `[AlgorithmName]Framework` (e.g., `LinearRegressionFramework`).
  - **Differentiable Algorithms**: MUST implement both `fit_tinygrad()` and `fit_pytorch()` methods. Each method must contain an explicit optimization loop following the canonical pattern: `zero_grad → forward → loss → backward → step`.
  - **Discrete Algorithms**: Use `scikit-learn` as a fallback (e.g., KNN, Naive Bayes, Decision Trees).

### 2.3 Tutorial Scripts

The `tutorial.py` script replaces the experimental Jupyter notebook format.

- **Requirement**: Use a standard Python script (`.py`) for better version control and scriptability.
- **Content**:
  - **Data Loading**: Prioritize real-world datasets (`load_breast_cancer`, `fetch_20newsgroups`).
  - **Training**: Compare the `FromScratch` results with the `Framework` results.
  - **Visualization**: Use `matplotlib` to generate premium, insightful plots.

### 2.4 Framework Implementation Standards

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

## 3. Documentation Standards

### 3.1 GitHub Markdown Math

Mathematical formulas MUST be formatted for correct rendering in GitHub:

- **Block Equations**: Placed on new lines, strictly sandwiched between dedicated \`$$\` opening and closing lines.
- **Inline Math**: Use single \`\$math\$\` without surrounding spaces.
- **Variable Escaping**: Variables with underscores (e.g., \`\\text{learning\\\\_rate}\`) must be wrapped in a text block with double-escaped underscores.

### 3.2 README Structure

Every module README must include:

1. **Title**: `# XX - Algorithm Name`.
2. **Description**: Brief intuition paragraph.
3. **Mathematical Core**: A numbered list with:
    - **The Hypothesis**: Equation for predictions.
    - **The Cost Function**: Optimization objective.
    - **Optimization Method**: Gradient Descent or specific solver.

### 3.3 Docstring Math

Include the core mathematical formula in the main class docstring using the same LaTeX format as the README.

## 4. Visualization Standards

### 4.1 Mermaid Diagrams

Use Mermaid diagrams for complex logic flows:

- Decision Tree split logic.
- Neural Network layer interactions.
- Optimization algorithm sequences (e.g., SMO).

### 4.2 Visual Consistency

All plots and diagrams MUST utilize a premium, consistent color palette:

- **Primary**: Slate greys and deep blues.
- **Accents**: Subtle neon highlights (orange, lime, or cyan) for key markers like Support Vectors.
