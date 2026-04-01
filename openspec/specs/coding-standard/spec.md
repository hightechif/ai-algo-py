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
- `framework.py`: Scikit-learn wrapper class.
- `tutorial.py`: Demonstration script.

### 2.2 Class Naming Convention

To maintain architectural cohesion, a strict naming convention is enforced:

- **Scratch Implementation**: `[AlgorithmName]FromScratch` (e.g., `LinearRegressionFromScratch`).
- **Framework Wrapper**: `[AlgorithmName]Framework` (e.g., `LinearRegressionFramework`).
  - **Differentiable Algorithms**: MUST implement both `fit_tinygrad` and `fit_pytorch`. Must explicitly define the optimization loop (Loss -> Backward -> Step).
  - **Discrete Algorithms**: Use `scikit-learn` as a fallback.

### 2.3 Tutorial Scripts

The `tutorial.py` script replaces the experimental Jupyter notebook format.

- **Requirement**: Use a standard Python script (`.py`) for better version control and scriptability.
- **Content**:
  - **Data Loading**: Prioritize real-world datasets (`load_breast_cancer`, `fetch_20newsgroups`).
  - **Training**: Compare the `FromScratch` results with the `Framework` results.
  - **Visualization**: Use `matplotlib` to generate premium, insightful plots.

## 3. Documentation Standards

### 3.1 GitHub Markdown Math

Mathematical formulas MUST be formatted for correct rendering in GitHub:

- **Block Equations**: Placed on new lines, strictly sandwiched between dedicated \`$$\` opening and closing lines.
- **Inline Math**: Use single \`\$math\$\` without surrounding spaces.
- **Variable Escaping**: Variables with underscores (e.g., \`\\text{learning\\\\_rate}\`) must be wrapped in a text block with double-escaped underscores.

### 3.2 README Structure

Every module README must include:

1.  **Title**: `# XX - Algorithm Name`.
2.  **Description**: Brief intuition paragraph.
3.  **Mathematical Core**: A numbered list with:
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
