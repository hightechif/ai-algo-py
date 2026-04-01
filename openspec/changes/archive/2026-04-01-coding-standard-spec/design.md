# Design: AI Algorithm Coding Standard Specification

## Structure of the Specification:
The spec will be located at `openspec/specs/coding-standard/spec.md`. It will mirror the rules in `.clinerules` but formatted as a formal project requirement.

### 1. Python Standards
- **Strict Static Typing**: Mandatory use of `typing` module and `np.ndarray` shapes.
- **Dependency Management**: Standardize on `numpy`, `matplotlib`, and `scikit-learn`.

### 2. Implementation Standards
- **Class Naming**:
    - `SomethingFromScratch` for implementations in `scratch.py`.
    - `SomethingFramework` for wrappers in `framework.py`.
- **Tutorial Scripts**:
    - Must be named `tutorial.py` (no `.ipynb`).
    - Must include sections for: Data Loading, Model Training (Scratch vs Framework), and Visualization.

### 3. Documentation Standards
- **README Structure**: Title, Description, Mathematical Core (Hypothesis, Cost, Optimization).
- **LaTeX Rendering**: Strict GitHub Markdown Math formatting (double `$$` block, single `$` inline).
- **Docstring Math**: Mandatory inclusion of core formulas in the main class docstring.

## Integration:
Once the spec is created, it will serve as the reference point for all `/opsx-apply` tasks involving code modification or module creation.
