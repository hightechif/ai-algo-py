# Tasks: Premium SVM Implementation (Module 1, Algorithm 05)

## Completed Tasks

- [x] Initial design and proposal.

## Implementation Phase (scratch.py)

- [x] **Define the SVM Class Skeleton**: Create the `SVM` class with strict static typing and a docstring including mathematical formulas.
- [x] **Parameter Logic**: Implement the model's initialization with hyperparameters for $C$, $d$ (kernel degree), and $c$ (kernel constant).
- [x] **Kernel Implementation**: Develop the vectorized `polynomial_kernel(x1, x2)` method.
- [x] **SMO Solver**: Implement the core `fit(X, y)` method using the Sequential Minimal Optimization algorithm:
  - [x] Initialize $\alpha_i = 0$ for all $i$.
  - [x] Implement the outer loop for heuristic selection of $\alpha_i$.
  - [x] Implement the inner loop for selecting and optimizing $\alpha_j$.
  - [x] Update $\alpha_i, \alpha_j$, and the bias $b$ until convergence.
- [x] **Prediction**: Implement the `predict(X)` method using the learned $\alpha_i$ and $b$, mapping new inputs through the polynomial kernel.

## Validation Phase (tutorial.ipynb)

- [x] **Data Prep**: Load a non-linearly separable dataset (e.g., `make_moons` or `make_circles` from `sklearn.datasets`).
- [x] **Training**: Fit the scratch-implemented SVM model to the data.
- [x] **Visualization**: Implement the custom `plot_svm_boundary` function:
  - [x] Generate a decision boundary plot.
  - [x] Highlight **Support Vectors** (where $\alpha_i > 10^{-5}$) with a distinct color (e.g., orange) and larger size.
- [x] **Comparison**: Use `framework.py` to compare results with `sklearn.svm.SVC(kernel='poly')`.

## Documentation Phase

- [x] **Update README**: Refine the `module_1_foundations/05_svm/README.md` to detail:
  - Polynomial Kernel mathematics.
  - Dual problem formulation and SMO overview.
  - Explanation of the $C$ (soft margin) parameter.
