# Design: Premium SVM with SMO and Polynomial Kernel

## Architectural Overview
The SVM implementation will reside in `scratch.py` and strictly follow the `.clinerules` for vectorization and typing. The model will solve the **Dual Optimization Problem** of the SVM using the SMO algorithm, allowing for efficient kernel-based classification.

## Key Components

### 1. Sequential Minimal Optimization (SMO)
The SMO algorithm decomposes the large QP problem into smaller ones that can be solved analytically.
- **Goal**: Find Lagrange multipliers $\alpha_i$ that maximize the dual objective function while satisfying constraints:
  - $0 \le \alpha_i \le C$
  - $\sum \alpha_i y_i = 0$
- **Strategy**: At each step, choose two multipliers $\alpha_i$ and $\alpha_j$ to optimize jointly. Keep all other multipliers fixed.
- **Analytical Solution**: An analytical update for $\alpha_j$ followed by a derivation for $\alpha_i$ based on the equality constraint.

### 2. The Kernel Trick (Polynomial Kernel)
Instead of operating in the original feature space, we map data points into a higher-dimensional space where a linear boundary might exist.
- **Kernel Function**: $K(x, x') = (x \cdot x' + c)^d$
  - $d$: degree of the polynomial.
  - $c$: free parameter.
- **Efficiency**: We compute the dot product in the high-dimensional space without ever calculating the mapping $\phi(x)$ itself.

### 3. Soft Margin & Regularization
The parameter $C$ (or `lambda_reg`) will be adjustable to control the trade-off between margin size and misclassification.
- High $C$: Small margin, few misclassifications (Harder margin).
- Low $C$: Large margin, more misclassifications (Softer margin).

### 4. Custom Visualization (tutorial.ipynb)
A specialized plotting function `plot_svm_boundary(model, X, y)` will:
- Generate a meshgrid to visualize the classification regions.
- Use a distinct color for the decision boundary and the margin lines ($\pm 1$).
- Highlight **Support Vectors** (where $\alpha_i > 0$) by drawing them in a separate color and with a larger marker size.

## Data Structures
- `alpha`: NumPy array of shape $(N,)$ representing Lagrange multipliers.
- `b`: Scaler representing the bias.
- `K`: Pre-computed (or lazy-loaded) kernel matrix of shape $(N, N)$.
