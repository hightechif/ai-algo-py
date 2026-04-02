# 05 - Support Vector Machines (SVM)

Support Vector Machines (SVM) is a powerful supervised learning model used for classification and regression. Unlike simple linear models, SVM aims to find the **optimal hyperplane** that maximizes the margin between different classes in a high-dimensional space.

## Mathematical Core

### 1. The Hypothesis (The Kernel Trick)

Instead of operating in the original feature space, we map data into a higher-dimensional space where classes are linearly separable using a **Polynomial Kernel**:

$$
K(x, x') = (x \cdot x' + c)^d
$$

The final decision for an input $x$ is:

$$
\hat{y} = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b\right)
$$

### 2. The Cost Function (Dual Problem)

We solve the dual optimization problem by finding the Lagrange multipliers $\alpha$ that maximize:

$$
W(\alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} y_i y_j \alpha_i \alpha_j K(x_i, x_j)
$$

Subject to $0 \le \alpha_i \le C$ (Soft Margin) and $\sum \alpha_i y_i = 0$.

### 3. Optimization (Sequential Minimal Optimization - SMO)

SMO breaks the complex quadratic programming (QP) problem into the smallest possible sub-problems, each involving only two multipliers $\alpha_i$ and $\alpha_j$. These sub-problems are solved analytically, allowing for efficient training without needing an external QP solver.

## Key Features of This Implementation

- **Sequential Minimal Optimization (SMO)**: A robust implementation of Platt's SMO for the "from-scratch" training of the Dual problem.
- **Polynomial Kernel**: Native support for mapping data to higher dimensions using $K(x, x') = (x \cdot x' + c)^d$.
- **Support Vector Highlighting**: The `tutorial.py` script visualizes and highlights **Support Vectors** ($|\beta_i| > 0$), making the mathematical theory explicit.
- **Soft Margin Balance**: Flexible $C$ parameter implementation to handle noisy data and overlapping boundaries.

## Framework Implementations

The `framework.py` file provides two high-performance differentiable implementations that solve the **Kernelized Primal SVM** using Stochastic Gradient Descent (SGD):

- **TinyGrad**: A minimal autograd engine where the Hinge Loss and kernel-space regularization ($\frac{1}{2}\alpha^T K \alpha$) are implemented from first principles.
- **PyTorch**: Utilizing industry-standard optimization techniques (Adam) and hardware acceleration (MPS/CUDA) to solve the non-linear objective.
