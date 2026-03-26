# 02 Logistic Regression Design

## Mathematical Foundation
- **Prediction (Sigmoid)**: $\hat{y} = \frac{1}{1 + e^{-(wX + b)}}$
- **Cost Function (Log Loss)**: $J(w,b) = -\frac{1}{N} \sum [y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$
- **Gradient Computation**: $dw = \frac{1}{N} X^T (\hat{y} - y)$, identically mirroring Linear Regression gradient structures.

## Component Architecture
1. `scratch.py`:
   - `class LogisticRegressionFromScratch`
   - `__init__(learning_rate: float, n_iters: int) -> None`
   - `fit(X: np.ndarray, y: np.ndarray) -> None`: Runs Gradient Descent on binary target probabilities.
   - `predict(X: np.ndarray) -> np.ndarray`: Applies sigmoid, then a 0.5 threshold to return explicit {0, 1} classes.
2. `framework.py`:
   - Utilizes `sklearn.linear_model.LogisticRegression`.
   - Returns both accuracy score and predicted data points, fully typed via `Tuple`.
3. `tutorial.ipynb`:
   - Imports Breast Cancer binary dataset from `sklearn.datasets`.
   - Directly evaluates the cross-entropy models.
