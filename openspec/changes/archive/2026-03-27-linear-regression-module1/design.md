# Design: Linear Regression

## Architecture & Data Flow
The module will utilize the `scikit-learn` built-in `fetch_california_housing` dataset. Specifically, we will extract a single feature (`MedInc`) to map against the target variable (`MedHouseVal`). This isolates the problem as a 2D Simple Linear Regression, allowing for easy visual plotting of the line of best fit.

### Component 1: `scratch.py`
Will contain a `LinearRegressionFromScratch` class consisting of:
- `__init__(self, learning_rate=0.01, n_iters=1000)`: Initialize weights and biases to 0.
- `fit(self, X, y)`:
  - Iterate `n_iters` times.
  - Calculate $\hat{y} = wX + b$.
  - Calculate gradients:
    - $dw = \frac{1}{N} \sum -2x(y - \hat{y})$
    - $db = \frac{1}{N} \sum -2(y - \hat{y})$
  - Update weights: $w_{new} = w - (lr \times dw)$
- `predict(self, X)`: Return $wX + b$.

### Component 2: `framework.py`
Will import `LinearRegression` from `sklearn.linear_model` and provide a clean helper function to rapidly fit and predict using the library defaults.

### Component 3: `tutorial.ipynb`
Will be the storytelling visualization tool utilizing `matplotlib` and `pandas`:
- **Load Data**: Fetch the subset of California Housing.
- **Run Model A**: Plot the dataset and the line of best fit derived from the `scratch.py` module.
- **Run Model B**: Plot the dataset and the line of best fit derived from the `framework.py` module side-by-side to prove mathematical consistency.
