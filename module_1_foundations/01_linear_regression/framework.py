import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional

class LinearRegressionFramework:
    """
    Wrapper for Scikit-Learn's Linear Regression to match our curriculum API.
    """
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Returns R^2 score for regression
        return float(self.model.score(X, y))
