import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Optional

class LogisticRegressionFramework:
    """
    Wrapper for Scikit-Learn's Logistic Regression to match our curriculum API.
    """
    def __init__(self, max_iter: int = 1000) -> None:
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))
