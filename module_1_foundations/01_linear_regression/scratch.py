import numpy as np
from typing import Optional

class LinearRegressionFromScratch:
    """
    Linear Regression implemented from scratch using NumPy.
    
    Hypothesis:
    $$ \hat{y} = wX + b $$
    """
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000) -> None:
        self.lr: float = learning_rate
        self.n_iters: int = n_iters
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Initialize parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Gradient Descent loop
        for _ in range(self.n_iters):
            # Calculate predictions: y_hat = wX + b
            y_pred = np.dot(X, self.w) + self.b
            
            # Deriving the Cost Function (Mean Squared Error)
            # Note: A Loss function is for one sample, a Cost function is the average 
            # over the whole dataset. We skip calculating the raw Cost value and jump 
            # straight to computing its partial derivatives (gradients).
            # dw = d(Cost)/dw = (1/N) * X.T * (y_pred - y)
            # db = d(Cost)/db = (1/N) * sum(y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b
