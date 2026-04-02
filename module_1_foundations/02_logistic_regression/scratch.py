import numpy as np
from typing import Optional

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

class LogisticRegressionFromScratch:
    """
    Logistic Regression implemented from scratch using NumPy.
    
    Hypothesis:
    $$ \hat{y} = \sigma(wX + b) $$
    where
    $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
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
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = sigmoid(linear_model)
            
            # Gradients (identical shape to Linear Regression)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = sigmoid(linear_model)
        # Apply 0.5 threshold to return explicit binary classes
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
