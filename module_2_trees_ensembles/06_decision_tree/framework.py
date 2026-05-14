from sklearn.tree import DecisionTreeClassifier
import numpy as np

class DecisionTreeFramework:
    """
    Framework implementation of Decision Tree using Scikit-learn.
    Provides a consistent API for benchmarking.
    """
    def __init__(self, **kwargs):
        """
        Initialize the scikit-learn DecisionTreeClassifier.
        
        Args:
            **kwargs: Arguments passed to DecisionTreeClassifier (e.g., max_depth).
        """
        self.model = DecisionTreeClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples in X."""
        return self.model.predict(X)
