from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestFramework:
    """
    Framework implementation of Random Forest using Scikit-learn.
    Provides a consistent API for benchmarking against the scratch implementation.
    """
    def __init__(self, **kwargs):
        """
        Initialize the scikit-learn RandomForestClassifier.
        
        Args:
            **kwargs: Arguments passed to RandomForestClassifier (e.g., n_estimators, max_depth).
        """
        # Ensure oob_score is enabled if not explicitly disabled
        if 'oob_score' not in kwargs:
            kwargs['oob_score'] = True
            
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples in X."""
        return self.model.predict(X)

    @property
    def oob_score_(self) -> float:
        """Return the Out-Of-Bag score calculated during training."""
        return getattr(self.model, 'oob_score_', None)
