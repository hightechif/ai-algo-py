import numpy as np
from typing import Optional

class NaiveBayesFromScratch:
    """
    Multinomial Naive Bayes implementation from scratch using NumPy.
    Utilizes Laplace smoothing and log-probabilities for numeric stability.
    
    Likelihood with Laplace Smoothing:
    $$ P(x_i | y) = \frac{N_{yi} + \alpha}{N_y + \alpha \cdot n_{features}} $$
    
    Log Posterior:
    $$ \log P(y|x) \propto \log P(y) + \sum_{i=1}^{n} x_i \log P(x_i | y) $$
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using X (term frequency matrix) and y (class labels).
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        _, n_features = X.shape
        
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            # Subset X by class
            X_c = X[y == c]
            
            # Prior: P(y) = N_c / N
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])
            
            # Word counts in class c summing across documents
            N_yi = X_c.sum(axis=0)
            N_y = N_yi.sum()
            
            # Likelihood with Laplace Smoothing
            # P(x_i | y) = (N_yi + alpha) / (N_y + alpha * n_features)
            smoothed_prob = (N_yi + self.alpha) / (N_y + self.alpha * n_features)
            self.feature_log_prob_[idx, :] = np.log(smoothed_prob)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for X.
        Uses the log sum approach to prevent underflow.
        """
        # Calculate log posterior for all classes
        # array broadcast sum: log(P(y)) + X @ log(P(x_i|y))^T
        log_posterior = self.class_log_prior_ + X @ self.feature_log_prob_.T
        
        # Get the class with highest posterior probability
        return self.classes_[np.argmax(log_posterior, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns average accuracy.
        """
        return float(np.mean(self.predict(X) == y))
