from sklearn.naive_bayes import MultinomialNB
import numpy as np

class NaiveBayesFramework:
    """
    Wrapper for Scikit-Learn's Multinomial Naive Bayes.
    """
    def __init__(self, alpha: float = 1.0):
        self.model = MultinomialNB(alpha=alpha)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the scikit-learn model.
        """
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions utilizing the scikit-learn model.
        """
        return self.model.predict(X)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns average accuracy.
        """
        return float(self.model.score(X, y))
