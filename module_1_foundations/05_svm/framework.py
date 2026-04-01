from sklearn.svm import SVC
import numpy as np

class SVMFramework:
    """
    Wrapper for Scikit-Learn's SVM implementation to match our curriculum API.
    """
    
    def __init__(self, C: float = 1.0, degree: int = 3, coef0: float = 1.0) -> None:
        self.model = SVC(kernel='poly', C=C, degree=degree, coef0=coef0)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
