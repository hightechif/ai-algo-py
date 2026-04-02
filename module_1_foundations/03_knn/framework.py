import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from typing import Optional, Union

class KNearestNeighborsFramework:
    """
    Wrapper for Scikit-Learn's KNN implementation to match our curriculum API.
    
    Euclidean Distance:
    $$ d(x, q) = \sqrt{\sum_{i=1}^{n} (x_i - q_i)^2} $$
    """
    def __init__(self, k: int = 3, task: str = 'classification', metric: str = 'minkowski') -> None:
        self.k = k
        self.task = task
        self.metric = metric
        if task == 'classification':
            self.model: Union[KNeighborsClassifier, KNeighborsRegressor] = KNeighborsClassifier(n_neighbors=k, metric=metric)
        elif task == 'regression':
            self.model = KNeighborsRegressor(n_neighbors=k, metric=metric)
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))
