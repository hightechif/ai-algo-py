import numpy as np
from collections import Counter
from typing import Optional

class KNearestNeighborsFromScratch:
    def __init__(self, k: int = 3, task: str = 'classification', metric: str = 'euclidean') -> None:
        self.k: int = k
        self.task: str = task
        self.metric: str = metric
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # KNN is a lazy learner. It simply memorizes the dataset.
        self.X_train = X
        self.y_train = y

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.metric == 'euclidean':
            return float(np.sqrt(np.sum((x1 - x2) ** 2)))
        elif self.metric == 'manhattan':
            return float(np.sum(np.abs(x1 - x2)))
        else:
            raise ValueError("Unsupported distance metric. Use 'euclidean' or 'manhattan'.")

    def _predict_single(self, x: np.ndarray) -> float:
        # Calculate distances from the query x to all training points
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and get indices of the first K neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.task == 'classification':
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            return float(most_common[0][0])
        elif self.task == 'regression':
            # Average value
            return float(np.mean(k_nearest_labels))
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
