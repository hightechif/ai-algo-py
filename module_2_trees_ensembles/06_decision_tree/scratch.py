from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

@dataclass
class Node:
    """A node in the decision tree."""
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[Any] = None

class DecisionTree:
    """
    Decision Tree Classifier (CART) implemented from scratch.
    
    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum samples required to split a node.
        min_impurity_decrease (float): Minimum gain required to split a node.
    """
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Optional[Node] = None

    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini Impurity for a label set."""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities**2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[Optional[int], Optional[float]]:
        """Find the optimal feature and threshold to split the data."""
        best_gain = -1.0
        split_idx: Optional[int] = None
        split_thresh: Optional[float] = None
        
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        current_impurity = self._gini(y)

        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                left_indices = np.where(X_column <= threshold)[0]
                right_indices = np.where(X_column > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                n_l, n_r = len(left_indices), len(right_indices)
                gini_l, gini_r = self._gini(y[left_indices]), self._gini(y[right_indices])
                weighted_impurity = (n_l / n_samples) * gini_l + (n_r / n_samples) * gini_r
                
                gain = current_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold

        if best_gain < self.min_impurity_decrease:
            return None, None
            
        return split_idx, split_thresh

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build the decision tree nodes."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Split data and grow children
        left_indices = np.where(X[:, feature_idx] <= threshold)[0]
        right_indices = np.where(X[:, feature_idx] > threshold)[0]

        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=feature_idx, threshold=threshold, left=left, right=right)

    def _most_common_label(self, y: np.ndarray) -> Any:
        """Find the most frequent label in the set."""
        if len(y) == 0:
            return None
        return np.bincount(y).argmax()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build the tree using the training data."""
        self.root = self._grow_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for a set of samples."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Optional[Node]) -> Any:
        """Traverse from root to leaf to find prediction for a single sample."""
        if node is None:
            return None
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
