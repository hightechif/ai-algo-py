import numpy as np
import os
import sys
from typing import Optional, List, Tuple

# Import the DecisionTree from Algorithm 06 using importlib to avoid module name conflicts
import importlib.util
dt_scratch_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '06_decision_tree', 'scratch.py')
spec = importlib.util.spec_from_file_location("dt_scratch", dt_scratch_path)
dt_scratch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt_scratch)
DecisionTree = dt_scratch.DecisionTree

class RandomForest:
    """
    Random Forest Classifier implemented from scratch.
    
    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum samples required to split a node.
        n_features (int or float or str): Number of features to consider at each split.
            If 'sqrt', uses sqrt(n_features).
        trees (List[DecisionTree]): List of trained decision trees.
        oob_score_ (float): Out-Of-Bag score.
    """
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        n_features: str = 'sqrt'
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees: List[DecisionTree] = []
        self.oob_score_: Optional[float] = None

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a bootstrap sample of the dataset."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        # Find out-of-bag indices
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        return X[indices], y[indices], oob_indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the random forest and calculate OOB score."""
        self.trees = []
        n_samples, n_total_features = X.shape
        
        # Determine number of features per tree
        num_features_subset = n_total_features
        if self.n_features == 'sqrt':
            num_features_subset = int(np.sqrt(n_total_features))
        elif isinstance(self.n_features, int):
            num_features_subset = self.n_features
        
        # Track OOB predictions for each sample
        # We will store a list of predictions for each sample from trees that didn't train on it
        oob_predictions: List[List[int]] = [[] for _ in range(n_samples)]
        
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=num_features_subset
            )
            X_samp, y_samp, oob_idx = self._bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
            
            # Predict for OOB samples
            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                for i, pred in zip(oob_idx, preds):
                    oob_predictions[i].append(pred)
                    
        # Calculate OOB score
        oob_correct = 0
        oob_total = 0
        for i in range(n_samples):
            if len(oob_predictions[i]) > 0:
                # Majority vote for this sample's OOB predictions
                pred = np.bincount(oob_predictions[i]).argmax()
                if pred == y[i]:
                    oob_correct += 1
                oob_total += 1
                
        if oob_total > 0:
            self.oob_score_ = oob_correct / oob_total
        else:
            self.oob_score_ = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate aggregated predictions."""
        # tree_preds will be of shape (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to shape (n_samples, n_trees)
        tree_preds = tree_preds.T
        
        # Majority vote
        y_pred = [np.bincount(preds).argmax() for preds in tree_preds]
        return np.array(y_pred)
