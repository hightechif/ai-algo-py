# Design: Decision Tree (Algorithm 06)

## 1. Technical Architecture

The implementation will follow a recursive node-based structure to handle the hierarchical nature of decision trees.

### 1.1 Node Structure
We will define a `Node` dataclass or class to store split information:
- `feature_index`: The index of the feature used for splitting.
- `threshold`: The value used for the binary split.
- `left`: Reference to the left child node.
- `right`: Reference to the right child node.
- `value`: (Leaf nodes only) The predicted class/probability.

### 1.2 Training Logic (fit)
The `fit` method in `scratch.py` will trigger a recursive function `_grow_tree`:
1. **Best Split Selection**: Iterate through every feature and every unique value in the feature to find the split that maximizes **Information Gain** (reduction in Gini Impurity).
2. **Stopping Conditions**: Check if current depth >= `max_depth` or samples < `min_samples_split`.
3. **Recursion**: Split the data and call `_grow_tree` for left and right branches.

## 2. Mathematical Core

### 2.1 Gini Impurity
Used to evaluate the quality of a split:
$$G = 1 - \sum_{i=1}^{c} p_i^2$$

### 2.2 Prediction
Traversal from root to leaf by comparing input features against node thresholds:
```python
if X[node.feature_index] <= node.threshold:
    return predict(node.left)
else:
    return predict(node.right)
```

## 3. Data Strategy
- **Dataset**: `load_breast_cancer` from Scikit-learn.
- **Preprocessing**: Handle continuous features using binary thresholds (e.g., $X_j \le t$).
