# 03 KNN Design

## Mathematical Foundation
KNN is a "lazy learning" algorithm that leverages mathematical distance metrics rather than updating parameters via gradients.
- **Euclidean Distance**: $d(p,q) = \sqrt{\sum (p_i - q_i)^2}$
- **Manhattan Distance**: $d(p,q) = \sum |p_i - q_i|$

## Component Architecture
1. `scratch.py`:
   - `class KNearestNeighborsFromScratch`
   - `__init__(k: int = 3, task: str = 'classification', metric: str = 'euclidean') -> None`
   - `fit(X: np.ndarray, y: np.ndarray) -> None`: Simply stores $X_{train}$ and $y_{train}$ in memory.
   - `predict(X: np.ndarray) -> np.ndarray`: Iterates through query elements, computes distance metrics point-to-point, argsorts distances, and extracts top $K$.
   - **Classification Mode**: Triggers `collections.Counter` to grab the majority vote.
   - **Regression Mode**: Determines the mathematical average (`np.mean`) of the neighbor labels.
2. `framework.py`:
   - Employs `sklearn.neighbors.KNeighborsClassifier` mapped linearly to the exact same classification prediction goal.
