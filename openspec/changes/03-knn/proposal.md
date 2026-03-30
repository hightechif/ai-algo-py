# 03 KNN Proposal

## Goal
Implement Algorithm 03: K-Nearest Neighbors (KNN) to support both Classification and Regression tasks.

## Scope
- Mathematical documentation of distance functions (Euclidean and Manhattan).
- A pure Python/NumPy implementation (`scratch.py`) capable of switching between classification and regression modes dynamically.
- A scikit-learn framework wrapper (`framework.py`) wrapping `KNeighborsClassifier`.
- A Jupyter Notebook tutorial (`tutorial.ipynb`) evaluating its multi-class classification power using a real-world dataset.

## Dataset
Following our `.clinerules` real-world priority:
- For Classification: The Iris Dataset from `sklearn.datasets` (predicting 3 specific flower classes: setosa, versicolor, virginica).
