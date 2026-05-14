# Proposal: CART Decision Tree Implementation (Module 2, Algorithm 06)

## Objective
Implement a robust Classification and Regression Tree (CART) from scratch that handles numerical data and demonstrates recursive partitioning, providing a foundation for future ensemble methods like Random Forests and Boosting.

## Problem Statement
Module 2 is currently empty. The Decision Tree is the fundamental building block for all subsequent algorithms in this module. We need a clean, recursive implementation that demonstrates impurity reduction (Gini) and optimal split finding while maintaining the project's strict coding standards.

## Proposed Solution
- **Algorithm**: Implement the **CART** algorithm with recursive binary splitting.
- **Criterion**: Use **Gini Impurity** as the primary metric for classification splits.
- **Stopping Criteria**: Implement `max_depth`, `min_samples_split`, and `min_impurity_decrease` to control tree complexity and prevent overfitting.
- **Framework Integration**: Wrap `scikit-learn`'s `DecisionTreeClassifier` in `framework.py` for performance benchmarking and API consistency.
- **Tutorial**: Use the **Titanic** dataset (or **Breast Cancer** for consistency) to demonstrate the tree's interpretability and decision boundaries.

## Expected Outcomes
- A recursive `scratch.py` implementation with strict static typing and NumPy-based logic.
- A `tutorial.py` showcasing the model's performance vs Scikit-learn.
- Updated `algorithm-curriculum/spec.md` containing formal requirements for Module 2.
