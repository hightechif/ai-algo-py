# Proposal: Random Forest Implementation (Module 2, Algorithm 07)

## Objective
Implement a Random Forest classifier from scratch, building upon our existing Decision Tree implementation. This ensemble method uses Bagging and Feature Randomness to reduce the high variance associated with individual Decision Trees.

## Problem Statement
Algorithm 06 (Decision Tree) is functional but suffers from high variance and tends to overfit. We need to introduce an ensemble approach (Random Forest) that leverages bootstrap aggregation and random feature subsets to improve generalization.

## Proposed Solution
- **Algorithm**: Implement the **Random Forest** algorithm (`07_random_forest/scratch.py`).
- **Core Concept 1: Bagging**: Train `n_trees` trees on bootstrap samples of the training data.
- **Core Concept 2: Feature Randomness**: Update the existing `06_decision_tree/scratch.py` (Option A) to accept an `n_features` parameter. At each split, the tree will only consider a random subset of features (defaulting to $\sqrt{m}$ for classification).
- **Advanced Feature: OOB Score**: Implement Out-Of-Bag error estimation. Since each tree uses a bootstrap sample, it leaves out ~37% of the data. We will use these OOB samples to evaluate the forest without a separate validation set.
- **Framework Integration**: Wrap `scikit-learn`'s `RandomForestClassifier` in `framework.py`.
- **Tutorial**: Use the **Breast Cancer** dataset to compare Scratch vs Framework performance, demonstrating the robustness of the forest and highlighting the OOB score.

## Expected Outcomes
- An updated `06_decision_tree/scratch.py` supporting `n_features`.
- A new `07_random_forest/scratch.py` acting as the ensemble orchestrator.
- A `tutorial.py` benchmarking both models and demonstrating OOB score calculation.
- Updated `algorithm-curriculum/spec.md` for Algorithm 07.
