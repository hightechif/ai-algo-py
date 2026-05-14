# Implementation Tasks: Random Forest

## Phase 1: Specification & Foundation Update
- [x] Update `openspec/specs/algorithm-curriculum/spec.md` with Algorithm 07 requirements.
- [x] Create `module_2_trees_ensembles/07_random_forest/README.md` explaining Bagging, Feature Randomness, and OOB.
- [x] Modify `06_decision_tree/scratch.py` to accept `n_features` in the constructor.
- [x] Update `_best_split` in `DecisionTree` to sample random feature indices when `n_features` is provided.

## Phase 2: Random Forest Implementation
- [x] Create `RandomForest` class in `07_random_forest/scratch.py`.
- [x] Implement `_bootstrap_sample` logic to select indices with replacement.
- [x] Implement `fit` method to loop over `n_trees`, apply bootstrap, and train individual trees.
- [x] Implement `predict` method to aggregate predictions via majority vote.

## Phase 3: Advanced Features (OOB Score)
- [x] Update `fit` to track Out-Of-Bag samples for each tree.
- [x] Implement OOB prediction aggregation.
- [x] Calculate and expose `self.oob_score_` after training.

## Phase 4: Framework & Evaluation
- [x] Create `07_random_forest/framework.py` wrapping `sklearn.ensemble.RandomForestClassifier`.
- [x] Create `07_random_forest/tutorial.py` using the Breast Cancer dataset.
- [x] Print accuracy metrics (Scratch, Framework, Scratch OOB Score).
- [x] Ensure mathematical alignment and strict typing.
