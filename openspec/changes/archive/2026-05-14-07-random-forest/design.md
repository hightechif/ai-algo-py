# Design: Random Forest (Algorithm 07)

## 1. Technical Architecture

### 1.1 Decision Tree Modification (Algorithm 06 Update)
To support Random Forest, the underlying `DecisionTree` class in `06_decision_tree/scratch.py` needs to be updated:
- **Parameter**: Add `n_features: Optional[int] = None` to the constructor.
- **Logic Change**: In `_best_split`, if `n_features` is provided, we randomly sample `n_features` column indices without replacement from the total available features. The search for the best split will only iterate over these sampled indices instead of `range(n_features)`.

### 1.2 Random Forest Class
A new `RandomForest` class in `07_random_forest/scratch.py` will act as an orchestrator.
- **Attributes**: 
  - `n_trees`: Number of trees in the forest.
  - `trees`: A list to store the trained `DecisionTree` instances.
  - Standard hyperparameters (`max_depth`, `min_samples_split`, `n_features`).
- **`fit` Method**:
  - Initializes an empty list for `trees`.
  - Loops `n_trees` times.
  - For each iteration, generates a bootstrap sample (randomly selecting `N` indices with replacement, where `N` is the number of rows).
  - Keeps track of OOB (Out-Of-Bag) samples (indices not selected in the bootstrap).
  - Instantiates a `DecisionTree`, fits it on the bootstrap sample, and stores it.
  - Computes the final OOB Score after all trees are trained.
- **`predict` Method**:
  - Gathers predictions from all trees in the forest for the given input `X`.
  - Transposes the prediction matrix to group by sample.
  - Applies a majority vote (`np.bincount(row).argmax()`) for each sample.

## 2. Advanced Feature: OOB Score
The Out-Of-Bag (OOB) score is computed during the `fit` process.
1. For each sample $i$ in the training set, we track which trees did *not* use sample $i$ during training.
2. We ask those specific trees to predict the label for sample $i$.
3. We take a majority vote among these OOB predictions.
4. We compare the final OOB predictions against the true labels to calculate the OOB accuracy.

## 3. Data Strategy
- **Dataset**: `load_breast_cancer` from Scikit-learn.
- **Heuristic**: `n_features` will default to $\lfloor \sqrt{\text{total features}} \rfloor$, which is standard for classification tasks.
