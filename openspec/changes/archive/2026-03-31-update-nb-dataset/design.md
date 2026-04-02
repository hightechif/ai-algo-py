# Design: Update Naive Bayes Dataset

## Approach
To demonstrate Multinomial Naive Bayes text classification correctly without having the user manually download CSV files, we will utilize `scikit-learn`'s built-in dataset fetching utility (`fetch_20newsgroups`). It fetches thousands of text posts out-of-the-box and handles caching automatically.

## Dataset Details
- **Source**: `sklearn.datasets.fetch_20newsgroups`
- **Subsets**: `train` and `test` to maintain the train/test split automatically.
- **Categories**: Rather than classifying all 20 categories, we will filter for two distinct categories (e.g., `['alt.atheism', 'comp.graphics']`) to mimic binary classification performance similarly to the original Spam/Ham scenario and keep the notebook lightweight.

## Implementation Details
1. **Importing**:
   ```python
   from sklearn.datasets import fetch_20newsgroups
   ```
2. **Loading**:
   ```python
   categories = ['alt.atheism', 'comp.graphics']
   train_data = fetch_20newsgroups(subset='train', categories=categories)
   test_data = fetch_20newsgroups(subset='test', categories=categories)
   X_train_text, y_train = train_data.data, train_data.target
   X_test_text, y_test = test_data.data, test_data.target
   ```
3. **Execution**:
   The existing `CountVectorizer` and train/test evaluation steps remain largely unchanged. The pipeline is simply applied to the new robust dataset structure.
