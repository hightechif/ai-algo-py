# Tasks: Update Naive Bayes Dataset

## Implementation Steps
- [x] 1. Modify `module_1_foundations/04_naive_bayes/tutorial.ipynb`.
    - Replace the mock data definition blocks with the `fetch_20newsgroups` import logic.
    - Set the categories to `['alt.atheism', 'comp.graphics']` (or similar) and generate `X_train_text`, `X_test_text`, `y_train`, `y_test`.
- [x] 2. Execute the `CountVectorizer` cells within the notebook and confirm the vocabulary scales significantly higher than before.
- [x] 3. Run the Scratch Model and Scikit-Learn evaluation blocks to ensure both implementations return comparable high-accuracy score metrics on this robust dataset.
- [x] 4. Clean up the markdown text in the notebook logically, removing references to "SMS Spam classification" and substituting discussions pertinent to "Categorizing Newsgroups".
