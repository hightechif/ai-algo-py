# Proposal: Update Naive Bayes Dataset

## Context
The current implementation of `module_1_foundations/04_naive_bayes` includes a tutorial Jupyter Notebook (`tutorial.ipynb`) that uses a very small mock array of 10 sentences for text classification (Spam vs. Ham). This resulted in an artificially low accuracy (33%) due to extreme sparsity in the term-frequency matrix during test time and Laplace smoothing domination.

## Objective
Update the `tutorial.ipynb` in `module_1_foundations/04_naive_bayes` to use a robust, publicly available text classification dataset. This will correctly demonstrate the effectiveness of the Naive Bayes algorithm on real-world text classification challenges, pushing the accuracy closer to expected high 90s.

## Scope
- Modify `tutorial.ipynb` within `module_1_foundations/04_naive_bayes/`.
- Replace the mock 10-sentence dataset with a public dataset import.
- Run both the scratch and framework Naive Bayes models against the new dataset.
- Display the updated, realistic accuracy metrics.
- Focus strictly on updating the dataset and generating accurate performance metrics; no refactoring of `scratch.py` or `framework.py` is needed.
