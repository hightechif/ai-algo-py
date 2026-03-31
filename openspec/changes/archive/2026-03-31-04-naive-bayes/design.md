# Design for 04-naive-bayes

## Architecture
The implementation will follow the standard curriculum pattern:
- `module_1_foundations/04_naive_bayes/`
  - `README.md`: Concept, Naive mapping assumption, Mathematical formulas for Multinomial variants.
  - `scratch.py`: The `MultinomialNaiveBayes` class built upon NumPy.
  - `framework.py`: A thin wrapper utilizing `sklearn.naive_bayes.MultinomialNB`.
  - `tutorial.ipynb`: Jupyter Notebook detailing data ingestion, vectorizing strings, and comparing predictions visually via evaluation arrays or confusion matrices.

## Key Technical Decisions
1. **Bag of Words / Tokenization**: The fundamental issue for Naive Bayes applied to text. We will leverage `sklearn.feature_extraction.text.CountVectorizer` solely in the Notebook so that the `scratch.py` remains a clean representation of the probabilistic model itself, relying on an already numerical array for $X$.
2. **Log-Sum-Exp Trick**: To counteract vanishing floating-point gradients (multiplying small likelihood probabilities iteratively causes Python float underflow), the prediction calculation natively calculates Log-Probabilities which mathematically translate sums back to comparative prediction labels.
3. **Laplace Smoothing Default**: By mathematically guaranteeing $n_c \ge 1$, we stop probability multiplication by 0 when new tokens are found in the unseen validation set.

## `scratch.py` Interface
```python
import numpy as np
from typing import Optional

class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing factor
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculates log prior class probabilities and log likelihoods of the tokens.
        """
        pass
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses log-sum-exp approach for posteriors to classify X vectors.
        """
        pass
```
