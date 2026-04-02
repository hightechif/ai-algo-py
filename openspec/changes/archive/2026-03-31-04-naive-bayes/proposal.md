# 04-naive-bayes: Multinomial Naive Bayes for SMS Spam Classification

## Problem Statement
Following the algorithm curriculum, the next step in Module 1: The Foundations & Distance (Supervised) is Naive Bayes. While it can be implemented with a Gaussian assumption for continuous problems, we've decided to tackle the Multinomial variant utilizing the SMS Spam Collection dataset to demonstrate the most classic (and practical) use-case for this algorithm: Natural Language Processing (text classification).

## Proposed Solution
We will implement Multinomial Naive Bayes from scratch using NumPy, focusing firmly on calculating priors, likelihoods (word counts with Laplace Smoothing), and utilizing the log-sum-exp trick during probability prediction to avoid underflow. Scikit-learn will be used in the tutorial for vectorization (Bag of Words via `CountVectorizer`) to maintain clarity in the algorithmic mathematics, and we will contrast our class output against `sklearn.naive_bayes.MultinomialNB`.

## Scope
- Implement pure NumPy `scratch.py` class for Multinomial NB.
- Provide theory block math in `README.md`.
- Implement `framework.py` with the Scikit-learn `MultinomialNB` equivalent.
- Create an end-to-end spam classification analysis in `tutorial.ipynb`.

## Success Criteria
- The class efficiently trains utilizing log-probabilities.
- The tutorial provides an explicit demonstration of word conversion (Tokenization).
- Predictions are identical or highly correlated with Scikit-Learn implementations.
