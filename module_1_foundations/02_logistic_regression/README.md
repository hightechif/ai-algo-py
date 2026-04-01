# 02 - Logistic Regression

Logistic regression is the foundational technique for binary classification in supervised machine learning. Despite its name, it is used for classification rather than regression, predicting the probability that an input belongs to a particular class (0 or 1) by mapping linear inputs through a sigmoid function.

## Mathematical Core

1. **The Hypothesis:**

    Our goal is to find a decision boundary for binary classification. We use the Sigmoid function to squash the linear combination into a probability between 0 and 1:

    $$
    \hat{y} = \frac{1}{1 + e^{-(wX + b)}}
    $$

    * $w$ represents the weights
    * $b$ represents the bias

2. **The Cost Function (Cross-Entropy / Log Loss):**

    To measure error in classification across our entire dataset, we compute the average of individual Cross-Entropy losses (penalizing confident but wrong probabilities):

    $$
    J(w,b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
    $$

3. **Optimization (Gradient Descent):**

    Miraculously, the derivative of Log Loss with respect to the weights results in the exact same gradient formula as Linear Regression:

    $$
    dw = \frac{1}{N} \sum (x \times (\hat{y} - y))
    $$

    $$
    w = w - (a \times dw)
    $$

## Framework Implementations
The `framework.py` file provides two differentiable implementations:
- **TinyGrad**: A minimal autograd engine that makes the optimization loop and Sigmoid activation explicit.
- **PyTorch**: The industry-standard deep learning library.
