# 02. Logistic Regression

## The Hypothesis
Our goal is to find a decision boundary for binary classification. We use the Sigmoid function to squash the linear combination into a probability between 0 and 1:

$$
\hat{y} = \frac{1}{1 + e^{-(wX + b)}}
$$

*   $w$ represents the weights
*   $b$ represents the bias

## The Loss Function (Cross-Entropy / Log Loss)
To measure error in classification, we penalize confident but wrong probabilities using Log Loss:

$$
J(w,b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

## Optimization (Gradient Descent)
Miraculously, the derivative of Log Loss with respect to the weights results in the exact same gradient formula as Linear Regression:

$$
dw = \frac{1}{N} \sum (x \times (\hat{y} - y))
$$

$$
w = w - (\text{learning\\_rate} \times dw)
$$
