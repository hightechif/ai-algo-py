# 01 - Linear Regression

Linear regression is the foundational technique of supervised machine learning. It assumes a linear relationship between the input variables (X) and the single output variable (y).

## Mathematical Core

1. **The Hypothesis:**

    Our goal is to find a line of best fit. The equation for that line is:

    $$
    \hat{y} = wX + b
    $$

    * $w$ represents the weights (the slope)
    * $b$ represents the bias (the intercept)

2. **The Cost Function (Mean Squared Error - MSE):**

    To measure how "wrong" our line is across the entire dataset, we calculate the average of the squared individual losses between our predictions and the actual truth to form the total cost.

    $$
    MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2
    $$

3. **Optimization (Gradient Descent):**

    We take the partial derivative of the MSE function with respect to $w$ and $b$ to find the "slope of the error". Then we step downwards by a small $a$ (learning rate) to minimize the error.

    $$
    dw = \frac{1}{N} \sum (x \times (\hat{y} - y))
    $$

    $$
    w = w - (a \times dw)
    $$

## Framework Implementations
The `framework.py` file provides two differentiable implementations:
- **TinyGrad**: A minimal autograd engine that makes the optimization loop explicit.
- **PyTorch**: The industry-standard deep learning library.
