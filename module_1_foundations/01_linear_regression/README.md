# 01 - Linear Regression

Linear regression is the foundational technique of supervised machine learning. It assumes a linear relationship between the input variables (X) and the single output variable (y). 

## Mathematical Core

1. **The Hypothesis:**
Our goal is to find a line of best fit. The equation for that line is:

$$ 
\hat{y} = wX + b 
$$

*   $w$ represents the weights (the slope)
*   $b$ represents the bias (the intercept)

2. **The Loss Function (Mean Squared Error - MSE):**
To measure how "wrong" our line is, we calculate the average squared difference between our predictions and the actual truth.

$$ 
MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2 
$$


3. **Optimization (Gradient Descent):**
We take the partial derivative of the MSE function with respect to $w$ and $b$ to find the "slope of the error". Then we step downwards by a small `learning_rate` to minimize the error.

$$ 
dw = \frac{1}{N} \sum (x \times (\hat{y} - y)) 
$$

$$ 
w = w - (\mathtt{learning\_rate} \times dw)
$$
