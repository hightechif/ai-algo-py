# 03 - K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a "lazy learning" algorithm. It doesn't build a massive internal mathematical model or search for a line of best fit over hundreds of epochs. Instead, it literally memorizes the entire training dataset and makes predictions based purely on proximity.

## Mathematical Core

### 1. The Hypothesis (Distance-Based)

Predicts the class or value of a new point by identifying its $K$ nearest neighbors in the training set and aggregating their labels.

$$
\hat{y} = \text{mode}(y_{neighbors}) \quad \text{(Classification)}
$$

$$
\hat{y} = \frac{1}{K} \sum y_{neighbors} \quad \text{(Regression)}
$$

### 2. The Cost Function (Distance Metric)

Since there is no training phase, there is no traditional cost function. Instead, KNN relies on a distance metric to define proximity. The most common is Euclidean distance:

$$
d(p,q) = \sqrt{\sum (p_i - q_i)^2}
$$

### 3. Optimization (Lazy Learning)

Optimization in KNN is non-parametric. It doesn't involve gradient descent, but rather hyperparameter tuning to find the optimal $K$ and distance metric (e.g., Manhattan vs. Euclidean) to minimize validation error.

## Classification vs Regression

KNN is beautifully versatile. Once it finds the $K$ nearest neighbors, it behaves differently based on the task:

- **Classification:** It takes a **majority vote** (e.g., if 2 neighbors are Square and 1 is Triangle, the point becomes a Square).
- **Regression:** It takes the **mathematical average** of the neighbors' values (e.g., if house prices nearby are 100k, 120k, and 110k, the prediction is exactly 110k).
