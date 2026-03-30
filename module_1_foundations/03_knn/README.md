# 03. K-Nearest Neighbors (KNN)

KNN is a "lazy learning" algorithm. It doesn't build a massive internal mathematical model or search for a line of best fit over hundreds of epochs. Instead, it literally memorizes the entire training dataset and makes predictions based purely on proximity.

## The Mathematical Distance
To find the nearest neighbors, we need a mathematical definition of "close". Depending on the problem, we use different distance formulas:

**Euclidean Distance (Straight Line):**

$$
d(p,q) = \sqrt{\sum (p_i - q_i)^2}
$$

**Manhattan Distance (City Block):**

$$
d(p,q) = \sum |p_i - q_i|
$$

## Classification vs Regression
KNN is beautifully versatile. Once it finds the $K$ nearest neighbors, it behaves differently based on the task:

*   **Classification:** It takes a **majority vote** (e.g., if 2 neighbors are Square and 1 is Triangle, the point becomes a Square).
*   **Regression:** It takes the **mathematical average** of the neighbors' values (e.g., if house prices nearby are 100k, 120k, and 110k, the prediction is exactly 110k).
