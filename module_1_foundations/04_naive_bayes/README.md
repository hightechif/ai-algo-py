# 04 - Multinomial Naive Bayes

Naive Bayes is a family of probabilistic algorithms based on applying Bayes' theorem with a "naive" assumption: that every pair of features is conditionally independent given the class variable.

Despite this oversimplified assumption, Naive Bayes classifiers perform extremely well in many complex real-world situations, particularly Natural Language Processing (NLP) tasks such as spam detection and document classification.

## Mathematical Core

### 1. The Hypothesis (Posterior Probability)

Predicts the class $y$ that maximizes the posterior probability $P(y|X)$ using Bayes' Theorem and the independence assumption.

$$
\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$

### 2. The Cost Function (Likelihood)

Instead of an iterative cost function, the model aims to maximize the Joint Probability of features and classes (MLE). For numerical stability, we maximize the **Log-Likelihood**:

$$
\log P(y \mid X) \propto \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y)
$$

### 3. Optimization (Maximum Likelihood Estimation)

Parameters (priors and likelihoods) are estimated directly from feature frequencies in the training set, using Laplace Smoothing ($\alpha$) to handle zero-count features:

$$
P(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}
$$

## Bayes' Theorem

Bayes' theorem defines the probability of an event, based on prior knowledge of conditions that might be related to the event. In classification, we calculate the posterior probability of a class $y$ given the predictor variables (features) $X$:

$$
P(y \mid X) = \frac{P(X \mid y) P(y)}{P(X)}
$$

Where:

- $P(y \mid X)$ is the **posterior probability** of class $y$ given feature matrix $X$.
- $P(y)$ is the **prior probability** of class $y$.
- $P(X \mid y)$ is the **likelihood** which is the probability of the feature vector $X$ given class $y$.
- $P(X)$ is the **marginal probability**. Since it's the same for all classes, we often drop it and use proportionality:

$$
P(y \mid X) \propto P(X \mid y) P(y)
$$

## The "Naive" Assumption

The unique aspect of Naive Bayes is assuming that all features $x_i$ inside $X = (x_1, \dots, x_n)$ are independent given $y$:

$$
P(x_i \mid y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i \mid y)
$$

Which simplifies the overall probability to a massive chain of products:

$$
P(y \mid X) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$
