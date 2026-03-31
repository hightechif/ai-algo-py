# 04. Multinomial Naive Bayes

Naive Bayes is a family of probabilistic algorithms based on applying Bayes' theorem with a "naive" assumption: that every pair of features is conditionally independent given the class variable.

Despite this oversimplified assumption, Naive Bayes classifiers perform extremely well in many complex real-world situations, paritcularly Natural Language Processing (NLP) tasks such as spam detection and document classification.

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

## Multinomial Naive Bayes

In the **Multinomial** variant, the features are assumed to be generated from a simple multinomial distribution. It's perfectly suited for discrete counts (e.g., word frequencies in text classification). The likelihood of observing feature $x_i$ belonging to class $y$ is calculated as the total count of $x_i$ occurring in class $y$, divided by the total sum of all feature counts in $y$.

$$
P(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}
$$

Where:
- $N_{yi}$: sum of $x_i$ appearing in class $y$.
- $N_y$: total sum of all features occurring in class $y$.
- $\alpha$: Laplace Smoothing parameter (prevents $0$ probabilities for unseen words).
- $n$: total number of distinct features (vocabulary size).

## The Log-Sum-Exp Solution

Because taking the product of many small probabilities causes floating-point math to underflow towards `0.0`, predicting a class is practically done by analyzing Log-Probabilities. Sums of logs substitute the products:

$$
\hat{y} = \arg\max_{y} \left( \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y) \right)
$$
