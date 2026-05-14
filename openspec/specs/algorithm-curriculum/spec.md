# AI Algorithm Curriculum Specification

## 1. Project Roadmap

The curriculum is divided into conceptual modules, where each subsequent module generally builds on the mathematics and theories of the prior one.

### 1.1 Module 1: Foundational Algorithms (Supervised)

1. **Linear Regression**: Fitting lines, Mean Squared Error, Gradient Descent.
2. **Logistic Regression**: Classification, Sigmoid function, Cross-Entropy Loss.
3. **K-Nearest Neighbors (KNN)**: Distance metrics (Euclidean, Manhattan).
4. **Naive Bayes**: Conditional probability, Bayes' Theorem.
5. **Support Vector Machine (SVM)**: Hyperplanes, margins, and the kernel trick.

### 1.2 Module 2: Trees & Ensembles (Supervised)

1. **Decision Tree**: Information Gain, Gini impurity, recursive splitting.
2. **Random Forest**: Bagging (Bootstrap Aggregation), feature randomness.
3. **AdaBoost**: Adaptive boosting, sample weighting.
4. **Gradient Boosting**: Minimizing residual errors sequentially.
5. **XGBoost**: Advanced boosting, regularization, tree pruning.
6. **Isolation Forest**: Utilizing random splits for anomaly detection.

### 1.3 Module 3: Finding Structure (Unsupervised)

1. **Principal Component Analysis (PCA)**: Eigenvectors, Eigenvalues, Dimensionality Reduction.
2. **t-SNE**: Non-linear manifold learning for visualization.
3. **k-Means Clustering**: Centroids, iterative distance minimization.
4. **k-Means++**: Optimized centroid initialization.
5. **Hierarchical Clustering**: Agglomerative methods, Dendrograms.
6. **DBSCAN**: Density-based spatial clustering.
7. **Gaussian Mixture Models (GMM)**: Expectation-Maximization (EM) algorithm.

### 1.4 Module 4: Deep Learning Foundations (Neural Networks)

1. **Artificial Neural Network (ANN)**: Perceptrons, Backpropagation, Activation Functions.
2. **Autoencoders**: Unsupervised representation learning, bottling data.
3. **Convolutional Neural Network (CNN)**: Kernels, filters, max-pooling for spatial data.
4. **Recurrent Neural Network (RNN)**: Handling sequence and time-series data.
5. **Long Short-Term Memory (LSTM)**: Gating mechanisms, solving vanishing gradients.
6. **Generative Adversarial Networks (GANs)**: Generator vs Discriminator architecture.
7. **Transformer**: Self-attention mechanisms, positional encoding.

### 1.5 Module 5: Learning via Reward (Reinforcement Learning)

1. **Markov Decision Process (MDP)**: States, actions, transition models, and rewards.
2. **Q-Learning**: Value-based iteration, temporal difference learning.
3. **SARSA**: On-policy learning.
4. **Deep Q-Network (DQN)**: Melding Neural Nets with Q-Learning.
5. **Policy Gradient**: Optimizing the policy directly.
6. **Actor-Critic**: Combining Policy-based and Value-based approaches.

### 1.6 Module 6: Evolutionary Computing

1. **Genetic Algorithm**: Chromosomes, crossover, mutation, fitness functions.

## 2. General Principles

### 2.1 Implementation Requirements

Every algorithm module in the curriculum MUST satisfy these core principles:

- **From Scratch vs Framework**: Every algorithm MUST provide a from-scratch implementation (NumPy) and a framework implementation (TinyGrad/PyTorch/Scikit-Learn).
- **Real-World Priority**: Tutorials MUST prioritize real-world datasets over synthetic generators.
- **Mathematical Transparency**: Implementation logic MUST be traceably aligned with the mathematical formulas in the module's `README.md`.

## 3. Module Functional Requirements

This section defines the specific functional requirements for completed algorithms.

### 3.1 Module 1: Foundational Algorithms

#### 3.1.1 Algorithm 01: Linear Regression

- **Requirement**: SHALL provide a from-scratch implementation using NumPy vectorization for gradient descent.
- **Requirement**: SHALL minimize the Mean Squared Error (MSE) cost function.
- **Requirement**: SHALL compare performance against an industry-standard framework.

#### Scenario: Real-world Dataset

- **WHEN** evaluating the model during a tutorial.
- **THEN** it MUST use the `California Housing` dataset.
- **AND** it MUST compare From-Scratch vs Framework results.

#### 3.1.2 Algorithm 02: Logistic Regression

- **Requirement**: SHALL implement the Sigmoid activation function for binary classification.
- **Requirement**: SHALL utilize the Binary Cross-Entropy (BCE) cost function.
- **Requirement**: Implementation SHALL use a gradient-based optimization approach.

#### Scenario: Feature Mapping

- **WHEN** training for binary classification.
- **THEN** it MUST utilize the `Breast Cancer` dataset.

#### 3.1.3 Algorithm 03: K-Nearest Neighbors (KNN)

- **Requirement**: SHALL support both Classification and Regression tasks.
- **Requirement**: SHALL allow selection between `Euclidean` and `Manhattan` distance metrics.

#### Scenario: Multi-class Evaluation

- **WHEN** testing classification accuracy.
- **THEN** it MUST use the `Iris` dataset for 3-class prediction.

#### 3.1.4 Algorithm 04: Naive Bayes

- **Requirement**: SHALL implement the `Multinomial` variant for discrete/text features.
- **Requirement**: SHALL utilize `Laplace Smoothing` to handle zero probabilities.
- **Requirement**: SHALL compute probabilities in log-space using the `log-sum-exp` trick to prevent numerical underflow.

#### Scenario: Text Classification

- **WHEN** demonstrating text analysis.
- **THEN** it MUST use the `SMS Spam Collection` dataset for binary labeling.

#### 3.1.5 Algorithm 05: Support Vector Machine (SVM)

- **Requirement**: SHALL solve the dual problem using the `Sequential Minimal Optimization (SMO)` algorithm.
- **Requirement**: SHALL support non-linear separation via the `Polynomial Kernel`.
- **Requirement**: SHALL explicitly highlight and visualize the **Support Vectors** in training plots.

#### Scenario: Non-linear Boundary

- **WHEN** testing kernelized separation.
- **THEN** it SHOULD use the `make_moons` dataset for complex boundary visualization.

### 3.2 Module 2: Trees & Ensembles

#### 3.2.1 Algorithm 06: Decision Tree

- **Requirement**: SHALL implement the CART (Classification and Regression Trees) algorithm using recursive binary splitting.
- **Requirement**: SHALL use Gini Impurity as the splitting criterion for classification.
- **Requirement**: SHALL support regularization parameters including `max_depth`, `min_samples_split`, and `min_impurity_decrease`.

#### Scenario: Interpretability and Visualization

- **WHEN** training for classification.
- **THEN** it MUST use the `Breast Cancer` dataset.
- **AND** it MUST generate a visualization of the decision boundaries.
