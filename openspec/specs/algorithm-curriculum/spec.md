# AI Algorithm Curriculum Specification

## 1. General Principles

### 1.1 Implementation Requirements

Every algorithm module in the curriculum MUST satisfy these core principles:

- **From Scratch vs Framework**: Every algorithm MUST provide a from-scratch implementation (NumPy) and a framework implementation (TinyGrad/PyTorch/Scikit-Learn).
- **Real-World Priority**: Tutorials MUST prioritize real-world datasets over synthetic generators.
- **Mathematical Transparency**: Implementation logic MUST be traceably aligned with the mathematical formulas in the module's `README.md`.

## 2. Module 1: Foundational Algorithms

This section defines the functional requirements for each algorithm in Module 1.

### 2.1 Algorithm 01: Linear Regression

- **Requirement**: SHALL provide a from-scratch implementation using NumPy vectorization for gradient descent.
- **Requirement**: SHALL minimize the Mean Squared Error (MSE) objective function.
- **Requirement**: SHALL compare performance against an industry-standard framework.

#### Scenario: Real-world Dataset
- **WHEN** evaluating the model during a tutorial.
- **THEN** it MUST use the `California Housing` dataset.
- **AND** it MUST compare From-Scratch vs Framework results.

### 2.2 Algorithm 02: Logistic Regression

- **Requirement**: SHALL implement the Sigmoid activation function for binary classification.
- **Requirement**: SHALL utilize the Binary Cross-Entropy (BCE) loss function.
- **Requirement**: Implementation SHALL use a gradient-based optimization approach.

#### Scenario: Feature Mapping
- **WHEN** training for binary classification.
- **THEN** it MUST utilize the `Breast Cancer` dataset.

### 2.3 Algorithm 03: K-Nearest Neighbors (KNN)

- **Requirement**: SHALL support both Classification and Regression tasks.
- **Requirement**: SHALL allow selection between `Euclidean` and `Manhattan` distance metrics.

#### Scenario: Multi-class Evaluation
- **WHEN** testing classification accuracy.
- **THEN** it MUST use the `Iris` dataset for 3-class prediction.

### 2.4 Algorithm 04: Naive Bayes

- **Requirement**: SHALL implement the `Multinomial` variant for discrete/text features.
- **Requirement**: SHALL utilize `Laplace Smoothing` to handle zero probabilities.
- **Requirement**: SHALL compute probabilities in log-space using the `log-sum-exp` trick to prevent numerical underflow.

#### Scenario: Text Classification
- **WHEN** demonstrating text analysis.
- **THEN** it MUST use the `SMS Spam Collection` dataset for binary labeling.

### 2.5 Algorithm 05: Support Vector Machine (SVM)

- **Requirement**: SHALL solve the dual problem using the `Sequential Minimal Optimization (SMO)` algorithm.
- **Requirement**: SHALL support non-linear separation via the `Polynomial Kernel`.
- **Requirement**: SHALL explicitly highlight and visualize the **Support Vectors** in training plots.

#### Scenario: Non-linear Boundary
- **WHEN** testing kernelized separation.
- **THEN** it SHOULD use the `make_moons` dataset for complex boundary visualization.
