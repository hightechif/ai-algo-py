# Design: Aggregating Curriculum Features

## Structure

The new specification will be organized by Module and Algorithm ID. Each algorithm will be defined by its **Core Feature Requirements** and **Scenario Requirements**.

### Capability: algorithm-curriculum

**Path**: `openspec/specs/algorithm-curriculum/spec.md`

### 1. General Principles

- **From Scratch vs Framework**: Mandatory implementation of both using a unified API.
- **Real-World Priority**: No synthetic datasets in tutorials.
- **Mathematical Transparency**: Implementation MUST reflect the `README.md` formulas.

### 2. Module 1: The Foundations & Distance (Supervised)

#### 01: Linear Regression

- Requirement: Mean Squared Error (MSE) minimization.
- Requirement: Gradient Descent optimization.
- Scenario: Using California Housing dataset.

#### 02: Logistic Regression

- Requirement: Sigmoid function for classification mapping.
- Requirement: Cross-Entropy Loss for optimization.
- Scenario: Using Breast Cancer dataset.

#### 03: K-Nearest Neighbors (KNN)

- Requirement: Dual Support (Classification and Regression).
- Requirement: Choice of Distance Metrics (Euclidean, Manhattan).
- Scenario: Multi-class classification using Iris dataset.

#### 04: Naive Bayes

- Requirement: Multinomial variant for discrete features.
- Requirement: Log-sum-exp trick for probability calculation.
- Scenario: Spam classification with SMS Spam Collection.

#### 05: Support Vector Machine (SVM)

- Requirement: Sequential Minimal Optimization (SMO) for solving the dual.
- Requirement: Kernel Trick (Polynomial kernel support).
- Requirement: Support Vector visualization.

## Aggregation Method

1. **Extract**: For each algorithm, scan `openspec/changes/archive/[date]-[id]-[name]/proposal.md` and `design.md`.
2. **Synthesize**: Convert implementation goals into formal "SHALL" requirements.
3. **Draft Delta**: Create a delta spec at `openspec/changes/aggregate-curriculum-features/specs/algorithm-curriculum/spec.md`.
4. **Sync**: Use `/opsx-sync` to merge the delta into the main spec file.
