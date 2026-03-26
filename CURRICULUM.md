# AI Algorithm Curriculum Spec

This document serves as the structural specification and learning path for mastering AI/ML concepts. The curriculum is divided into conceptual modules, where each subsequent module generally builds on the mathematics and theories of the prior one.

## Implementation Guidelines

For every algorithm listed in the curriculum below, the codebase will maintain a consistent, educational structure. 

### Folder Structure per Algorithm
Each algorithm will have its own directory containing standard files:
```text
ai-algo-py/
└── module_1_foundations/
    └── 01_linear_regression/
        ├── README.md               # Explanation of math, theory, and usage
        ├── scratch.py              # The pure NumPy/math implementation
        ├── framework.py            # Implementation using scikit-learn/PyTorch
        └── tutorial.ipynb          # Jupyter notebook walking through an example end-to-end
```

### Defining "From Scratch"
*   **Modules 1-3 (Machine Learning):** Pure Python + `NumPy`. No ML frameworks. Manually calculating gradients, matrix splits, and distance metrics.
*   **Module 4 (Deep Learning):** Building the architecture using raw PyTorch/TensorFlow tensors, but heavily restricting the use of pre-packaged layers like `nn.Linear` or `nn.Transformer` unless absolutely necessary for complex architectures (like self-attention).

---

## The Curriculum Path

### Module 1: The Foundations & Distance (Supervised)
*Before learning neural nets, we must master basic optimization, distance metrics, and probabilty.*
1. **Linear Regression**: Fitting lines, Mean Squared Error, Gradient Descent.
2. **Logistic Regression**: Classification, Sigmoid function, Cross-Entropy Loss.
3. **K-Nearest Neighbors (KNN)**: Distance metrics (Euclidean, Manhattan).
4. **Naive Bayes**: Conditional probability, Bayes' Theorem.
5. **Support Vector Machine (SVM)**: Hyperplanes, margins, and the kernel trick.

### Module 2: Trees & Ensembles (Supervised)
*Branching logic and the power of combining many weak learners into strong ones.*
6. **Decision Tree**: Information Gain, Gini impurity, recursive splitting.
7. **Random Forest**: Bagging (Bootstrap Aggregation), feature randomness.
8. **AdaBoost**: Adaptive boosting, sample weighting.
9. **Gradient Boosting**: Minimizing residual errors sequentially.
10. **XGBoost**: Advanced boosting, regularization, tree pruning.
11. **Isolation Forest**: Utilizing random splits for anomaly detection.

### Module 3: Finding Structure (Unsupervised)
*Extracting patterns from unlabeled data.*
12. **Principal Component Analysis (PCA)**: Eigenvectors, Eigenvalues, Dimensionality Reduction.
13. **t-SNE**: Non-linear manifold learning for visualization.
14. **k-Means Clustering**: Centroids, iterative distance minimization.
15. **k-Means++**: Optimized centroid initialization.
16. **Hierarchical Clustering**: Agglomerative methods, Dendrograms.
17. **DBSCAN**: Density-based spatial clustering.
18. **Gaussian Mixture Models (GMM)**: Expectation-Maximization (EM) algorithm.

### Module 4: Deep Learning Foundations (Neural Networks)
*Moving into continuous representations and universal function approximators.*
19. **Artificial Neural Network (ANN)**: Perceptrons, Backpropagation, Activation Functions.
20. **Autoencoders**: Unsupervised representation learning, bottling data.
21. **Convolutional Neural Network (CNN)**: Kernels, filters, max-pooling for spatial data.
22. **Recurrent Neural Network (RNN)**: Handling sequence and time-series data.
23. **Long Short-Term Memory (LSTM)**: Gating mechanisms, solving vanishing gradients.
24. **Generative Adversarial Networks (GANs)**: Generator vs Discriminator architecture.
25. **Transformer**: Self-attention mechanisms, positional encoding.

### Module 5: Learning via Reward (Reinforcement Learning)
*Teaching agents to make decisions through trial and error in environments.*
26. **Markov Decision Process (MDP)**: States, actions, transition models, and rewards.
27. **Q-Learning**: Value-based iteration, temporal difference learning.
28. **SARSA**: On-policy learning.
29. **Deep Q-Network (DQN)**: Melding Neural Nets with Q-Learning.
30. **Policy Gradient**: Optimizing the policy directly.
31. **Actor-Critic**: Combining Policy-based and Value-based approaches.

### Module 6: Evolutionary Computing
*Optimization inspired by biological evolution.*
32. **Genetic Algorithm**: Chromosomes, crossover, mutation, fitness functions.
