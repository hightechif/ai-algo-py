# AI Algorithms From Scratch (ai-algo-py)

A comprehensive, progression-based curriculum for mastering Artificial Intelligence and Machine Learning algorithms. This project implements 30+ algorithms from foundational mathematics to advanced deep learning.

## 🚀 The Mission

1. **Master the Math**: Implement every algorithm using only fundamental libraries (`NumPy`) to understand optimization, gradients, and cost functions.
2. **Bridge the Gap**: Compare scratch implementations with industry-standard frameworks (`PyTorch`, `TinyGrad`, `Scikit-Learn`).
3. **Curriculum Based**: Follow a structured learning path from Foundations to Reinforcement Learning.

## 🏛️ Project Architecture

To maintain the highest educational standards, this project follows a formal **OpenSpec** system.

- **[Master Roadmap](./openspec/specs/algorithm-curriculum/spec.md)**: The full 32-algorithm curriculum across 6 major modules.
- **[Algorithm Requirements](./openspec/specs/algorithm-curriculum/spec.md#3-module-functional-requirements)**: Functional requirements for each algorithm (Datasets, Optimization, Scenarios).
- **[Coding Standards](./openspec/specs/coding-standard/spec.md)**: Strict standards for typing, file structure, and documentation.

Below is the structured path detailing how all 32 AI Algorithms are mapped across the 6 major learning modules:

```text
=========================[ AI ALGORITHMS CURRICULUM ]=========================

[1] FOUNDATIONS (Supervised)          [2] TREES & ENSEMBLES
 ├─ Linear Regression    [Reg]         ├─ Decision Tree           [Cls/Reg]
 ├─ Logistic Regression  [Cls]         ├─ Random Forest           [Cls/Reg]
 ├─ K-Nearest Neighbors  [Cls/Reg]     ├─ AdaBoost                [Cls/Reg]
 ├─ Naive Bayes          [Cls]         ├─ Grad Boost / XGBoost    [Cls/Reg]
 └─ Support Vector Mach. [Cls/Reg]     └─ Isolation Forest        [Anomaly]

[3] FINDING STRUCTURE (Unsup.)        [4] DEEP LEARNING (NNs)
 ├─ PCA / t-SNE          [Dim Reduct]  ├─ Artificial Neural Net   [Deep]
 ├─ K-Means / K-Means++  [Cluster]     ├─ Autoencoders            [Gen/Deep]
 ├─ Hierarch. Clustering [Cluster]     ├─ Convolutional NN (CNN)  [Vis/Deep]
 └─ DBSCAN / GMM         [Cluster]     ├─ Recurrent NN / LSTM     [Seq/Deep]
                                       └─ GANs / Transformers     [Gen/Deep]

[5] REINFORCEMENT LEARNING            [6] EVOLUTIONARY COMPUTING
 ├─ Markov Decision Proc [RL]          ├─ Genetic Algorithm       [Opt]
 ├─ Q-Learning / SARSA   [RL]          └─ (Future Expansions)
 ├─ Deep Q-Network (DQN) [RL]               
 └─ Actor-Critic         [RL]     

Legend: [Reg] Regression | [Cls] Classification | [Cluster] Clustering 
        [RL] Reinforcement | [Deep] Deep Learning | [Opt] Optimization
==============================================================================
```

## 📦 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/hightechif/ai-algo-py.git

# Install dependencies
pip install -r requirements.txt
```

### Navigating the Modules
Each algorithm resides in its own module directory:
`module_X/YY_name/`
- `scratch.py`: Pure NumPy implementation.
- `framework.py`: Multi-engine framework implementation.
- `tutorial.py`: Comparison and visualization script.

---

*Follow the formal requirements in `openspec/specs/` to ensure your contributions meet the curriculum standards.*
