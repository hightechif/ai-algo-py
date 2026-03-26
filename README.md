# AI Algorithms from Scratch (ai-algo-py)

A comprehensive, progression-based curriculum and code repository for learning Artificial Intelligence and Machine Learning algorithms. This repository aims to document the journey of implementing 30+ classic and modern AI algorithms.

## The Goal

The purpose of this project serves a dual goal:

1. **Personal Learning:** A disciplined approach to understanding the underlying mathematics and mechanics of AI algorithms.
2. **Public Documentation:** Providing clean, heavily-commented, and structural code to serve as a curriculum for others trying to master these same concepts.

## Approach: "From Scratch" vs "Framework"

To gain a deep understanding, each algorithm in this repository is implemented twice:

* **From Scratch:** Using fundamental libraries (like `NumPy` in Python) to manually program matrix multiplications, gradients, backwards propagation, loss functions, and optimization loops.
* **With Frameworks:** Utilizing industry-standard libraries like `scikit-learn`, `PyTorchT`, or `TensorFlow` to solve the exact same problem, highlighting how the "black boxes" abstract away the scratch implementations.

## Getting Started

The project is outlined in `CURRICULUM.md`, which defines the exact order and modules to progress through. The raw list of algorithms can be found in `algo-list.txt`.

## Algorithm Curriculum Roadmap

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
