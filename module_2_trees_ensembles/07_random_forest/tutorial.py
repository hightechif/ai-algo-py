import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys

# Import our implementations
try:
    from scratch import RandomForest
    from framework import RandomForestFramework
except ImportError:
    # If running from a different directory, adjust the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from scratch import RandomForest
    from framework import RandomForestFramework

# Also import our simple DecisionTree for baseline comparison
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '06_decision_tree'))
    import scratch as dt_scratch
    DecisionTree = dt_scratch.DecisionTree
    sys.path.pop(0)
except ImportError:
    pass

def run_tutorial():
    """
    Demonstrates the Random Forest implementation on the Breast Cancer dataset.
    Compares Scratch vs Framework performance and showcases OOB Score.
    """
    print("--- Algorithm 07: Random Forest Tutorial ---")

    # 1. Load Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # We use all features this time to show the power of Random Forest!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset: Breast Cancer (All {X.shape[1]} features)")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # 2. Baseline: Single Decision Tree
    print("Training Baseline (Single Decision Tree)...")
    dt_model = DecisionTree(max_depth=5)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)

    # 3. Train Scratch Random Forest
    print("\nTraining Scratch Random Forest (10 trees)...")
    # Using 10 trees to keep tutorial fast, in practice 100+ is better
    scratch_rf = RandomForest(n_trees=10, max_depth=5, n_features='sqrt')
    scratch_rf.fit(X_train, y_train)
    scratch_preds = scratch_rf.predict(X_test)
    scratch_acc = accuracy_score(y_test, scratch_preds)

    # 4. Train Framework Random Forest
    print("Training Framework Random Forest (10 trees)...")
    framework_rf = RandomForestFramework(n_estimators=10, max_depth=5, max_features='sqrt', random_state=42)
    framework_rf.fit(X_train, y_train)
    framework_preds = framework_rf.predict(X_test)
    framework_acc = accuracy_score(y_test, framework_preds)

    # 5. Results
    print(f"\n--- Results ---")
    print(f"Single Tree Test Accuracy:  {dt_acc:.4f}")
    print(f"Scratch RF Test Accuracy:   {scratch_acc:.4f}")
    print(f"Framework RF Test Accuracy: {framework_acc:.4f}")
    
    print(f"\n--- Advanced: Out-Of-Bag (OOB) Scores ---")
    print(f"Scratch RF OOB Score:       {scratch_rf.oob_score_:.4f}")
    print(f"Framework RF OOB Score:     {framework_rf.oob_score_:.4f}")
    
    # Note: For visualizing decision boundaries, we usually reduce to 2D.
    # Since RF's true power is in high-dimensional space, we skip the 2D contour plot here
    # but acknowledge the variance reduction in the printouts!

if __name__ == "__main__":
    run_tutorial()
