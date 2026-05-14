import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Import our implementations
try:
    from scratch import DecisionTree
    from framework import DecisionTreeFramework
except ImportError:
    # If running from a different directory, adjust the path
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from scratch import DecisionTree
    from framework import DecisionTreeFramework

def run_tutorial():
    """
    Demonstrates the Decision Tree implementation on the Breast Cancer dataset.
    Compares the from-scratch implementation with the scikit-learn framework.
    """
    print("--- Algorithm 06: Decision Tree Tutorial ---")

    # 1. Load Dataset
    # We use only two features for visualization purposes initially
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Feature indices for 'mean radius' and 'mean texture'
    X_viz = X[:, :2]
    X_train, X_test, y_train, y_test = train_test_split(X_viz, y, test_size=0.2, random_state=42)

    print(f"Dataset: Breast Cancer (Subset: 2 features)")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 2. Train Scratch Model
    print("\nTraining Scratch Model...")
    scratch_model = DecisionTree(max_depth=5, min_samples_split=2)
    scratch_model.fit(X_train, y_train)
    scratch_preds = scratch_model.predict(X_test)
    scratch_acc = accuracy_score(y_test, scratch_preds)

    # 3. Train Framework Model
    print("Training Framework Model...")
    framework_model = DecisionTreeFramework(max_depth=5, min_samples_split=2)
    framework_model.fit(X_train, y_train)
    framework_preds = framework_model.predict(X_test)
    framework_acc = accuracy_score(y_test, framework_preds)

    # 4. Results
    print(f"\n--- Results ---")
    print(f"Scratch Accuracy:   {scratch_acc:.4f}")
    print(f"Framework Accuracy: {framework_acc:.4f}")

    # 5. Visualization
    print("\nGenerating decision boundary visualization...")
    plt.figure(figsize=(10, 6))
    plot_decision_boundaries(X_viz, y, scratch_model, "Decision Tree (Scratch) - Breast Cancer")
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "decision_boundary.png")
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

def plot_decision_boundaries(X, y, model, title):
    """Plot the decision boundaries for the given model."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict over the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour and training points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Mean Radius")
    plt.ylabel("Mean Texture")

if __name__ == "__main__":
    run_tutorial()
