import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scratch import SVMFromScratch
from framework import SVMFramework

def plot_svm_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "SVM Decision Boundary") -> None:
    """
    Custom visualization function that highlights Support Vectors.
    """
    # Create a mesh grid to plot in
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Standardize predictions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(Z <= 0, -1, 1).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    # Highlight Support Vectors
    # In SMO, Support Vectors are points where alpha > 1e-5
    if hasattr(model, 'alphas') and model.alphas is not None:
        sv_idx = np.where(model.alphas > 1e-5)[0]
        plt.scatter(X[sv_idx, 0], X[sv_idx, 1], s=150, 
                    linewidth=2, facecolors='none', edgecolors='orange', 
                    label='Support Vectors')
    
    plt.title(title)
    plt.legend()
    plt.show()

def main() -> None:
    # 1. Data Preparation
    print("Preparing Moons dataset...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert to {-1, 1} for SVM

    # 2. Training Scratch SVM
    print("Training SVMFromScratch (Polynomial Kernel + SMO)...")
    scratch_svm = SVMFromScratch(C=1.0, degree=3, coef0=1.0, max_iter=200)
    scratch_svm.fit(X, y)
    plot_svm_boundary(scratch_svm, X, y, title="Scratch SVM (Polynomial Kernel + SMO)")

    # 3. Training Framework SVM
    print("Training SVMFramework (Scikit-Learn SVC)...")
    framework_svm = SVMFramework(C=1.0, degree=3, coef0=1.0)
    framework_svm.fit(X, y)
    plot_svm_boundary(framework_svm, X, y, title="Framework SVM (Scikit-Learn)")

if __name__ == "__main__":
    main()
