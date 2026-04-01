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
    preds = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(preds <= 0, -1, 1).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    # Highlight Support Vectors (Manual SMO)
    if hasattr(model, 'alphas') and model.alphas is not None:
        sv_idx = np.where(model.alphas > 1e-5)[0]
        plt.scatter(X[sv_idx, 0], X[sv_idx, 1], s=150, 
                    linewidth=2, facecolors='none', edgecolors='orange', 
                    label='Support Vectors (Manual)')
    
    plt.title(title)
    plt.legend()
    plt.show()

def main() -> None:
    # 1. Data Preparation
    print("Preparing Moons dataset...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    # Binary classification labels {0, 1} for training, but plots use -1, 1

    # 2. Training Scratch SVM (Manual SMO)
    print("Training SVMFromScratch (Polynomial Kernel + SMO)...")
    scratch_svm = SVMFromScratch(C=1.0, degree=3, coef0=1.0, max_iter=200)
    # Scratch uses {-1, 1} internally or maps it
    scratch_svm.fit(X, np.where(y == 0, -1, 1))
    plot_svm_boundary(scratch_svm, X, np.where(y == 0, -1, 1), title="Scratch SVM (Polynomial SMO)")

    # 3. Training Framework SVM (TinyGrad)
    print("Training SVMFramework via TinyGrad...")
    framework_tg = SVMFramework(C=1.0, learning_rate=0.01, n_iters=1000)
    framework_tg.fit_tinygrad(X, y)
    plot_svm_boundary(framework_tg, X, np.where(y == 0, -1, 1), title="Framework SVM (TinyGrad Hinge Loss)")

    # 4. Training Framework SVM (PyTorch)
    print("Training SVMFramework via PyTorch...")
    framework_pt = SVMFramework(C=1.0, learning_rate=0.01, n_iters=1000)
    framework_pt.fit_pytorch(X, y)
    plot_svm_boundary(framework_pt, X, np.where(y == 0, -1, 1), title="Framework SVM (PyTorch Hinge Loss)")

if __name__ == "__main__":
    main()
