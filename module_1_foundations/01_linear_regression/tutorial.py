import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from scratch import LinearRegressionFromScratch
from framework import LinearRegressionFramework

def main() -> None:
    # 1. Load Data
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data[:, 0:1] # MedInc feature
    y = data.target

    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training Scratch Model (Manual implementation)
    print("Training LinearRegressionFromScratch...")
    scratch_model = LinearRegressionFromScratch(learning_rate=0.01, n_iters=1000)
    scratch_model.fit(X_train, y_train)
    scratch_preds = scratch_model.predict(X_test)
    
    # 4. Training Framework Model (TinyGrad)
    print("Training LinearRegressionFramework via TinyGrad...")
    framework_tg = LinearRegressionFramework(learning_rate=0.01, n_iters=1000)
    framework_tg.fit_tinygrad(X_train, y_train)
    tg_preds = framework_tg.predict(X_test)
    tg_score = framework_tg.score(X_test, y_test)

    # 5. Training Framework Model (PyTorch)
    print("Training LinearRegressionFramework via PyTorch...")
    framework_pt = LinearRegressionFramework(learning_rate=0.01, n_iters=1000)
    framework_pt.fit_pytorch(X_train, y_train)
    pt_preds = framework_pt.predict(X_test)
    pt_score = framework_pt.score(X_test, y_test)

    # 6. Visualization
    print("\nVisualizing results...")
    plt.figure(figsize=(12, 7))
    plt.scatter(X_test, y_test, color='lightgray', alpha=0.5, label='Actual Data')
    plt.plot(X_test, scratch_preds, color='blue', linewidth=2, label='Scratch (Manual GD)')
    plt.plot(X_test, tg_preds, color='orange', linestyle='--', linewidth=2, label='Framework (TinyGrad)')
    plt.plot(X_test, pt_preds, color='green', linestyle=':', linewidth=2, label='Framework (PyTorch)')
    plt.xlabel('Median Income')
    plt.ylabel('House Value')
    plt.title('Linear Regression Comparison: Scratch vs TinyGrad vs PyTorch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 7. Quantitative Comparison
    print(f"\nTinyGrad R^2 Score: {tg_score:.4f}")
    print(f"PyTorch R^2 Score: {pt_score:.4f}")
    
    # Check similarity
    corr_tg = np.corrcoef(scratch_preds.flatten(), tg_preds.flatten())[0, 1]
    corr_pt = np.corrcoef(scratch_preds.flatten(), pt_preds.flatten())[0, 1]
    print(f"Correlation (Scratch vs TinyGrad): {corr_tg:.4f}")
    print(f"Correlation (Scratch vs PyTorch): {corr_pt:.4f}")

if __name__ == "__main__":
    main()
