import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scratch import LogisticRegressionFromScratch
from framework import LogisticRegressionFramework

def main() -> None:
    # 1. Load Data
    print("Loading Breast Cancer dataset...")
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # 3. Scaling (Important for Gradient Descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Training Scratch Model
    print("Training LogisticRegressionFromScratch...")
    scratch_model = LogisticRegressionFromScratch(learning_rate=0.1, n_iters=1000)
    scratch_model.fit(X_train_scaled, y_train)
    scratch_acc = np.mean(scratch_model.predict(X_test_scaled) == y_test)

    # 5. Training Framework Model (TinyGrad)
    print("Training LogisticRegressionFramework via TinyGrad...")
    framework_tg = LogisticRegressionFramework(learning_rate=0.1, n_iters=1000)
    framework_tg.fit_tinygrad(X_train_scaled, y_train)
    tg_acc = framework_tg.score(X_test_scaled, y_test)

    # 6. Training Framework Model (PyTorch)
    print("Training LogisticRegressionFramework via PyTorch...")
    framework_pt = LogisticRegressionFramework(learning_rate=0.1, n_iters=1000)
    framework_pt.fit_pytorch(X_train_scaled, y_train)
    pt_acc = framework_pt.score(X_test_scaled, y_test)

    # 7. Comparison Summary
    print(f"\nResults (Accuracy):")
    print(f"Scratch:   {scratch_acc:.4f}")
    print(f"TinyGrad:  {tg_acc:.4f}")
    print(f"PyTorch:   {pt_acc:.4f}")

if __name__ == "__main__":
    main()
