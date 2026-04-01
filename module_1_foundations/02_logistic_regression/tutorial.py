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
    scratch_preds = scratch_model.predict(X_test_scaled)
    scratch_acc = np.mean(scratch_preds == y_test)

    # 5. Training Framework Model
    print("Training LogisticRegressionFramework...")
    framework_model = LogisticRegressionFramework(max_iter=1000)
    framework_model.fit(X_train_scaled, y_train)
    framework_acc = framework_model.score(X_test_scaled, y_test)

    # 6. Results
    print(f"\nResults:")
    print(f"Scratch Accuracy:   {scratch_acc:.4f}")
    print(f"Framework Accuracy: {framework_acc:.4f}")

    assert np.isclose(scratch_acc, framework_acc, atol=1e-2), "Accuracy gap too large!"

if __name__ == "__main__":
    main()
