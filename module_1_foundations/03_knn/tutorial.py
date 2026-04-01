import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scratch import KNearestNeighborsFromScratch
from framework import KNearestNeighborsFramework

def run_classification() -> None:
    print("--- KNN Classification (Iris Dataset) ---")
    # 1. Load Data
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Training
    print("Training KNearestNeighborsFromScratch (k=5)...")
    scratch_model = KNearestNeighborsFromScratch(k=5, task='classification', metric='euclidean')
    scratch_model.fit(X_train_scaled, y_train)
    scratch_acc = np.mean(scratch_model.predict(X_test_scaled) == y_test)

    print("Training KNearestNeighborsFramework (k=5)...")
    framework_model = KNearestNeighborsFramework(k=5, task='classification')
    framework_model.fit(X_train_scaled, y_train)
    framework_acc = framework_model.score(X_test_scaled, y_test)

    print(f"Scratch Accuracy:   {scratch_acc:.4f}")
    print(f"Framework Accuracy: {framework_acc:.4f}\n")

def run_regression() -> None:
    print("--- KNN Regression (Diabetes Dataset) ---")
    # 1. Load Data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Training
    print("Training KNearestNeighborsFromScratch (k=5)...")
    scratch_reg = KNearestNeighborsFromScratch(k=5, task='regression', metric='euclidean')
    scratch_reg.fit(X_train_scaled, y_train)
    scratch_preds = scratch_reg.predict(X_test_scaled)
    scratch_mse = np.mean((scratch_preds - y_test) ** 2)

    print("Training KNearestNeighborsFramework (k=5)...")
    framework_reg = KNearestNeighborsFramework(k=5, task='regression')
    framework_reg.fit(X_train_scaled, y_train)
    framework_preds = framework_reg.predict(X_test_scaled)
    framework_mse = np.mean((framework_preds - y_test) ** 2)

    print(f"Scratch MSE:   {scratch_mse:.4f}")
    print(f"Framework MSE: {framework_mse:.4f}\n")

def main() -> None:
    run_classification()
    run_regression()

if __name__ == "__main__":
    main()
