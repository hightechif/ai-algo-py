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
    X = data.data[:, 0].reshape(-1, 1)  # MedInc feature
    y = data.target

    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training Scratch Model
    print("Training LinearRegressionFromScratch...")
    scratch_model = LinearRegressionFromScratch(learning_rate=0.01, n_iters=1000)
    scratch_model.fit(X_train, y_train)
    scratch_preds = scratch_model.predict(X_test)
    scratch_r2 = scratch_model.score(X_test, y_test) if hasattr(scratch_model, 'score') else None
    
    # 4. Training Framework Model
    print("Training LinearRegressionFramework...")
    framework_model = LinearRegressionFramework()
    framework_model.fit(X_train, y_train)
    framework_preds = framework_model.predict(X_test)
    framework_r2 = framework_model.score(X_test, y_test)

    # 5. Visualization
    print("\nVisualizing results...")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data')
    plt.plot(X_test, scratch_preds, color='blue', linewidth=2, label='Scratch Prediction')
    plt.plot(X_test, framework_preds, color='orange', linestyle='--', linewidth=2, label='Framework Prediction')
    plt.xlabel('Median Income')
    plt.ylabel('House Value')
    plt.title('Linear Regression: Scratch vs Framework')
    plt.legend()
    plt.show()

    # 6. Quantitative Comparison
    print(f"Framework R^2 Score: {framework_r2:.4f}")
    
    # Simple check for similarity
    correlation = np.corrcoef(scratch_preds.flatten(), framework_preds.flatten())[0, 1]
    print(f"Correlation between predictions: {correlation:.4f}")

if __name__ == "__main__":
    main()
