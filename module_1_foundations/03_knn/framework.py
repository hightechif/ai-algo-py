import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Tuple, Any

def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, task: str = 'classification', k: int = 3) -> Tuple[Any, np.ndarray, float]:
    """
    Trains a sklearn KNN model and returns predictions and the evaluation metric.
    Classification returns Accuracy. Regression returns MSE.
    """
    if task == 'classification':
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = float(accuracy_score(y_test, y_pred))
    elif task == 'regression':
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = float(mean_squared_error(y_test, y_pred))
    else:
        raise ValueError("Unsupported task. Use 'classification' or 'regression'.")

    return model, y_pred, score
