import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Tuple

def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[LogisticRegression, np.ndarray, float]:
    """
    Trains a sklearn Logistic Regression model and returns predictions and accuracy score.
    
    Returns:
        model: The trained scikit-learn model
        y_pred: Predictions on the X_test dataset
        accuracy: The percentage accuracy of classifications
    """
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    
    return model, y_pred, accuracy
