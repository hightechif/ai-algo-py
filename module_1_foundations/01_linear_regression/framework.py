import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple

def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[LinearRegression, np.ndarray]:
    """
    Trains a sklearn Linear Regression model and returns the predictions.
    
    Returns:
        model: The trained scikit-learn model
        y_pred: Predictions on the X_test dataset
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, y_pred
