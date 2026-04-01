import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as tg_optim
from typing import Optional

class LogisticRegressionFramework:
    """
    Logistic Regression Framework implementation using PyTorch and TinyGrad.
    
    Hypothesis:
    $$ \hat{y} = \sigma(wX + b) $$
    """
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit_pytorch(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implementation using PyTorch."""
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        # Single layer + Sigmoid activation
        model = nn.Sequential(
            nn.Linear(X.shape[1], 1),
            nn.Sigmoid()
        )
        
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        for _ in range(self.n_iters):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Extract weights and bias
        linear_layer = model[0]
        self.w = linear_layer.weight.detach().numpy().flatten()
        self.b = float(linear_layer.bias.detach().numpy()[0])

    def fit_tinygrad(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implementation using TinyGrad."""
        X_tg = Tensor(X.astype(np.float32))
        y_tg = Tensor(y.astype(np.float32).reshape(-1, 1))

        # Parameters
        w = Tensor.zeros(X.shape[1], 1, requires_grad=True)
        b = Tensor.zeros(1, requires_grad=True)
        
        optimizer = tg_optim.SGD([w, b], lr=self.lr)

        for _ in range(self.n_iters):
            optimizer.zero_grad()
            # Logits: wX + b
            logits = X_tg.matmul(w) + b
            # Activation: Sigmoid
            out = logits.sigmoid()
            # Binary Cross-Entropy Loss
            # BCE = -[y*log(p) + (1-y)*log(1-p)]
            loss = (-(y_tg * out.log() + (1-y_tg) * (1-out).log())).mean()
            loss.backward()
            optimizer.step()

        self.w = w.numpy().flatten()
        self.b = float(b.numpy()[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model must be fitted before prediction.")
        
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = 1 / (1 + np.exp(-linear_model))
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
