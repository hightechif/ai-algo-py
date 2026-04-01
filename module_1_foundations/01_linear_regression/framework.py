import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as tg_optim
from typing import Optional

class LinearRegressionFramework:
    """
    Linear Regression Framework implementation using PyTorch and TinyGrad.
    
    Hypothesis:
    $$ \hat{y} = wX + b $$
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

        model = nn.Linear(X.shape[1], 1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        for _ in range(self.n_iters):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.w = model.weight.detach().numpy().flatten()
        self.b = float(model.bias.detach().numpy()[0])

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
            # y = wX + b
            out = X_tg.matmul(w) + b
            # MSE Loss
            loss = ((out - y_tg)**2).mean()
            loss.backward()
            optimizer.step()

        self.w = w.numpy().flatten()
        self.b = float(b.numpy()[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model must be fitted before prediction.")
        return np.dot(X, self.w) + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return float(1 - u/v)
