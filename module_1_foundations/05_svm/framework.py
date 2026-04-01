import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as tg_optim
from typing import Optional

class SVMFramework:
    """
    Support Vector Machine Framework implementation using PyTorch and TinyGrad.
    Solves the Soft-Margin SVM objective using Hinge Loss.
    
    Objective:
    $$ L(w, b) = \lambda ||w||^2 + \sum \max(0, 1 - y_i(wX_i + b)) $$
    """
    def __init__(self, C: float = 1.0, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        self.C = C
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit_pytorch(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implementation using PyTorch."""
        X_tensor = torch.from_numpy(X.astype(np.float32))
        # SVM traditionally uses {-1, 1}
        y_tensor = torch.from_numpy(np.where(y <= 0, -1, 1).astype(np.float32)).view(-1, 1)

        model = nn.Linear(X.shape[1], 1)
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        for _ in range(self.n_iters):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            
            # Hinge Loss: L = max(0, 1 - y * pred)
            hinge_loss = torch.mean(torch.clamp(1 - y_tensor * outputs, min=0))
            # L2 Regularization (soft margin)
            l2_reg = 0.5 * torch.sum(model.weight ** 2) / self.C
            loss = hinge_loss + l2_reg
            
            loss.backward()
            optimizer.step()

        self.w = model.weight.detach().numpy().flatten()
        self.b = float(model.bias.detach().numpy()[0])

    def fit_tinygrad(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implementation using TinyGrad."""
        X_tg = Tensor(X.astype(np.float32))
        y_tg = Tensor(np.where(y <= 0, -1, 1).astype(np.float32).reshape(-1, 1))

        # Parameters
        w = Tensor.zeros(X.shape[1], 1, requires_grad=True)
        b = Tensor.zeros(1, requires_grad=True)
        
        optimizer = tg_optim.SGD([w, b], lr=self.lr)

        for _ in range(self.n_iters):
            optimizer.zero_grad()
            # Predict
            preds = X_tg.matmul(w) + b
            
            # Hinge Loss: max(0, 1 - y * preds)
            hinge = (1 - y_tg * preds).relu().mean()
            # L2 weight decay / margin optimization
            l2 = 0.5 * (w * w).sum() / self.C
            loss = hinge + l2
            
            loss.backward()
            optimizer.step()

        self.w = w.numpy().flatten()
        self.b = float(b.numpy()[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model must be fitted before prediction.")
        
        linear_model = np.dot(X, self.w) + self.b
        return np.where(linear_model >= 0, 1, 0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
