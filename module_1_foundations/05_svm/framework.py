import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as tg_optim
from typing import Optional, Literal

class SVMFramework:
    """
    Support Vector Machine (SVM) Framework implementation using PyTorch and TinyGrad.
    Leverages the Kernel Trick in the Primal formulation for non-linear separation.
    
    Objective:
    $$ L(\alpha, b) = \frac{1}{2}\alpha^T K \alpha + C \sum \max(0, 1 - y_i(\sum \alpha_j K_{ij} + b)) $$
    
    This implementation aligns with the high-performance scratch version but 
    utilizes modern autograd engines for flexible optimization.
    """
    
    def __init__(
        self, 
        C: float = 1.0, 
        kernel: Literal['linear', 'poly'] = 'poly',
        degree: int = 3, 
        coef0: float = 1.0,
        learning_rate: float = 0.05, 
        n_iters: int = 1000,
        tol: float = 1e-4
    ) -> None:
        """
        Initialize the SVM framework.
        
        Args:
            C: Regularization parameter (Soft Margin).
            kernel: Kernel type ('linear' or 'poly').
            degree: Degree for the Polynomial kernel.
            coef0: Independent term in polynomial kernel.
            learning_rate: Learning rate for optimization.
            n_iters: Maximum number of gradient descent iterations.
            tol: Convergence tolerance for the loss value.
        """
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.coef0 = coef0
        self.lr = learning_rate
        self.n_iters = n_iters
        self.tol = tol
        
        # State: Learned weights in the dual-space (represented in primal)
        self.learned_alphas: Optional[np.ndarray] = None
        self.b: float = 0.0
        
        # Stored training data (Essential for Kernel trick)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    @property
    def alphas(self) -> Optional[np.ndarray]:
        """Provides compatibility with support vector visualization tools."""
        if self.learned_alphas is None:
            return None
        return np.abs(self.learned_alphas)

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Efficiently compute the kernel matrix using NumPy."""
        if self.kernel_type == 'poly':
            return (np.dot(X1, X2.T) + self.coef0) ** self.degree
        return np.dot(X1, X2.T)

    def fit_pytorch(self, X: np.ndarray, y: np.ndarray) -> None:
        """Non-linear SVM training via PyTorch autograd."""
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                             "cuda" if torch.cuda.is_available() else "cpu")
        
        self.X_train = X
        self.y_train = np.where(y <= 0, -1, 1).astype(np.float32)
        n_samples = X.shape[0]
        
        # Precompute the kernel matrix for the training set
        K = torch.from_numpy(self._compute_kernel(X, X).astype(np.float32)).to(device)
        y_torch = torch.from_numpy(self.y_train).view(-1, 1).to(device)
        
        # Optimize alphas and bias directly in the Kernel space
        alphas = torch.zeros((n_samples, 1), requires_grad=True, device=device)
        b = torch.zeros(1, requires_grad=True, device=device)
        
        optimizer = optim.Adam([alphas, b], lr=self.lr)
        
        for _ in range(self.n_iters):
            optimizer.zero_grad()
            
            # f(x) = K @ alphas + b
            preds = K @ alphas + b
            
            # 1/2 * w^T w  => 1/2 * alpha^T @ K @ alpha
            reg = 0.5 * (alphas.T @ K @ alphas).sum()
            
            # Hinge Loss: C * sum(max(0, 1 - y * pred))
            hinge = torch.clamp(1 - y_torch * preds, min=0).sum()
            
            loss = reg + self.C * hinge
            loss.backward()
            optimizer.step()
            
            if loss.item() < self.tol:
                break
        
        self.learned_alphas = alphas.detach().cpu().numpy().flatten()
        self.b = float(b.detach().cpu().numpy()[0])

    def fit_tinygrad(self, X: np.ndarray, y: np.ndarray) -> None:
        """Non-linear SVM training via TinyGrad engine."""
        self.X_train = X
        self.y_train = np.where(y <= 0, -1, 1).astype(np.float32)
        n_samples = X.shape[0]
        
        K = Tensor(self._compute_kernel(X, X).astype(np.float32))
        y_tg = Tensor(self.y_train.reshape(-1, 1))
        
        alphas = Tensor.zeros(n_samples, 1, requires_grad=True)
        b = Tensor.zeros(1, requires_grad=True)
        
        optimizer = tg_optim.Adam([alphas, b], lr=self.lr)
        
        Tensor.training = True
        for _ in range(self.n_iters):
            optimizer.zero_grad()
            
            preds = K.matmul(alphas) + b
            reg = 0.5 * (alphas.transpose().matmul(K.matmul(alphas))).sum()
            hinge = (1 - y_tg * preds).relu().sum()
            
            loss = reg + self.C * hinge
            loss.backward()
            optimizer.step()
        Tensor.training = False
        
        self.learned_alphas = alphas.numpy().flatten()
        self.b = float(b.numpy()[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using the kernel-mapped decision boundary."""
        if self.learned_alphas is None or self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        
        K_test = self._compute_kernel(X, self.X_train)
        predictions = np.dot(K_test, self.learned_alphas) + self.b
        return np.where(predictions >= 0, 1, 0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
