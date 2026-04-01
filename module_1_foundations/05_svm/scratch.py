import numpy as np
from typing import Optional

class SVMFromScratch:
    """
    Support Vector Machine (SVM) implementation from scratch using SMO.
    
    Dual Optimization Objective:
    $$
    W(\\alpha) = \\sum_{i=1}^{n} \\alpha_i - \\frac{1}{2} \\sum_{i,j=1}^{n} y_i y_j \\alpha_i \\alpha_j K(x_i, x_j)
    $$
    
    Subject to:
    - $0 \\le \\alpha_i \\le C$
    - $\\sum_{i=1}^{n} \\alpha_i y_i = 0$
    """
    
    def __init__(
        self, 
        C: float = 1.0, 
        degree: int = 3, 
        coef0: float = 1.0, 
        tol: float = 1e-3, 
        max_iter: int = 100
    ) -> None:
        """
        Initialize the SVM model with SMO hyperparameters.
        
        Args:
            C: Regularization parameter (Soft Margin).
            degree: Degree for the Polynomial kernel.
            coef0: Independent term in polynomial kernel.
            tol: Tolerance for KKT conditions and convergence.
            max_iter: Maximum number of passes through the training set without change.
        """
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        # State variables (Learned Parameters)
        self.alphas: Optional[np.ndarray] = None
        self.b: float = 0.0
        
        # Stored training data (Required for dual prediction)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def _polynomial_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute the Polynomial kernel between vectors.
        
        Formula:
        $$
        K(x, x') = (x \\cdot x' + c)^d
        $$
        """
        # Supports both single vectors and matricized dot products
        return (np.dot(x1, x2.T) + self.coef0) ** self.degree

    def _get_error(self, i: int) -> float:
        """Calculate prediction error for sample i."""
        if self.alphas is None or self.X is None or self.y is None:
            return 0.0
            
        # f(x) = sum(alpha_j * y_j * K(x_j, x)) + b
        kernels = self._polynomial_kernel(self.X, self.X[i])
        prediction = np.sum(self.alphas * self.y * kernels) + self.b
        return float(prediction - self.y[i])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM using the SMO algorithm.
        """
        self.X = X
        # Ensure y is in {-1, 1}
        self.y = np.where(y <= 0, -1, 1).astype(float)
        n_samples = X.shape[0]
        
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        
        passes = 0
        while passes < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                Ei = self._get_error(i)
                
                # Check KKT conditions
                if (self.y[i] * Ei < -self.tol and self.alphas[i] < self.C) or \
                   (self.y[i] * Ei > self.tol and self.alphas[i] > 0):
                    
                    # Heuristic: Select j != i randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    Ej = self._get_error(j)
                    old_alpha_i, old_alpha_j = self.alphas[i].copy(), self.alphas[j].copy()
                    
                    # Compute L and H boundaries
                    if self.y[i] != self.y[j]:
                        L = max(0.0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0.0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    kii = self._polynomial_kernel(self.X[i], self.X[i])
                    kjj = self._polynomial_kernel(self.X[j], self.X[j])
                    kij = self._polynomial_kernel(self.X[i], self.X[j])
                    eta = 2.0 * kij - kii - kjj
                    
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alphas[j] -= (self.y[j] * (Ei - Ej)) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - old_alpha_j) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alphas[i] += self.y[i] * self.y[j] * (old_alpha_j - self.alphas[j])
                    
                    # Update bias b
                    b1 = self.b - Ei - self.y[i] * (self.alphas[i] - old_alpha_i) * kii - \
                         self.y[j] * (self.alphas[j] - old_alpha_j) * kij
                    b2 = self.b - Ej - self.y[i] * (self.alphas[i] - old_alpha_i) * kij - \
                         self.y[j] * (self.alphas[j] - old_alpha_j) * kjj
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = float(b1)
                    elif 0 < self.alphas[j] < self.C:
                        self.b = float(b2)
                    else:
                        self.b = float((b1 + b2) / 2.0)
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for a set of samples.
        """
        if self.alphas is None or self.X is None or self.y is None:
            raise ValueError("Model must be fitted before prediction.")
            
        # Vectorized prediction: sign(sum(alpha_i * y_i * K(X_i, X)) + b)
        kernels = self._polynomial_kernel(self.X, X)
        predictions = np.dot((self.alphas * self.y), kernels) + self.b
        return np.where(predictions >= 0, 1, 0)
