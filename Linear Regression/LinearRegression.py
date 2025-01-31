import numpy as np
from itertools import combinations_with_replacement


class LinearRegression:

    def __init__(self, model_type='linear', lr=0.001, n_iters=1000, lambada=0.1, degree=1):
        self.model_type = model_type
        self.lr = lr
        self.n_iters = n_iters
        self.lambada = lambada
        self.degree = degree
        self.weights = None

    def fit(self, X, y, gradient=True):
        if self.degree > 1:
            X = self._polynomial_features(X, self.degree)  # Transform features if degree > 1
        if self.model_type in ['linear', 'lasso', 'ridge']:
            self._regression(X, y, gradient)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, X):
        if self.model_type in ['linear', 'lasso', 'ridge']:
            return X @ self.weights
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _regression(self, X, y, gradient=True):
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(-1, 1, (n_features, 1))
        self.weights = np.hstack((np.ones((n_features, 1)), self.weights))  # Add bias term
        if gradient:
            for _ in range(self.n_iters):
                y_pred = np.dot(X, self.weights)
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
                if self.model_type == 'lasso':
                    dw += self.lambada * np.sign(self.weights)  # L1 regularization
                elif self.model_type == 'ridge':
                    dw += self.lambada * np.sum(self.weights ** 2)  # L2 regularization
                self.weights -= self.lr * dw
        else:
            if self.model_type == 'linear':
                self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
            elif self.model_type == 'ridge':
                self.weights = np.linalg.inv(X.T @ X + self.lambada * np.eye(X.shape[1])) @ X.T @ y
            elif self.model_type == 'lasso':
                raise ValueError("Normal Equation doesn't support L1 regularization")

    def _polynomial_features(self, X, degree):
        n_samples, n_features = np.shape(X)
        def index_combinations():
            combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
            flat_combs = [item for sublist in combs for item in sublist]
            return flat_combs
        combinations = index_combinations()
        n_output_features = len(combinations)
        X_new = np.empty((n_samples, n_output_features))
        for i, index_combs in enumerate(combinations):
            X_new[:, i] = np.prod(X[:, index_combs], axis=1)
        return X_new