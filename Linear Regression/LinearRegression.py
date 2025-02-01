import numpy as np
from itertools import combinations_with_replacement
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, model_type='linear', lr=0.01, n_iters=1000, lambada=0.1, degree=1):
        self.model_type = model_type
        self.lr = lr
        self.n_iters = n_iters
        self.lambada = lambada
        self.degree = degree
        self.weights = None

    def fit(self, X, y, gradient=True):
        if self.degree > 1:
            X = self._polynomial_features(X, self.degree)  # Transform features if degree > 1
        X = np.hstack((np.ones(X.shape[0]).reshape(-1,1), X))  # Add bias term
        if self.model_type in ['linear', 'lasso', 'ridge']:
            self._regression(X, y, gradient)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, X):
        if self.degree > 1:
            X = self._polynomial_features(X, self.degree)
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        if self.model_type in ['linear', 'lasso', 'ridge']:
            return X @ self.weights
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _regression(self, X, y, gradient=True):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)
        self.weights = np.random.uniform(-1, 1, (n_features, 1))  # Initialize weights with random values
        if gradient:
            for _ in range(self.n_iters):
                y_pred = np.dot(X, self.weights)
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
                if self.model_type == 'lasso':
                    dw += self.lambada * np.sign(self.weights)  # L1 regularization
                elif self.model_type == 'ridge':
                    dw += self.lambada * self.weights  # L2 regularization
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
    

# 1. Linear Regression (Gradient Descent)
x, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=5)
model = LinearRegression(model_type='linear', lr=0.01, n_iters=1000)
model.fit(x, y)
plt.scatter(x, y, label='Data')
plt.plot(x, model.predict(x), color='red', label='Linear Regression')
plt.title('Linear Regression (Gradient Descent)')
plt.legend()
plt.show()

# 2. Linear Regression (Normal Equation)
x, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=5)
model = LinearRegression(model_type='linear', lr=0.01, n_iters=1000)
model.fit(x, y, gradient=False)  # Use normal equation
plt.scatter(x, y, label='Data')
plt.plot(x, model.predict(x), color='red', label='Linear Regression (Normal Equation)')
plt.title('Linear Regression (Normal Equation)')
plt.legend()
plt.show()

# 3. Ridge Regression (Gradient Descent)
# Generate synthetic data
x, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=5)
model = LinearRegression(model_type='ridge', lr=0.01, n_iters=1000, lambada=0.1)
model.fit(x, y)
plt.scatter(x, y, label='Data')
plt.plot(x, model.predict(x), color='red', label='Ridge Regression')
plt.title('Ridge Regression (Gradient Descent)')
plt.legend()
plt.show()

# 4. Ridge Regression (Normal Equation)
x, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=5)
model = LinearRegression(model_type='ridge', lr=0.01, n_iters=1000, lambada=0.1)
model.fit(x, y, gradient=False)  # Use normal equation
plt.scatter(x, y, label='Data')
plt.plot(x, model.predict(x), color='red', label='Ridge Regression (Normal Equation)')
plt.title('Ridge Regression (Normal Equation)')
plt.legend()
plt.show()

# 5. Lasso Regression (Gradient Descent)
x, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=5)
model = LinearRegression(model_type='lasso', lr=0.01, n_iters=1000, lambada=0.1)
model.fit(x, y)
plt.scatter(x, y, label='Data')
plt.plot(x, model.predict(x), color='red', label='Lasso Regression')
plt.title('Lasso Regression (Gradient Descent)')
plt.legend()
plt.show()

# 6. Polynomial Regression (Quadratic)
np.random.seed(42)
x = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points between -5 and 5
y = 2 * x**2 + 3 * x + 5 + np.random.normal(0, 10, size=x.shape)  # Quadratic relationship with noise
model = LinearRegression(model_type='linear', lr=0.01, n_iters=1000, degree=2)
model.fit(x, y)
x_test = np.linspace(-5, 5, 100).reshape(-1, 1)  # Test data for plotting
y_pred = model.predict(x_test)
plt.scatter(x, y, label='Data', color='blue')
plt.plot(x_test, y_pred, label='Polynomial Regression (Degree=2)', color='red')
plt.title('Polynomial Regression on Non-Linear Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 7. Polynomial Regression (Cubic)
np.random.seed(42)
x = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points between -5 and 5
y = x**3 - 2 * x**2 + 3 * x + 5 + np.random.normal(0, 10, size=x.shape)  # Cubic relationship with noise
model_cubic = LinearRegression(model_type='linear', lr=0.01, n_iters=1000, degree=3)
model_cubic.fit(x, y)
x_test = np.linspace(-5, 5, 100).reshape(-1, 1)  # Test data for plotting
y_pred_cubic = model_cubic.predict(x_test)
plt.scatter(x, y, label='Data', color='blue')
plt.plot(x_test, y_pred_cubic, label='Polynomial Regression (Degree=3)', color='red')
plt.title('Polynomial Regression on Cubic Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()