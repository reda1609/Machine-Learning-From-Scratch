import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles


class LogisticRegression:

    def __init__(self, lr=0.01, n_iters=1000, degree=1):
        self.lr = lr
        self.n_iters = n_iters
        self.degree = degree
        self.weights = None

    def fit(self, X, y):
        if self.degree > 1:
            X = self._polynomial_features(X, self.degree)  # Transform features if degree > 1
        X = np.hstack((np.ones(X.shape[0]).reshape(-1,1), X))  # Add bias term
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)
        self.weights = np.random.uniform(-1, 1, (n_features, 1))  # Initialize weights with random values
        for _ in range(self.n_iters):
            y_pred = self.sigmoid(np.dot(X, self.weights))
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            self.weights -= self.lr * dw

    def predict(self, X):
        if self.degree > 1:
            X = self._polynomial_features(X, self.degree)
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        y_pred = self.sigmoid(X @ self.weights)
        return (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    def _polynomial_features(self, X, degree):
        n_samples, n_features = np.shape(X)
        def index_combinations():
            combs = [combinations_with_replacement(range(n_features), i) for i in range(1, degree + 1)]
            flat_combs = [item for sublist in combs for item in sublist]
            return flat_combs
        combinations = index_combinations()
        n_output_features = len(combinations)
        X_new = np.empty((n_samples, n_output_features))
        for i, index_combs in enumerate(combinations):
            X_new[:, i] = np.prod(X[:, index_combs], axis=1)
        return X_new
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
# Function to plot decision boundary
def plot_decision_boundary(X, y, model, degree):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title(f"Decision Boundary (Degree {degree})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Generate synthetic data
# X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
# X, y = make_moons(n_samples=200, noise=0.2, random_state=42)  # For non-linear data
X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title("Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Degree 1 (Linear Decision Boundary)
model_degree_1 = LogisticRegression(lr=0.01, n_iters=1000, degree=1)
model_degree_1.fit(X, y)
plot_decision_boundary(X, y, model_degree_1, degree=1)

# Degree 2 (Non-linear Decision Boundary)
model_degree_2 = LogisticRegression(lr=0.01, n_iters=2000, degree=2)
model_degree_2.fit(X, y)
plot_decision_boundary(X, y, model_degree_2, degree=2)