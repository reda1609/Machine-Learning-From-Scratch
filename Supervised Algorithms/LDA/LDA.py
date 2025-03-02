import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X, y):
        """ Compute LDA for Dataset """
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Compute overall mean
        self.mean_ = np.mean(X, axis=0)

        # Compute within-class scatter matrix (S_W)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for label in class_labels:
            X_class = X[y == label]
            mean_class = np.mean(X_class, axis=0)
            n_i = X_class.shape[0]  # Number of samples in class

            # Compute within-class scatter
            S_W += np.dot((X_class - mean_class).T, (X_class - mean_class))

            # Compute between-class scatter
            mean_diff = (mean_class - self.mean_).reshape(-1, 1)
            S_B += n_i * np.dot(mean_diff, mean_diff.T)

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, idx[:self.n_components]]

        # Compute explained variance ratio
        self.explained_variance_ratio_ = eigenvalues[idx[:self.n_components]] / np.sum(eigenvalues)

    def transform(self, X):
        """ Project new data using stored eigenvectors """
        return np.dot(X - self.mean_, self.components_)

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Apply LDA
lda = LDA(n_components=2)
lda.fit(X, y)
X_lda = lda.transform(X)

# Plot the transformed data
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.title("LDA of Iris Dataset (Implemented from Scratch)")
plt.colorbar(label='Target Class')
plt.show()

# Print explained variance ratio
print("Explained Variance Ratio:", lda.explained_variance_ratio_)
