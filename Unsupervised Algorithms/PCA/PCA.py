import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """ Compute PCA on the dataset """
        # Standardize the dataset
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_standardized = (X - self.mean) / self.std

        # Compute covariance matrix
        cov_matrix = np.cov(X_standardized.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # Correct sorting

        # Store the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        
        # Compute explained variance ratio
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def transform(self, X):
        """ Apply the dimensionality reduction to the dataset """
        # Standardize the dataset
        X_standardized = (X - self.mean) / self.std
        # Apply the transformation
        return X_standardized @ self.components  # Correct projection


# Load dataset
data = load_iris()
X = data.data  # Features

# Apply PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Plot the transformed data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target, cmap='viridis', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset (Implemented from Scratch)")
plt.colorbar(label='Target Class')
plt.show()

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
