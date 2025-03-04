import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _compute_high_dim_affinities(self, X, tol=1e-5):
        """Compute pairwise similarities in high-dimensional space using Gaussian distribution."""
        n = X.shape[0]
        P = np.zeros((n, n))
        sigma = np.ones(n)  # Initial sigma for each point

        def binary_search_sigma(i):
            """Binary search to find sigma for a given perplexity."""
            sigma_min, sigma_max = 1e-20, 1e5
            target_entropy = np.log(self.perplexity)

            for _ in range(50):  # Limit iterations
                distances = np.delete(cdist([X[i]], X, 'sqeuclidean')[0], i)
                probs = np.exp(-distances / (2 * sigma[i] ** 2))
                probs /= np.sum(probs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))

                if np.abs(entropy - target_entropy) < tol:
                    break
                if entropy > target_entropy:
                    sigma_max = sigma[i]
                else:
                    sigma_min = sigma[i]
                sigma[i] = (sigma_min + sigma_max) / 2
            
            return np.concatenate([probs[:i], [0], probs[i:]])

        for i in range(n):
            P[i] = binary_search_sigma(i)

        return (P + P.T) / (2 * n)  # Symmetric P distribution

    def _compute_low_dim_affinities(self, Y):
        """Compute pairwise similarities in low-dimensional space using Student's t-distribution."""
        n = Y.shape[0]
        distances = cdist(Y, Y, 'sqeuclidean')
        Q = (1 + distances) ** -1
        np.fill_diagonal(Q, 0)
        return Q / np.sum(Q)

    def _compute_gradient(self, P, Q, Y):
        """Compute the gradient of KL divergence."""
        PQ_diff = P - Q
        grad = np.zeros_like(Y)

        for i in range(Y.shape[0]):
            grad[i] = 4 * np.sum((PQ_diff[:, i][:, np.newaxis] * (Y[i] - Y)), axis=0)

        return grad

    def fit_transform(self, X):
        """Compute t-SNE embedding of X."""
        # Compute P (high-dimensional similarities)
        P = self._compute_high_dim_affinities(X)

        # Initialize Y randomly in low-dimensional space
        Y = np.random.randn(X.shape[0], self.n_components)

        # Initialize gradient descent parameters
        momentum = 0.9
        velocity = np.zeros_like(Y)

        for iter in range(self.n_iter):
            # Compute Q (low-dimensional similarities)
            Q = self._compute_low_dim_affinities(Y)

            # Compute gradient
            grad = self._compute_gradient(P, Q, Y)

            # Update Y with momentum
            velocity = momentum * velocity - self.learning_rate * grad
            Y += velocity

            # Print progress
            if iter % 100 == 0:
                kl_divergence = np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))
                print(f"Iteration {iter}: KL Divergence = {kl_divergence:.4f}")

        return Y

# Example Usage with Digits Dataset
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=500)
Y_tsne = tsne.fit_transform(X_scaled)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(Y_tsne[:, 0], Y_tsne[:, 1], c=y, cmap='jet', alpha=0.7)
plt.colorbar()
plt.title("t-SNE Visualization (from Scratch)")
plt.show()
