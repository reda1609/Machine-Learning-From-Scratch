import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def _euclidean_distance(self, datapoint, centroids):
        return np.sqrt(np.sum((datapoint - centroids) ** 2, axis=1))

    def _initialize_centroids(self, X):
        centroid_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[centroid_indices]

    def _assign_clusters(self, X):
        distances = np.array([self._euclidean_distance(x, self.centroids) for x in X])
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, clusters):
        for i in range(self.k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)

    def fit(self, X, max_iter=100):
        self._initialize_centroids(X)
        for _ in range(max_iter):
            clusters = self._assign_clusters(X)
            prev_centroids = self.centroids.copy()
            self._update_centroids(X, clusters)
            if np.all(prev_centroids == self.centroids):  # Check for convergence
                break

    def predict(self, X_new):
        distances = self._euclidean_distance(X_new, self.centroids)
        return np.argmin(distances)

# Example Usage
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 points in 2D space

kmeans = KMeansClustering(k=3)
kmeans.fit(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=kmeans._assign_clusters(X), cmap='viridis', marker='o')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.legend()
plt.show()
