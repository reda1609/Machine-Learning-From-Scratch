import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class DBSCAN:

    def __init__(self, eps=0.5, MinPts=10):
        self.eps = eps
        self.MinPts = MinPts
        self.labels = None

    def _get_neighbours(self, X, idx):
        """ Find all points within eps distance of point X[idx] """
        distances = np.linalg.norm(X - X[idx], axis=1)  # Euclidean distance
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id):
        """ Expand the cluster by adding all points within eps distance of any point in the cluster """
        self.labels[idx] = cluster_id
        queue = list(neighbors)     # Queue of points to process
        while queue:
            current = queue.pop()
            if self.labels[current] == -1:  # If no label, assign current cluster id (Border Point)
                self.labels[current] = cluster_id
            elif self.labels[current] >= 0:  # Already part of a cluster
                continue
            self.labels[current] = cluster_id  # Assign cluster
            new_neighbors = self._get_neighbours(X, current)
            if len(new_neighbors) >= self.MinPts:  # Core point
                queue.extend(new_neighbors)  # Add neighbors to the queue
        

    def fit(self, X):
        no_samples = X.shape[0]
        self.labels = np.full(no_samples, -1)
        cluster_id = 0
        for i in range(no_samples):
            if self.labels[i] != -1:
                continue    # already visited
            neighbours = self._get_neighbours(X, i)
            if len(neighbours) < self.MinPts:
                continue    # noise
            # Core point, start a new cluster
            self._expand_cluster(X, i, neighbours, cluster_id)
            cluster_id += 1


# Generate sample dataset
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Run DBSCAN
dbscan = DBSCAN(eps=0.2, MinPts=5)
dbscan.fit(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels, cmap='viridis', edgecolors='k')
plt.title("DBSCAN Clustering (Implemented from Scratch)")
plt.show()