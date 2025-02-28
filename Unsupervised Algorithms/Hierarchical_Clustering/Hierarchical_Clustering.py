import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs


class HierarchicalClustering:

    def __init__(self, linkage_type='complete', verbose=False):
        """
        Initializes the Hierarchical Clustering model.

        Parameters:
        - linkage_type (str): Type of linkage ('complete', 'single', or 'average').
        - verbose (bool): If True, prints debug information.
        """
        if linkage_type not in {'complete', 'single', 'average'}:
            raise ValueError("linkage_type must be 'complete', 'single', or 'average'")
        self.linkage_type = linkage_type
        self.verbose = verbose
    
    def argmin(self, D):
        """ Finds the minimum value (excluding the diagonal) in a distance matrix. """
        np.fill_diagonal(D, np.inf)  # Ignore diagonal values
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        return D[min_idx], min_idx[0], min_idx[1]
    
    def cluster_distance(self, X, cluster_members):
        """ Computes the Euclidean distance between clusters based on the linkage method. """
        keys = list(cluster_members.keys())
        n = len(keys)
        Distance = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dists = euclidean_distances(X[cluster_members[keys[i]]], X[cluster_members[keys[j]]])
                
                if self.linkage_type == 'complete':
                    Distance[i, j] = np.max(dists)  # Max distance
                elif self.linkage_type == 'single':
                    Distance[i, j] = np.min(dists)  # Min distance
                elif self.linkage_type == 'average':
                    Distance[i, j] = np.mean(dists)  # Mean distance (average linkage)

        return Distance
    
    def fit(self, X):
        """
        Performs hierarchical clustering and generates a dendrogram.

        Parameters:
        - X (np.array): Dataset of shape (n_samples, n_features).

        Returns:
        - Z (np.array): Linkage matrix.
        """
        self.n_samples = X.shape[0]
        cluster_members = {i: [i] for i in range(self.n_samples)}   # Start with each sample in its own cluster
        Z = np.zeros((self.n_samples - 1, 4))

        for i in range(self.n_samples - 1):
            if self.verbose:
                print(f"\nIteration {i+1}/{self.n_samples-1}")

            D = self.cluster_distance(X, cluster_members)
            min_dist, x_idx, y_idx = self.argmin(D)

            keys = list(cluster_members.keys())
            x, y = keys[x_idx], keys[y_idx]

            Z[i] = [x, y, min_dist, len(cluster_members[x]) + len(cluster_members[y])]

            cluster_members[self.n_samples + i] = cluster_members[x] + cluster_members[y]
            del cluster_members[x]
            del cluster_members[y]

        self.Z = Z
        return self.Z
    
    def predict(self, n_clusters):
        """ Assigns labels to data points based on the hierarchical tree. """
        labels = np.zeros(self.n_samples, dtype=int)
        cluster_members = {i: [i] for i in range(self.n_samples)}

        for i in range(self.n_samples - n_clusters):
            x, y = int(self.Z[i, 0]), int(self.Z[i, 1])
            cluster_members[self.n_samples + i] = cluster_members[x] + cluster_members[y]
            del cluster_members[x]
            del cluster_members[y]

        for cluster_label, members in enumerate(cluster_members.values()):
            labels[members] = cluster_label

        return labels
    
    def plot_dendrogram(self):
        """ Plots the dendrogram using the linkage matrix. """
        plt.figure(figsize=(10, 5))
        dendrogram(self.Z)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.show()


# === TESTING ON A SYNTHETIC DATASET ===
if __name__ == "__main__":
    # Generate synthetic data
    X, _ = make_blobs(n_samples=30, centers=4, random_state=42, cluster_std=1.5)

    # Apply hierarchical clustering
    model = HierarchicalClustering(linkage_type='average', verbose=True)
    Z = model.fit(X)

    # Plot the dendrogram
    model.plot_dendrogram()

    # Predict clusters for n=3
    labels = model.predict(n_clusters=4)

    # Visualize the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, edgecolors='k')
    plt.title("Hierarchical Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
