import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:

    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _initialize_centroids(self, X):
        np.random.seed(42)  # Ensuring reproducibility
        centroid_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[centroid_indices]

    def _assign_clusters(self, X):
        distances = np.array([[self._euclidean_distance(x, centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, X):
        self._initialize_centroids(X)
        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X_new):
        distances = np.array([self._euclidean_distance(X_new, centroid) for centroid in self.centroids])
        return np.argmin(distances)

    def inertia(self, X):  # WCSS Calculation
        return np.sum([self._euclidean_distance(X[i], self.centroids[self.labels[i]])**2 for i in range(X.shape[0])])

    def silhouette_score(self, X):
        unique_clusters = np.unique(self.labels)
        silhouette_scores = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            own_cluster = self.labels[i]
            own_cluster_points = X[self.labels == own_cluster]
            
            # Compute a(i): Mean intra-cluster distance
            if len(own_cluster_points) > 1:
                a_i = np.mean([self._euclidean_distance(X[i], p) for p in own_cluster_points if not np.array_equal(p, X[i])])
            else:
                a_i = 0

            # Compute b(i): Mean nearest-cluster distance
            b_i = float("inf")
            for cluster in unique_clusters:
                if cluster != own_cluster:
                    other_cluster_points = X[self.labels == cluster]
                    mean_distance = np.mean([self._euclidean_distance(X[i], p) for p in other_cluster_points])
                    b_i = min(b_i, mean_distance)

            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        
        return np.mean(silhouette_scores)
    
    @staticmethod
    def elbow_method(X, max_k=10):
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeansClustering(k)
            kmeans.fit(X)
            wcss.append(kmeans.inertia(X))
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS (Inertia)')
        plt.title('Elbow Method for Optimal k')
        plt.show()

# Example Usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 points in 2D space
    
    print("Running Elbow Method to determine optimal k...")
    KMeansClustering.elbow_method(X, max_k=10)
    
    kmeans = KMeansClustering(k=3)
    kmeans.fit(X)
    
    print(f"WCSS (Inertia): {kmeans.inertia(X):.4f}")
    print(f"Silhouette Score: {kmeans.silhouette_score(X):.4f}")
