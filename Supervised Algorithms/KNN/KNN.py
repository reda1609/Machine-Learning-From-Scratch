import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            nearest_labels = self._predict(x)
            predictions.append(Counter(nearest_labels).most_common(1)[0][0])
        return predictions

    def _predict(self, x):
        distances = self._calculate_distance(x, self.X_train)
        indices = np.argsort(distances)[:self.k]
        return [self.y_train[i] for i in indices]

    def _calculate_distance(self, X, y):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((X - y) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(X - y), axis=1)
        else:
            raise ValueError("Invalid distance metric")
        

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Flatten the grid and predict
    Z = np.array(model.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # Plot contour and data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("KNN Decision Boundary")
    plt.show()

# Generate synthetic dataset (binary classification)
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features for better distance calculation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train KNN
knn = KNN(k=5, distance_metric='euclidean')
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

plot_decision_boundary(knn, X_test, y_test)