import numpy as np
from DecisionTree import DecisionTree


class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alphas = []  # Store the weights of weak classifiers
        self.weak_classifiers = []  # Store the weak classifiers

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initialize weights equally
        weights = np.ones(n_samples) / n_samples
        for t in range(self.n_estimators):
            # Train a weak classifier (decision stump)
            stump = DecisionTree(max_depth=1)
            stump.fit(X, y, sample_weight=weights)
            self.weak_classifiers.append(stump)
            # Predict and compute weighted error
            y_pred = stump.predict(X)
            # Use (0-1) Loss
            misclassified = (y_pred != y)
            error = np.sum(weights * misclassified) / np.sum(weights)
            # Compute alpha (classifier weight) or Performance
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.alphas.append(alpha)
            # Update weights
            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)  # Normalize weights

    def predict(self, X):
        # Aggregate predictions from all weak classifiers
        final_prediction = np.zeros(X.shape[0])
        for alpha, classifier in zip(self.alphas, self.weak_classifiers):
            final_prediction += alpha * classifier.predict(X)
        return np.sign(final_prediction)