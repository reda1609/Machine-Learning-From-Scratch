from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

class NaiveBayes:
    def __init__(self, model_type="gaussian"):
        if model_type not in ["gaussian", "multinomial"]:
            raise ValueError("Invalid model type. Choose 'gaussian' or 'multinomial'.")
        self.model_type = model_type

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._priors = np.zeros(n_classes, dtype=np.float64)

        if self.model_type == "gaussian":
            self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
            self._var = np.zeros((n_classes, n_features), dtype=np.float64)

            for idx, c in enumerate(self._classes):
                X_c = X[y == c]
                self._mean[idx, :] = X_c.mean(axis=0)
                self._var[idx, :] = X_c.var(axis=0) + 1e-9  # Stability fix
                self._priors[idx] = X_c.shape[0] / float(n_samples)

        elif self.model_type == "multinomial":
            self._feature_probs = np.zeros((n_classes, n_features), dtype=np.float64)

            for idx, c in enumerate(self._classes):
                X_c = X[y == c]
                # Compute probabilities (Laplace smoothing to avoid zero probabilities)
                self._feature_probs[idx, :] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)
                self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])

            if self.model_type == "gaussian":
                likelihood = np.sum(np.log(self._pdf(idx, x) + 1e-9))  # Avoid log(0)
            elif self.model_type == "multinomial":
                likelihood = np.sum(x * np.log(self._feature_probs[idx] + 1e-9))  # Avoid log(0)

            posteriors.append(prior + likelihood)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    # Gaussian Naïve Bayes test
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    nb_gaussian = NaiveBayes(model_type="gaussian")
    nb_gaussian.fit(X_train, y_train)
    predictions = nb_gaussian.predict(X_test)

    print("Gaussian Naïve Bayes Accuracy:", accuracy(y_test, predictions))

    # Multinomial Naïve Bayes test (simulated count data)
    np.random.seed(123)
    X_multinomial = np.random.randint(0, 10, (1000, 10))  # Simulated count-based features
    y_multinomial = np.random.randint(0, 2, 1000)

    X_train, X_test, y_train, y_test = train_test_split(X_multinomial, y_multinomial, test_size=0.2, random_state=123)

    nb_multinomial = NaiveBayes(model_type="multinomial")
    nb_multinomial.fit(X_train, y_train)
    predictions = nb_multinomial.predict(X_test)

    print("Multinomial Naïve Bayes Accuracy:", accuracy(y_test, predictions))