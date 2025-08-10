import torch


class NaiveBayesClassifier:
    def __init__(self):
        # We will store the learned probabilities and statistics here.
        self.priors = None
        self.means = None
        self.stds = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_classes = len(torch.unique(y))

        # Initialize tensors to store the means and standard deviations for each class and feature.
        self.means = torch.zeros((self.n_classes, n_features), dtype=torch.float32)
        self.stds = torch.zeros((self.n_classes, n_features), dtype=torch.float32)
        self.priors = torch.zeros(self.n_classes, dtype=torch.float32)

        # Calculate the prior probability for each class and the mean and std for each feature within each class.
        for c in range(self.n_classes):
            X_c = X[y == c]
            self.priors[c] = float(len(X_c)) / n_samples
            self.means[c, :] = torch.mean(X_c, axis=0)
            self.stds[c, :] = torch.std(X_c, axis=0)

    def _gaussian_density(self, x, mean, std):
        # Calculate the probability density function for a Gaussian distribution.
        numerator = torch.exp(-0.5 * ((x - mean) / std) ** 2)
        denominator = torch.sqrt(2 * torch.pi * std**2)
        return numerator / denominator

    def predict(self, X):
        # Calculate the posterior probability for each class and for each data point in X.
        n_samples, _ = X.shape
        posteriors = torch.zeros((n_samples, self.n_classes))

        for c in range(self.n_classes):
            # Calculate the log of the prior probability.
            log_prior = torch.log(self.priors[c])
            # Calculate the log of the likelihood for each feature.
            log_likelihood = torch.sum(
                torch.log(self._gaussian_density(X, self.means[c, :], self.stds[c, :])),
                axis=1,
            )
            # Combine them to get the posterior. Using log probabilities avoids numerical instability.
            posteriors[:, c] = log_prior + log_likelihood

        # The predicted class is the one with the highest log posterior probability.
        return torch.argmax(posteriors, dim=1)


# 1. Prepare the Data using PyTorch tensors.
# X: features (height, weight)
# y: class labels (0 for male, 1 for female)
X = torch.tensor(
    [[180, 80], [175, 75], [178, 85], [160, 55], [165, 60], [158, 50]],
    dtype=torch.float32,
)
y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

# 2. Create and Train the classifier.
model = NaiveBayesClassifier()
model.fit(X, y)

# 3. Make a prediction for a new person.
new_person = torch.tensor([[170, 65]], dtype=torch.float32)
prediction = model.predict(new_person)

# 4. Print the result.
print(f"Features of the new person: {new_person[0].tolist()} cm and kg.")
print(f"Predicted class label (0=Male, 1=Female): {prediction.item()}")
