import numpy as np
from sklearn.utils import resample
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
#from ctgan import CTGAN

class SyntheticDataGenerator:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.models = {}

    def random_sampling(self, n_samples: int) -> tuple:
        """Generate synthetic data using random sampling with replacement."""
        x_resampled, y_resampled = resample(self.x_train, self.y_train, n_samples=n_samples, replace=True)
        return x_resampled, y_resampled

    def interpolate(self, n_samples: int) -> tuple:
        """Generate synthetic data using interpolation between existing samples."""
        synthetic_x = []
        synthetic_y = []
        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(len(self.x_train), 2, replace=False)
            interpolated_x = (self.x_train[idx1] + self.x_train[idx2]) / 2
            interpolated_y = (self.y_train[idx1] + self.y_train[idx2]) / 2  # Average target, adjust as needed
            synthetic_x.append(interpolated_x)
            synthetic_y.append(interpolated_y)
        return np.array(synthetic_x), np.array(synthetic_y)

    def gaussian_mixture(self, n_samples: int) -> tuple:
        """Generate synthetic data using Gaussian Mixture Models."""
        gmm = GaussianMixture(n_components=3)  # Adjust number of components as needed
        gmm.fit(self.x_train)
        synthetic_x, _ = gmm.sample(n_samples)
        # For the target, we can randomly sample from y_train or create a model-based approach
        synthetic_y = np.random.choice(self.y_train, n_samples)
        return synthetic_x, synthetic_y

    def gaussian_copula(self, n_samples: int) -> tuple:
        """Generate synthetic data using Gaussian Copula."""
        model = GaussianCopula()
        model.fit(np.hstack((self.x_train, self.y_train.reshape(-1, 1))))  # Combine features and target
        synthetic_data = model.sample(n_samples)
        synthetic_x = synthetic_data[:, :-1]  # All but last column as features
        synthetic_y = synthetic_data[:, -1]   # Last column as target
        return synthetic_x, synthetic_y

    def ctgan(self, n_samples: int) -> tuple:
        """Generate synthetic data using Conditional GAN (CTGAN)."""
        model = CTGAN()
        # Concatenate features and target into one array for CTGAN
        data = np.hstack((self.x_train, self.y_train.reshape(-1, 1)))
        model.fit(data)
        synthetic_data = model.sample(n_samples)
        synthetic_x = synthetic_data[:, :-1]
        synthetic_y = synthetic_data[:, -1]
        return synthetic_x, synthetic_y

    def decision_tree(self, n_samples: int) -> tuple:
        """Generate synthetic data using a decision tree model."""
        dt = DecisionTreeClassifier()
        dt.fit(self.x_train, self.y_train)
        synthetic_x = []
        synthetic_y = []
        
        for _ in range(n_samples):
            random_row = np.random.rand(self.x_train.shape[1])  # Generate random feature vector
            synthetic_y_pred = dt.predict([random_row])         # Predict target using the decision tree
            synthetic_x.append(random_row)
            synthetic_y.append(synthetic_y_pred[0])             # Append prediction
            
        return np.array(synthetic_x), np.array(synthetic_y)

    def bootstrapping(self, n_samples: int) -> tuple:
        """Generate synthetic data using bootstrapping."""
        return resample(self.x_train, self.y_train, n_samples=n_samples, replace=True)
