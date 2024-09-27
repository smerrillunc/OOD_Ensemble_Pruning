import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.mixture import GaussianMixture
#from sdv.tabular import GaussianCopula, CTGAN
from sklearn.tree import DecisionTreeClassifier

class SyntheticDataGenerator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models = {}

    def random_sampling(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using random sampling with replacement."""
        return resample(self.data, n_samples=n_samples, replace=True)

    def interpolate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using interpolation between existing samples."""
        synthetic_data = []
        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(self.data.index, 2, replace=False)
            interpolated_row = (self.data.loc[idx1] + self.data.loc[idx2]) / 2
            synthetic_data.append(interpolated_row)
        return pd.DataFrame(synthetic_data, columns=self.data.columns)

    def gaussian_mixture(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using Gaussian Mixture Models."""
        gmm = GaussianMixture(n_components=3)  # Adjust number of components as needed
        gmm.fit(self.data)
        synthetic_data = gmm.sample(n_samples)[0]
        return pd.DataFrame(synthetic_data, columns=self.data.columns)

    def gaussian_copula(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using Gaussian Copula."""
        model = GaussianCopula()
        model.fit(self.data)
        synthetic_data = model.sample(n_samples)
        return synthetic_data

    def ctgan(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using Conditional GAN (CTGAN)."""
        model = CTGAN()
        model.fit(self.data)
        synthetic_data = model.sample(n_samples)
        return synthetic_data

    def decision_tree(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using a decision tree model."""
        features = self.data.drop(columns=self.data.select_dtypes(include='object').columns)
        target = self.data.select_dtypes(include='object')
        
        dt = DecisionTreeClassifier()
        dt.fit(features, target)
        synthetic_data = []
        
        for _ in range(n_samples):
            random_row = np.random.rand(len(features.columns))
            synthetic_row = dt.predict([random_row])
            synthetic_data.append(np.concatenate((random_row, synthetic_row)))
        return pd.DataFrame(synthetic_data, columns=self.data.columns)

    def bootstrapping(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using bootstrapping."""
        return resample(self.data, n_samples=n_samples, replace=True)

# Example Usage
if __name__ == "__main__":
    # Load or create your DataFrame here
    data = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.randint(0, 100, size=100),
        'C': np.random.choice(['X', 'Y', 'Z'], size=100)
    })

    generator = SyntheticDataGenerator(data)
    
    synthetic_random = generator.random_sampling(10)
    print("Synthetic Random Sampling:\n", synthetic_random)
    
    synthetic_interp = generator.interpolate(10)
    print("Synthetic Interpolation:\n", synthetic_interp)
    
    synthetic_gmm = generator.gaussian_mixture(10)
    print("Synthetic GMM:\n", synthetic_gmm)
    
    synthetic_gaussian_copula = generator.gaussian_copula(10)
    print("Synthetic Gaussian Copula:\n", synthetic_gaussian_copula)
    
    synthetic_ctgan = generator.ctgan(10)
    print("Synthetic CTGAN:\n", synthetic_ctgan)
    
    synthetic_decision_tree = generator.decision_tree(10)
    print("Synthetic Decision Tree:\n", synthetic_decision_tree)