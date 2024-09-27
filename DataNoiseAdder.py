import numpy as np
import pandas as pd
from scipy.stats import norm
from copulas.multivariate import GaussianMultivariate

class DataNoiseAdder:
    def __init__(self, df):
        self.df = df.copy()

    def add_concept_shift(self, shift_type="boundary_shift", shift_params=None):
        """
        Simulate concept shift by changing the relationship between features and the target.
        - shift_type: Type of shift ("boundary_shift", "label_flip").
        - shift_params: Parameters controlling the shift.
        """
        if shift_type == "boundary_shift":
            # Change the decision boundary (example for binary classification)
            feature_col = shift_params.get("feature_col", self.df.columns[0])
            threshold_shift = shift_params.get("threshold_shift", 0.5)
            
            # Original threshold is usually some median or mean
            original_threshold = np.median(self.df[feature_col])
            new_threshold = original_threshold + threshold_shift
            
            print(f"Original threshold: {original_threshold}, New threshold: {new_threshold}")
            
            # Reassign labels based on the new, shifted threshold
            self.df[self.target_col] = np.where(self.df[feature_col] > new_threshold, 1, 0)
        
        elif shift_type == "label_flip":
            # Randomly flip a percentage of labels
            flip_prob = shift_params.get("flip_prob", 0.1)
            flip_mask = np.random.rand(len(self.df)) < flip_prob
            self.df.loc[flip_mask, self.target_col] = 1 - self.df.loc[flip_mask, self.target_col]
        
        return self.df

    def add_covariate_shift(self, columns, shift_type="scaling", shift_params=None):
        """
        Simulate covariate shift by changing the input distribution for specified columns.
        - shift_type: Type of shift ("scaling", "distribution", "noise").
        - shift_params: Parameters controlling the shift.
        """
        if shift_type == "scaling":
            # Apply a scaling factor to the selected features
            scale_factor = shift_params.get("scale_factor", 1.2)
            for col in columns:
                self.df[col] *= scale_factor
        elif shift_type == "distribution":
            # Sample from a different distribution (e.g., Uniform instead of Gaussian)
            dist_type = shift_params.get("dist_type", "uniform")
            if dist_type == "uniform":
                for col in columns:
                    self.df[col] = np.random.uniform(np.min(self.df[col]), np.max(self.df[col]), size=len(self.df))
            elif dist_type == "exponential":
                for col in columns:
                    self.df[col] = np.random.exponential(scale=np.mean(self.df[col]), size=len(self.df))
        elif shift_type == "noise":
            # Add noise from a different distribution (e.g., Uniform noise instead of Gaussian)
            noise_type = shift_params.get("noise_type", "uniform")
            noise_level = shift_params.get("noise_level", 0.1)
            for col in columns:
                if noise_type == "uniform":
                    self.df[col] += np.random.uniform(-noise_level, noise_level, size=len(self.df))
                elif noise_type == "exponential":
                    self.df[col] += np.random.exponential(scale=noise_level, size=len(self.df))
        return self.df

    def add_gaussian_noise(self, columns, std_dev=1.0):
        for col in columns:
            self.df[col] += np.random.normal(0, std_dev, size=len(self.df))
        return self.df

    def add_laplace_noise(self, columns, scale=1.0):
        for col in columns:
            self.df[col] += np.random.laplace(0, scale, size=len(self.df))
        return self.df

    def add_uniform_noise(self, columns, noise_range=(-1, 1)):
        for col in columns:
            self.df[col] += np.random.uniform(noise_range[0], noise_range[1], size=len(self.df))
        return self.df

    def add_multiplicative_noise(self, columns, factor_range=(0.9, 1.1)):
        for col in columns:
            self.df[col] *= np.random.uniform(factor_range[0], factor_range[1], size=len(self.df))
        return self.df

    def add_dropout_noise(self, columns, dropout_prob=0.1, null_value=0):
        for col in columns:
            mask = np.random.rand(len(self.df)) < dropout_prob
            self.df.loc[mask, col] = null_value
        return self.df

    def add_categorical_noise(self, columns, noise_prob=0.1):
        for col in columns:
            unique_vals = self.df[col].unique()
            mask = np.random.rand(len(self.df)) < noise_prob
            random_categories = np.random.choice(unique_vals, size=mask.sum())
            self.df.loc[mask, col] = random_categories
        return self.df

    def add_ordinal_noise(self, columns, shift_range=(-1, 1), min_val=1, max_val=5):
        for col in columns:
            shift = np.random.randint(shift_range[0], shift_range[1] + 1, size=len(self.df))
            self.df[col] = np.clip(self.df[col] + shift, min_val, max_val)
        return self.df

    def add_jitter(self, columns, jitter_amount=0.01):
        for col in columns:
            self.df[col] += np.random.uniform(-jitter_amount, jitter_amount, size=len(self.df))
        return self.df

    def add_differential_privacy_noise(self, columns, epsilon=1.0, sensitivity=1.0):
        for col in columns:
            scale = sensitivity / epsilon
            self.df[col] += np.random.laplace(0, scale, size=len(self.df))
        return self.df

    def add_multivariate_gaussian_noise(self, columns, mean=None, covariance=None):
        """
        Add correlated Gaussian noise to specified columns based on the provided covariance matrix.
        """
        if mean is None:
            mean = [0] * len(columns)
        if covariance is None:
            covariance = np.eye(len(columns))

        # Generate correlated Gaussian noise
        correlated_noise = np.random.multivariate_normal(mean, covariance, size=len(self.df))

        # Add noise to the specified columns
        for i, col in enumerate(columns):
            self.df[col] += correlated_noise[:, i]

        return self.df

    def add_cholesky_correlated_noise(self, columns, covariance=None):
        """
        Add correlated noise using Cholesky decomposition to specified columns.
        """
        if covariance is None:
            covariance = np.eye(len(columns))

        # Cholesky decomposition
        L = np.linalg.cholesky(covariance)

        # Generate independent Gaussian noise
        independent_noise = np.random.normal(0, 1, size=(len(self.df), len(columns)))

        # Introduce correlations
        correlated_noise = independent_noise @ L.T

        # Add noise to the specified columns
        for i, col in enumerate(columns):
            self.df[col] += correlated_noise[:, i]

        return self.df

    def add_correlated_uniform_noise(self, columns, noise_range=(-1, 1), covariance=None):
        """
        Add correlated uniform noise to specified columns based on the provided covariance matrix.
        """
        if covariance is None:
            covariance = np.eye(len(columns))

        # Cholesky decomposition
        L = np.linalg.cholesky(covariance)

        # Generate independent uniform noise
        independent_noise = np.random.uniform(noise_range[0], noise_range[1], size=(len(self.df), len(columns)))

        # Introduce correlations
        correlated_noise = independent_noise @ L.T

        # Add noise to the specified columns
        for i, col in enumerate(columns):
            self.df[col] += correlated_noise[:, i]

        return self.df

    def add_ar1_noise(self, columns, phi=0.8, noise_std=1.0):
        """
        Add AR(1) correlated noise to time series data for specified columns.
        """
        def generate_ar1_noise(data_len, phi, noise_std):
            noise = [np.random.normal(0, noise_std)]  # Initial noise term
            for i in range(1, data_len):
                next_noise = phi * noise[-1] + np.random.normal(0, noise_std)
                noise.append(next_noise)
            return np.array(noise)

        # Add AR(1) noise to each column
        for col in columns:
            ar1_noise = generate_ar1_noise(len(self.df), phi, noise_std)
            self.df[col] += ar1_noise

        return self.df

    def add_copula_based_noise(self, columns):
        """
        Add correlated noise using a copula based on the specified columns.
        """
        # Fit a Gaussian copula to the data in the specified columns
        copula = GaussianMultivariate()
        copula.fit(self.df[columns])

        # Sample noise from the copula
        sampled_noise = copula.sample(len(self.df))

        # Add the sampled noise to the original data
        for col in columns:
            self.df[col] += sampled_noise[col]

        return self.df

    def apply_noise(self, noise_type, columns, **kwargs):
        if noise_type == 'gaussian':
            return self.add_gaussian_noise(columns, **kwargs)
        elif noise_type == 'laplace':
            return self.add_laplace_noise(columns, **kwargs)
        elif noise_type == 'uniform':
            return self.add_uniform_noise(columns, **kwargs)
        elif noise_type == 'multiplicative':
            return self.add_multiplicative_noise(columns, **kwargs)
        elif noise_type == 'dropout':
            return self.add_dropout_noise(columns, **kwargs)
        elif noise_type == 'categorical':
            return self.add_categorical_noise(columns, **kwargs)
        elif noise_type == 'ordinal':
            return self.add_ordinal_noise(columns, **kwargs)
        elif noise_type == 'jitter':
            return self.add_jitter(columns, **kwargs)
        elif noise_type == 'differential_privacy':
            return self.add_differential_privacy_noise(columns, **kwargs)
        elif noise_type == 'multivariate_gaussian':
            return self.add_multivariate_gaussian_noise(columns, **kwargs)
        elif noise_type == 'cholesky':
            return self.add_cholesky_correlated_noise(columns, **kwargs)
        elif noise_type == 'correlated_uniform':
            return self.add_correlated_uniform_noise(columns, **kwargs)
        elif noise_type == 'ar1':
            return self.add_ar1_noise(columns, **kwargs)
        elif noise_type == 'copula':
            return self.add_copula_based_noise(columns, **kwargs)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

if __name__ == '__main__':
    # Example usage
    df = pd.DataFrame({
        'temperature': [20, 21, 19, 22, 20],
        'humidity': [0.7, 0.65, 0.72, 0.75, 0.68],
        'rating': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'C', 'B']
    })

    noise_adder = DataNoiseAdder(df)

    # Adding multivariate Gaussian noise to 'temperature' and 'humidity'
    cov_matrix = [[1.0, 0.5], [0.5, 1.0]]  # Correlation between temperature and humidity
    noisy_df = noise_adder.apply_noise('multivariate_gaussian', ['temperature', 'humidity'], covariance=cov_matrix)

    # Adding AR(1) correlated noise to 'temperature'
    noisy_df = noise_adder.apply_noise('ar1', ['temperature'], phi=0.8, noise_std=0.5)

    print(noisy_df)