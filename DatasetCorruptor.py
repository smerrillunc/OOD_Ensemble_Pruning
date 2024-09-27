import pandas as pd
import numpy as np
import random

class DatasetCorruptor:
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def clip_values(self, column, min_value=None, max_value=None):
        """Clip numeric values in a column to a specified range."""
        if min_value is not None:
            self.dataframe[column] = self.dataframe[column].clip(lower=min_value)
        if max_value is not None:
            self.dataframe[column] = self.dataframe[column].clip(upper=max_value)

    def random_value_replacement(self, column, replacement_fraction=0.1, new_value_range=(0, 100)):
        """Replace a fraction of numeric values with random values from a specified range."""
        n_replacements = int(len(self.dataframe) * replacement_fraction)
        indices = random.sample(range(len(self.dataframe)), n_replacements)
        self.dataframe.loc[indices, column] = np.random.uniform(new_value_range[0], new_value_range[1], size=n_replacements)

    def introduce_outliers(self, column, n_outliers=5, outlier_range=(100, 200)):
        """Introduce outliers in a numeric column."""
        outlier_indices = random.sample(range(len(self.dataframe)), n_outliers)
        self.dataframe.loc[outlier_indices, column] = np.random.uniform(outlier_range[0], outlier_range[1], size=n_outliers)

    def random_categorical_replacement(self, column, replacement_fraction=0.1):
        """Replace a fraction of categorical values with random values from the same column."""
        n_replacements = int(len(self.dataframe) * replacement_fraction)
        unique_categories = self.dataframe[column].unique()
        indices = random.sample(range(len(self.dataframe)), n_replacements)
        self.dataframe.loc[indices, column] = np.random.choice(unique_categories, size=n_replacements)

    def introduce_label_noise(self, column, noise_fraction=0.1):
        """Introduce noise in categorical labels by changing them randomly."""
        n_changes = int(len(self.dataframe) * noise_fraction)
        unique_categories = self.dataframe[column].unique()
        indices = random.sample(range(len(self.dataframe)), n_changes)
        for idx in indices:
            original_value = self.dataframe.at[idx, column]
            other_categories = unique_categories[unique_categories != original_value]
            self.dataframe.at[idx, column] = np.random.choice(other_categories)

    def swap_categories(self, column, swap_fraction=0.05):
        """Swap values between two categories for a specified fraction."""
        n_swaps = int(len(self.dataframe) * swap_fraction)
        unique_categories = self.dataframe[column].unique()
        if len(unique_categories) < 2:
            raise ValueError("Not enough unique categories to swap.")
        
        category_a, category_b = random.sample(unique_categories.tolist(), 2)
        indices_a = self.dataframe[self.dataframe[column] == category_a].sample(n=n_swaps).index
        indices_b = self.dataframe[self.dataframe[column] == category_b].sample(n=n_swaps).index
        
        self.dataframe.loc[indices_a, column] = category_b
        self.dataframe.loc[indices_b, column] = category_a

    def shuffle_column(self, column):
        """Shuffle values within a numeric or categorical column."""
        shuffled = self.dataframe[column].sample(frac=1).reset_index(drop=True)
        self.dataframe[column] = shuffled

    def delete_random_rows(self, fraction=0.05):
        """Randomly delete a fraction of rows in the dataset."""
        n_deletions = int(len(self.dataframe) * fraction)
        indices_to_delete = random.sample(range(len(self.dataframe)), n_deletions)
        self.dataframe.drop(indices_to_delete, inplace=True)

    def corrupt(self, corruption_config):
        """Corrupt the dataset based on a provided configuration."""
        for action, params in corruption_config.items():
            if hasattr(self, action):
                getattr(self, action)(**params)
            else:
                print(f"Action '{action}' not recognized.")

# Example Usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'numeric_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'categorical_column': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(data)
    corruptor = DatasetCorruptor(df)

    # Define corruption actions
    corruption_config = {
        'add_noise': {'column': 'numeric_column', 'noise_level': 0.2},
        'clip_values': {'column': 'numeric_column', 'min_value': 0, 'max_value': 10},
        'random_value_replacement': {'column': 'numeric_column', 'replacement_fraction': 0.3, 'new_value_range': (0, 100)},
        'introduce_outliers': {'column': 'numeric_column', 'n_outliers': 2, 'outlier_range': (100, 200)},
        'random_categorical_replacement': {'column': 'categorical_column', 'replacement_fraction': 0.3},
        'introduce_label_noise': {'column': 'categorical_column', 'noise_fraction': 0.2},
        'swap_categories': {'column': 'categorical_column', 'swap_fraction': 0.1},
        'shuffle_column': {'column': 'numeric_column'},
        'delete_random_rows': {'fraction': 0.1}
    }

    # Corrupt the dataset
    corruptor.corrupt(corruption_config)

    # Display the corrupted DataFrame
    print(corruptor.dataframe)
