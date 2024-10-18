import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import tqdm
import collections
import os

class DecisionTreeEnsemble:
    def __init__(self, num_classifiers=10000, feature_fraction=0.5, data_fraction=0.8,
                 max_depth=10, min_samples_leaf=4, random_state=0, cache_size=100, save_dir="models_batches"):
        """
        Initialize the ensemble of decision tree classifiers.
        
        :param num_classifiers: Number of decision tree classifiers to train.
        :param feature_fraction: Fraction of features to use for each classifier.
        :param data_fraction: Fraction of data to use for training each classifier.
        :param max_depth: The maximum depth of the tree (to control tree complexity).
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param random_state: Seed for reproducibility (optional).
        :param cache_size: Maximum number of classifiers to hold in memory at a time.
        :param save_dir: Directory where classifier batches will be saved.
        """
        self.num_classifiers = num_classifiers
        self.feature_fraction = feature_fraction
        self.data_fraction = data_fraction
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.cache_size = cache_size
        self.save_dir = save_dir
        
        self.feature_subsets = []
        self.data_subsets = []
        self.classifier_cache = collections.OrderedDict()  # Cache for classifiers
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists

    def _get_random_subsets(self, X):
        """
        Generate random subsets of features and samples.
        
        :param X: Feature matrix (2D array).
        :return: A tuple (feature_indices, sample_indices).
        """
        n_features = int(X.shape[1] * self.feature_fraction)
        n_samples = int(X.shape[0] * self.data_fraction)
        
        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
        sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
        
        return feature_indices, sample_indices

    def train(self, X, y, batch_size=100):
        """
        Train the ensemble of decision trees in batches and save each batch to disk.
        
        :param X: Feature matrix (2D array).
        :param y: Target vector (1D array).
        :param batch_size: The number of classifiers to train in each batch.
        """
        for batch_start in range(0, self.num_classifiers, batch_size):
            batch_classifiers = []
            batch_feature_subsets = []
            batch_data_subsets = []
            
            for i in tqdm.tqdm(range(batch_start, min(batch_start + batch_size, self.num_classifiers))):
                # Get random subsets of features and samples
                feature_indices, sample_indices = self._get_random_subsets(X)
                X_subset = X[sample_indices][:, feature_indices]
                y_subset = y[sample_indices]

                # Train a decision tree classifier
                clf = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=(self.random_state + i) if self.random_state else None
                )
                clf.fit(X_subset, y_subset)

                batch_classifiers.append(clf)
                batch_feature_subsets.append(feature_indices)
                batch_data_subsets.append(sample_indices)

            # Save the batch to disk
            batch_file = os.path.join(self.save_dir, f'batch_{batch_start // batch_size}.pkl')
            with open(batch_file, 'wb') as f:
                pickle.dump({
                    'classifiers': batch_classifiers,
                    'feature_subsets': batch_feature_subsets,
                    'data_subsets': batch_data_subsets
                }, f)

            print(f"Saved batch {batch_start // batch_size} to {batch_file}")

            # Store feature and data subsets for prediction
            self.feature_subsets.extend(batch_feature_subsets)
            self.data_subsets.extend(batch_data_subsets)

    def _load_classifier(self, index):
        """
        Load a classifier from disk or return it from cache if available.
        
        :param index: The index of the classifier to load.
        :return: The classifier object.
        """
        if index in self.classifier_cache:
            # If the classifier is in cache, move it to the end (most recently used)
            self.classifier_cache.move_to_end(index)
            return self.classifier_cache[index]

        # Otherwise, load the classifier from disk
        batch_num = index // 1000
        batch_file = os.path.join(self.save_dir, f'batch_{batch_num}.pkl')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f)

        classifier_index_in_batch = index % 1000
        clf = batch['classifiers'][classifier_index_in_batch]

        # Add to cache
        self.classifier_cache[index] = clf
        if len(self.classifier_cache) > self.cache_size:
            self.classifier_cache.popitem(last=False)  # Remove the oldest item from cache

        return clf

    def predict(self, X):
        """
        Make predictions using the ensemble of classifiers with majority voting.
        
        :param X: Feature matrix (2D array).
        :return: Aggregated predictions from the ensemble (majority voted).
        """
        all_predictions = np.zeros((X.shape[0], self.num_classifiers))
        
        for i in tqdm.tqdm(range(self.num_classifiers)):
            clf = self._load_classifier(i)
            feature_indices = self.feature_subsets[i]
            all_predictions[:, i] = clf.predict(X[:, feature_indices])
        
        # Aggregate predictions (majority voting)
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=all_predictions)

        return final_predictions

    def predict_proba(self, X):
        """
        Predict class probabilities using the ensemble. 
        This returns the proportion of classifiers that predict each class.
        
        :param X: Feature matrix (2D array).
        :return: A 2D array of shape (n_samples, 2) where each row contains
                 the probability for class 0 and class 1.
        """
        all_predictions = np.zeros((X.shape[0], self.num_classifiers))

        for i in tqdm.tqdm(range(self.num_classifiers)):
            clf = self._load_classifier(i)
            feature_indices = self.feature_subsets[i]
            all_predictions[:, i] = clf.predict(X[:, feature_indices])

        # Calculate probabilities as the proportion of classifiers that predict each class
        prob_class_1 = np.mean(all_predictions == 1, axis=1)  # Proportion predicting class 1
        prob_class_0 = 1 - prob_class_1                       # Proportion predicting class 0

        return np.vstack([prob_class_0, prob_class_1]).T

    def get_classifier_info(self, index):
        """
        Get information about a specific classifier (features and samples used).
        
        :param index: The index of the classifier.
        :return: Dictionary with 'model', 'features', and 'samples' used for training.
        """
        if index >= len(self.feature_subsets):
            raise IndexError("Classifier index out of range.")
        
        clf = self._load_classifier(index)
        return {
            'model': clf,
            'features': self.feature_subsets[index],
            'samples': self.data_subsets[index]
        }

    def save(self, file_path):
        """
        Save the entire state of the ensemble (except classifiers) to a file.
        
        :param file_path: The path where the ensemble state will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'num_classifiers': self.num_classifiers,
                'feature_fraction': self.feature_fraction,
                'data_fraction': self.data_fraction,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state,
                'feature_subsets': self.feature_subsets,
                'data_subsets': self.data_subsets
            }, f)
        print(f"Ensemble state saved to {file_path}")
    
    @classmethod
    def load(cls, file_path):
        """
        Load the ensemble from a saved file.
        
        :param file_path: The path to the saved ensemble file.
        :return: An instance of DecisionTreeEnsemble with the loaded state.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create an instance of the class
        ensemble = cls(
            num_classifiers=data['num_classifiers'],
            feature_fraction=data['feature_fraction'],
            data_fraction=data['data_fraction'],
            max_depth=data['max_depth'],
            min_samples_leaf=data['min_samples_leaf'],
            random_state=data['random_state']
        )
        
        # Restore the saved classifiers, feature, and data subsets
        ensemble.classifiers = data['classifiers']
        ensemble.feature_subsets = data['feature_subsets']
        ensemble.data_subsets = data['data_subsets']
        
        print(f"Ensemble loaded from {file_path}")
        
        return ensemble