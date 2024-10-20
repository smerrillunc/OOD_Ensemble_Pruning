import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import tqdm
import gc
import numbers

class DecisionTreeEnsemble:
    def __init__(self, num_classifiers=10000, feature_fraction=0.5, data_fraction=0.8, 
                 max_depth=10, min_samples_leaf=4, random_state=0):
        """
        Initialize the ensemble of decision tree classifiers.
        
        :param num_classifiers: Number of decision tree classifiers to train.
        :param feature_fraction: Fraction of features to use for each classifier.
        :param data_fraction: Fraction of data to use for training each classifier.
        :param max_depth: The maximum depth of the tree (to control tree complexity).
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param random_state: Seed for reproducibility (optional).
        """
        self.num_classifiers = num_classifiers
        self.feature_fraction = feature_fraction
        self.data_fraction = data_fraction
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.classifiers = []
        #self.feature_subsets = []
        #self.data_subsets = []
        
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

    def train(self, X, y):
        """
        Train the ensemble of decision trees.
        
        :param X: Feature matrix (2D array).
        :param y: Target vector (1D array).
        """
        n_features = int(X.shape[1] * self.feature_fraction)
        n_samples = int(X.shape[0] * self.data_fraction)

        for i in tqdm.tqdm(range(self.num_classifiers)):
            # Get random subsets of features and samples
    
            random_state = (self.random_state + i) if self.random_state else i
            random_instance = DecisionTreeEnsemble.check_random_state(random_state)

            feature_indices = random_instance.choice(X.shape[1], n_features, replace=False)
            sample_indices = random_instance.choice(X.shape[0], n_samples, replace=False)

            
            # Subset the data
            X_subset = np.take(X, sample_indices, axis=0)[:, feature_indices]
            y_subset = np.take(y, sample_indices)

            # Train the decision tree with additional hyperparameters
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=random_state
            )
            clf.fit(X_subset, y_subset)
            # Store the trained classifier and corresponding feature/sample indices
            self.classifiers.append(clf)
            del X_subset, y_subset, clf
            gc.collect()  

            #self.feature_subsets.append(feature_indices)
            #self.data_subsets.append(sample_indices)
            
            if (i + 1) % 1000 == 0:  # Print progress every 1000 classifiers
                print(f"Trained {i + 1} classifiers.")
                
    def predict(self, X):
        """
        Make predictions using the trained ensemble of classifiers with majority voting.
        
        :param X: Feature matrix (2D array).
        :return: Aggregated predictions from the ensemble (majority voted).
        """
        # Store predictions for each classifier
        all_predictions = np.zeros((X.shape[0], len(self.classifiers)))
        n_features = int(X.shape[1] * self.feature_fraction)

        for i, clf in enumerate(self.classifiers):
            random_instance = DecisionTreeEnsemble.check_random_state(clf.random_state)
            feature_indices = random_instance.choice(X.shape[1], n_features, replace=False)
            all_predictions[:, i] = clf.predict(X[:, feature_indices])

        # Aggregate predictions (majority voting here)
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
        # Store predictions for each classifier
        all_predictions = np.zeros((X.shape[0], len(self.classifiers)))
        n_features = int(X.shape[1] * self.feature_fraction)

        for i, clf in enumerate(self.classifiers):
            random_instance = DecisionTreeEnsemble.check_random_state(clf.random_state)
            feature_indices = random_instance.choice(X.shape[1], n_features, replace=False)

            all_predictions[:, i] = clf.predict(X[:, feature_indices])

        # Calculate probabilities as the proportion of classifiers that predict each class
        prob_class_1 = np.mean(all_predictions == 1, axis=1)  # Proportion predicting class 1
        prob_class_0 = 1 - prob_class_1                       # Proportion predicting class 0

        # Return probabilities in a 2D array
        return np.vstack([prob_class_0, prob_class_1]).T
    
    def get_individual_predictions(self, X):
        """
        Get predictions from each individual model.
        
        :param X: Feature matrix (2D array).
        :return: A 2D array where each row corresponds to a sample, and each column
                 corresponds to the predictions from a particular classifier.
        """
        # Store predictions for each classifier
        all_predictions = np.zeros((X.shape[0], len(self.classifiers)))
        n_features = int(X.shape[1] * self.feature_fraction)

        for i, clf in enumerate(self.classifiers):
            random_instance = DecisionTreeEnsemble.check_random_state(clf.random_state)
            feature_indices = random_instance.choice(X.shape[1], n_features, replace=False)
            all_predictions[:, i] = clf.predict(X[:, feature_indices])

        return np.array(all_predictions)

    def get_individual_probabilities(self, X):
        """
        Get the predicted probability estimates from each individual model.
        
        :param X: Feature matrix (2D array).
        :return: A list of 3D arrays where each element corresponds to the predicted
                 probabilities from a particular classifier. Each 3D array has shape
                 (n_samples, n_classes).
        """
        all_probabilities = []
        n_features = int(X.shape[1] * self.feature_fraction)

        for i, clf in enumerate(self.classifiers):
            random_instance = DecisionTreeEnsemble.check_random_state(clf.random_state)
            feature_indices = random_instance.choice(X.shape[1], n_features, replace=False)
            # Predict probabilities for each sample
            probabilities = clf.predict_proba(X[:, feature_indices])
            all_probabilities.append(probabilities)

        return np.array(all_probabilities)

    def get_classifier_info(self, index):
        """
        Get information about a specific classifier (features and samples used).
        
        :param index: The index of the classifier.
        :return: Dictionary with 'model', 'features', and 'samples' used for training.
        """
        if index >= len(self.classifiers):
            raise IndexError("Classifier index out of range.")
        
        return {
            'model': self.classifiers[index],
            #'features': self.feature_subsets[index],
            #'samples': self.data_subsets[index]
        }

    def save(self, file_path):
        """
        Save the entire state of the ensemble to a file.
        
        :param file_path: The path where the model will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'num_classifiers': self.num_classifiers,
                'feature_fraction': self.feature_fraction,
                'data_fraction': self.data_fraction,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state,
                'classifiers': self.classifiers,
                #'feature_subsets': self.feature_subsets,
               # 'data_subsets': self.data_subsets
            }, f)
        print(f"Ensemble saved to {file_path}")
    

    @staticmethod
    def check_random_state(seed):
        """Turn seed into a np.random.RandomState instance.

        Parameters
        ----------
        seed : None, int or instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

        Returns
        -------
        :class:`numpy:numpy.random.RandomState`
            The random state object based on `seed` parameter.

        Examples
        --------
        >>> from sklearn.utils.validation import check_random_state
        >>> check_random_state(42)
        RandomState(MT19937) at 0x...
        """
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, numbers.Integral):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState instance" % seed
        )


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
        #ensemble.feature_subsets = data['feature_subsets']
        #ensemble.data_subsets = data['data_subsets']
        
        print(f"Ensemble loaded from {file_path}")
        
        return ensemble