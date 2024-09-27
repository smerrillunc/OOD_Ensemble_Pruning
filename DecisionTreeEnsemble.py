import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeEnsemble:
    def __init__(self, num_classifiers=10000, feature_fraction=0.5, data_fraction=0.8, 
                 max_depth=None, min_samples_leaf=1, random_state=None):
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
        self.feature_subsets = []
        self.data_subsets = []
        
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
        for i in range(self.num_classifiers):
            # Get random subsets of features and samples
            feature_indices, sample_indices = self._get_random_subsets(X)
            
            # Subset the data
            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]
            
            # Train the decision tree with additional hyperparameters
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=(self.random_state + i) if self.random_state else None
            )
            clf.fit(X_subset, y_subset)
            
            # Store the trained classifier and corresponding feature/sample indices
            self.classifiers.append(clf)
            self.feature_subsets.append(feature_indices)
            self.data_subsets.append(sample_indices)
            
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

        for i, clf in enumerate(self.classifiers):
            feature_indices = self.feature_subsets[i]
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

        for i, clf in enumerate(self.classifiers):
            feature_indices = self.feature_subsets[i]
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

        for i, clf in enumerate(self.classifiers):
            feature_indices = self.feature_subsets[i]
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

        for i, clf in enumerate(self.classifiers):
            feature_indices = self.feature_subsets[i]
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
            'features': self.feature_subsets[index],
            'samples': self.data_subsets[index]
        }