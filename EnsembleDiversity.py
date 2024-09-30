import numpy as np
from scipy.stats import entropy, pearsonr
from sklearn.metrics import brier_score_loss, roc_auc_score, hamming_loss
from scipy.spatial.distance import hamming
from sklearn.metrics import accuracy_score, log_loss

class EnsembleDiversity:
    def __init__(self, y_true, predictions):
        """
        Initialize with the true labels and predictions from the ensemble members.
        
        Args:
        y_true: Actual labels (array-like).
        predictions: A 2D array where each row corresponds to the predictions from a decision tree.
        """
        self.y_true = np.array(y_true)
        self.predictions = np.array(predictions)

    def q_statistic(self):
        """Compute Q-statistic for pairwise diversity between trees."""
        n_trees = self.predictions.shape[0]
        q_stats = np.zeros((n_trees, n_trees))
        
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                N11 = np.sum((self.predictions[i] == 1) & (self.predictions[j] == 1))
                N00 = np.sum((self.predictions[i] == 0) & (self.predictions[j] == 0))
                N10 = np.sum((self.predictions[i] == 1) & (self.predictions[j] == 0))
                N01 = np.sum((self.predictions[i] == 0) & (self.predictions[j] == 1))

                q_stats[i, j] = (N11 * N00 - N10 * N01) / (N11 * N00 + N10 * N01 + 1e-10)  # Avoid division by zero

        return q_stats

    def correlation_coefficient(self):
        """Calculate Pearson correlation coefficient for each pair of trees."""
        n_trees = self.predictions.shape[0]
        corr_matrix = np.zeros((n_trees, n_trees))

        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                corr_matrix[i, j], _ = pearsonr(self.predictions[i], self.predictions[j])

        return corr_matrix

    def entropy(self):
        """Calculate entropy of the ensemble predictions."""
        # Average probabilities for each class across the trees
        mean_probs = np.mean(self.predictions, axis=0)
        return entropy(mean_probs)

    def diversity_measure(self):
        """Calculate the diversity measure for the ensemble."""
        n_trees = self.predictions.shape[0]
        dm = 0
        
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                dm += hamming(self.predictions[i], self.predictions[j])
        
        return dm / (n_trees * (n_trees - 1))

    def hamming_distance(self):
        """Calculate pairwise Hamming distance between tree predictions."""
        n_trees = self.predictions.shape[0]
        hamming_matrix = np.zeros((n_trees, n_trees))

        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                hamming_matrix[i, j] = hamming(self.predictions[i], self.predictions[j])

        return hamming_matrix

    def error_rate(self):
        """Compute the error rate for each individual tree."""
        error_rates = np.mean(self.predictions != self.y_true, axis=1)
        return error_rates

    def auc(self):
        """Calculate AUC for each individual tree."""
        aucs = []
        for pred in self.predictions:
            try:
                auc = roc_auc_score(self.y_true, pred)
            except:
                auc = np.nan
            aucs.append(auc)
        return aucs

    def brier_score(self):
        """Compute the Brier score for each individual tree."""
        brier_scores = []
        for pred in self.predictions:
            brier = brier_score_loss(self.y_true, pred)
            brier_scores.append(brier)
        return brier_scores

    def ensemble_variance(self):
        """Compute the variance of the predictions across the ensemble."""
        return np.var(self.predictions, axis=0)

    def kl_divergence(self, tree_idx1, tree_idx2):
        """Calculate the Kullback-Leibler divergence between two trees."""
        p1 = np.clip(self.predictions[tree_idx1], 1e-10, 1 - 1e-10)
        p2 = np.clip(self.predictions[tree_idx2], 1e-10, 1 - 1e-10)
        return np.sum(p1 * np.log(p1 / p2))
