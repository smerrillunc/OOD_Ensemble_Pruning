import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class OutlierDetector:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.n_outliers = int(0.1 * len(x_train))  # 10% of the dataset
        
    def frequency_based_outliers(self, threshold=0.01):
            """
            Detect outliers based on frequency analysis of individual features
            and rare category detection, implemented in a memory-efficient way.
            """
            n_samples, n_features = self.x_train.shape
            outlier_scores = np.zeros(n_samples, dtype=np.float32)

            for feature_idx in range(n_features):
                # Extract the current feature
                feature = self.x_train[:, feature_idx]

                # Count frequencies
                value_counts = Counter(feature)
                total_count = n_samples

                # Identify rare categories (below threshold)
                rare_categories = {val for val, count in value_counts.items() 
                                   if count / total_count < threshold}

                # Update outlier scores
                outlier_scores += np.isin(feature, list(rare_categories)).astype(np.float32)

            # Sort by outlier score and select top 10%
            outlier_indices = np.argsort(outlier_scores)[-self.n_outliers:]

            return outlier_indices
    
    def isolation_forest_outliers(self):
        """Detect outliers using the Isolation Forest algorithm"""
        clf = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = clf.fit_predict(self.x_train)
        outlier_indices = np.where(outlier_labels == -1)[0]
        return outlier_indices

    def lof_outliers(self):
        """Detect outliers using LOF with Gower distance"""
        gower_dist = self._gower_distance(self.x_train)
        lof = LocalOutlierFactor(n_neighbors=20, metric='precomputed', contamination=0.1)
        outlier_labels = lof.fit_predict(gower_dist)
        outlier_indices = np.where(outlier_labels == -1)[0]
        return outlier_indices
    
    
    def logistic_regression_outliers(self):
            """
            Detect outliers using logistic regression decision boundary distance.
            """
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.x_train)

            # Fit logistic regression
            lr = LogisticRegression(random_state=42)
            lr.fit(X_scaled, self.y_train)

            # Get decision function scores
            decision_scores = np.abs(lr.decision_function(X_scaled))

            # Get the indices of the top 10% points furthest from the decision boundary
            outlier_indices = np.argsort(decision_scores)[-self.n_outliers:]

            return outlier_indices
        
    def dbscan_outliers(self, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.x_train)
        outlier_indices = np.where(labels == -1)[0]
        return outlier_indices

    def _gower_distance(self, X):
        """Compute Gower distance for mixed data types"""
        df = pd.DataFrame(X)
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        
        def gower_dist(xi, xj):
            num_diff = np.abs(xi[num_cols] - xj[num_cols]).sum() / len(num_cols) if len(num_cols) > 0 else 0
            cat_diff = (xi[cat_cols] != xj[cat_cols]).sum() / len(cat_cols) if len(cat_cols) > 0 else 0
            return (num_diff + cat_diff) / 2
        
        dist_matrix = pdist(df.values, metric=gower_dist)
        return squareform(dist_matrix)
