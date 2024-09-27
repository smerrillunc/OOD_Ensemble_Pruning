import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
#from sklearn.mixture import GMM
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class Clustering:
    def __init__(self, data):
        """
        Initializes the Clustering class with data.
        
        :param data: A 2D numpy array or pandas DataFrame of shape (n_samples, n_features).
        """
        self.data = data
        self.labels = None

    def standardize_data(self):
        """Standardizes the data."""
        self.data = StandardScaler().fit_transform(self.data)

    def k_means(self, n_clusters=3):
        """Applies K-Means clustering.
        
        :param n_clusters: Number of clusters to form.
        :return: Cluster labels.
        """
        kmeans = KMeans(n_clusters=n_clusters)
        self.labels = kmeans.fit_predict(self.data)
        return self.labels

    def hierarchical_clustering(self, n_clusters=3):
        """Applies Agglomerative Hierarchical Clustering.
        
        :param n_clusters: Number of clusters to form.
        :return: Cluster labels.
        """
        hier = AgglomerativeClustering(n_clusters=n_clusters)
        self.labels = hier.fit_predict(self.data)
        return self.labels

    def dbscan(self, eps=0.5, min_samples=5):
        """Applies DBSCAN clustering.
        
        :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
        :param min_samples: The number of samples in a neighborhood for a point to be considered a core point.
        :return: Cluster labels.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = dbscan.fit_predict(self.data)
        return self.labels

    def mean_shift(self):
        """Applies Mean Shift clustering.
        
        :return: Cluster labels.
        """
        mean_shift = MeanShift()
        self.labels = mean_shift.fit_predict(self.data)
        return self.labels

    def gaussian_mixture(self, n_components=3):
        """Applies Gaussian Mixture Models.
        
        :param n_components: The number of mixture components.
        :return: Cluster labels.
        """
        gmm = GMM(n_components=n_components)
        self.labels = gmm.fit_predict(self.data)
        return self.labels

    def spectral_clustering(self, n_clusters=3):
        """Applies Spectral Clustering.
        
        :param n_clusters: Number of clusters to form.
        :return: Cluster labels.
        """
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        self.labels = spectral.fit_predict(self.data)
        return self.labels

    def plot_clusters(self):
        """Plots the clusters in 2D (only works if data has 2 dimensions)."""
        if self.data.shape[1] != 2:
            raise ValueError("Data must have 2 dimensions for plotting.")
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', marker='o')
        plt.title('Cluster Plot')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    clustering = Clustering(data)
    clustering.standardize_data()

    # Perform K-Means
    kmeans_labels = clustering.k_means(n_clusters=4)
    print("K-Means Labels:", kmeans_labels)

    # Plot K-Means clusters
    clustering.plot_clusters()