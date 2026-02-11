import random
import numpy as np


class KMeans:
    def __init__(self, n_clusters: int = 2, max_iter: int = 100):
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.centroids = None

    def fit_predict(self, X):
        random_idx = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_idx]

        for i in range(self.max_iter):
            # assign clusters
            cluster_group = self.assign_cluster(X)
            old_centroids = self.centroids
            # move centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # check finish
            if (old_centroids == self.centroids).all():
                break

        return cluster_group

    def assign_cluster(self, X) -> np.array:
        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))

            min_dist = min(distances)
            index_pos = distances.index(min_dist)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for t in cluster_type:
            new_centroids.append(X[cluster_group == t].mean(axis=0))

        return np.array(new_centroids)

    def estimate_n() -> np.array:
        return []
