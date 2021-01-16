import sys
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

ERR_TAG = '[ERROR]'
TAG = '[Clustering]'


class Clustering:
    def __init__(self, data, max_clusters, cluster_alg):
        self.data = data
        self.max_clusters = max_clusters
        self.cluster_alg = cluster_alg
        self.threshold = np.round(np.linalg.norm(np.array(data).max(0) - np.array(data).min(0)) / 2)

    @staticmethod
    def closest_groups(groups, means):
        c1 = None
        c2 = None
        max_dist = sys.maxsize

        for i in range(len(groups)):
            for j in range(len(groups)):
                if i != j:
                    dist = np.linalg.norm(means[j] - means[i])
                    if dist < max_dist:
                        c1 = i
                        c2 = j
                        max_dist = dist

        return max_dist, c1, c2

    def join_groups(self, groups):
        groups_copy = groups[:]
        means = [np.array(g).mean(0) for g in groups]

        while True:
            dist, c1, c2 = self.closest_groups(groups_copy, means)

            if dist > self.threshold:
                break

            g1 = groups[c1]
            g2 = groups[c2]

            indices = c1, c2
            groups_copy = [i for j, i in enumerate(groups_copy) if j not in indices]

            groups_copy.append(g1 + g2)
            means = [np.array(g).mean(0) for g in groups]

        return groups_copy

    def make_groups(self, labels):

        groups = [[] for _ in range(len(set(labels)))]

        for i in range(len(labels)):
            groups[labels[i]].append(self.data[i])

        return self.join_groups(groups)

    def get_labels(self, clusters):
        if self.cluster_alg == 'kmeans':
            labels = KMeans(clusters).fit_predict(self.data)
        elif self.cluster_alg == 'gmm':
            labels = GaussianMixture(clusters).fit_predict(self.data)
        else:
            labels = AgglomerativeClustering(clusters).fit_predict(self.data)

        return labels

    def cluster(self):
        best_labels = None
        best_eval = 0
        n = 0

        for i in range(2, self.max_clusters):
            labels = self.get_labels(i)

            if labels is None:
                print(ERR_TAG, TAG, 'Labels of', self.cluster_alg, 'are None!')
                return -1, []

            score = metrics.silhouette_score(self.data, labels, metric='euclidean')

            if score > best_eval:
                best_labels = labels
                best_eval = score
                n = i

        return n, self.make_groups(best_labels)
