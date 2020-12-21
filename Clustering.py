from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

class Clustering:

    def __init__(self, data):
        self.data = data

    def kmeans(self, clusters):
    	return KMeans(clusters).fit_predict(self.data)

    def GMM(self, clusters):
    	return GaussianMixture(clusters).fit_predict(self.data)

    def hierarchical(self, clusters):
    	return AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=clusters).fit_predict(self.data)

    def DBSCAN(self, eps, minPoints):
    	return  DBSCAN(algorithm='auto', eps=eps, min_samples=minPoints).fit_predict(self.data)
