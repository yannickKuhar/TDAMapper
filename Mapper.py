import numpy as np

from Clustering import Clustering


class Mapper:
    def __init__(self, data, resolution, overlap, cluster_alg, max_clusters, filter):
        self.data = data
        self.resolution = resolution
        self.overlap = overlap
        self.cluster_alg = cluster_alg
        self.filter = filter
        self.max_clusters = max_clusters

    @staticmethod
    def split_interval(minimum, maximum, resolution, overlap):
        step = (maximum - minimum) * resolution
        r_step = step * overlap
        subintervals = []

        for m in np.arange(minimum, maximum, step):
            left = minimum if (m - r_step) < minimum else m - r_step
            right = maximum if (m + step) > maximum else m + step

            subintervals.append((left, right))

        return step, subintervals

    def make_bins(self, data, resolution, overlap):
        interval = {self.filter(data[i]): data[i] for i in range(len(data))}

        step, subintervals = self.split_interval(min(interval), max(interval), resolution, overlap)

        bins = {subint: [] for subint in subintervals}

        for val in interval:
            for bin in bins:
                if bin[0] <= val <= bin[1]:
                    bins[bin].append(interval[val])

        return bins

    @staticmethod
    def node_to_set(node):
        candidate = list(map(tuple, node))
        candidate_set = set()
        for c in candidate:
            candidate_set.add(c)

        return candidate_set

    def make_graph(self, nodes):
        graph = {i: [] for i in range(len(nodes))}

        for g in graph:
            candidate = self.node_to_set(nodes[g])
            for i in range(len(nodes)):
                if i != g:
                    if len(set(self.node_to_set(nodes[i])).intersection(candidate)) > 0:
                        graph[g].append(i)

        return graph

    def run(self):
        bins = self.make_bins(self.data, resolution=self.resolution, overlap=self.overlap)

        nodes = []

        for s in bins.values():
            c = Clustering(data=np.array(s), max_clusters=self.max_clusters, cluster_alg=self.cluster_alg)
            n, groups = c.cluster()

            for g in groups:
                nodes.append(g)

        return self.make_graph(nodes)
