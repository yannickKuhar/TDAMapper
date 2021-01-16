import sys
import time
import numpy as np


from Mapper import Mapper
from ReadPlyFile import ReadPlyFile
from Visualization import Visualization
from sklearn.datasets import make_circles


def filter_x(point):
    return point[0]


def main(args):
    # data = make_circles(100, shuffle=True)[0]

    # data = ReadPlyFile('data/bun000.ply', 1.0).get_data()
    # data = ReadPlyFile('data/drill_1.6mm_0_cyb.ply', 1.0).get_data()
    data = ReadPlyFile('data/dragonStandRight_0.ply', 1.0).get_data()

    print(len(data))

    def filter_norm(point):
        return np.linalg.norm(point - np.array(data).min(0))

    mapper = Mapper(data, resolution=0.2, overlap=0.4, cluster_alg='kmeans', max_clusters=5, filter=filter_norm)
    graph = mapper.run()
    viz = Visualization(graph)
    viz.draw()


if __name__ == '__main__':
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    print('%s seconds' % round(end_time - start_time, 3))
