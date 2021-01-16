import sys

from Mapper import Mapper
from ReadPlyFile import ReadPlyFile
from sklearn.datasets import make_circles


def filter_x(point):
    return point[0]


def main(args):
    data = make_circles(100, shuffle=True)[0]

    # print(data)

    data = ReadPlyFile('data/bun000.ply', 0.02).get_data()

    mapper = Mapper(data, resolution=0.2, overlap=0.4, cluster_alg='kmeans', max_clusters=5, filter=filter_x)
    graph = mapper.run()
    print(graph)


if __name__ == '__main__':
    main(sys.argv)
