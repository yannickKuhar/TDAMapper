import sys

from sklearn.datasets import load_iris
from Clustering import Clustering


def main(args):
	data = load_iris()	

	c = Clustering(data['data'])

	print(c.kmeans(3))
	print(c.GMM(3))
	print(c.hierarchical(3))
	print(c.DBSCAN(0.5, 7)) 


if __name__ == '__main__':
	main(sys.argv)
