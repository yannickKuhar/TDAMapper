import numpy as np

from plyfile import PlyData


class ReadPlyFile:
    def __init__(self, path):
        self.path = path

    def sample_data(self, X, percentage):
            idxs = np.random.choice(len(X), int(len(X) * percentage))
            sample = X[idxs]

            return sample

    def get_data(self):
        f = open(self.path, 'rb')
        data = PlyData.read(f)
        S = []

        for i in range(len(data['vertex']['x'])):
            S.append((data['vertex']['x'][i], data['vertex']['y'][i], data['vertex']['z'][i]))

        return np.array(S)
