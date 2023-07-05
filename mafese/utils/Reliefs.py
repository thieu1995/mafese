import numpy as np
from data_loader import *
import warnings
from sklearn.neighbors import NearestNeighbors
import copy
import sys
'''
This source code is developed by Ngoc Hung Nguyen.

This algorithm refers to : https://www.sciencedirect.com/science/article/pii/S1532046418301400
 
'''


class Reliefs:
    methods = ["relief", 'reliefF']

    def __init__(self, method='relief', data=([], [])):
        self.request_method = method
        self.data_x, self.data_y = data[0], data[1]
        self.run()

    def run(self):
        try:
            request_method = getattr(self, self.request_method)
        except Exception as e:
            print('An error is raised: ', e)
            print('The method is not supported please refer to', Reliefs.methods)
            exit(0)
        if len(self.data_x) == 0:
            warnings.warn('input data set is null, please check it again !!!')
        weights = request_method(num_neighbors=10, m=200)
        return weights

    def relief(self, m=100, continuous=False, num_neighbors=5):

        def value(I1, I2):
            if I1.__class__.__name__ == 'NoneType':
                return np.abs(I2)
            elif I2.__class__.__name__ == 'NoneType':
                return np.abs(I1)
            return np.abs(I1 - I2)

        def distance(I1, I2):
            return np.linalg.norm(I1 - I2)

        def diff(A, I1, I2):
            if continuous == False:
                if (value(A, I1) == value(A, I2)).all():
                    return 0
                else:
                    return 1
        data_x = copy.copy(self.data_x)
        data_y = copy.copy(self.data_y)

        num_samples, _ = data_x.shape
        weights = [0] * num_samples
        neighbors = NearestNeighbors(n_neighbors=num_neighbors).fit(data_x)
        for i in range(m):
            nearest_hit = None
            nearest_miss = None

            # random select on point
            idx = np.random.randint(0, num_samples)
            R_i = data_x[idx]
            target_endpoint = data_y[idx]
            distances, indices = neighbors.kneighbors([R_i])

            min_distance_hit = sys.maxsize
            min_distance_miss = sys.maxsize
            for idx, j in enumerate(indices[0]):
                if data_y[j] == target_endpoint and j != i and min_distance_hit > distances[0][idx]:
                    nearest_hit = data_x[j]
                    min_distance_hit = distances[0][idx]
                elif data_y[j] != target_endpoint and min_distance_miss > distances[0][idx]:
                    nearest_miss = data_x[j]
                    min_distance_miss = distances[0][idx]
            for A in range(num_samples):
                if (R_i == data_x[A]).all():
                    continue
                weights[A] = -diff(data_x[A], R_i, nearest_hit)/m + diff(data_x[A], R_i, nearest_miss)/m
        print(weights)
        return weights


data = get_dataset("Arrhythmia")
x_tr, x_te, y_tr, y_te = data.split_train_test(inplace=False)
a = Reliefs('relief', (x_tr, y_tr))
