import numpy as np

from sktime.classifiers.distance_based import proximity_forest
from sktime.classifiers.distance_based import elastic_ensemble
from sktime.distances import elastic
from sktime.distances import elastic_cython

class EDSplitter:

    def __init__(self, tree, params):
        self.tree = tree
        self.params = params
        self.measure = None
        self.exemplars = {}
        self._exemplar_indices = []

    def init_data(self):
        print('init data')

    def split(self, X_train, y_train, **extra_data):

        #select a random measure
        # self.measure = elastic.dtw_distance
        self.measure = self.euclidean

        #select a random param

        #select random exemplars
        cls_indices = extra_data['class_indices']
        for cls in cls_indices:
            exemplar_index =  int(np.random.choice(cls_indices[cls][0], 1))
            self._exemplar_indices.append(exemplar_index)
            self.exemplars[int(cls)] = X_train.iloc[exemplar_index]

        print(f'exemplars: {self._exemplar_indices}')
        #partition based on similarity to the exemplars
        distances = {}
        splits = {}
        min_distance = np.inf

        for i in range(X_train.shape[0]):
            for j in self.exemplars:
                e = self.exemplars[j]
                s = X_train.iloc[i]
                distances[j] =  self.measure(s, e)
                if (distances[j] <= min_distance):
                    min_distance = distances[j]
                    nearest_e = j
                if nearest_e not in splits.keys():
                    splits[nearest_e] = []
            # print(distances)
            splits[nearest_e].append(i)



        return splits

    def predict(self, query, qi):
        distances = {}
        min_distance = np.inf
        for j in self.exemplars:
            e = self.exemplars[j]
            print(f'e {e[0][0]} --> q {query[0][0]}' )
            distances[j] = self.measure(query, e)
            if (distances[j] <= min_distance):
                min_distance = distances[j]
                nearest_e = j
        print(distances)
        return nearest_e


    def euclidean(self, a, b):
        dimension = 0
        return np.sum((a[dimension] - b[dimension]) ** 2) #skip sqrt

    def dtw(self, a, b):

        return 0;