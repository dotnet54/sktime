from random import random
import numpy as np
from scipy import stats

from sktime.classifiers.distance_based import proximity_forest
from sktime.classifiers.distance_based import elastic_ensemble
from sktime.distances import elastic
from sktime.distances import elastic_cython
from sktime.utils import dataset_properties


class EDSplitter:

    def __init__(self, node, tree):
        self.node = node
        self.tree = tree
        self.enabled_measures = [
            self.euclidean,
            # self.dtw, self.ddtw, self.wdtw, self.wddtw,
            # self.lcss, self.msm, self.twe, self.erp
        ]
        self.similarity_measure = None
        self.measure_params = None
        self.exemplars = {}
        self._exemplar_indices = []

    def split(self, x_train, y_train, **extra_data):
        candidate_splits_gini = [None] * self.tree.num_similarity_candidate_splits  # keeping all gini for statistics
        min_gini = np.inf
        best_split = None

        for k in range(0, self.tree.num_similarity_candidate_splits):

            # select a random measure and initialize it
            self.similarity_measure = np.random.choice(self.enabled_measures)(X=x_train)
            # select a random param
            self.measure_params = self.get_random_parammeters(self.similarity_measure)

            # select random exemplars
            cls_indices = extra_data['class_indices']
            for cls in cls_indices:
                exemplar_index = int(np.random.choice(cls_indices[cls][0], 1))
                self._exemplar_indices.append(exemplar_index)
                self.exemplars[int(cls)] = x_train.iloc[exemplar_index]

            print(f'exemplars: {self._exemplar_indices}')
            # partition based on similarity to the exemplars
            distances = {}
            current_split = {}
            min_distance = np.inf

            for i in range(x_train.shape[0]):
                for j in self.exemplars:
                    e = self.exemplars[j]
                    s = x_train.iloc[i]
                    # TODO skip self distance
                    distances[j] = self.similarity_measure['measure'](s, e, **self.measure_params)
                    if distances[j] <= min_distance:
                        min_distance = distances[j]
                        nearest_e = j
                    if nearest_e not in current_split.keys():
                        current_split[nearest_e] = []
                # print(distances)
                current_split[nearest_e].append(i)

            candidate_splits_gini[k] = self.node._weighted_gini(current_split, y_train)
            # TODO tie break randomly
            if candidate_splits_gini[k] <= min_gini:
                min_gini = candidate_splits_gini[k]
                best_split = current_split

        return best_split

    def predict(self, query, qi):
        distances = {}
        min_distance = np.inf
        for j in self.exemplars:
            e = self.exemplars[j]
            print(f'e {e[0][0]} --> q {query[0][0]}')
            distances[j] = self.similarity_measure['measure'](query, e, **self.measure_params)
            if distances[j] <= min_distance:
                min_distance = distances[j]
                nearest_e = j
        print(distances)
        return nearest_e

    def euclidean(self, **kwargs):

        def measure(a, b, dim=0):
            return np.sum((a[dim] - b[dim]) ** 2)  # skip sqrt

        return {
            'name': 'Euclidean',
            'measure': measure,
            'params': []
        }

    def dtw(self, **kwargs):
        return {
            'name': 'DTW',
            'measure': proximity_forest.cython_wrapper(elastic_cython.dtw_distance),
            'params': [{'w': stats.uniform(0, 0.25)}]
        }

    def ddtw(self, **kwargs):
        return {
            'name': 'DDTW',
            'measure': proximity_forest.cython_wrapper(elastic_cython.ddtw_distance),
            'params': [{'w': stats.uniform(0, 0.25)}]
        }

    def wdtw(self, **kwargs):
        return {
            'name': 'WDTW',
            'measure': proximity_forest.cython_wrapper(elastic_cython.wdtw_distance),
            'params': [{'g': stats.uniform(0, 1)}]
        }

    def wddtw(self, **kwargs):
        return {
            'name': 'WDDTW',
            'measure': proximity_forest.cython_wrapper(elastic_cython.wddtw_distance),
            'params': [{'g': stats.uniform(0, 1)}]
        }

    def lcss(self, **kwargs):
        X = kwargs['X']
        stdp = dataset_properties.stdp(X)
        instance_length = dataset_properties.max_instance_length(X)
        max_raw_warping_window = np.floor((instance_length + 1) / 4)
        return {
            'name': 'LCSS',
            'measure': proximity_forest.cython_wrapper(elastic_cython.lcss_distance),
            'params': [{'epsilon': stats.uniform(0.2 * stdp, stdp - 0.2 * stdp),
                        'delta': stats.randint(low=0, high=max_raw_warping_window + 1)}]
        }

    def msm(self, **kwargs):
        return {
            'name': 'MSM',
            'measure': proximity_forest.cython_wrapper(elastic_cython.msm_distance),
            'params': [{'c': [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375, 0.0475,
                              0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125, 0.085, 0.08875,
                              0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46,
                              0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784, 0.82, 0.856, 0.892, 0.928,
                              0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68,
                              6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2, 20.8,
                              24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8, 60.4, 64, 67.6, 71.2, 74.8, 78.4,
                              82, 85.6, 89.2, 92.8, 96.4, 100]}]
        }

    def twe(self, **kwargs):
        return {
            'name': 'TWE',
            'measure': proximity_forest.cython_wrapper(elastic_cython.twe_distance),
            'params': [{'penalty': [0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
                                    0.077777778, 0.088888889, 0.1],
                        'stiffness': [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}]
        }

    def erp(self, **kwargs):
        X = kwargs['X']
        stdp = dataset_properties.stdp(X)
        instance_length = dataset_properties.max_instance_length(X)
        max_raw_warping_window = np.floor((instance_length + 1) / 4)
        return {
            'name': 'ERP',
            'measure': proximity_forest.cython_wrapper(elastic_cython.erp_distance),
            'params': [{'g': stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
                        'band_size': stats.randint(low=0, high=max_raw_warping_window + 1)}]
        }

    # similar to from pick_rand_param_perm_from_dict in proximity_forest class
    def get_random_parammeters(self, distance_measure):
        params = {}
        # TODO random state

        for param in distance_measure['params']:
            for k, v in param.items():
                if isinstance(v, list):
                    params[k] = np.random.choice(v)
                elif hasattr(v, 'rvs'):
                    params[k] = v.rvs()
                else:
                    print('unknown param type')

        return params
