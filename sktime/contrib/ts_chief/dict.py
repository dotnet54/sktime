import numpy as np
import random

from sktime.classifiers.dictionary_based.boss import boss_distance
from sktime.transformers.dictionary_based import SFA
from sktime.transformers.dictionary_based.SAX import BitWord
from sktime.transformers.base import BaseTransformer

class BOSSDataStore:

    def __init__(self, **params):
        self.params = params
        self.max_num_boss_transformations = 5
        self.usable_num_boss_transformations_usable = self.max_num_boss_transformations
        self.alphabet_size = 4
        self.word_lengths = [6,8,10,12,14,16]
        self.norm_options = [True,False]
        self.min_window_size = 10
        self.remove_repeat_words = True

        self.sfa_transforms = {}
        self.train_boss_transforms = {}
        self.test_boss_transforms = {}

    def initialize_before_train(self, tree, x_train, y_train):
        print('initializeBeforeTrain BOSSplitter')
        series_length = x_train.iloc[0].iloc[0].shape[0]
        print(series_length)
        self.possible_boss_params = self.get_boss_params(series_length)
        self.usable_num_boss_transformations_usable = min(len(self.possible_boss_params),
                                               self.max_num_boss_transformations)

        self.rand_boss_transfroms_used = random.choices(self.possible_boss_params,
                                                        k=self.usable_num_boss_transformations_usable)


        for param in self.rand_boss_transfroms_used:
            sfa = SFA.SFA(*param)
            sfa.fit(x_train)
            x_train_boss = sfa.transform(x_train)
            self.train_boss_transforms[param] = x_train_boss
            self.sfa_transforms[param] = sfa


    def get_boss_params(self, series_length):
        params = []

        for window_size in range(self.min_window_size, series_length):
            for word_length in self.word_lengths:
                for norm in self.norm_options:
                    param = (word_length,self.alphabet_size, window_size, norm,
                             self.remove_repeat_words, True)
                    params.append(param)

        return params

    def initialize_before_test(self, tree, x_test):
        print('initializeBeforeTest BOSSplitter')
        # sfa = SFA.SFA(*self.selected_random_boss_transformation)
        # sfa.fit(x_test)
        # x_test_boss = sfa.transform(x_test)
        # self.test_boss_transforms[param] = x_test_boss


class BOSSplitter:

    def __init__(self, tree, params):
        self.tree = tree
        self.params = params
        self.boss_data_store = tree.root.boss_data_store
        self.transformer = None
        self.measure = self.boss_dist
        self.exemplars = {}
        self._exemplar_indices = []

    def split(self, x_train, y_train, **extra_data):
        x_train_indices = x_train.index.values.tolist()

        # select one transform
        _keys = list(self.boss_data_store.train_boss_transforms.keys())
        self.selected_random_boss_transformation = random.choices(_keys, k=1)[0]
        self.random_boss_transformed_dataset = self.boss_data_store.train_boss_transforms[self.selected_random_boss_transformation]
        x_train_transformed = self.random_boss_transformed_dataset.iloc[x_train_indices]
        self.transformer = self.boss_data_store.sfa_transforms[self.selected_random_boss_transformation]

        cls_indices = extra_data['class_indices']
        for cls in cls_indices:
            exemplar_index =  int(np.random.choice(cls_indices[cls][0], 1))
            self._exemplar_indices.append(exemplar_index)
            self.exemplars[int(cls)] = x_train_transformed.iloc[exemplar_index]

        print(f'exemplars: {self._exemplar_indices}')
        #partition based on similarity to the exemplars
        distances = {}
        splits = {}
        min_distance = np.inf

        for i in range(x_train_transformed.shape[0]):
            for j in self.exemplars:
                e = self.exemplars[j]
                s = x_train_transformed.iloc[i]
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

        transformed_query = self.transformer.transform_query(query)
        distances = {}
        min_distance = np.inf
        for j in self.exemplars:
            e = self.exemplars[j]
            print(f'e {e[0].iloc[0]} --> q {transformed_query[0].iloc[0]}' )
            distances[j] = self.measure(transformed_query, e)
            if (distances[j] <= min_distance):
                min_distance = distances[j]
                nearest_e = j
        print(distances)
        return nearest_e

    def boss_dist(self, a, b):
        a = a[0].to_dict()
        b = b[0].to_dict()
        return boss_distance(a,b)
