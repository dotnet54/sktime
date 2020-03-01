import numpy as np
import random

from sktime.classifiers.dictionary_based.boss import boss_distance
from sktime.transformers.dictionary_based import SFA
from sktime.transformers.dictionary_based.SAX import BitWord
from sktime.transformers.base import BaseTransformer


class BOSSDataStore:

    def __init__(self, tree,
                 boss_max_num_transformations=1000,
                 min_window_size=10,
                 word_lengths=[6, 8, 10, 12, 14, 16],
                 alphabet_size=4,
                 norm_options=[True, False],
                 remove_repeat_words=True):
        # params
        self.tree = tree
        self.boss_max_num_transformations = boss_max_num_transformations
        self.min_window_size = min_window_size
        self.word_lengths = word_lengths
        self.alphabet_size = alphabet_size
        self.norm_options = norm_options
        self.remove_repeat_words = remove_repeat_words

        # initialize
        self.all_possible_boss_transform_params = None
        self.num_transformations_used = self.boss_max_num_transformations
        self.transformations_used = []
        self.sfa_transforms = {}
        self.train_boss_transformations = {}
        self.test_boss_transformations = {}

    def initialize_before_train(self, x_train):

        series_length = x_train.iloc[0].iloc[0].shape[0]  # TODO multivariate, variable length
        self.all_possible_boss_transform_params = self.get_all_boss_params(series_length)

        self.num_transformations_used = min(len(self.all_possible_boss_transform_params),
                                            self.boss_max_num_transformations)

        self.transformations_used = random.choices(self.all_possible_boss_transform_params,
                                                   k=self.num_transformations_used)

        # TODO use multi threading here
        print(f'Computing BOSS {self.num_transformations_used} transforms')
        for i, param in enumerate(self.transformations_used):
            sfa = SFA.SFA(*param)
            sfa.fit(x_train)
            x_train_boss = sfa.transform(x_train)
            self.sfa_transforms[param] = sfa
            self.train_boss_transformations[param] = x_train_boss
            print(f'{i}.', end='')

        print(f'\nall possible {self.num_transformations_used} BOSS transforms done')


    def initialize_before_test(self, x_test):
        pass

    def get_all_boss_params(self, series_length):
        params = []
        safe_words = False  # sfa implementation param -- not really needed

        for window_size in range(self.min_window_size, series_length):
            for word_length in self.word_lengths:
                for norm in self.norm_options:
                    param = (word_length, self.alphabet_size, window_size, norm,
                             self.remove_repeat_words, safe_words)
                    params.append(param)

        return params


class BOSSplitter:

    def __init__(self, node, tree):
        self.node = node
        self.tree = tree
        self.boss_data_store = tree.boss_data_store
        self.transformer = None
        self.histogram_similarity_measure = self.boss_dist
        self.exemplars = {}
        self.exemplar_indices = {}
        self.random_boss_transformation_params = None
        self.random_boss_transformation = None

    def split(self, x_train, y_train, class_indices):
        candidate_splits_gini = [None] * self.tree.n_dictionary_candidate_splits  # keeping all gini for statistics
        min_gini = np.inf
        best_split = None

        for candidate_split_index in range(0, self.tree.n_dictionary_candidate_splits):
            current_split = {}

            # select one random boss transform
            _boss_transform_params = list(self.boss_data_store.train_boss_transformations.keys())
            self.random_boss_transformation_params = random.choices(_boss_transform_params, k=1)[0]
            self.random_boss_transformation = self.boss_data_store.\
                train_boss_transformations[self.random_boss_transformation_params]

            x_train_indices = x_train.index.values.tolist()  # TODO refactor to use indices
            x_train_transformed = self.random_boss_transformation.iloc[x_train_indices]
            self.transformer = self.boss_data_store.sfa_transforms[self.random_boss_transformation_params]

            for cls in class_indices:
                _exemplar_index = int(np.random.choice(class_indices[cls][0], 1))
                self.exemplar_indices[int(cls)] = _exemplar_index
                # TODO TEST make a deep copy, make sure that there is no pointer
                #  to x_train so that it can be garbage collected
                self.exemplars[int(cls)] = x_train_transformed.iloc[_exemplar_index].copy(deep=True)
                current_split[cls] = []

            # partition based on similarity to the exemplars
            nearest_class = None
            for i in range(x_train.shape[0]):
                # dont need to store all distances, just storing for debugging
                distances = {}
                min_distance = np.inf
                for exemplar_class, exemplar in self.exemplars.items():
                    transformed_series = x_train_transformed.iloc[i]
                    # using exemplar_indices to speed up the equality check
                    if i == self.exemplar_indices[exemplar_class]:
                        nearest_class = exemplar_class
                        break
                    else:
                        distances[exemplar_class] = self.histogram_similarity_measure(transformed_series, exemplar)

                    # TODO tie break randomly
                    if distances[exemplar_class] <= min_distance:
                        min_distance = distances[exemplar_class]
                        nearest_class = exemplar_class
                # print(distances)
                current_split[nearest_class].append(i)

            candidate_splits_gini[candidate_split_index] = self.node._weighted_gini(current_split, y_train)
            # TODO tie break randomly
            if candidate_splits_gini[candidate_split_index] <= min_gini:
                min_gini = candidate_splits_gini[candidate_split_index]
                best_split = current_split

        return best_split

    def predict(self, query, qi):
        # TODO instead of calculating the transform we are just fetching in memory transform
        # TODO this is temporary -- change this to be independent of the fit function
        transformed_query = self.transformer.transform_query(query)
        distances = {}
        min_distance = np.inf
        nearest_class = None
        for exemplar_class, exemplar in self.exemplars.items():
            distances[exemplar_class] = self.histogram_similarity_measure(transformed_query, exemplar)
            # TODO tie break randomly
            if distances[exemplar_class] <= min_distance:
                min_distance = distances[exemplar_class]
                nearest_class = exemplar_class
        # print(distances)
        return nearest_class, distances

    def boss_dist(self, a, b):
        a = a[0].to_dict()
        b = b[0].to_dict()
        return boss_distance(a, b)
