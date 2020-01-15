#%%
import random
from math import inf

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import RidgeClassifierCV
from sktime.classifiers.shifaz.similarity import EDSplitter
from sktime.classifiers.shifaz.dict import BOSSplitter

class ChiefTree:
    _node_id = 0

    def __init__(self, **params):
        self.params = params
        self.max_depth = params.get('max_depth', 20)
        self.lowest_gini = params.get('lowest_gini', 0.01)
        self.Ce = params.get('Ce', 0)
        self.Cb = params.get('Cb', 0)
        self.Cr = params.get('Cr', 0)

        #initialize
        self.children = {}
        self.split_functions = []
        self.best_split_function = None
        self.best_split = None
        self.leaf = False
        self.parent = None
        self.depth = 0
        self._height = 0
        self._root = self
        self._node_id = ChiefTree._node_id + 1

    # @classmethod
    # def as_root(cls, **params):
    #     tree = cls(**params)
    #     tree.parent = None
    #     tree.depth = 0
    #     tree._height = 0
    #     tree._root = self
    #     return tree

    @classmethod
    def as_child(cls, parent, **params):
        tree = cls(**params)
        tree.parent = parent
        tree.depth = parent.depth + 1
        tree._height = 0
        tree.root = parent._root
        return tree

    def fit(self, X_train, y_train):
        self.node_gini = self.gini(y_train)
        if self.stop_building(y_train):
            self.label = self.make_label(y_train)
            self.leaf = True
            return

        class_indices = self.get_class_indices(X_train, y_train)
        self.generate_split_functions(X_train, y_train)
        candidate_splits = []
        for splitter in self.split_functions:
            candidate_splits.append(splitter.split(X_train, y_train,
                                                   class_indices = class_indices))

        best_splitter_index = self.argmin(candidate_splits, self.weighted_gini, y_train)
        self.best_split_function = self.split_functions[best_splitter_index]
        #memory intensive,.. dont keep all candidate splits
        splits = candidate_splits[best_splitter_index]

        for key, indices in splits.items():
            if len(indices) > 0:
                self.children[key] = ChiefTree.as_child(self, **self.params)
                self.children[key].fit(X_train.iloc[indices], y_train[indices])
            else:
                print(f'no data in the split {key} : {len(indices)}')

        return self

    def stop_building(self, y_train):

        if (self.node_gini <= self.lowest_gini):
            return True
        elif (self.parent is not None and self.node_gini == self.parent.node_gini):
            print(f'Warn no improvemen to gini {self.depth}, {y_train.shape},' +
                   f' {self.node_gini}, {self.parent.node_gini}, {np.unique(y_train, return_counts=True)[1] }, {self.make_label(y_train)}')
            return True
        elif (self.max_depth != -1 and self.depth > self.max_depth):
            # debug
            print(self._print_suffix + f'Error recursion too deep {self.depth}, {self.node_gini}')
            raise KeyboardInterrupt
            return True

        return False

    def make_label(self, y_train):
        # print(np.random.choice(stats.mode(y_train)[0], 1))
        if self.depth > self.root._height:
            self.root._height = self.depth
        label = int(np.random.choice(stats.mode(y_train)[0], 1))  # int(y_train.value_counts().max())
        print(f'new leaf {self.depth}, {self.node_gini}, {label}, {y_train.shape}, {y_train}')
        return label

    def get_class_indices(self, X, y):
        split_indices = {}

        for cls in np.unique(y):
            grp = np.where(y == cls)
            split_indices[cls] = grp

        return split_indices

    def  generate_split_functions(self, X_train, y_train):

        for e in range(self.Ce):
            splitter = EDSplitter(self, self.params)
            # splitter.split(X_train, y_train)
            self.split_functions.append(splitter)

        # for b in self.Cb:
        #     splitter = BOSSplitter(self.params)
        #     splitter.split(X_train, y_train)
        #     self.splitters.append(splitter)

        return None

    def predict(self, X_test, y_test):
        return self.best_split_function.predict(X_test, y_test)

    def gini(self, y):
        if y.shape[0] == 0:
            return 0
        #     print(y.shape[0])
        #     print(np.unique(y, return_counts=True)[1])
        #     print(np.unique(y, return_counts=True)[1] / y.shape[0])
        #     print(np.power(np.unique(y, return_counts=True)[1] / y.shape[0] , 2))
        #     print(np.sum(np.power(np.unique(y, return_counts=True)[1] / y.shape[0] , 2)))
        return 1 - np.sum(np.power(np.unique(y, return_counts=True)[1] / y.shape[0], 2))

    def weighted_gini(self, splits, y):
        wg = 0
        for k, s in splits.items():
            wg += len(s) / y.shape[0] * self.gini(y[s])
        return wg

    def argmin(self, candidate_splits, func_selector, parent_data):
        min_arg = None
        min_gini = inf

        candidate_ginis = np.empty(len(candidate_splits))

        for i, cs in enumerate(candidate_splits):
            candidate_ginis[i] = func_selector(cs, parent_data)

        #         print(candidate_splits)
        # print(candidate_ginis)
        min_gini = candidate_ginis.min()
        min_arg = candidate_ginis.argmin()
        # print(f'min_gini:{min_gini}, arg_min: {min_arg}')

        return min_arg

    def __repr__(self):
        return f'ChiefTree (leaf={self.leaf}, label={self.label}, children={len(self.children)}, depth={self.depth})'


class ChiefForest:

    def __init__(self, **kwargs):

        # read args
        self.kwargs = kwargs
        self.k = int(kwargs.get('k'))

        # init
        self.trees = []
        for i in range(self.k):
            _t = ChiefTree(**kwargs)
            self.trees.append(_t)

    def fit(self, X_train, y_train):

        for i, _t in enumerate(self.trees):
            print(f'{i}.', end ="")
            _t.fit(X_train, y_train)
        print('')

        return self

    def predict(self, X_test, y_test):
        scores = [None] * len(self.trees)
        for i, _t in enumerate(self.trees):
            print(f'{i}.', end ="")
            scores[i] = _t.predict(X_test, y_test)
        print('')

        return scores

    def __repr__(self):
        return f'\n------------------------ChiefForest (k={self.k})----------------------'