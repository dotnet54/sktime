#%%
import random
from math import inf

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sktime.classifiers.shifaz.similarity import EDSplitter
from sktime.classifiers.shifaz.dict import BOSSplitter

class ChiefTree:
    params = None

    Ce = 0
    Cb = 0
    Cr = 0
    lowest_gini = 0.01
    max_depth = -1  # unlimited

    children = None
    leaf = None
    label = None
    parent = None
    splitters = None

    #informational
    depth = 0
    node_gini = None
    best_splitter = None
    best_split = None
    class_indices = None
    class_distribution = None

    def __init__(self, **params):
        self.params = params
        self.max_depth = params.get('max_depth', 20)
        self.Ce = params.get('Ce', 0)
        self.Cb = params.get('Cb', 0)
        self.Cr = params.get('Cr', 0)

        #initialize
        self.children = []
        self.splitters = []
        self.leaf = False
        self.best_splitter = None
        self.best_split = None

    @classmethod
    def as_root(cls, parent, **params):
        tree = cls(**params)
        tree.parent = None
        tree.depth = 0
        return tree

    @classmethod
    def as_child(cls, parent, **params):
        tree = cls(**params)
        tree.parent = parent
        tree.depth = parent.depth + 1
        return tree

    def fit(self, X_train, y_train):
        self.node_gini = self.gini(y_train)
        if self.stop_building(y_train):
            self.label = self.make_label(y_train)
            self.leaf = True
            return

        self.class_indices = self.get_class_indices(X_train, y_train)
        candidate_splits = []
        self.splitters = self.generate_splitters(X_train, y_train)
        for splitter in self.splitters:
            candidate_splits.append(splitter.split())

        self.best_splitter = self.argmin(candidate_splits, self.weighted_gini, y_train)
        splits = candidate_splits[self.best_splitter]

        for key, indices in splits.items():
            #             print(f'training split {key} : {indices}')
            if len(indices) > 0:
                _child = ChiefTree.as_child(self, **self.params)
                self.children.append(_child)
                # vprint(
                #     self._print_suffix + f"child {key}:{len(indices)} :{'{0:.4f}'.format(self.gini(y_train.iloc[indices]))}  =" + str(
                #         self.children))
                _child.fit(X_train.iloc[indices], y_train.iloc[indices])
            #             print(_print_suffix + str(self.children))
            else:
                print(self._print_suffix + f'no data in the split {key} : {len(indices)}')

        return self

    def stop_building(self, y_train):

        if (self.node_gini <= self.lowest_gini):
            return True
        elif (self.parent is not None and self.node_gini == self.parent.node_gini):
            print(self._print_suffix + f'Warn no improvemen to gini {self.depth}, {y_train.shape},' +
                   f' {self.node_gini}, {self.parent.node_gini}, {y_train.value_counts()}, {self.make_label(y_train)}')
            return True
        elif (self.max_depth != -1 and self.depth > self.max_depth):
            # debug
            print(self._print_suffix + f'Error recursion too deep {self.depth}, {self.node_gini}')
            raise KeyboardInterrupt
            return True

        return False

    def make_label(self, y_train):
        label = int(y_train.mode())  # int(y_train.value_counts().max())
        # vprint(self._print_suffix + f'new leaf {self.depth}, {self.node_gini}, {label}, {y_train.shape}')
        return label

    def get_class_indices(self, X, y):
        split_indices = {}

        for cls in np.unique(y):
            grp = np.where(y == cls)
            split_indices[cls] = grp

        return split_indices

    def generate_splitters(self, X_train, y_train):

        for e in self.Ce:
            splitter = EDSplitter(self, self.params)
            splitter.split(X_train, y_train)
            self.splitters.append(splitter)

        # for b in self.Cb:
        #     splitter = BOSSplitter(self.params)
        #     splitter.split(X_train, y_train)
        #     self.splitters.append(splitter)

        return []

    def predict(self, X_test, y_test):
        scores = np.zeros(X_test.shape[0])
        X_test_feature_sampled = X_test.iloc[:, self.features[self.best_candidate]]
        # print(f'RocketTree: predict: {X_test.shape}, X_test_feature_sampled{X_test_feature_sampled.shape}')

        for i, row in X_test_feature_sampled.iterrows():
            #             print(i)
            node = self
            label = None
            while (not node.leaf):
                #                 print(node)
                # branch = np.random.randint(0,len(node.children) #debug
                query = row.values.reshape(1, -1)
                branch = self.clf[self.best_candidate].predict(query)[0]
                # print('branch' + str(branch))
                if branch in node.children:
                    node = node.children[branch]
                else:
                    label = branch
                    break;

            if label is None:
                scores[i] = node.label
            else:
                scores[i] = label
        #         print(scores)
        return scores



    def splitRidge(self, X, y, split_number):
        print(f'def m={self.m},X = {X.shape}', v=4)

        _features = list(range(0, X.shape[1]))
        _features = np.random.choice(_features, self.m, replace=False)
        self.features[split_number] = _features
        X_feature_sampled = X.iloc[:, _features]
        # vprint(self.features, v=4)
        #         raise KeyboardInterrupt

        # vprint(f'new m={self.m},X = {X.shape}, X_feature_sampled={X_feature_sampled.shape}', v=4)

        if self.params.get('cross_validate', True):
            self.clf[split_number] = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)
        else:
            alphas = 10 ** np.linspace(-3, 3, 10)
            a = alphas[random.randint(0, len(alphas))]
            self.clf[split_number] = RidgeClassifierCV(alphas=a, normalize=True)

        self.clf[split_number].fit(X_feature_sampled, y)
        _d = self.clf[split_number].predict(X_feature_sampled)
        df = pd.Series(_d)

        classes = df.unique()
        splits = {}  # [None]  * classes.shape[0]

        for i, row in df.iteritems():
            if row not in splits:
                splits[row] = []
            splits[row].append(i)
        #         print(row)

        return splits

    def gini(self, y):
        if y.shape[0] == 0:
            return 0

        return 1 - (((y.value_counts() / y.shape[0]).pow(2)).sum())

    def weighted_gini(self, splits, y):
        wg = 0
        for k, s in splits.items():
            wg += len(s) / y.shape[0] * self.gini(y.iloc[s])
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
    k = 1
    trees = None
    kwargs = None

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