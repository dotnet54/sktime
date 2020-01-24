# %%
import inspect
from math import inf

import numpy as np
import pandas as pd
from scipy import stats

from sktime.contrib.ts_chief.similarity import EDSplitter
from sktime.contrib.ts_chief.dict import BOSSplitter, BOSSDataStore
from sktime.contrib.ts_chief.interval import RISESplitter

from sktime.classifiers.base import BaseClassifier

__author__ = "Ahmed Shifaz"
__all__ = ["TSChiefForest", "TSChiefTree"]

DEFAULT_LOWEST_GINI = 1e-7
DEFAULT_MAX_DEPTH = np.inf


class TSChiefForest(BaseClassifier):

    def __init__(self, check_inputs=True,
                 num_trees=1,
                 num_similarity_candidate_splits=1,
                 num_dictionary_candidate_splits=0,
                 num_interval_candidate_splits=0,
                 verbosity=1,
                 **kwargs):
        # params
        self.kwargs = kwargs  # TODO remove
        self.num_trees = num_trees
        self.verbosity = verbosity

        # initialize
        self.trees = []

        # args = inspect.getfullargspec(self.__init__) TODO
        for i in range(self.num_trees):
            tree = TSChiefTree(num_similarity_candidate_splits=num_similarity_candidate_splits,
                               num_dictionary_candidate_splits=num_dictionary_candidate_splits,
                               num_interval_candidate_splits=num_interval_candidate_splits)
            self.trees.append(tree)

    def fit(self, X, y, input_checks=True):
        # TODO input_checks
        for i, tree in enumerate(self.trees):
            if self.verbosity > 0:
                print(f'{i}.', end="")
                tree.fit(X, y)
        if self.verbosity > 0:
            print('')

        return self

    def predict(self, X, input_checks=True, **debug):
        # TODO input_checks
        predictions = [None] * len(self.trees)
        for i, tree in enumerate(self.trees):
            if self.verbosity > 0:
                print(f'{i}.', end="")
                predictions[i] = tree.predict(X, **debug)
        if self.verbosity > 0:
            print('')
        ensemble_prediction = self.majority_vote(predictions)
        return ensemble_prediction

    def predict_proba(self, X, input_checks=True):
        pass

    # TODO hacky - redo
    @staticmethod
    def majority_vote(score, save=False, file_prefix='forest.score.csv'):
        df = pd.DataFrame(score)
        if save:
            df.T.to_csv(file_prefix + ".score.csv")
        # display(df)
        # take a majority vote and randomly select from ties
        # predictions = df.T # for series
        y_pred = df.mode().fillna(method='ffill').sample(1)
        return y_pred.T

    def __repr__(self):
        return f'TSChief-Forest: (k={self.num_trees})'

    def _setup_before_train(self, x_train, y_train, **kwargs):
        pass

    def _setup_before_test(self, x_test, y_test, **kwargs):
        pass


class TSChiefTree(BaseClassifier):

    def __init__(self, check_inputs=True,
                 max_depth=DEFAULT_MAX_DEPTH, lowest_gini=DEFAULT_LOWEST_GINI,
                 num_similarity_candidate_splits=1,
                 num_dictionary_candidate_splits=0,
                 num_interval_candidate_splits=0,
                 verbosity=0,
                 ensemble=None):

        # ts_chief parameters
        self.verbosity = verbosity
        self.max_depth = max_depth  # maximum depth of trees
        self.lowest_gini = lowest_gini  # gini at which a node is converted to a leaf
        self.num_similarity_candidate_splits = num_similarity_candidate_splits
        self.num_dictionary_candidate_splits = num_dictionary_candidate_splits
        self.num_interval_candidate_splits = num_interval_candidate_splits
        # pointer to the ensemble or forest if this tree is part of an ensemble,
        # some initializations may be delegated a forest level if this tree is part of an ensemble
        self.ensemble = ensemble

        # initialize
        self.children = {}
        self.split_functions = []
        self.best_split_function = None
        self.best_split = None
        self.leaf = False
        self.label = None
        self.parent = None
        self.root = self
        self.depth = 0
        self.splitters_initialized_before_train = False
        self.splitters_initialized_before_test = False
        self.node_gini = None
        self.boss_data_store = None

        # development
        self.node_count = 0
        self.tree_height = 0  # max depth

    def fit(self, X, y, input_checks=True):
        # TODO input_checks

        if not self.splitters_initialized_before_train and self.num_dictionary_candidate_splits > 0:
            self.boss_data_store = BOSSDataStore()
            self.boss_data_store.initialize_before_train(self, X, y)

        self.splitters_initialized_before_train = True

        self.root = TSChiefNode(tree=self, parent_node=None)
        self.root.fit(X, y)

        return self

    def predict(self, X, input_checks=True, **debug):
        # TODO input_checks

        if not self.splitters_initialized_before_test and self.num_dictionary_candidate_splits > 0:
            self.boss_data_store.initialize_before_test(self, X)

        self.splitters_initialized_before_test = True
        scores = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            query = X.iloc[i]
            node = self.root
            label = None
            while not node.leaf:
                branch = node.best_split_function.predict(query, i)
                if self.verbosity > 1:
                    print(f'branch: {branch}, true label:  {debug["y"[i]]}')
                if branch in node.children:
                    node = node.children[branch]
                else:
                    label = branch
                    break
            if label is None:
                scores[i] = node.label
            else:
                scores[i] = label
        return scores

    def predict_proba(self, X, input_checks=True):
        pass

    def __repr__(self):
        return f'TSChiefTree:'


class TSChiefNode:
    # development notes:
    # keeping TSChiefNode separately from TSChiefTree class, though it looks unnecessary it improves code readability

    def __init__(self, tree=None, parent_node=None):
        self.tree = tree
        self.parent = parent_node
        self.children = {}
        self.split_functions = []
        self.best_split_function = None
        self.best_split = None
        self.leaf = False
        self.label = None
        self.node_gini = None  # gini of the data that reached this node
        self.class_distribution = None  # TODO class distribution of the data that reached this node
        self.class_indices = None  # indices of items belonging to each class

        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    def fit(self, X, y):
        self.node_gini = self._gini(y)
        self.class_indices = self._get_class_indices(X, y)

        if self._check_stop_conditions(y):
            self.label = self._make_label(y)
            self.leaf = True
            return self

        if self.tree.num_similarity_candidate_splits > 0:
            ed_splitter = EDSplitter(self, self.tree)
            self.split_functions.append(ed_splitter)

        if self.tree.num_dictionary_candidate_splits > 0:
            boss_splitter = BOSSplitter(self, self.tree)
            self.split_functions.append(boss_splitter)

        if self.tree.num_interval_candidate_splits > 0:
            rise_splitter = RISESplitter(self.self.tree)
            self.split_functions.append(rise_splitter)

        candidate_splits = []
        for splitter in self.split_functions:
            candidate_splits.append(splitter.split(X, y, class_indices=self.class_indices))

        best_splitter_index = self._argmin(candidate_splits, self.weighted_gini, y)
        self.best_split_function = self.split_functions[best_splitter_index]

        # TODO improve for memory, currently stores the top best splits for each type of splitter in memory
        splits = candidate_splits[best_splitter_index]

        for key, indices in splits.items():
            if len(indices) > 0:
                self.children[key] = TSChiefTree.as_child(self, **self.params)
                self.children[key].fit(X.iloc[indices], y[indices])
            else:
                # TODO debug only
                if self.tree.verbosity > 1:
                    print(f'no data in the split {key} : {len(indices)}')

        return self

    def _check_stop_conditions(self, y):

        if self.node_gini <= self.tree.lowest_gini:
            return True
        elif self.parent is not None and self.node_gini == self.parent.node_gini:
            if self.tree.verbosity > 1:
                print(f'no improvement to gini {self.depth}, {y.shape}, {self.node_gini}, {self.parent.node_gini}, '
                      f'{np.unique(y, return_counts=True)[1]}, {self.make_label(y)}')
            return True
        elif self.depth > self.tree.max_depth:
            # TODO debug only
            if self.tree.verbosity > 1:
                print(self._print_suffix + f'Error recursion too deep {self.depth}, {self.node_gini}')

            raise KeyboardInterrupt
            return True

        return False

    def _make_label(self, y_train):
        label = int(np.random.choice(stats.mode(y_train)[0], 1))
        if self.tree.verbosity > 1:
            print(f'new leaf {self.depth}, {self.node_gini}, {label}, {y_train.shape}, {y_train}')
        return label

    def _get_class_indices(self, X, y):
        split_indices = {}

        for cls in np.unique(y):
            grp = np.where(y == cls)
            split_indices[cls] = grp

        return split_indices

    def _gini(self, y):
        if y.shape[0] == 0:
            return 0
        #     print(y.shape[0])
        #     print(np.unique(y, return_counts=True)[1])
        #     print(np.unique(y, return_counts=True)[1] / y.shape[0])
        #     print(np.power(np.unique(y, return_counts=True)[1] / y.shape[0] , 2))
        #     print(np.sum(np.power(np.unique(y, return_counts=True)[1] / y.shape[0] , 2)))
        return 1 - np.sum(np.power(np.unique(y, return_counts=True)[1] / y.shape[0], 2))

    def _weighted_gini(self, splits, y):
        wg = 0
        for k, s in splits.items():
            wg += len(s) / y.shape[0] * self._gini(y[s])
        return wg

    def _argmin(self, candidate_splits, func_selector, parent_data):
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
        return f'TSChiefNode (leaf={self.leaf}, label={self.label}, children={len(self.children)}, depth={self.depth})'
