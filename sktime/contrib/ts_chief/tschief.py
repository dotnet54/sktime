# %%

from math import inf
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import inspect
from copy import deepcopy

from sktime.contrib.ts_chief.similarity import EDSplitter
from sktime.contrib.ts_chief.dict import BOSSplitter, BOSSDataStore
from sktime.contrib.ts_chief.interval import RISESplitter
from sktime.classifiers.base import BaseClassifier
from sktime.utils.validation.supervised import validate_X, validate_X_y
from sklearn.utils.multiclass import class_distribution

__author__ = "Ahmed Shifaz"
__all__ = ["TSChiefForest", "TSChiefTree"]

DEFAULT_LOWEST_GINI = 1e-7
DEFAULT_MAX_DEPTH = np.inf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TSChiefForest(BaseClassifier):

    def __init__(self, check_inputs=True,
                 random_state=None,
                 n_threads=0,
                 n_trees=100,
                 n_similarity_candidate_splits=5,
                 n_dictionary_candidate_splits=100,
                 n_interval_candidate_splits=100,
                 pf_similarity_measures=[],
                 boss_max_n_transformations=1000,
                 boss_min_window_length=10,
                 boss_word_lengths=[6, 8, 10, 12, 14, 16],
                 boss_norm_option=[True, False],
                 boss_alphabet_size=4,
                 rise_transforms=[],
                 rise_min_interval_length=16,
                 rise_max_lag=100,
                 rise_acf_min_values=4,
                 verbosity=1,
                 **debug_info):
        # params
        self.debug_info = debug_info  # TODO remove
        self.verbosity = verbosity
        self.n_trees = n_trees
        self.n_similarity_candidate_splits = n_similarity_candidate_splits
        self.n_dictionary_candidate_splits = n_dictionary_candidate_splits
        self.n_interval_candidate_splits = n_interval_candidate_splits
        self.splitters_initialized_before_train = False
        self.splitters_initialized_before_test = False
        self.boss_max_n_transformations = boss_max_n_transformations
        self.boss_data_store = None

        # initialize
        # args = inspect.getfullargspec(self.__init__) TODO
        self.series_length = 0
        self.n_instances = 0
        self.n_classes = 0
        self.class_distribution = None
        self.trees = [None] * self.n_trees
        for i in range(self.n_trees):
            self.trees[i] = TSChiefTree(ensemble=self,
                                        verbosity=verbosity,
                                        n_similarity_candidate_splits=n_similarity_candidate_splits,
                                        n_dictionary_candidate_splits=n_dictionary_candidate_splits,
                                        n_interval_candidate_splits=n_interval_candidate_splits,
                                        boss_max_n_transformations=boss_max_n_transformations)

    def fit(self, X, y, input_checks=True):

        if input_checks:
            # validate_X_y(X, y)
            if isinstance(X, pd.DataFrame):
                if X.shape[1] > 1:
                    raise TypeError("TS-CHIEF cannot handle multivariate problems yet")
                elif isinstance(X.iloc[0, 0], pd.Series):
                    # X = np.asarray([a.values for a in X.iloc[:, 0]])  # TODO dont flattten
                    pass
                else:
                    raise TypeError(
                        "Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series \
                        objects (TS-CHIEF cannot yet handle multivariate problems")

        self.n_instances, self.series_length = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.class_distribution = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        if not self.splitters_initialized_before_train and self.n_dictionary_candidate_splits > 0:
            self.boss_data_store = BOSSDataStore(self, self.boss_max_n_transformations)
            self.boss_data_store.initialize_before_train(X)

        for i, tree in enumerate(self.trees):
            if self.verbosity > 0:
                print(f'{i}.', end="")
                tree.fit(X, y)
        if self.verbosity > 0:
            print('')

        return self

    def predict(self, X, input_checks=True, **debug):
        if input_checks:
            # validate_X(X)
            if isinstance(X, pd.DataFrame):
                if X.shape[1] > 1:
                    raise TypeError("TS-CHIEF cannot handle multivariate problems yet")
                elif isinstance(X.iloc[0, 0], pd.Series):
                    # X = np.asarray([a.values for a in X.iloc[:, 0]])  # TODO dont flattten
                    pass
                else:
                    raise TypeError(
                        "Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series \
                        objects (TS-CHIEF cannot yet handle multivariate problems")

        n_test_instances, series_length = X.shape

        if series_length != self.series_length:
            raise TypeError("ERROR: number of attributes in the train data does not match to the test data")

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
        return f'TSChief-Forest: (k={self.n_trees})'

    def _setup_before_train(self, x_train, y_train, **kwargs):
        pass

    def _setup_before_test(self, x_test, y_test, **kwargs):
        pass


class TSChiefTree(BaseClassifier):

    def __init__(self, check_inputs=True,
                 max_depth=DEFAULT_MAX_DEPTH, lowest_gini=DEFAULT_LOWEST_GINI,
                 n_similarity_candidate_splits=1,
                 n_dictionary_candidate_splits=0,
                 n_interval_candidate_splits=0,
                 boss_max_n_transformations=1000,
                 verbosity=0,
                 ensemble=None):

        # ts_chief parameters
        self.verbosity = verbosity
        self.max_depth = max_depth  # maximum depth of trees
        self.lowest_gini = lowest_gini  # gini at which a node is converted to a leaf
        self.n_similarity_candidate_splits = n_similarity_candidate_splits
        self.n_dictionary_candidate_splits = n_dictionary_candidate_splits
        self.n_interval_candidate_splits = n_interval_candidate_splits
        self.boss_max_n_transformations = boss_max_n_transformations
        # pointer to the ensemble or forest if this tree is part of an ensemble,
        # some initializations may be delegated a forest level if this tree is part of an ensemble
        self.ensemble = ensemble

        # initialize
        self.children = {}
        self.split_functions = []
        self.best_split_function = None
        self.best_split = None  # holds the indicies of the splits
        self.root = None
        self.splitters_initialized_before_train = False
        self.splitters_initialized_before_test = False
        self.boss_data_store = None
        self.is_fitted = False

        # development
        self.node_count = 0
        self.tree_height = 0  # max depth

    def fit(self, X, y, input_checks=True):
        # TODO input_checks

        if not self.splitters_initialized_before_train and self.n_dictionary_candidate_splits > 0:
            if self.ensemble is None:
                # if this tree is on its own so do some initialization work
                self.boss_data_store = BOSSDataStore(self, self.boss_max_n_transformations)
                self.boss_data_store.initialize_before_train(X)
            else:
                # if this tree is part of a forest, assume that the forest class did the initialization
                self.boss_data_store = self.ensemble.boss_data_store

        self.splitters_initialized_before_train = True

        self.root = TSChiefNode(tree=self, parent_node=None)
        self.root.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X, input_checks=True, **debug):
        # TODO input_checks
        # TODO check is_fitted

        if not self.splitters_initialized_before_test and self.n_dictionary_candidate_splits > 0:
            self.boss_data_store.initialize_before_test(X)

        self.splitters_initialized_before_test = True
        scores = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            query = X.iloc[i]
            node = self.root
            label = None
            while not node.leaf:
                branch, _ = node.best_split_function.predict(query, i)
                if self.verbosity > 2:
                    if branch != debug["y"][i]:
                        print(f'branch: {branch}, true label:  {debug["y"][i]}')
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
        self.num_instances_reached = 0  # refer to stopping criteria
        self.class_distribution = None  # TODO class distribution of the data that reached this node
        self.class_indices = None  # indices of items belonging to each class

        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        self.branch_ = None  # TODO debug only

    def fit(self, X, y):
        self.num_instances_reached = y.shape[0]
        self.node_gini = self._gini(y)
        self.class_indices = self.get_class_indices(y)

        if self._check_stop_conditions(y):
            self.label = self._make_label(y)
            self.leaf = True
            return self

        if self.tree.n_similarity_candidate_splits > 0:
            ed_splitter = EDSplitter(self, self.tree)
            self.split_functions.append(ed_splitter)

        if self.tree.n_dictionary_candidate_splits > 0:
            boss_splitter = BOSSplitter(self, self.tree)
            self.split_functions.append(boss_splitter)

        if self.tree.n_interval_candidate_splits > 0:
            rise_splitter = RISESplitter(self, self.tree)
            self.split_functions.append(rise_splitter)

        top_candidate_splits = []
        for splitter in self.split_functions:
            candidate_split = splitter.split(X, y, class_indices=self.class_indices)
            top_candidate_splits.append(candidate_split)

        best_splitter_index = self._argmin(top_candidate_splits, self._weighted_gini, y)
        self.best_split_function = self.split_functions[best_splitter_index]

        # TODO improve for memory, currently stores the top best splits for each type of splitter in memory
        best_split = top_candidate_splits[best_splitter_index]

        for split_key, split_indices in best_split.items():
            if len(split_indices) > 0:
                self.children[split_key] = TSChiefNode(tree=self.tree, parent_node=self)
                self.children[split_key].branch_ = split_key  # TODO debug only
                self.children[split_key].fit(X.iloc[split_indices], y[split_indices])
            else:
                # TODO debug only
                if self.tree.verbosity > 1:
                    print(f'no data in the split {split_key} : {len(split_indices)}')

        return self

    def _check_stop_conditions(self, y):

        if self.node_gini <= self.tree.lowest_gini:
            return True
        elif self.parent is not None \
                and self.num_instances_reached == self.parent.num_instances_reached \
                and self.node_gini == self.parent.node_gini:
            if self.tree.verbosity > 1:
                print(f'Warn: no improvement to gini '
                      f'depth: {self.depth}, '
                      f'num_instances_reached: {self.num_instances_reached}, '
                      f'parent.num_instances_reached: {self.parent.num_instances_reached}, '
                      f'node_gini: {self.node_gini}, '
                      f'parent_gini: {self.parent.node_gini}, '
                      f'classes: {np.unique(y, return_counts=True)[1]}, '
                      f'label: {self._make_label(y)}')
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
        if self.tree.verbosity > 2:
            print(f'new leaf {self.depth}, {self.node_gini}, {label}, {y_train.shape}, {y_train}')
        return label

    def get_class_indices(self, y):
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
