from random import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import class_distribution
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AR


# based on rise implementation in the package
# TODO multivariate, variable length

class RISESplitter:

    def __init__(self, node, tree,
                 random_state=None,
                 min_interval=16,
                 acf_lag=100,
                 acf_min_values=4,):
        self.tree = tree
        self.node = node
        self.random_state = random_state
        # random.seed(random_state)  # TODO
        self.min_interval = min_interval
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values
        self.maxlag = 16 #  TODO

        # These are all set in fit
        self.n_classes = 0
        self.classifiers = []
        self.intervals = []
        self.lags = []
        self.classes_ = []
        self.split_attribute = None
        self.split_threshold = None

        #init base splitter
        # TODO replace this with BestSplitter class from sklearn
        # Decision Stump which samples random features
        self.decision_stump = DecisionTreeClassifier(max_depth=1, max_features=self.tree.num_interval_candidate_splits)

    def split(self, X, y, class_indices):
        series_length = X.iloc[0].iloc[0].shape[0]

        num_attributes_generated = 0
        sub_features = []
        i = 0
        while num_attributes_generated <= self.tree.num_interval_candidate_splits:
            interval = self.generate_n_random_intervals(self.min_interval, series_length, 1)[0]
            if i % 4 == 0:
                sub_feature = self.ar(X, interval, maxlag=self.maxlag)
            elif i % 4 == 1:
                sub_feature = self.ar(X, interval, maxlag=self.maxlag)
            elif i % 4 == 2:
                sub_feature = self.ar(X, interval, maxlag=self.maxlag)
            elif i % 4 == 3:
                sub_feature = self.ar(X, interval, maxlag=self.maxlag)
            i += 1
            self.intervals.append(interval)
            num_attributes_generated = num_attributes_generated + sub_feature.shape[1]
            sub_features.append(sub_feature)

        all_features = np.hstack(sub_features)
        # attributes_to_use = np.random.choice(range(1, num_attributes_generated),
        #                                      self.tree.num_interval_candidate_splits,
        #                                      replace=False)
        # X_transformed = all_features[:,attributes_to_use]
        X_transformed = all_features
        self.decision_stump.fit(X_transformed, y)
        pred_y = self.decision_stump.predict(X_transformed)  # TODO training performance on a stump

        splits = self.node.get_class_indices(pred_y)

        return splits

    def predict(self, query, qi):

        sub_features = []
        i = 0
        for interval in self.intervals:
            if i % 4 == 0:
                sub_features.append(self.ar(query, interval, maxlag=self.maxlag))
            elif i % 4 == 1:
                sub_features.append(self.ar(query, interval, maxlag=self.maxlag))
            elif i % 4 == 2:
                sub_features.append(self.ar(query, interval, maxlag=self.maxlag))
            elif i % 4 == 3:
                sub_features.append(self.ar(query, interval, maxlag=self.maxlag))
            i += 1

        query_transformed = np.hstack(sub_features).ravel()
        query_transformed = query_transformed.reshape(1, -1)
        branch = self.decision_stump.predict(query_transformed)[0]

        return branch, None

    # TODO from segment class
    def generate_n_random_intervals(self, min_length, series_length, n_intervals=1):
        starts = np.random.randint(series_length - min_length + 1, size=n_intervals)
        if n_intervals == 1:
            starts = [starts]  # make it an iterable
        ends = [start + np.random.randint(min_length, series_length - start + 1) for start in starts]
        return np.column_stack([starts, ends])


    def ar(self, X, interval, maxlag=1):

        def _ar(x, interval, maxlag = 1):
            x = np.asarray(x).ravel()[0]
            nlags = np.minimum(len(x) - 1, maxlag)
            model = AR(endog=x)
            return model.fit(maxlag=nlags).params.ravel()

        if isinstance(X, pd.DataFrame):
            xt =[]
            for i in range(0, X.shape[0]):
                x = X.iloc[i]
                xt.append(_ar(x, interval, maxlag))
            X_transformed = pd.DataFrame(xt)
        else:
            X_transformed = pd.DataFrame(_ar(X, interval, maxlag))

        return X_transformed

    def acf(self, X, interval, maxlag=1):

        def _acf(x, interval, maxlag = 1):
            x = np.asarray(x).ravel()
            nlags = np.minimum(len(x) - 1, maxlag)
            return acf(x, nlags=nlags).ravel()

        if isinstance(X, pd.DataFrame):
            xt =[]
            for i in range(0, X.shape[0]):
                x = X.iloc[i]
                xt.append(_acf(x, interval, maxlag))
            X_transformed = pd.DataFrame(xt)
        else:
            X_transformed = pd.DataFrame(_acf(X, interval, maxlag))

        return X_transformed

    def pacf(self, X, interval):
        def _pacf(x, interval, maxlag = 1):
            # TODO
            return x

        if isinstance(X, pd.DataFrame):
            xt =[]
            for i in range(0, X.shape[0]):
                x = X.iloc[i]
                xt.append(_pacf(x, interval))
            X_transformed = pd.DataFrame(xt)
        else:
            X_transformed = pd.DataFrame(_pacf(X, interval))

        return X_transformed

    def ps(self, X, interval):

        def _ps(x, interval, maxlag = 1):
            x = np.asarray(x).ravel()
            fft = np.fft.fft(x)
            ps = fft.real * fft.real + fft.imag * fft.imag
            return ps[:ps.shape[0] // 2].ravel()

        if isinstance(X, pd.DataFrame):
            xt =[]
            for i in range(0, X.shape[0]):
                x = X.iloc[i]
                xt.append(_ps(x, interval))
            X_transformed = pd.DataFrame(xt)
        else:
            X_transformed = pd.DataFrame(_ps(X, interval))

        return X_transformed


def _ar_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    model = AR(endog=x)
    return model.fit(maxlag=nlags).params.ravel()


def _acf_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    return acf(x, nlags=nlags).ravel()


def _powerspectrum(x, **kwargs):
    x = np.asarray(x).ravel()
    fft = np.fft.fft(x)
    ps = fft.real * fft.real + fft.imag * fft.imag
    return ps[:ps.shape[0] // 2].ravel()


def _acf(x, max_lag):
    """ autocorrelation function transform, currently calculated using standard stats method.
    We could use inverse of power spectrum, especially given we already have found it, worth testing for speed and correctness
    HOWEVER, for long series, it may not give much benefit, as we do not use that many ACF terms

    Parameters
    ----------
    x : array-like shape = [interval_width]
    max_lag: int, number of ACF terms to find

    Return
    ----------
    y : array-like shape = [max_lag]

    """
    y = np.zeros(max_lag)
    length = len(x)
    for lag in range(1, max_lag + 1):
        # Do it ourselves to avoid zero variance warnings
        s1 = np.sum(x[:-lag])
        ss1 = np.sum(np.square(x[:-lag]))
        s2 = np.sum(x[lag:])
        ss2 = np.sum(np.square(x[lag:]))
        s1 = s1 / (length - lag)
        s2 = s2 / (length - lag)
        y[lag - 1] = np.sum((x[:-lag] - s1) * (x[lag:] - s2))
        y[lag - 1] = y[lag - 1] / (length - lag)
        v1 = ss1 / (length - lag) - s1 * s1
        v2 = ss2 / (length - lag) - s2 * s2
        if v1 <= 0.000000001 and v2 <= 0.000000001:  # Both zero variance, so must be 100% correlated
            y[lag - 1] = 1
        elif v1 <= 0.000000001 or v2 <= 0.000000001:  # One zero variance the other not
            y[lag - 1] = 0
        else:
            y[lag - 1] = y[lag - 1] / (math.sqrt(v1) * math.sqrt(v2))
    return np.array(y)


#        y[lag - 1] = np.corrcoef(x[lag:], x[:-lag])[0][1]
#        if np.isnan(y[lag - 1]) or np.isinf(y[lag-1]):
#            y[lag-1]=0


def _matrix_acf(x, num_cases, max_lag):
    """ autocorrelation function transform, currently calculated using standard stats method.
    We could use inverse of power spectrum, especially given we already have found it, worth testing for speed and correctness
    HOWEVER, for long series, it may not give much benefit, as we do not use that many ACF terms

     Parameters
    ----------
    x : array-like shape = [num_cases, interval_width]
    max_lag: int, number of ACF terms to find

    Return
    ----------
    y : array-like shape = [num_cases,max_lag]

    """

    y = np.empty(shape=(num_cases, max_lag))
    for lag in range(1, max_lag + 1):
        # Could just do it ourselves ... TO TEST
        #            s1=np.sum(x[:-lag])/x.shape()[0]
        #            ss1=s1*s1
        #            s2=np.sum(x[lag:])
        #            ss2=s2*s2
        #
        y[lag - 1] = np.corrcoef(x[:, lag:], x[:, -lag])[0][1]
        if np.isnan(y[lag - 1]) or np.isinf(y[lag - 1]):
            y[lag - 1] = 0
    return y


def _ps(x):
    """ power spectrum transform, currently calculated using np function.
    It would be worth looking at ff implementation, see difference in speed to java
    Parameters
    ----------
    x : array-like shape = [interval_width]

    Return
    ----------
    y : array-like shape = [len(x)/2]
    """
    fft = np.fft.fft(x)
    fft = fft.real * fft.real + fft.imag * fft.imag
    fft = fft[:int(len(x) / 2)]
    return np.array(fft)
