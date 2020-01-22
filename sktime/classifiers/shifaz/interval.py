from random import random
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import class_distribution

#based on rise implementation in the package
class RISESplitter:

    def __init__(self, tree,
                 random_state=None,
                 min_interval=16,
                 acf_lag=100,
                 acf_min_values=4,
                 num_candidate_splits=1):
        self.tree = tree
        # TODO replace this with BestSplitter class from sklearn
        self.base_splitter = DecisionTreeClassifier
        self.random_state = random_state
        random.seed(random_state)
        self.min_interval=min_interval
        self.acf_lag=acf_lag
        self.acf_min_values=acf_min_values
        self.num_candidate_splits = num_candidate_splits
        # These are all set in fit
        self.n_classes = 0
        self.series_length = 0
        self.classifiers = []
        self.intervals=[]
        self.lags=[]
        self.classes_ = []

    def init_data(self):
        print('init data')

    def split(self, X, y, **extra_data):

        n_instances, self.series_length = X.shape

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.intervals = np.zeros((self.n_trees, 2), dtype=int)
        self.intervals[0][0] = 0
        self.intervals[0][1] = self.series_length
        for i in range(1, self.n_trees):
            self.intervals[i][0] = random.randint(self.series_length - self.min_interval)
            self.intervals[i][1] = random.randint(self.intervals[i][0] + self.min_interval, self.series_length)
        # Check lag against global properties
        if self.acf_lag > self.series_length - self.acf_min_values:
            self.acf_lag = self.series_length - self.acf_min_values
        if self.acf_lag < 0:
            self.acf_lag = 1
        self.lags = np.zeros(self.n_trees, dtype=int)
        for i in range(0, self.n_trees):
            temp_lag = self.acf_lag
            if temp_lag > self.intervals[i][1] - self.intervals[i][0] - self.acf_min_values:
                temp_lag = self.intervals[i][1] - self.intervals[i][0] - self.acf_min_values
            if temp_lag < 0:
                temp_lag = 1
            self.lags[i] = int(temp_lag)
            acf_x = np.empty(shape=(n_instances, self.lags[i]))
            ps_len = (self.intervals[i][1] - self.intervals[i][0]) / 2
            ps_x = np.empty(shape=(n_instances, int(ps_len)))
            for j in range(0, n_instances):
                acf_x[j] = acf(X[j, self.intervals[i][0]:self.intervals[i][1]], temp_lag)
                ps_x[j] = ps(X[j, self.intervals[i][0]:self.intervals[i][1]])
            transformed_x = np.concatenate((acf_x, ps_x), axis=1)
            #            transformed_x=acf_x
            tree = deepcopy(self.base_estimator)
            tree.fit(transformed_x, y)
            self.classifiers.append(tree)



        return []

    def predict(self, query, qi):


        return 0




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
    length=len(x)
    for lag in range(1, max_lag + 1):
# Do it ourselves to avoid zero variance warnings
        s1=np.sum(x[:-lag])
        ss1=np.sum(np.square(x[:-lag]))
        s2=np.sum(x[lag:])
        ss2=np.sum(np.square(x[lag:]))
        s1=s1/(length-lag)
        s2 = s2 / (length - lag)
        y[lag-1] = np.sum((x[:-lag]-s1)*(x[lag:]-s2))
        y[lag - 1] = y[lag - 1]/ (length - lag)
        v1 = ss1/(length - lag)-s1*s1
        v2 = ss2/(length-lag)-s2*s2
        if v1 <= 0.000000001 and v2 <= 0.000000001: # Both zero variance, so must be 100% correlated
            y[lag - 1]=1
        elif v1 <= 0.000000001 or v2 <= 0.000000001: # One zero variance the other not
            y[lag - 1] = 0
        else:
            y[lag - 1] = y[lag - 1]/(math.sqrt(v1)*math.sqrt(v2))
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

    y = np.empty(shape=(num_cases,max_lag))
    for lag in range(1, max_lag + 1):
        # Could just do it ourselves ... TO TEST
        #            s1=np.sum(x[:-lag])/x.shape()[0]
        #            ss1=s1*s1
        #            s2=np.sum(x[lag:])
        #            ss2=s2*s2
        #
        y[lag - 1] = np.corrcoef(x[:,lag:], x[:,-lag])[0][1]
        if np.isnan(y[lag - 1]) or np.isinf(y[lag-1]):
            y[lag-1]=0
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
    fft=np.fft.fft(x)
    fft=fft.real*fft.real+fft.imag*fft.imag
    fft=fft[:int(len(x)/2)]
    return np.array(fft)