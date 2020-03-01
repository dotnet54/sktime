
import pandas as pd
import numpy as np

from sktime.contrib.ts_chief.shifaz_dev.test import load_dataset
from sktime.contrib.ts_chief.similarity import EDSplitter
from sktime.classifiers.distance_based import proximity_forest
from sktime.distances import elastic_cython
from sktime.utils import dataset_properties


def main():
    dataset_name = 'Coffee'
    x_train, y_train, x_test, y_test = load_dataset(dataset_name)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    print(f'Train: {x_train.shape}, Test: {x_test.shape}')

    # test_euclidean(x_train, dataset_name = dataset_name, type = 'train')
    test_ddtw(x_train, dataset_name = dataset_name, type = 'train')

def test_euclidean(dataset, **params):
    measure = EDSplitter.euclidean

    results = []
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            dist = measure(None, dataset.iloc[i], dataset.iloc[j])
            row = []
            row.append(i)
            row.append(j)
            row.append(dist)
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/euc_{params['dataset_name']}_{params['type']}.csv")
    return df

def test_dtw(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.dtw_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]
    windows = [w for w in range(1, series_len, 10)]
    print(f'testing {len(windows)} windows')
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for w in windows:
                dist =  measure(dataset.iloc[i], dataset.iloc[j], w=w)
                row = []
                row.append(i)
                row.append(j)
                row.append(dist)
                row.append(w)
                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/dtw_{params['dataset_name']}_{params['type']}.csv")
    return df

def test_ddtw(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.ddtw_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]
    windows = [w for w in range(1, series_len, 10)]
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for w in windows:
                dist =  measure(dataset.iloc[i], dataset.iloc[j], w=w)
                row = []
                row.append(i)
                row.append(j)
                row.append(dist)
                row.append(w)
                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/ddtw_{params['dataset_name']}_{params['type']}.csv")
    return df

def test_wdtw(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.wdtw_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]
    wdtw_g = np.random.uniform(0,1,100)
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for c in wdtw_cost:
                dist =  measure(dataset.iloc[i], dataset.iloc[j], g=g)
                row = []
                row.append(i)
                row.append(j)
                row.append(dist)
                row.append(g)
                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/wdtw_{params['dataset_name']}_{params['type']}.csv")
    return df

def test_wddtw(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.wddtw_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]
    wdtw_g = np.random.uniform(0,1,100)
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for g in wdtw_g:
                dist =  measure(dataset.iloc[i], dataset.iloc[j], g=g)
                row = []
                row.append(i)
                row.append(j)
                row.append(dist)
                row.append(g)
                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/wddtw_{params['dataset_name']}_{params['type']}.csv")
    return df

def test_twe(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.twe_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]
    # penalty = np.random.uniform(0,1,100)
    # stiffness = np.random.uniform(0,1,100)
    penalty =[0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
                    0.077777778, 0.088888889, 0.1]
    stiffness = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for twe_nu in stiffness:
                for twe_lambda in penalty:
                    dist =  measure(dataset.iloc[i], dataset.iloc[j],
                                    stiffness=twe_nu, penalty=twe_lambda)
                    row = []
                    row.append(i)
                    row.append(j)
                    row.append(dist)
                    row.append(twe_nu)
                    row.append(twe_lambda)
                    results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/twe_{params['dataset_name']}_{params['type']}.csv")
    return df

def test_msm(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.msm_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]
    msm_c = [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325,
              0.03625, 0.04, 0.04375, 0.0475, 0.05125,
              0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775,
              0.08125, 0.085, 0.08875, 0.0925, 0.09625,
              0.1, 0.136, 0.172, 0.208,
              0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496,
              0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748,
              0.784, 0.82, 0.856,
              0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8,
              3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68,
              6.04, 6.4, 6.76, 7.12,
              7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2,
              20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46,
              49.6, 53.2, 56.8, 60.4,
              64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4,
              100]
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for c in msm_c:
                dist =  measure(dataset.iloc[i], dataset.iloc[j], c=c)
                row = []
                row.append(i)
                row.append(j)
                row.append(dist)
                row.append(c)
                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/msm_{params['dataset_name']}_{params['type']}.csv")
    return df


def test_lcss(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.lcss_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]

    stdp = dataset_properties.stdp(dataset)
    instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    epsilon = stats.uniform(0.2 * stdp, stdp - 0.2 * stdp)
    window = stats.randint(low=0, high=max_raw_warping_window + 1)

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for w in window:
                for e in epsilon:
                    dist =  measure(dataset.iloc[i], dataset.iloc[j], delta=w, epsilon=e)
                    row = []
                    row.append(i)
                    row.append(j)
                    row.append(dist)
                    row.append(e)
                    row.append(w)
                    results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/lcss_{params['dataset_name']}_{params['type']}.csv")
    return df

def test_erp(dataset, **params):
    measure = proximity_forest.cython_wrapper(elastic_cython.erp_distance)
    results = []
    series_len = dataset.iloc[0].iloc[0].shape[0]
    stdp = dataset_properties.stdp(X)
    instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
    n_dimensions = 1  # todo use other dimensions
    erp_g = stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp)
    band_size = stats.randint(low=0, high=max_raw_warping_window + 1)

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            for w in band_size:
                for g in erp_g:
                    dist =  measure(dataset.iloc[i], dataset.iloc[j], band_size=w, g=g)
                    row = []
                    row.append(i)
                    row.append(j)
                    row.append(dist)
                    row.append(w)
                    row.append(g)
                    results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"E:/tmp/erp_{params['dataset_name']}_{params['type']}.csv")
    return df

main()