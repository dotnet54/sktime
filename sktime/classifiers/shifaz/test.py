import pandas as pd
import numpy as np

from sktime.utils.load_data import load_from_tsfile_to_dataframe

def load_dataset(dataset_name):
    # train_x, train_y = load_from_tsfile_to_dataframe("../sktime/datasets/data/GunPoint/GunPoint_TRAIN.ts")
    # test_x, test_y = load_from_tsfile_to_dataframe("../sktime/datasets/data/GunPoint/GunPoint_TEST.ts")

    data_path = "/data/"
    archive_name ='Univariate_ts'

    train_file = data_path + archive_name + '/' + dataset_name + '/' +  dataset_name + '_TRAIN.ts'
    test_file = data_path + archive_name + '/' + dataset_name + '/' +  dataset_name + '_TEST.ts'

    train_x, train_y = load_from_tsfile_to_dataframe(train_file)
    test_x, test_y = load_from_tsfile_to_dataframe(test_file)

    return train_x, train_y, test_x, test_y


def test_ee():

    return None


def test_boss():

    return None


def test_rise():

    return None

def test():

    train_x, train_y, test_x, test_y = load_dataset('GunPoint')

    # print(train_x)

# test()