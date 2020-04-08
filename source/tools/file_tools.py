import json

import pandas as pd
import os
import struct
import numpy as np
import _pickle as pickle

# get cwd
def get_cwd():
    return os.getcwd()


# join path
def join_path(*args):
    path = ''
    for v in args:
        path = os.path.join(path, v)
    return path


# root path.  os.path.pardir = ..
# basic_path = join_path(get_cwd(), os.path.pardir, 'data')
basic_path = join_path(get_cwd(), 'resource')


# judge whether file exist
def is_exist(*args):
    full_path = join_path(basic_path, *args)
    flag = os.path.exists(full_path)
    return flag


# tans Dataform to nparray
def df2np(df):
    return np.array(df)


# load csv
def load_csv(path, is_to_np=False):
    data = pd.read_csv(path)
    if is_to_np:
        data = df2np(data)
    return data


# save csv
def save_csv(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, sep=',')


# load json
def load_json(*args):
    with open(join_path(basic_path, *args), 'r') as load_f:
        load_dict = json.load(load_f)
        return load_dict


def save_pickle(file, path):
    file_buffer = open(path, 'wb')
    pickle.dump(file, file_buffer)
    file_buffer.close()


def load_pickle(path):
    file_buffer = open(path, 'rb')
    file = pickle.load(file_buffer)
    file_buffer.close()
    return file


# load digit data
def load_digit(trans=True, is_unified=True):
    sample_data = load_csv(join_path(basic_path, 'digit', 'sample_submission.csv'))

    test_data = load_csv(join_path(basic_path, 'digit', 'test.csv'))
    # test_data = pd.read_csv(join_path(basic_path, 'digit', 'test.csv'), nrows=700)

    train_data = load_csv(join_path(basic_path, 'digit', 'train.csv'))
    if trans:
        sample_data = np.array(sample_data)
        test_data = np.array(test_data)
        train_data = np.array(train_data, dtype=np.uint16)
    if is_unified:
        train_data = unify_digit_format(train_data)
    return sample_data, test_data, train_data


# unify digit format
def unify_digit_format(train_data):
    label = train_data[:, 0]
    rest = train_data[:, 1:]
    result_data = np.column_stack((rest, label))
    return result_data


