import json

import pandas as pd
import os
import struct
import numpy as np


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




