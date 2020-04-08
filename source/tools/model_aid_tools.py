import numpy as np
import pandas as pd
import random


def drop_col(df, cols):
    """
        drop cols from dataFrame
    :param df: dataFrame
    :param cols: list, col names
    :return: after_drop
    """
    after_drop = df.drop(cols, axis=1)
    return after_drop


def check_miss(df):
    nan_num = df.isnull().sum()
    has_nan = False
    for i, v in nan_num.items():
        if v > 0:
            has_nan = True
            print('index: ', i, 'nan: ', v)
    if not has_nan:
        print('there is not nan in this data set')
    return has_nan


def drop_miss(df):
    return df.dropna()


def get_column_names(df, is_print=False):
    columns_names = df.columns.values.tolist()
    if is_print:
        print(" ".join(columns_names))
    return columns_names


def get_random_int(a, b):
    """
        get a random int between a and b
    :param a:
    :param b:
    :return:
    """
    return random.randint(a, b)


def get_n_random_int(a, b, n):
    """
        get n random int between a and b non-repeatedly
    :param a:
    :param b:
    :param n:
    :return:
    """
    indexList = list(range(a, b))
    randomIndex = random.sample(indexList, n)
    return randomIndex


def remove_repeat_element_and_sort_list(l, key=None, reverse=False):
    remove_repeat_list = list(set(l))
    remove_repeat_list.sort(key=key, reverse=reverse)
    return remove_repeat_list
