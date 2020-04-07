import numpy as np
import pandas as pd


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


def get_column_names(df):
    columns_names = df.columns.values.tolist()
    print(" ".join(columns_names))
    return columns_names
