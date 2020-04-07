import time
import numpy as np
import pandas as pd
from source.tools import file_tools as ft
from source.tools import kmean_tools as kmt
from source.tools import pca_tools as pt
from source.tools import knn_tools as kt
from source.tools import model_aid_tools as mt


def digit(is_split=False, k=5, is_find_k=False, is_norm=False):
    start_time = time.time()
    digit_path = ft.join_path(ft.basic_path, 'resource', 'digit')
    train_data = ft.load_csv(ft.join_path(digit_path, 'train.csv'), is_to_np=True)
    test_data = ft.load_csv(ft.join_path(digit_path, 'test.csv'), is_to_np=True)

    # normalize train_data
    if is_norm:
        train_data = kt.mm_normalization(train_data, contain_label=True)

    # split train data into two parts: new train data and test data
    if is_split:
        train_data, test_data, train_index, test_index = kt.split_train_test(train_data)

    # find proper k
    if is_find_k:
        max_ac, k = kt.find_k(train_data)

    result_data = kt.classify(test_data, train_data, k, dim=3, is_auto_dim=False, is_multiprocess=True)
    result_data_path = ft.join_path(digit_path, 'result.csv')
    ft.save_csv(result_data, result_data_path)

    # calculate confusion matrix
    is_calculate = is_split
    if is_calculate:
        confusion = kt.digit_confusion_matrix(result_data)
        confusion_path = ft.join_path(digit_path, 'confusion.csv')
        ft.save_csv(confusion, confusion_path)
    print('digit execution in ' + str(time.time() - start_time), 'seconds')


def human_action():
    # load data
    data_path = ft.join_path(ft.basic_path, 'human_action')
    train_set = ft.load_csv(ft.join_path(data_path, 'train.csv'))

    sample = train_set.sample(8)

    # remove index and label
    labels = train_set['activity']
    clean_train_set = mt.drop_col(train_set, ['rn', 'activity'])

    # remove null
    has_nan = mt.check_miss(clean_train_set)
    if has_nan:
        clean_train_set = mt.drop_miss(clean_train_set)

    # find optimal k by elbow
    # k_model = kmt.Kmean(np.array(clean_train_set))
    # plot_path = ft.join_path(data_path, 'kmean_find_optimal_k.png')
    # k_model.find_optimal_k(20, path=plot_path)

    # k-mean
    # by finding optimal k, 2 and 4 is good choice
    for k in (2, 4):
        k_model = kmt.Kmean(np.array(clean_train_set))
        node_classification = k_model.kmean(k=k)
        df = pd.DataFrame({'clust_label': node_classification, 'orig_label': labels.tolist()})
        ct = pd.crosstab(df['clust_label'], df['orig_label'])
        cluster_path = ft.join_path(data_path, 'kmean_cluster_%d.csv' % k)
        ct.to_csv(cluster_path)

    # pca
    # pca_model = pt.PCA(np.array(clean_train_set), k=2)
    # pca_train_set = pca_model.pca()
    # k_model = kmt.Kmean(pca_train_set)
    # classifications = k_model.kmean(k=pca_train_set.shape[1])
    #
    # plot_path = ft.join_path(data_path, 'kmean_pca.png')
    # pca_model.plot_distribution(pca_train_set, classifications, path=plot_path)


if __name__ == '__main__':
    human_action()
    pass
