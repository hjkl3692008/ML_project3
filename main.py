import time
import numpy as np
import pandas as pd
from source.tools import file_tools as ft
from source.tools import kmean_tools as kmt
from source.tools import pca_tools as pt
from source.tools import knn_tools as kt
from source.tools import model_aid_tools as mt
from source.tools import random_forest_tools as rt


def digit_RF():
    # load data
    digit_path = ft.join_path(ft.basic_path, 'digit')
    train_data = ft.load_csv(ft.join_path(digit_path, 'train.csv'))
    test_data = ft.load_csv(ft.join_path(digit_path, 'test.csv'))

    # RF
    rf = rt.RandomForest(train_data)
    m = 5
    rf.train(m=m, is_multiprocess=True, path=digit_path)

    # importance of features
    # tree_path = ft.join_path(digit_path, 'RF_importance_features.png')
    # rf.plot_feature_importance(rf.trees[0], path=tree_path)

    # prediction
    # tree0 = ft.load_pickle(ft.join_path(digit_path, 'random_forest_tree_0.pickle'))
    # rf.trees = [tree0]
    predict_labels = rf.predict(np.array(test_data))
    predict_file_path = ft.join_path(digit_path, 'RF_predict_%d.csv' % m)
    ft.save_csv(np.array(predict_labels).T, predict_file_path)


def digit_knn(is_split=False, k=5, is_find_k=False, is_norm=False, is_pca=False):
    start_time = time.time()
    digit_path = ft.join_path(ft.basic_path, 'digit')
    sample_data, test_data, train_data = ft.load_digit()

    # normalize train_data
    if is_norm:
        train_data = kt.mm_normalization(train_data, contain_label=True)

    # do pca
    if is_pca:
        # divide train and label
        train_label = train_data[:, train_data.shape[1] - 1]
        train_data = train_data[:, :train_data.shape[1] - 1]
        # do pca
        pca_model = pt.PCA(train_data, k=331)
        train_data = pca_model.do_pca()
        # plot when p = 2
        # plot_path = ft.join_path(digit_path, 'knn_pca_%d.png' % pca_model.k)
        # pca_model.plot_distribution(train_data, train_label, path=plot_path)
        # plot_path = ft.join_path(digit_path, 'knn_pca_variance_%d.png' % pca_model.k)
        # pca_model.plot_variance(pca_model.variance_ratio, path=plot_path)
        # plot_path = ft.join_path(digit_path, 'knn_pca_variance_line_%d.png' % pca_model.k)
        # pca_model.plot_variance_line(pca_model.variance_ratio, path=plot_path)
        # combine train and label
        train_data = np.column_stack((train_data, train_label))
        test_data = pca_model.trans_by_pca_mat(test_data)

    # split train data into two parts: new train data and test data
    if is_split:
        train_data, test_data, train_index, test_index = kt.split_train_test(train_data)

    # find proper k
    if is_find_k:
        max_ac, k = kt.find_k(train_data)

    result_data = kt.classify(test_data, train_data, k, dim=2, is_auto_dim=False, is_multiprocess=True)
    result_data_path = ft.join_path(digit_path, 'knn_result_k%d.csv' % k)
    ft.save_csv(result_data, result_data_path)

    # calculate confusion matrix
    is_calculate = is_split
    if is_calculate:
        confusion = kt.digit_confusion_matrix(result_data)
        confusion_path = ft.join_path(digit_path, 'knn_confusion_k%d.csv' % k)
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
    k_model = kmt.Kmean(np.array(clean_train_set))
    plot_path = ft.join_path(data_path, 'kmean_find_optimal_k.png')
    k_model.find_optimal_k(20, path=plot_path)

    # k-mean
    # by finding optimal k, 2 and 4 is good choice
    for k in (2, 4):
        k_model = kmt.Kmean(np.array(clean_train_set))
        node_classification = k_model.train(k=k)
        df = pd.DataFrame({'clust_label': node_classification, 'orig_label': labels.tolist()})
        ct = pd.crosstab(df['clust_label'], df['orig_label'])
        cluster_path = ft.join_path(data_path, 'kmean_cluster_%d.csv' % k)
        ct.to_csv(cluster_path)

    # pca
    for k in (2, 4):
        pca_model = pt.PCA(np.array(clean_train_set), k=2)
        pca_train_set = pca_model.do_pca()
        k_model = kmt.Kmean(pca_train_set)
        classifications = k_model.train(k=k)

        df = pd.DataFrame({'clust_label': classifications, 'orig_label': labels.tolist()})
        ct = pd.crosstab(df['clust_label'], df['orig_label'])
        cluster_path = ft.join_path(data_path, 'kmean_cluster_pca_%d.csv' % k)
        ct.to_csv(cluster_path)

        plot_path = ft.join_path(data_path, 'kmean_pca_%d.png' % k)
        pca_model.plot_distribution(pca_train_set, classifications, path=plot_path)
        plot_path = ft.join_path(data_path, 'kmean_pca_variance_%d.png' % k)
        pca_model.plot_variance(pca_model.variance_ratio, path=plot_path)


if __name__ == '__main__':
    # human_action()
    digit_RF()
    # digit_knn(is_split=False, k=3, is_find_k=False, is_norm=True, is_pca=True)
    pass
