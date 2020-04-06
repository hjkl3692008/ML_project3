import operator
import random
import time

from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

from source.tools import file_tools as ft


# split train data and test data
def split_train_test(d, percentage=0.9, is_save=False, path=''):
    # num of total data
    shape = d.shape
    num = shape[0]

    # shuffle index
    index = np.arange(num)
    random.shuffle(index)

    # calculate train's num and test's num
    train_num = int(num * percentage)
    test_num = num - train_num

    # get train's index and test's index
    train_index = index[0:train_num]
    test_index = index[train_num:num]

    # split data
    train_data = d[train_index]
    test_data = d[test_index]

    # save data
    if is_save:
        final_path = ft.join_path(ft.basic_path, 'resource', path)
        ft.save_csv(train_data, final_path)
        ft.save_csv(test_data, final_path)

    return train_data, test_data, train_index, test_index


class Tdtd:
    test_data = None
    train_data = None

    def __init__(self, test_data, train_data):
        self.test_data = test_data
        self.train_data = train_data


# split train data and test data by fold
def split_by_fold(d, fold):
    num = d.shape[0]
    index = np.arange(num)
    random.shuffle(index)

    d_list = {}
    for i in range(fold):
        i_range = range(0 + i, num, fold)
        train_data = d[index[i_range]]
        d_list[i] = train_data

    tdtd_list = []
    for k1, v1 in d_list.items():
        ov = np.array([])
        for k2, v2 in d_list.items():
            if k2 != k1:
                if ov.size == 0:
                    ov = v2
                else:
                    ov = np.concatenate((ov, v2))
        t = Tdtd(v1, ov)
        tdtd_list.append(t)

    return tdtd_list


# knn
def classify(test_data, train_data, k, dim=2, is_auto_dim=False, is_multiprocess=False):

    label_list = []
    if is_multiprocess:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(classify_one_data, i, train_data, k, dim=dim, is_auto_dim=is_auto_dim) for i in test_data]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                label_list.append(result)
    else:
        for i in test_data:
            label = classify_one_data(i, train_data, k, dim=dim, is_auto_dim=is_auto_dim)
            label_list.append(label)

    result_data = np.column_stack((test_data, np.array(label_list).T))
    return result_data


# 1-knn
def classify_one_data(d, train_data, k, dim=2, is_auto_dim=True):
    assert len(d) == train_data.shape[1] or len(d) == (train_data.shape[1] - 1)
    assert k <= train_data.shape[0]

    start_time = time.time()

    if is_auto_dim:
        dim = train_data.shape[1] - 1

    feature = train_data.shape[1] - 1

    d_array = []
    for i in train_data:
        i_distance = minkowski(d[0:feature], i[0:feature], dim)
        d_array.append(i_distance)

    s = np.argsort(d_array)

    class_array = train_data[s[0:k], train_data.shape[1] - 1]

    class_count = {}
    for i in class_array:
        label = i
        class_count[label] = class_count.get(label, 0) + 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    d_label = sorted_class_count[0][0]

    print('d_label: ' + str(d_label))
    print('classify_one_data execution in ' + str(time.time() - start_time), 'seconds')
    return d_label


# plot basic line
def plot_line(data, label=None, title='the basic line'):
    plt.plot(data[:, 0], data[:, 1], 'r--', label=label)
    plt.title(title)
    plt.legend()
    plt.show()


# plot multiple hist
def plot_hist(data, is_show=True):
    col = data.shape[1]
    col_names = data.columns.values
    d = np.array(data)
    for i in range(col):
        plt.subplot(col, 1, i + 1)
        plot_single_hist(d[:, i], False, col_names[i])
    if is_show:
        plt.show()


# hist
def plot_single_hist(data, is_show=True, legend=None):
    plt.hist(data, density=True, orientation='horizontal', color='#67abfe', label=legend)
    if legend is not None:
        plt.legend(loc='best')
    if is_show:
        plt.show()


# heat_map  data(dataFrame)
def heat_map(data, title='heat map table'):
    # get basic parameters from data
    col_num = data.shape[1]
    names = data.columns.values
    correction = data.corr()
    # plot correlation matrix
    ax = sns.heatmap(correction, cmap=plt.cm.Greys, linewidths=0.05, vmax=1, vmin=0, annot=True,
                     annot_kws={'size': 6, 'weight': 'bold'})
    plt.xticks(np.arange(col_num) + 0.5, names)
    plt.yticks(np.arange(col_num) + 0.5, names)
    ax.set_title(title)

    plt.show()


# find K data(nparray)
def find_k(data, test_range=30, is_plot=False, fold=10):
    k_range = range(1, test_range + 1, 2)  # ignore oven number
    avg_accuracy = []
    for k in k_range:
        accuracy = []
        td = split_by_fold(data, fold)
        for t in td:
            test_data = t.test_data
            train_data = t.train_data
            result_data = classify(test_data, train_data, k)
            fp, fn, tp, tn = fpntpn(result_data)
            sen, spec, acc = ssa(fp, fn, tp, tn)
            accuracy.append(acc)
        avg_accuracy.append(np.mean(accuracy))

    max_ac = max(avg_accuracy)
    k_list = np.array(list(k_range))
    k = k_list[avg_accuracy.index(max_ac)]

    if is_plot:
        aa = np.array(avg_accuracy)
        k_array = np.column_stack((k_list.T, aa.T))
        plot_line(k_array, label='k', title='the average accuracy in different k')

    return max_ac, k


# minkowski distance
def minkowski(a, b, dim=2):
    return distance.minkowski(a, b, dim)


# false positive, false negative, true positive, true negative
def fpntpn(d, c=None):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    index = d.shape[1] - 1
    if c is not None:
        d = d[np.where(d[:, index - 1] == c)]
    for i in d:
        if i[index - 1] == 1:
            if i[index] == i[index - 1]:
                tp = tp + 1
            else:
                fn = fn + 1
        if i[index - 1] == 0:
            if i[index] == i[index - 1]:
                tn = tn + 1
            else:
                fp = fp + 1
    return fp, fn, tp, tn


# fpntpn digit from 0~9
def fpntpn_digit(d):
    c_range = range(0, 10)
    fpntpn_list = np.array([])
    for c in c_range:
        fp, fn, tp, tn = fpntpn(d, c)
        ftpn = np.array([fp, fn, tp, tn])
        if c == 0:
            fpntpn_list = ftpn
        else:
            fpntpn_list = np.hstack((fpntpn_list, ftpn))
    return fpntpn_list


# sensitivity & specificity & accuracy
def ssa(fp, fn, tp, tn):
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return sen, spec, accuracy


# ssa digit
def ssa_digit(fpntpn_list):
    fp = fpntpn_list[:, 0]
    fn = fpntpn_list[:, 1]
    tp = fpntpn_list[:, 2]
    tn = fpntpn_list[:, 3]
    sen, spec, accuracy = ssa(fp, fn, tp, tn)
    np.hstack((fpntpn_list, sen, spec, accuracy))
    return fpntpn_list


# digit confusion matrix
def digit_confusion_matrix(result_data):
    labels = list(range(0, 10))

    confusion = np.array([])
    for label in labels:
        labels_count = one_label(result_data, label, labels)
        if confusion.size == 0:
            confusion = labels_count
        else:
            confusion = np.vstack((confusion, labels_count))

    return confusion


# one label confusion
def one_label(result_data, label, labels):

    col = result_data.shape[1]
    d = result_data[np.where(result_data[:, col - 2] == label)]

    class_count = {}
    for lab in labels:
        class_count[lab] = 0

    for i in d:
        lab = i[col-1]
        class_count[lab] = class_count.get(lab, 0) + 1

    labels_count = sorted(class_count.items(), key=operator.itemgetter(0))

    count_list = [label]
    for count in labels_count:
        count_list.append(count[1])
    count_array = np.array(count_list)

    return count_array


# normalizing functions start ###############

# todo:// unified entrance
def normalizing(n_type):
    0


# (feature - min)/(max - min)
def mm_normalization(d, contain_label=False):
    assert type(d) == np.ndarray
    n_d = np.array([])
    for i in range(d.shape[1]):
        feature = d[:, i]
        max_f = np.max(feature)
        min_f = np.min(feature)

        n_dc = None
        if contain_label & (i == d.shape[1] - 1):
            n_dc = d[:, i]
        elif max_f == min_f:
            n_dc = d[:, i] / 255  # todo:// 255 not a good choice
        else:
            n_dc = (feature - min_f)/(max_f - min_f)

        if i == 0:
            n_d = n_dc
        else:
            n_d = np.vstack((n_d, n_dc))

        # if i == 0:
        #     n_d = (feature - min_f)/(max_f - min_f)
        # elif contain_label & (i == d.shape[1] - 1):
        #     n_d = np.vstack((n_d, d[:, i]))
        # else:
        #     norm = (feature - min_f)/(max_f - min_f)
        #     n_d = np.vstack((n_d, norm))

    return n_d.T


# (x-μ)/σ
def z_score(d, contain_label=False):
    assert type(d) == np.ndarray
    n_d = np.array([])
    for i in range(d.shape[1]):
        feature = d[:, i]
        mu = np.average(feature)
        std = np.std(feature)

        n_dc = None
        if contain_label & (i == d.shape[1] - 1):
            n_dc = d[:, i]
        elif std == 0:
            n_dc = np.zeros(d[:, i].shape)
        else:
            n_dc = (feature - mu)/std

        if i == 0:
            n_d = n_dc
        else:
            n_d = np.vstack((n_d, n_dc))

        # if i == 0:
        #     n_d = (feature - mu)/std
        # elif contain_label & (i == d.shape[1] - 1):
        #     n_d = np.vstack((n_d, d[:, i]))
        # else:
        #     norm = (feature - mu)/std
        #     n_d = np.vstack((n_d, norm))

    return n_d.T


# 1/(1+sigmoid)
def sigmoid(d, contain_label=False):
    assert type(d) == np.ndarray
    n_d = np.array([])
    for i in range(d.shape[1]):
        if i == 0:
            n_d = 1.0 / (1 + np.exp(-d[:, i]))
        elif contain_label & (i == d.shape[1]-1):
            n_d = np.vstack((n_d, d[:, i]))
        else:
            norm = 1.0 / (1 + np.exp(-d[:, i]))
            n_d = np.vstack((n_d, norm))
    return n_d.T
# normalizing functions end ###############
