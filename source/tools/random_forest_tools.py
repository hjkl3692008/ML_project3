import numpy as np
import pandas as pd
import time
from collections import Counter
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from source.tools import model_aid_tools as mt
from source.tools import file_tools as ft

DIVIDE_TYPE_CHOOSE_THEN_MAX = 1
DIVIDE_TYPE_MAX_THEN_CHOOSE = 2


class RandomForest(object):
    m = None
    d = None
    trees = None
    bagging_data = None
    bagging_label = None

    def __init__(self, df):
        self.df = df
        self.shape = df.shape
        self.col_names = mt.get_column_names(df)
        self.original_data = np.array(df)

    def train(self, m, d=None, divide_type=1, is_multiprocess=False, path=None):
        """

        :param path: save tree path
        :param is_multiprocess: whether run in multiprocess
        :param divide_type:
                    1: random choose d features without replace, then choose max entropy to divide
                    2: choose d features with max entropy, then random choose one
        :param m: create m new data-set
        :param d: choose d features
        :return:
        """
        start_time = time.time()
        # set d
        if d is None:
            self.d = int(np.log2(self.original_data.shape[1]))

        # bagging data
        bagging_df = bagging(self.df, m)
        self.bagging_data = []
        self.bagging_label = []
        for i in range(m):
            data, label = divide_df(bagging_df[i])
            self.bagging_data.append(data)
            self.bagging_label.append(label)

        # create trees
        self.trees = []
        row_index = np.array(range(self.shape[0]))
        if is_multiprocess:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.CART, i, row_index, self.d, divide_type) for i in range(m)]
                i = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    self.trees.append((i, result))
                    i = i + 1
        else:
            for i in range(m):
                tree = self.CART(i, row_index, self.d, divide_type)
                ft.save_pickle((i, tree), ft.join_path(path, 'random_forest_tree_%d.pickle' % i))
                self.trees.append((i, tree))
                print('the', i, ' tree execution in ' + str(time.time() - start_time), 'seconds')

        print('RF execution in ' + str(time.time() - start_time), 'seconds')

    def predict(self, data):
        predict_labels = []
        for row in data:
            votes = []
            for i, tree in self.trees:
                v = self.tree_predict(row, tree)
                votes.append(v)
            c = Counter(votes)
            mode_of_votes = c.most_common(1)[0][0]
            predict_labels.append(mode_of_votes)
        return predict_labels

    def tree_predict(self, row, tree):
        feature = tree.get('feature')
        if feature == -1:
            labels = tree.get('label')
            c = Counter(labels)
            label = c.most_common(1)[0][0]
            return label

        divide_point = tree.get('divide_point')
        if row[feature] <= divide_point:
            return self.tree_predict(row, tree.get('left'))
        else:
            # row[feature] > divide_point
            return self.tree_predict(row, tree.get('right'))

    def get_data_by_index(self, data_index, row_index):
        return self.bagging_data[data_index][row_index], self.bagging_label[data_index][row_index]

    def CART(self, data_index, row_index, d, divide_type):

        divide = {
            'information': 0,
            'feature': -1,
            'divide_point': -1,
            'left_row_indexes': None,
            'right_row_indexes': None
        }

        data, label = self.get_data_by_index(data_index, row_index)
        can_continue_divide = self.check_stop_condition(data, label)
        if not can_continue_divide:
            # divide['data'] = data
            divide['label'] = label
            # print(divide)
            return divide

        if divide_type == DIVIDE_TYPE_CHOOSE_THEN_MAX:
            indexes = mt.get_n_random_int(0, self.shape[1], min(d, self.shape[1]))
            information_list = self.find_feature_with_max_gain(data_index, row_index, indexes)
            divide = information_list[0]
        elif divide_type == DIVIDE_TYPE_MAX_THEN_CHOOSE:
            information_list = self.find_feature_with_max_gain(data_index, row_index, indexes=None)
            random_index = mt.get_random_int(0, min(d, len(information_list)))
            divide = information_list[random_index]

        divide['left'] = self.CART(data_index, divide['left_row_indexes'], d, divide_type)
        divide['right'] = self.CART(data_index, divide['right_row_indexes'], d, divide_type)

        return divide

    def information_of_feature(self, data_index, row_index, index):
        return_dict = {
            'information': float("inf"),
            'feature': -1,
            'divide_point': -1,
            'left_row_indexes': None,
            'right_row_indexes': None
        }

        data, label = self.get_data_by_index(data_index, row_index)
        feature = data[:, index]
        points = mt.remove_repeat_element_and_sort_list(feature)

        information = float("inf")
        for i in range(len(points) - 1):

            point = (points[i] + points[i + 1]) / 2

            left_indexes = np.argwhere(feature < point).flatten()
            right_indexes = np.argwhere(feature > point).flatten()

            left_label = label[left_indexes]
            right_label = label[right_indexes]

            # todo: Gini or other standard
            left_gini = Gini(left_label)
            right_gini = Gini(right_label)

            total_len = len(label)
            left_len = len(left_label)
            right_len = len(right_label)
            gini = (left_len / total_len) * left_gini + (right_len / total_len) * right_gini

            if gini < information:
                return_dict.update({
                    'information': gini,
                    'feature': index,
                    'divide_point': point,
                    'left_row_indexes': row_index[left_indexes],
                    'right_row_indexes': row_index[left_indexes]
                })

        return return_dict

    def check_stop_condition(self, data, label):
        can_continue_divide = True

        def count(x):
            return len(set(x))

        # there are three stop condition
        # 1.only one data
        if data.shape[0] == 1:
            can_continue_divide = False
            return can_continue_divide
        # 2.all labels are the same
        if count(label) == 1:
            can_continue_divide = False
            return can_continue_divide
        # todo: bigger than max depth
        # 3.all features are the same
        is_same = True
        for i in range(data.shape[1]):
            if count(data[:, i]) == 1:
                continue
            else:
                is_same = False
                break
        if is_same:
            can_continue_divide = False
            return can_continue_divide

        return can_continue_divide

    def find_feature_with_max_gain(self, data_index, row_index, indexes=None, is_multiprocess=False):
        if indexes is None:
            # traverse all features
            traverse_range = range(0, len(row_index))
        else:
            traverse_range = indexes

        information_list = []
        if is_multiprocess:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.information_of_feature, data_index, row_index, i) for i in
                           traverse_range]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    information_list.append(result)
        else:
            for i in traverse_range:
                information_dict = self.information_of_feature(data_index, row_index, i)
                information_list.append(information_dict)

        information_list.sort(key=lambda x: x.get('information'))

        return information_list

    @staticmethod
    def plot_feature_importance(tree, path=None):
        """
            plot feature importance
        :param tree:
        :param path: save path
        :return:
        """

        def extract_data_from_tree(tree):
            feature = tree.get('feature')
            if feature == -1:
                return None
            information = tree.get('information')
            left = tree.get('left')
            right = tree.get('right')
            tree_data = [[feature, information]]
            left_data = extract_data_from_tree(left)
            if left_data is not None:
                tree_data.extend(left_data)
            right_data = extract_data_from_tree(right)
            if right_data is not None:
                tree_data.extend(right_data)
            return tree_data

        data = np.array(extract_data_from_tree(tree))
        data = data[data[:, 1].argsort()]

        plt.scatter(data[:, 0], data[:, 1], marker='o', alpha=0.6)
        if path is not None:
            savefig(path)
        plt.show()


def divide_df(df):
    label = np.array(df['label'], dtype=np.uint16)
    data = np.array(mt.drop_col(df, ['label']), dtype=np.uint16)
    return data, label


def bagging(df, m):
    """
    :param df: dataFrame, the original data
    :param m: create m new data-set from input dataFrame
    :return: m * data-set
    """
    col_names = mt.get_column_names(df)
    data = np.array(df)
    data_set = []
    for i in range(m):
        size = data.shape[0]
        sample_index = np.random.choice(size, size=size)
        sample_data = data[sample_index]
        sample_df = pd.DataFrame(sample_data, columns=col_names)
        data_set.append(sample_df)
    return data_set


def entropy(labels):
    length = len(labels)
    c = dict(Counter(labels))
    ent = 0.0
    for key, value in c.items():
        p = value / length
        ent -= p * np.log2(p)
    return ent


def Gini(labels):
    length = len(labels)
    assert length != 0
    c = dict(Counter(labels))
    gi = 0.0
    for key, value in c.items():
        p = value / length
        gi -= p ** 2
    gi = 1 - gi
    return gi


def ID3():
    pass


def C45():
    pass
