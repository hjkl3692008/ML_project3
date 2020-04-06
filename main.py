import time
from source.tools import file_tools as ft
from source.tools import knn_tools as kt


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


if __name__ == '__main__':
    pass