import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib import cm

class PCA(object):

    mat_T_sub = None
    cov_mat = None
    featValue = None
    featVec = None
    selectVec = None
    variance_ratio = None
    mat_pca = None

    def __init__(self, mat, k=None, percentage=None):
        self.mat = mat
        self.m, self.n = mat.shape
        if k is not None and k > self.n:
            raise ValueError('the k is bigger than features: %s > %s' % k, self.n)
        self.k = k
        self.percentage = percentage

    @staticmethod
    def avg_row(mat):
        return np.mean(mat, axis=1)

    # def subtract_avg(self):
    #     self.avg = self.avg_row(self.mat_T)
    #     self.avg_mat = np.tile(self.avg, (self.m, 1)).T
    #     return self.mat_T - self.avg_mat

    def subtract_avg(self, mat):
        avg = self.avg_row(mat.T)
        avg_mat = np.tile(avg, (mat.shape[0], 1)).T
        return mat.T - avg_mat

    @staticmethod
    def cov(mat):
        return np.cov(mat)

    @staticmethod
    def feat_value_and_vector(cov_mat):
        # use eigh only when mat is symmetrical
        # if else, use eig
        featValue, featVec = np.linalg.eigh(cov_mat)
        return featValue, featVec

    def calculate_variance_ratio(self):
        variance_sum = sum(self.featValue)
        self.variance_ratio = self.featValue / variance_sum

    def do_pca(self):
        """
            calculate the pca from the original matrix
            pca decrease the dimension of the original matrix
        :return:
            mat_pca : mat after pca
        """
        self.mat_T_sub = self.subtract_avg(self.mat)
        self.cov_mat = self.cov(self.mat_T_sub)
        self.featValue, self.featVec = self.feat_value_and_vector(self.cov_mat)
        # diag = np.dot(np.dot(self.featVec.T, self.cov_mat), self.featVec)
        index = np.argsort(-self.featValue)

        # if percentage is not None, calculate k by percentage
        self.calculate_variance_ratio()
        if self.percentage is not None:

            i = 0
            variance_contribution = 0
            while variance_contribution < self.percentage:
                variance_contribution += self.variance_ratio[index[i]]
                i += 1

            self.k = i

        self.selectVec = np.array(self.featVec.T[index[:self.k]])
        self.mat_pca = np.dot(self.selectVec, self.mat_T_sub).T
        return self.mat_pca

    def trans_by_pca_mat(self, mat):
        mat_T_sub = self.subtract_avg(mat)
        return np.dot(self.selectVec, mat_T_sub).T

    @staticmethod
    def plot_distribution(data, classification, path=None):
        """
            plot distribution when dimension = 2
        :param path: save path
        :param classification: list, classifications of nodes
        :param data: npArray, dimension = 2
        :return:
        """

        plt.scatter(data[:, 0], data[:, 1], marker='o', c=classification, cmap='rainbow')  # summer
        if path is not None:
            savefig(path)
        plt.show()

    @staticmethod
    def plot_variance(variance_ratio, path=None):
        """
            plot distribution
        :param variance_ratio: npArray
        :param path: save path
        :return:
        """
        variance_ratio = variance_ratio[np.where(variance_ratio > 0.01)]

        map_vir = cm.get_cmap(name='viridis')
        color = map_vir(range(len(variance_ratio)))
        plt.bar(range(len(variance_ratio)), variance_ratio, color=color)
        if path is not None:
            savefig(path)
        plt.show()

    @staticmethod
    def plot_variance_line(variance_ratio, path):
        """
            plot distribution line
        :param variance_ratio: npArray
        :param path: save path
        :return:
        """
        flatten_ratio = variance_ratio.flatten()

        index = np.argsort(-flatten_ratio)
        ordered_ratio = flatten_ratio[index]
        sum_ratio = np.zeros(ordered_ratio.shape)

        sum_num = 0
        for i in range(len(ordered_ratio)):
            ratio = ordered_ratio[i]
            sum_num += ratio
            sum_ratio[i] = sum_num

        plt.plot(range(len(sum_ratio)), sum_ratio, 'r')
        if path is not None:
            savefig(path)
        plt.show()
