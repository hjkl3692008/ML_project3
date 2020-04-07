import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

class PCA(object):
    avg = None
    avg_mat = None
    mat_T_sub = None
    cov_mat = None
    featValue = None
    featVec = None
    selectVec = None
    variance_radio = None
    mat_pca = None

    def __init__(self, mat, k=None, percentage=None):
        self.mat = mat
        self.m, self.n = mat.shape
        if k is not None and k > self.n:
            raise ValueError('the k is bigger than features: %s > %s' % k, self.n)
        self.mat_T = mat.T
        self.k = k
        self.percentage = percentage

    @staticmethod
    def avg_row(mat):
        return np.mean(mat, axis=1)

    def subtract_avg(self):
        self.avg = self.avg_row(self.mat_T)
        self.avg_mat = np.tile(self.avg, (self.m, 1)).T
        return self.mat_T - self.avg_mat

    @staticmethod
    def cov(mat):
        return np.cov(mat)

    @staticmethod
    def feat_value_and_vector(cov_mat):
        # use eigh only when mat is symmetrical
        # if else, use eig
        featValue, featVec = np.linalg.eigh(cov_mat)
        return featValue, featVec

    def pca(self):
        """
            calculate the pca from the original matrix
            pca decrease the dimension of the original matrix
        :return:
            mat_pca : mat after pca
        """
        self.mat_T_sub = self.subtract_avg()
        self.cov_mat = self.cov(self.mat_T_sub)
        self.featValue, self.featVec = self.feat_value_and_vector(self.cov_mat)
        # diag = np.dot(np.dot(self.featVec.T, self.cov_mat), self.featVec)
        index = np.argsort(-self.featValue)

        # if percentage is not None, calculate k by percentage
        if self.percentage is not None:

            variance_sum = sum(self.featValue)
            self.variance_radio = self.featValue / variance_sum

            i = 0
            variance_contribution = 0
            while variance_contribution < self.percentage:
                variance_contribution += self.variance_radio[index[i]]
                i += 1

            self.k = i

        self.selectVec = np.array(self.featVec.T[index[:self.k]])
        self.mat_pca = np.dot(self.selectVec, self.mat_T_sub).T
        return self.mat_pca

    @staticmethod
    def plot_distribution(data, classification, path=None):
        """
            plot distribution when dimension = 2
        :param path: save path
        :param classification: list, classifications of nodes
        :param data: npArray, dimension = 2
        :return:
        """

        plt.scatter(data[:, 0], data[:, 1], marker='o', c=classification, cmap='summer')
        if path is not None:
            savefig(path)
        plt.show()
