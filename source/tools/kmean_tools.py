import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig


class Kmean(object):
    k = None
    centers = None
    total_distance = None
    classification = None

    def __init__(self, mat, threshold=0.01):
        self.mat = mat
        self.m, self.n = mat.shape
        self.threshold = threshold

    @staticmethod
    def distance_2nodes(a, b, dim=2):
        return distance.minkowski(a, b, dim)

    def init_centers(self):
        index = np.random.permutation(self.m)
        return self.mat[index[:self.k], :]

    def nodes_belong2_centers(self, nodes, centers):
        node_classification = []
        total_distance = 0
        for node in nodes:
            distances = []
            # find distance between the node and all centers
            for center in centers:
                dis = self.distance_2nodes(center, node)
                distances.append(dis)
            # order distances, find min-distance and set it to the node
            node_classification.append(np.argmin(distances))
            total_distance += min(distances)
        return np.array(node_classification), total_distance

    def get_new_centers(self, node_classification):
        centers = []
        for i in range(self.k):
            index = np.where(node_classification == i)
            nodes = self.mat[index]
            center = np.mean(nodes, axis=0)
            centers.append(center)
        return centers

    @staticmethod
    def check(total_distance, last_total_distance, threshold, node_classification, last_node_classification):
        # stop condition
        # the change between two iterations is less than threshold
        check1 = np.abs(total_distance - last_total_distance) < threshold
        # the classification does not change in two iterations
        check2 = (node_classification == last_node_classification).all()
        return check1 or check2

    def train(self, k):

        self.k = k

        centers = self.init_centers()
        last_node_classification, last_total_distance = self.nodes_belong2_centers(self.mat, centers)
        centers = self.get_new_centers(last_node_classification)
        node_classification, total_distance = self.nodes_belong2_centers(self.mat, centers)
        while not self.check(total_distance, last_total_distance, self.threshold, node_classification,
                             last_node_classification):
            # if not pass the check, continue
            last_node_classification, last_total_distance = node_classification, total_distance
            centers = self.get_new_centers(last_node_classification)
            node_classification, total_distance = self.nodes_belong2_centers(self.mat, centers)

        self.centers = centers
        self.total_distance = total_distance
        self.classification = node_classification
        return node_classification

    def predict(self, test_set):
        centers = self.centers
        node_classification, total_distance = self.nodes_belong2_centers(test_set, centers)
        return node_classification

    def find_optimal_k(self, k_range, path=None):

        distances = []
        k_range = list(range(1, k_range))
        for k in k_range:
            k_model = Kmean(self.mat, self.threshold)
            k_model.train(k)
            total_distance = k_model.total_distance
            distances.append(total_distance)

        plt.plot(k_range, distances, '-o')
        plt.xlabel('Number of clusters, k')
        plt.ylabel('total_distance')
        plt.xticks(k_range)
        if path is not None:
            savefig(path)
        plt.show()
