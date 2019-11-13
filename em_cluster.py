from operator import add
import sys
import random
import numpy as np
from scipy.stats import multivariate_normal

class Factory:
    def __init__(self, path):
        self.path = path
        self.data = np.loadtxt(path)[:,:-1]
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
class Cluster:
    def __init__(self, clusters, epoch, factory):
        self.clusters = int(clusters)
        self.epochs = int(epoch)
        self.factory = factory
        self.init_params()
        self.train()
    def init_params(self):
        self.prob = [[0 for _ in range(self.clusters)] for _ in range(self.factory.rows)]
        for i in range(len(self.prob)):
            self.prob[i][random.randint(0, self.clusters - 1)] = 1
    def train(self):
        for _ in range(self.epochs):
            means, weights, covariances = self.m_step()
            self.e_step(means, weights, covariances)
        with open("points.txt", "w") as f:
             for i,x in enumerate(self.factory.data):
                 f.write(f"{x[0]} {x[1]}")
                 for j in range(self.clusters):
                     f.write(f" {self.prob[i][j]}")
                 f.write("\n")
    def e_step(self, means, weights, covariances):
        bottom = []
        probabilities = []
        for x in self.factory.data:
            total = 0
            prob_j = []
            for i in range(self.clusters):
                gaussian = self.multivariate_gaussian(np.array(means[i]), np.array(covariances[i]), x)
                final = gaussian * weights[i]
                prob_j.append(final)
                total += final
            prob_j = [x/total for x in prob_j]
            probabilities.extend([prob_j])
        self.prob = probabilities
    def m_step(self):
        probabilities = np.array(self.prob)
        weights = []
        N_k = []
        for x in range(self.clusters):
            column_sum = np.sum(probabilities[:,x])
            array_sum = probabilities.sum()
            weights.append(column_sum / array_sum)
            N_k.append(column_sum)
        means = []
        for i in range(self.clusters):
            pij = N_k[i]
            temp = np.zeros((1,self.factory.cols))
            for x in self.factory.data:
                temp += pij * x
            temp /= pij
            means.extend(temp)
        means = np.array(means).tolist()
        covariances = []
        means = np.array(means)
        for cluster in range(self.clusters):
            temp = np.zeros((self.factory.cols, self.factory.cols))
            for i, x in enumerate(self.factory.data):
                prob = self.prob[i][cluster]
                means_cluster = means[cluster, :].reshape(1,self.factory.cols)
                first = (x - means_cluster).T.reshape(self.factory.cols, 1)
                second = (x - means_cluster).reshape(1, self.factory.cols)
                final = prob * (first @ second)
                temp += final
            if N_k[cluster != 0]:
                    temp = temp / N_k[cluster]
            for r in range(self.factory.cols):
                for x in range(self.factory.cols):
                    if r == x and temp[r][x] < 0.0001:
                        print("GOT ONE")
                        temp[r][x] = 0.0001
            covariances.extend([temp])
        covariances = np.array(covariances).tolist()
        means = means.tolist()
        return means, weights, covariances
    def multivariate_gaussian(self, mean, covariance, x):
        first = 1 / ( np.sqrt( np.power(2 * np.pi, self.factory.cols) * np.linalg.det(covariance) ) )
        second = np.exp( -0.5 * (x - mean).T @ np.linalg.pinv(covariance) @ (x - mean) )
        return first * second
        var = multivariate_normal(mean = mean, cov = covariance)
        return var.pdf(x)
def main():
    if len(sys.argv) < 4:
        print("usage: [path to file] k number_of_iterations")
        return
    clusters = Cluster(*sys.argv[2:4], Factory(sys.argv[1]))
if __name__ == "__main__":
    main()
