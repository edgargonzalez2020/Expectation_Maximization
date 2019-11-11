import random
import sys
import numpy as np

class DataFactory:
    def __init__(self, path):
        self.path = path
        self.data = np.loadtxt(self.path, delimiter=",")[:,:-1]
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
class EM:
    def __init__(self, number_of_clusters, iterations, factory):
        self.number_of_clusters = int(number_of_clusters)
        self.iterations = int(iterations)
        self.factory = factory
        self.train()
    def train(self):
        probabilities = [[0 for x in range(self.number_of_clusters)]
                for _ in range(self.factory.data.shape[0])]
        for x in range(len(probabilities)):
            probabilities[x][random.randint(0, self.number_of_clusters-1)] = 1
        print(probabilities)
        for x in range(1):
            weights, means, probabilities, covariances = self.m_step(means, weights, probabilities, covariances)
            self.e_step(means, weights, probabilities, covariances)
    def e_step(self, means, weights, probabilities, covariances):
        for x in self.factory.data:
            for y in range(self.number_of_clusters):
                probabilities[tuple(x)][y] = (self.gaussian(x, means[y], covariances[y]) * weights[y]) / (1 / self.factory.rows)
    def m_step(self, means, weights, probabilities, covariances):
        means = [0] * self.number_of_clusters
        for x in range(self.number_of_clusters):
            top = 0
            for y in self.factory.data:
                prob = probabilities[tuple(y)]
                means[x] = top / sum(prob)
        return weights, means, probabilities, covariances
    def get_covariances(self, probabilities, means):
        covariances = []
        shape = self.factory.data.shape[1]
        total_prob = [0] * self.number_of_clusters
        for x in probabilities:
            for i, y in enumerate(probabilities[x]):
                total_prob[i] += y
        for x in range(self.number_of_clusters):
            cov = [[0] * shape] * shape
            term = 0
            for y in self.factory.data:
                for z in range(shape):
                    for a in range(shape):
                        prob = probabilities[tuple(y)][x]
                        term += prob * (y[z] - means[x][z]) * (y[a] - means[x][a])
                        cov[z][a] = term / total_prob[x]
                        if z == a and cov[z][a] < .0001:
                            cov[z][a] = .0001
            covariances.append(cov)
        return covariances
    def gaussian(self, x, mean, covariance):
        d = x.shape[0]
        first_term = 1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(covariance))
        second_term = np.exp(-0.5 * ( x - mean ).T * np.linalg.pinv(covariance) * ( x - mean ))
        return first_term * second_term
    def calculate_covariance(self):
        pass
def main():
    factory = DataFactory(sys.argv[1])
    cluster = EM(*sys.argv[2:5],factory)
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: [path_to_file] k iterations")
    main()
