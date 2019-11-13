import sys
import random
import numpy as np
class Factory:
    def __init__(self, path):
        self.data = np.loadtxt(path)[:,:-1]
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
class EmCluster:
    def __init__(self, number_of_clusters, iterations, factory):
        self.number_of_clusters = int(number_of_clusters)
        self.iterations = int(iterations)
        self.factory = factory
    def train(self):
        self.prob = [[0 for _ in range(self.number_of_clusters)] for _ in range(self.factory.rows)]
        for x in range(self.factory.rows):
            self.prob[x][random.randint(0, self.number_of_clusters - 1)] = 1
        for i in range(1,self.iterations + 1):
            means, weights, covariances = self.m_step()
            self.e_step(means, weights, covariances)
        with open("points.txt", "w") as f:
            for i,x in enumerate(self.factory.data):
                f.write(f"{x[0]} {x[1]}")
                for j in range(self.number_of_clusters):
                    f.write(f" {self.prob[i][j]}")
                f.write("\n")
        #    print(f"After iteration {i}: ")
        #    count = 1
        #    for y in range(self.number_of_clusters):
        #        print(f"weight {count} = {weights[y]:.4f}, mean = {means[y-1]}")
        #        count += 1
        #print("After final iteration")
    def e_step(self, means, weights, covariances):
        print(np.array(covariances)[0].shape)
        bottom = []
        probabilities = []
        for x in self.factory.data:
            total = 0
            prob_j = []
            for i in range(self.number_of_clusters):
                gaussian = self.multivariate_gaussian(x,np.array(means[i]), np.array(covariances)[i])
                final = gaussian * weights[i]
                prob_j.append(final)
                total += final
            prob_j = [x/total for x in prob_j]
            probabilities.extend([prob_j])
        self.prob = probabilities

        #for cluster in range(self.number_of_clusters):
        #    for i,x in enumerate(self.factory.data):
        #        gaussian_term = self.multivariate_gaussian(x,np.array(means[cluster]),np.array(covariances[cluster]))
        #        upper = gaussian_term * weights[cluster]
        #        probabilities[i][cluster] = upper
        for i in range(len(probabilities)):
            print(probabilities[i])
            total = sum(probabilities[i])
            for j in range(len(probabilities[i])):
                probabilities[i][j] /= total
    def multivariate_gaussian(self, x, mean,covariance):
        d = self.factory.cols
        first = 1 / ( np.sqrt( np.power(2 * np.pi, d) * np.linalg.det(covariance) ) )
        second = np.exp( -0.5 * (x - mean).T @ np.linalg.pinv(covariance) @ (x - mean))
        return first * second
    def m_step(self):
        weights = [0 for _ in range(self.number_of_clusters)]
        means = []
        pij = [0 for _ in range(self.number_of_clusters)]
        covariances = []
        denominator = 0
        for cluster in range(self.number_of_clusters):
            numerator = 0
            for i, x in enumerate(self.factory.data):
                 numerator += self.prob[i][cluster]
                 denominator += self.prob[i][cluster]
            pij[cluster] = numerator
            weights[cluster] = numerator
        weights = [x/denominator for x in weights]
        self.prob = np.array(self.prob)
        for cluster in range(self.number_of_clusters):
            numerator = np.zeros((1, self.factory.cols))
            for i,x in enumerate(self.factory.data):
                prob = self.prob[i][cluster]
                numerator += prob * np.array(x)
            means.extend(numerator / pij[cluster])
        means = np.array(means).tolist()
        for cluster in range(self.number_of_clusters):
            temp = [[0 for _ in range(self.factory.cols)] for _ in range(self.factory.cols)]
            for r in range(self.factory.cols):
                for c in range(self.factory.cols):
                    prob = self.prob[:,cluster]
                    mu_r = means[cluster][r]
                    mu_c = means[cluster][c]
                    x_r = self.factory.data[:,r] - mu_r
                    x_c = self.factory.data[:,c] - mu_c
                    upper = np.sum(prob * x_r * x_c)
                    if r == c and (upper / pij[cluster]) < 0.0001:
                        temp[r][c] = 0.0001
                    else:
                        temp[r][c] = upper / pij[cluster]
            covariances.append(temp)
        return means, weights, covariances
def main():
    if len(sys.argv) < 4:
        print("usage: [path_to_file] k iterations")
    factory = Factory(sys.argv[1])
    cluster = EmCluster(*sys.argv[2:5], factory)
    cluster.train()
if __name__ == "__main__":
    main()
