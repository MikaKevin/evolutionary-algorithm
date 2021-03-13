import numpy as np
import math

class ESwithCMA:
    def __init__(self, fitness, mu, lmbd, m, sigma, useCMA):
        self.fitness = fitness
        self.dimension = len(m)
        # population size
        self.mu = mu
        self.lmbd = lmbd
        # mean, global step size, evolution path, and covariance matrix
        self.m = m
        self.sigma = sigma
        self.path = np.zeros(self.dimension)
        self.C = np.eye(self.dimension)
        self.useCMA = useCMA
        # learning rate
        self.eta = mu / (2.0 * self.dimension * self.dimension)
        # weights
        w = [math.log(self.mu) - math.log(i+1) for i in range(self.mu)]
        w /= np.sum(w)
        self.w = w

    def step(self):
        # TODO: implement this function
        pass

    def mean(self):
        # return the mean of the search distribution
        return self.m

    def stepsize(self):
        # return the "global" step size
        return self.sigma

    def covarianceMatrix(self):
        # return the covariance matrix
        return self.C