import numpy as np
import math

def sphere(x):
    return np.linalg.norm(x)**2

def feasible(x):
    return x[0] >= 1 and x[1] >= 1 and x[2] >= 1 and x[3] >= 1

class ESwithCMA:
    def __init__(self, fitness, feasible, mu, lmbd, m, sigma):
        self.fitness = fitness
        self.feasible = feasible
        self.dimension = len(m)
        # population size
        self.mu = mu
        self.lmbd = lmbd
        # mean, global step size, and covariance matrix
        self.m = m
        self.sigma = sigma
        self.C = np.eye(self.dimension)
        # learning rate
        self.eta = mu / (2.0 * self.dimension * self.dimension)
        # weights
        self.w = [mu - r + 0.5 for r in range(1, mu+1)]
        self.w /= np.sum(self.w)
        self.numFitnessEvaluations = 0
        self.numConstraintEvaluations = 0

    def step(self):
        D, U = np.linalg.eig(self.C)
        A = np.real(U @ np.diag(np.sqrt(np.abs(D))))   # better numerical stability
        z = [None] * self.lmbd
        x = [None] * self.lmbd
        fx = [None] * self.lmbd
        for i in range(self.lmbd):
            # sample offspring
            z[i] = np.random.randn(self.dimension)
            x[i] = self.m + self.sigma * np.dot(A, z[i])

            # evaluate offspring
            fx[i] = self.fitness(x[i])
            self.numFitnessEvaluations += 1

        # sort offspring by fitness
        order = np.argsort(fx)

        # adapt covariance matrix
        C_hat = np.zeros([self.dimension, self.dimension])
        for i in range(self.mu):
            delta = (x[order[i]] - self.m)
            C_hat += self.w[i] * np.outer(delta, delta)
        self.C = (1.0 - self.eta) * self.C + (self.eta / self.sigma**2) * C_hat

        # adapt the step size
        s = 0.0
        for i in range(self.mu):
            s += self.w[i] * np.linalg.norm(z[order[i]])**2
        self.sigma *= math.exp(s / self.dimension - 1.0)

        # adapt the mean
        m_hat = np.zeros(self.dimension)
        for i in range(self.mu):
            m_hat += self.w[i] * x[order[i]]
        self.m = m_hat

    def mean(self):
        return self.m

    def stepsize(self):
        return self.sigma

    def covarianceMatrix(self):
        return self.C

    def numberOfFitnessEvaluations(self):
        return self.numFitnessEvaluations

    def numberOfConstraintEvaluations(self):
        return self.numConstraintEvaluations
    
class ESwithCMA_Resampling:
    def __init__(self, fitness, feasible, mu, lmbd, m, sigma):
        self.fitness = fitness
        self.feasible = feasible
        self.dimension = len(m)
        # population size
        self.mu = mu
        self.lmbd = lmbd
        # mean, global step size, and covariance matrix
        self.m = m
        self.sigma = sigma
        self.C = np.eye(self.dimension)
        # learning rate
        self.eta = mu / (2.0 * self.dimension * self.dimension)
        # weights
        self.w = [mu - r + 0.5 for r in range(1, mu+1)]
        self.w /= np.sum(self.w)
        self.numFitnessEvaluations = 0
        self.numConstraintEvaluations = 0

    def step(self):
        # sample offspring
        D, U = np.linalg.eig(self.C)
        A = np.real(U @ np.diag(np.sqrt(np.abs(D))))   # better numerical stability
        z = [None] * self.lmbd
        x = [None] * self.lmbd
        fx = [None] * self.lmbd
        for i in range(self.lmbd):
            # sample offspring
            while True:
                z[i] = np.random.randn(self.dimension)
                x[i] = self.m + self.sigma * np.dot(A, z[i])
                self.numConstraintEvaluations += 1
                if self.feasible(x[i]):
                    break

            # evaluate offspring
            fx[i] = self.fitness(x[i])
            self.numFitnessEvaluations += 1

        # sort offspring by fitness
        order = np.argsort(fx)

        # adapt covariance matrix
        C_hat = np.zeros([self.dimension, self.dimension])
        for i in range(self.mu):
            delta = (x[order[i]] - self.m)
            C_hat += self.w[i] * np.outer(delta, delta)
        self.C = (1.0 - self.eta) * self.C + (self.eta / self.sigma**2) * C_hat

        # adapt the step size
        s = 0.0
        for i in range(self.mu):
            s += self.w[i] * np.linalg.norm(z[order[i]])**2
        self.sigma *= math.exp(s / self.dimension - 1.0)

        # adapt the mean
        m_hat = np.zeros(self.dimension)
        for i in range(self.mu):
            m_hat += self.w[i] * x[order[i]]
        self.m = m_hat

    def mean(self):
        # return the mean of the search distribution
        return self.m

    def stepsize(self):
        # return the "global" step size
        return self.sigma

    def covarianceMatrix(self):
        # return the covariance matrix
        return self.C

    def numberOfFitnessEvaluations(self):
        return self.numFitnessEvaluations

    def numberOfConstraintEvaluations(self):
        return self.numConstraintEvaluations
