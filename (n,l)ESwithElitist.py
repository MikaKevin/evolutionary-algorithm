import math
import numpy as np
class NonElitistES:
    def __init__(self, fitness, m, sigma):
        self.fitness = fitness
        self.dimension = len(m)
        self.lbd = 4 + int(math.floor(3 * math.log(self.dimension)))
        self.mu = self.lbd // 2
        self.m = m
        self.sigma = sigma
        self.path = np.zeros(self.dimension)
        self.best_x = None
        self.best_f = None
        w = [math.log(self.mu) - math.log(i+1) for i in range(self.mu)]
        w /= np.sum(w)
        self.w = w

    def step(self):
        # sample lambda offspring
        x = self.sigma * np.random.randn(self.lbd, self.dimension) + self.m

        # evaluate offspring
        fx = [self.fitness(x[i]) for i in range(self.lbd)]
        order = np.argsort(fx)

        # update best point so far
        if self.best_f is None or fx[order[0]] < self.best_f:
            self.best_x = x[order[0]]
            self.best_f = fx[order[0]]

        # weighted recombination
        new_m = np.zeros(self.dimension)
        for i in range(self.mu):
            new_m += self.w[i] * x[order[i]]

        # evolution path update
        c = 1.0 / math.sqrt(self.dimension)
        self.path = (1 - c) * self.path + math.sqrt(c * (2 - c)) / self.sigma * (new_m - self.m)

        # cumulative step size adaptation
        self.sigma *= math.exp(c / self.dimension * (np.dot(self.path, self.path) - self.dimension))

        # overwrite mean
        self.m = new_m

    def getMu(self):
        return self.mu

    def getLambda(self):
        return self.lbd

    def bestPoint(self):
        # return the best known search point
        return self.best_x

    def bestFitness(self):
        # return the fitness of the best known point
        return self.best_f

    def stepsize(self):
        return self.sigma
