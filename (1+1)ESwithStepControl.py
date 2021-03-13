import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt 
%matplotlib inline

class ElitistES:
    def __init__(self, fitness, x, sigma):
        self.fitness = fitness
        self.dimension = len(x)
        self.x = x
        self.fx = fitness(x)
        self.sigma = sigma

    def step(self):
        # perform a step
        pass   # TODO

    def bestPoint(self):
        # return the best known search point
        return self.x

    def bestFitness(self):
        # return the fitness of the best known point
        return self.fx

    def stepsize(self):
        return self.sigma

class ElitistES:
    def __init__(self, fitness, x, sigma):
        self.fitness = fitness
        self.dimension = len(x)
        self.x = x
        self.fx = fitness(x)
        self.sigma = sigma

    def step(self):
        # perform a step
        y = self.x + self.sigma * np.random.randn(self.dimension)
        fy = self.fitness(y)
        if fy <= self.fx:
            self.x = y
            self.fx = fy
            self.sigma = self.sigma * (1.0 + 1.0 / self.dimension)**4
        else:
            self.sigma = self.sigma / (1.0 + 1.0 / self.dimension)

    def bestPoint(self):
        # return the best known search point
        return self.x

    def bestFitness(self):
        # return the fitness of the best known point
        return self.fx

    def stepsize(self):
        return self.sigma

