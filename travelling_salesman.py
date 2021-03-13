import numpy as np
class TSP(object):
    def __init__(self, n):
        self.cities = np.random.uniform(size=(n, 2))

    def size(self):
        return self.cities.shape[0]

    def fitness(self, x):
        n = self.size()
        ret = np.linalg.norm(self.cities[x[n-1],:] - self.cities[x[0],:])
        for i in range(1, n):
            ret += np.linalg.norm(self.cities[x[i-1],:] - self.cities[x[i],:])
        return ret

from copy import deepcopy

def exhaustiveSearch(instance):
    n = instance.size()
    x = [0] * n
    bestF = 1e100
    bestX = deepcopy(x)
    while True:
        # check constraints
        feasible = True
        for i in range(1, n):
            if not feasible:
                break
            for j in range(0, i):
                if x[i] == x[j]:
                    feasible = False
                    break

        # evaluate fitness
        if feasible:
            f = instance.fitness(x)
            if f < bestF:
                bestF = f
                bestX = deepcopy(x)

        # move to next tour
        i = 0
        while i < n:
            if x[i]+1 < n:
                x[i] += 1
                break
            x[i] = 0
            i += 1
        if i == n:
            break
    return (bestF, bestX)
