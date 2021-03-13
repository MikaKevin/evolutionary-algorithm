import numpy as np

class VertexCover(object):
    def __init__(self, n):
        connected = np.zeros((n, n), dtype='bool')
        idx = np.random.randint(0, n, (n, 2))
        for i in range(n):
            for j in range(2):
                k = idx[i, j]
                if i != k:
                    connected[i, k] = True
                    connected[k, i] = True
        self._edge = []
        for i in range(n):
            e = []
            for j in range(n):
                if connected[i, j]:
                    e.append(j)
            self._edge.append(e)

    # fitness (to be minimized)
    # In case of constraint violations a very bad fitness is returned,
    # namely the number of constraint violations + n.
    def fitness(self, x):
        fit = 0   # number of selected vertices
        vio = 0   # number of constraint violations
        n = len(x)
        for i in range(n):
            if x[i]:
                fit += 1
            for j in self._edge[i]:
                if not x[i] and not x[j]:
                    vio += 1
        return n + vio if vio > 0 else fit
# GA Solution:
# random initialization
pop = np.random.choice([True, False], (100, 50))
# evaluation
fit = [instance.fitness(pop[i,:]) for i in range(100)]

# evolutionary loop
for generation in range(100):
    # find elitist
    elite = np.argmin(fit)

    # output progress
    print("generation: " + str(generation) + "   elite fitness: " + str(fit[elite]))

    # create new offspring generation
    offspring = pop
    for i in range(100):
        if i != elite:
            # tournament selection
            a = np.random.randint(0, 100)
            b = np.random.randint(0, 100)
            c = np.random.randint(0, 100)
            if fit[b] < fit[a]:
                a = b
            if fit[c] < fit[a]:
                a = c
            offspring[i,:] = pop[a,:]

            # standard bit-flip mutation
            flip = np.random.uniform(size=(100))
            for j in range(50):
                if flip[j] < 0.02:
                    offspring[i,j] = not offspring[i,j]

    # replace parents
    pop = offspring

    # evaluate
    fit = [instance.fitness(pop[i,:]) for i in range(100)]