class MOES:
    def __init__(self, fitness, x, sigma):
        self.fitness = fitness
        self.popsize = x.shape[0]
        self.dimension = x.shape[1]
        self.x = x                                             # matrix with individuals as rows
        self.fx = [fitness(x[i]) for i in range(x.shape[0])]   # matrix with fitness vectors as rows
        self.sigma = sigma                                     # vector of step sizes

        # add space for a single offspring, which is maintained at the end of the population arrays
        self.x = np.append(self.x, np.zeros((1, self.dimension)), axis = 0)
        self.fx = np.append(self.fx, np.zeros((1, 2)), axis = 0)
        self.sigma = np.append(self.sigma, np.zeros(1))

    # This function returns the list of indices of the individuals that
    # are non-dominated, excluding the offspring slot
    def firstrank(self):
        ret = []
        for r in range(self.popsize):
            # check whether the objective vector self.fx[r] dominates another objective vector
            dominated = False
            for s in range(self.popsize):
                if self.fx[r, 0] > self.fx[s, 0] and self.fx[r, 1] > self.fx[s, 1]:
                    # fx[s] dominates fx[r]
                    dominated = True
                    break
            if not dominated:
                # add r to the first rank
                ret.append(r)
        assert len(ret) > 0
        return ret

    # This function returns a list of indices of the individuals which
    # don't dominate any other individuals: the last non-dominance rank.
    def lastrank(self):
        ret = []
        for r in range(self.popsize + 1):
            # check whether the objective vector self.fx[r] dominates another objective vector
            dominating = False
            for s in range(popsize + 1):
                if self.fx[r, 0] < self.fx[s, 0] and self.fx[r, 1] < self.fx[s, 1]:
                    # fx[r] dominates fx[s]
                    dominating = True
                    break
            if not dominating:
                # add r to the last rank
                ret.append(r)
        return ret

    # This function returns the least contributor from the last non-dominance rank.
    # 'front' is supposed to contain the fitness values of these individuals.
    # The index of one of the least contributors is returned. The function avoids
    # to return "extreme" solutions, which are minimal in one objective.
    def minimalContributor(self, front):
        n = front.shape[0]
        if n <= 2:
            return np.random.randint(n)
        order = np.argsort(front[:,0])
        min_contrib = 1e100
        index = -1
        for i in range(1, n-1):
            contrib = (front[order[i+1]][0] - front[order[i]][0]) * (front[order[i-1]][1] - front[order[i]][1])
            assert contrib >= 0
            if contrib < min_contrib:
                min_contrib = contrib
                index = order[i]
        assert index != -1
        return index

    def step(self):
        pass   # TODO: implement this function

    # return the non-dominated solutions
    def points(self):
        return self.x[self.firstrank()]

    # return the non-dominated fitnesses
    def fitnesses(self):
        return self.fx[self.firstrank()]

    # return the parent population's step sizes
    def stepsize(self):
        return self.sigma[0:self.popsize]