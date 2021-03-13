

import numpy as np
def sample(lmbd, m, sigma, C):
    d = len(m)
    A = np.linalg.cholesky(C)
    z = [np.random.randn(d) for i in range(lmbd)]
    delta = [np.dot(A, z[i]) for i in range(lmbd)]
    x = [m + sigma * delta[i] for i in range(lmbd)]
    return z, delta, x