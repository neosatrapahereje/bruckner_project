import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean(float[:, ::1] X, float[:, ::1] Y):
    """
    Pairwise Euclidean Distance
    """
    cdef int M = X.shape[0]
    cdef int N = Y.shape[0]
    cdef int L = X.shape[1]
    cdef float diff, dist
    cdef float[:, ::1] D = np.empty((M, N), dtype=np.float32)


    for i in range(M):
        for j in range(N):
            dist = 0.0
            for k in range(L):
                diff = X[i, k] - Y[j, k]
                dist += diff * diff
            D[i, j] = sqrt(dist)
    return np.asarray(D)



    
