import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean_cdist(float[:, ::1] X, float[:, ::1] Y):
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


@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean(double[::1] X, double[::1] Y):
    """
    Euclidean Distance
    """
    cdef int M = X.shape[0]
    cdef double diff, dist

    dist = 0.0
    for i in range(M):
        diff = X[i] - Y[i]
        dist += diff * diff
    return sqrt(dist)


@cython.boundscheck(False)
@cython.wraparound(False)
def cosine(double[::1] X, double[::1] Y):
    """
    Cosine Distance
    """
    cdef int M = X.shape[0]
    cdef double dot, cos, dist, norm_x, norm_y

    dot = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(M):
        dot += (X[i] * Y[i])
        norm_x += X[i] ** 2
        norm_y += Y[i] ** 2

    cos = dot / max(sqrt(norm_x) * sqrt(norm_y), 1e-10)

    dist = 1 - cos

    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def l1(double[::1] X, double[::1] Y):
    """
    L1- norm
    """
    cdef int M = X.shape[0]
    cdef double diff, dist

    dist = 0.0
    for i in range(M):
        diff = X[i] - Y[i]
        dist += abs(diff)
    return dist






    
