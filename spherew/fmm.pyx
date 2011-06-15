cimport numpy as np
cimport cython
import numpy as np

cdef extern from "fmm1d.h":
    void fastsht_fmm1d(double *x_grid,  double *gamma,
                       double *q, size_t nx,
                       double *y_grid,  double *omega,
                       double *phi, size_t ny, size_t nvecs)

cdef extern from "math.h":
    double exp(double)

cdef extern from "mkl_vml.h":
    cdef enum:
        VML_HA
        VML_LA
        VML_EP
        VML_ERRMODE_STDERR
    int vmlSetMode(int)
    void vdExp(Py_ssize_t n, double *a, double *y)
    

def fmm1d(np.ndarray[double, mode='c'] x_grid,
          np.ndarray[double, ndim=2, mode='c'] q,
          np.ndarray[double, mode='c'] y_grid,
          np.ndarray[double, mode='c'] omega=None,
          np.ndarray[double, mode='c'] gamma=None,
          np.ndarray[double, ndim=2, mode='c'] out=None,
          int repeat=1):
    cdef int i
    if out is None:
        out = np.zeros((y_grid.shape[0], q.shape[1]))
    if omega is None:
        omega = y_grid * 0 + 1
    if gamma is None:
        gamma = x_grid * 0 + 1
    if (not x_grid.shape[0] == q.shape[0] == gamma.shape[0]
        or not y_grid.shape[0] == out.shape[0] == omega.shape[0]
        or q.shape[1] != out.shape[1]):
        raise ValueError("Shapes do not conform")
    for i in range(repeat):
        fastsht_fmm1d(<double*>x_grid.data, <double*>gamma.data, <double*>q.data, x_grid.shape[0],
                      <double*>y_grid.data, <double*>omega.data, <double*>out.data, y_grid.shape[0],
                      q.shape[1])
    return out
                  
@cython.boundscheck(False)
@cython.wraparound(False)
def bench_libc_exp(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    for i in range(repeat):
        for j in range(x.shape[0]):
            out[j] = exp(x[j])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bench_inv(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    for i in range(repeat):
        for j in range(x.shape[0]):
            out[j] = 1.0 / x[j]

@cython.boundscheck(False)
@cython.wraparound(False)
def bench_mul(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    for i in range(repeat):
        for j in range(x.shape[0]):
            out[j] = x[j] * x[j]

@cython.boundscheck(False)
@cython.wraparound(False)
def bench_vml_exp(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    vmlSetMode(VML_LA | VML_ERRMODE_STDERR)
    for i in range(repeat):
        for j in range(x.shape[0]):
            vdExp(28, <double*>x.data, <double*>out.data)

