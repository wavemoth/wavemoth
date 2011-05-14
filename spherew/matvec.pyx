cimport numpy as np
import numpy as np

cdef extern from "matvec.h":
    void dmat_zvec_ "dmat_zvec"(size_t m, size_t n, double *A, double *x, double *y)
    
def dmat_zvec(np.ndarray[double, ndim=2, mode='c'] A,
              np.ndarray[double complex, mode='c'] x,
              np.ndarray[double complex, mode='c'] out=None):
    if out is None:
        out = np.zeros(A.shape[0], dtype=np.complex)
    if A.shape[0] != out.shape[0] or A.shape[1] != x.shape[0]:
        raise ValueError('unconforming shapes')
    dmat_zvec_(A.shape[0], A.shape[1], <double*>A.data,
              <double*>x.data, <double*>out.data)
    return out
