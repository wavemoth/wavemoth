# Wrap certain BLAS functions for benchmark and unit test purposes

cimport numpy as np

cdef extern from "blas.h":
    ctypedef int int32_t
    void dgemm_crc(double *A, double *B, double *C,
                   int32_t m, int32_t n, int32_t k,
                   double beta)

def benchmark_dgemm_crc(np.ndarray[double, ndim=2, mode='fortran'] A,
                        np.ndarray[double, ndim=2, mode='c'] B,
                        np.ndarray[double, ndim=2, mode='fortran'] C,
                        int repeats=1, double beta=0):
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == C.shape[0]
    cdef int i
    for i in range(repeats):
        dgemm_crc(<double*>A.data, <double*>B.data, <double*>C.data,
                  C.shape[0], C.shape[1], A.shape[1], beta)
