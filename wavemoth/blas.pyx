# Wrap certain BLAS functions for benchmark and unit test purposes

cimport numpy as np

def dgemm_crc(np.ndarray[double, ndim=2, mode='fortran'] A,
              np.ndarray[double, ndim=2, mode='c'] B,
              np.ndarray[double, ndim=2, mode='fortran'] C,
              int repeat=1, double beta=0):
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == C.shape[0]
    cdef int i
    for i in range(repeat):
        dgemm_crc_(<double*>A.data, <double*>B.data, <double*>C.data,
                   C.shape[0], C.shape[1], A.shape[1], beta)

def dgemm_ccc(np.ndarray[double, ndim=2, mode='fortran'] A,
              np.ndarray[double, ndim=2, mode='fortran'] B,
              np.ndarray[double, ndim=2, mode='fortran'] C,
              int repeat=1, double beta=0):
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == C.shape[0]
    cdef int i
    for i in range(repeat):
        dgemm_ccc_(<double*>A.data, <double*>B.data, <double*>C.data,
                   C.shape[0], C.shape[1], A.shape[1], beta)
