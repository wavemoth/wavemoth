# Wrap certain BLAS functions for benchmark purposes

cdef extern from "blas.h":
    void dgemm_crr(double *A, double *X, double *Y,
                   int32_t m, int32_t n, int32_t k,
                   double beta)
    void dgemm_rrr(double *A, double *X, double *Y,
                   int32_t m, int32_t n, int32_t k,
                   double beta)

def benchmark_dgemm_crr(np.ndarray[double, ndim=2, mode='fortran'] A,
                        np.ndarray[double, ndim=2, mode='c'] X,
                        np.ndarray[double, ndim=2, mode='c'] Y,
                        int repeats=0):
    assert A.shape[1] == X.shape[0]
    assert A.shape[0] == Y.shape[0]
    cdef int i
    for i in range(repeats):
        dgemm_crr(<double*>A.data, <double*>X.data, <double*>Y.data,
                  Y.shape[0], Y.shape[1], A.shape[1], 0.0)
   
def benchmark_dgemm_rrr(np.ndarray[double, ndim=2, mode='c'] A,
                        np.ndarray[double, ndim=2, mode='c'] X,
                        np.ndarray[double, ndim=2, mode='c'] Y,
                        int repeats=0):
    assert A.shape[1] == X.shape[0]
    assert A.shape[0] == Y.shape[0]
    cdef int i
    for i in range(repeats):
        dgemm_rrr(<double*>A.data, <double*>X.data, <double*>Y.data,
                  Y.shape[0], Y.shape[1], A.shape[1], 0.0)

    
