cdef extern from "blas.h":
    ctypedef int int32_t
    void dgemm_crc_ "dgemm_crc"(double *A, double *B, double *C,
                   int32_t m, int32_t n, int32_t k,
                   double beta)
    void dgemm_ccc_ "dgemm_ccc"(double *A, double *B, double *C,
                   int32_t m, int32_t n, int32_t k,
                   double beta)
