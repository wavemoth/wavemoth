/*!
C convenience wrappers around a very restricted subset of Fortran
BLAS, since that appears to be slightly more portable than CLAPACK.
 */

#ifndef _BLAS_WRAPPER_H
#define _BLAS_WRAPPER_H

#include <stdint.h>

/*
We rely on this header only being included once for correct
results both with and without INLINE being defined.
*/

#ifndef INLINE
# if __STDC_VERSION__ >= 199901L
#  define INLINE inline
# else
#  define INLINE
# endif
#endif



void dgemm_(char *transa, char *transb, int *m, int *n,
            int *k, double *alpha, double *a, int *lda, 
            double *b, int *ldb, double *beta, double *c,
            int *ldc);

static INLINE void dgemm(char transa, char transb, int m, int n, int k,
                         double alpha, double *a, int lda, double *b,
                         int ldb, double beta, double *c, int ldc) {
  dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b,
         &ldb, &beta, c, &ldc);
}

/*
Simplified dgemm interfaces. Computes

Y <- A * X + beta * Y

where A, X and Y are row-major.  Y is m-by-n, A is m-by-k,
X is k-by-n.
*/
static INLINE void dgemm_rrr(double *A, double *X, double *Y,
                             int32_t m, int32_t n, int32_t k,
                             double beta) {
  /* We compute X^T A^T + Y^T, which Fortran sees as X A + Y */
  dgemm('N', 'N', n, m, k, 1.0, X, (n > 0) ? n : 1, A, (k > 0) ? k : 1, beta, Y, n);
}


/*
Simplified dgemm interface. Computes

C <- A * B + beta * C

where A is col-major, B is row-major, and C is col-major.

Y is m-by-n, A is m-by-k, B is k-by-n.
*/
static INLINE void dgemm_crc(double *A, double *B, double *C,
                             int32_t m, int32_t n, int32_t k,
                             double beta) {
  /* This is supported directly by Fortran BLAS by passing transpose
     flag on second matrix. */
  dgemm('N', 'T', m, n, k, 1.0, A, (m > 0) ? m : 1, B, (n > 0) ? n : 1,
        beta, C, (m > 0) ? m : 1);
}

static INLINE void dgemm_ccc(double *A, double *B, double *C,
                             int32_t m, int32_t n, int32_t k,
                             double beta) {
  dgemm('N', 'N', m, n, k, 1.0, A, (m > 0) ? m : 1, B, (k > 0) ? k : 1,
        beta, C, (m > 0) ? m : 1);
}

#endif
