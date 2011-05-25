/*!
C convenience wrappers around a very restricted subset of Fortran
BLAS, since that appears to be slightly more portable than CLAPACK.
 */

#ifndef _BLAS_WRAPPER_H
#define _BLAS_WRAPPER_H

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

where A is col-major and Y and X row-major.  Y is m-by-n, A is m-by-k,
X is k-by-n.
*/
static INLINE void dgemm_crr(double *A, double *X, double *Y,
                             int32_t m, int32_t n, int32_t k,
                             double beta) {
  /* We compute X^T A^T + Y^T, which Fortran sees as X A^T + Y */
  dgemm('N', 'T', n, m, k, 1.0, X, n, A, m, beta, Y, n);
}

#endif