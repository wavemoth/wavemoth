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

#endif
