#include "butterfly.h"
#include "stdio.h"

typedef int (*apply_dz_func_t)(char *matrixdata, 
                               double *x, double *y,
                               bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec);
/*
LAPACK headers
*/
void dgemm_(char *transa, char *transb, int *m, int *n,
            int *k, double *alpha, double *a, int *lda, 
            double *b, int *ldb, double *beta, double *c,
            int *ldc);

static void dgemm(char transa, char transb, int m, int n, int k,
           double alpha, double *a, int lda, double *b,
           int ldb, double beta, double *c, int ldc) {
  dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b,
         &ldb, &beta, c, &ldc);
}

/*
Type implementations
*/

static int zero_right_d(char *matrixdata, double *x, double *y,
                        bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  bfm_index_t i;
  for (i = 0; i != nrow * nvec; ++i) {
    y[i] = 0.0;
  }
  return 0;
}


static int dense_rowmajor_right_d(char *matrixdata, double *x, double *y,
                                  bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  double *matrix = (double*)(matrixdata + 16);
  /* dgemm uses Fortran-order, so do transposed multiply;
     C-order: y^T = x^T * matrix^T
     Fortran-order: y = x * matrix
  */
  int m = nvec, n = nrow, k = ncol;
  dgemm('N', 'N', m, n, k, 1.0, x, m, matrix, k, 0.0, y, m);
  return 0;
}

static int hstack_right_d(char *matrixdata, double *x, double *y,
                          bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  return -1;
}

/*
DISPATCH
*/

static apply_dz_func_t dispatch_table_dz[BFM_MAX_TYPE] = {
  zero_right_d,
  dense_rowmajor_right_d,
  hstack_right_d
};

int bfm_apply_right_d(char *matrixdata, double *x, double *y,
                       bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  int type = ((int*)matrixdata)[0];
  return (*dispatch_table_dz[type])(matrixdata, x, y, nrow, ncol, nvec);
}


