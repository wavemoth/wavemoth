#include "butterfly.h"
#include "stdio.h"
#include "blas.h"
#include "assert.h"

typedef int (*apply_dz_func_t)(char *matrixdata, 
                               double *x, double *y,
                               bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec);

static INLINE char *aligned(char *ptr) {
   return ptr + (size_t)ptr % 16;
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


/*
Simplified dgemm interfaces. Computes

Y <- A * X + Y

where A is col-major and Y and X row-major.  Y is m-by-n, A is m-by-k,
X is k-by-n.
*/
static void INPLACE dgemm_crr(double *A, double *X, double *Y,
                              int32_t m, int32_t n, int32_t k) {
  /* We compute X^T A^T + Y^T, which Fortran sees as X A^T + Y */
  dgemm('N', 'T', n, m, k, 1.0, X, n, A, m, 1.0, Y, n);
}


static int hstack_right_d(char *matrixdata, double *x, double *y,
                          bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  return -1;
}


static char *filter_vectors(char *filter, double *x, double *a, double *b,
                            int32_t alen, int32_t blen, int32_t nvec) {
  int j;
  char *end = filter + alen + blen;
  while (filter != end) {
    if (*filter++) {
      for (j = 0; j != nvec; ++j) {
        *a++ = *x++;
      }
    } else {
      for (j = 0; j != nvec; ++j) {
        *b++ = *x++;
      }
    }
  }
  return filter;
}

static int butterfly_right_d(char *data, double *x, double *y,
                             bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  BFM_ButterflyHeader header = *(BFM_ButterflyHeader*)data;
  double buf[header.nrow_buf * nvec];
  double LR_out[(header.nrow_L_p + header.nrow_R_p) * nvec];
  int i;
  data += sizeof(BFM_ButterflyHeader);

  /* Filter x into LR_out, which is the part hit by the identity matrix,
     and buf, which is the input to the interpolation matrix. */
  data = filter_vectors(data, x, LR_out, buf, header.nrow_L_p, header.ncol_L_p, nvec);
  data = aligned(data);
  /* LR_out <- L_p * buf + LR_out */
  dgemm_crr(L_p, buf, LR_out, header.nrow_L_P, nvec, header.ncol_L_p);

  /* early debug return */
  for (i = 0; i != nrow * nvec; ++i) {
    y[i] = LR_out[i];
  }
}

/*
DISPATCH
*/

static apply_dz_func_t dispatch_table_dz[BFM_MAX_TYPE] = {
  zero_right_d,
  dense_rowmajor_right_d,
  hstack_right_d,
  butterfly_right_d
};

int bfm_apply_right_d(char *matrixdata, double *x, double *y,
                       bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  int32_t type = ((int32_t*)matrixdata)[0];
  return (*dispatch_table_dz[type])(matrixdata, x, y, nrow, ncol, nvec);
}


