#include "butterfly.h"
#include "stdio.h"
#include "blas.h"

#undef NDEBUG
#include "assert.h"


typedef int (*apply_dz_func_t)(char *matrixdata, 
                               double *x, double *y,
                               bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec);

static INLINE char *aligned(char *ptr) {
  size_t m = (size_t)ptr % 16;
  if (m == 0) {
    return ptr;
  } else { 
    return ptr + 16 - m;
  }
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
static INLINE void dgemm_crr(double *A, double *X, double *Y,
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
  char group;
  char *end = filter + alen + blen;
  printf("%d %d %d\n", alen, blen, nvec);
  while (filter != end) {
    switch (*filter++) {
    case 0:
      for (j = 0; j != nvec; ++j) {
        *a++ = *x++;
      }
      break;
    case 1:
      for (j = 0; j != nvec; ++j) {
        *b++ = *x++;
      }
      break;
    default:
      assert(0);
      break;
    }
  }
  return filter;
}

static int butterfly_right_d(char *data, double *x, double *y,
                             bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) { 
  BFM_ButterflyHeader info = *(BFM_ButterflyHeader*)data;
  double buf[info.nrow_buf * nvec];
  double LR_out[(info.nrow_L_ip + info.nrow_R_ip) * nvec];
  int i; 
  data += sizeof(BFM_ButterflyHeader);
  printf("%d %d %d %d\n", info.nrow_L_ip, info.ncol_L_ip, info.nrow_buf, nvec);
  /* Filter x into LR_out, which is the part hit by the identity matrix,
     and buf, which is the input to the interpolation matrix. */
  data = filter_vectors(data, x, buf, LR_out, info.ncol_L_ip, info.nrow_L_ip, nvec);
  data = aligned(data);
  //  for (i = 0; i != 6; ++i) printf("%f ", LR_out[i]);
  //printf("\n");
  //for (i = 0; i != 3*18; ++i) printf("%f ", buf[i]);
  //printf("\n");
  //for (i = 0; i != 2*18; ++i) printf("%f ", ((double*)data)[i]);
  //printf("\n");
  /* LR_out <- L_ip * buf + LR_out */
  dgemm_crr((double*)data, buf, LR_out, info.nrow_L_ip, nvec, info.ncol_L_ip);
  data = aligned(data + info.nrow_L_ip * info.ncol_L_ip * sizeof(double));

  /* early debug return */ 
  printf("adsf\n");
  for (i = 0; i != info.nrow_L_ip * nvec; ++i) {//info.nrow_L_ip * nvec
    printf("%d\n", i);

    
    y[i] = LR_out[i];
  }
  return 0;
}

/*
DISPATCH
*/

static apply_dz_func_t dispatch_table_dz[BFM_MAX_TYPE + 1] = {
  zero_right_d,
  dense_rowmajor_right_d,
  hstack_right_d,
  butterfly_right_d
};

int bfm_apply_right_d(char *matrixdata, double *x, double *y,
                       bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec) {
  int32_t type = ((int32_t*)matrixdata)[0];
  assert((size_t)matrixdata % 16 == 0);
  return (*dispatch_table_dz[type])(matrixdata, x, y, nrow, ncol, nvec);
}


