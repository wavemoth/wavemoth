/**
 * Routines for matrix-vector products.
 * 
 */

#include <stdlib.h>

#include "matvec.h"

/**
 * Multiply a double-precision real matrix with complex vector.
 *
 */
void dmat_zvec(size_t m, size_t n, double *A, double *x, double *y) {
  size_t i, j, ix, iy;
  double real_acc, imag_acc;
  iy = 0;
  for (i = 0; i != m; ++i) {
    real_acc = imag_acc = 0;
    ix = 0;
    for (j = 0; j != n; ++j) {
      real_acc += A[i * n + j] * x[ix++];
      imag_acc += A[i * n + j] *x[ix++];
    }
    y[iy++] = real_acc;
    y[iy++] = imag_acc;
  }
}
