#include <assert.h>

#include "legendre_transform.h"

void fastsht_associated_legendre_transform(size_t nx, size_t nl,
                                           size_t nvecs,
                                           size_t *il_start, 
                                           double *a,
                                           double *y,
                                           double *x_squared, 
                                           double *c, double *d,
                                           double *c_inv,
                                           double *P, double *Pp1) {
  size_t ix, il, j, il_start_val;
  double Pval, Pval_prev, Pval_prevprev;

  assert(nl >= 2);
  for (ix = 0; ix != nx; ++ix) {
    /* First get away with the precomputed values. This also zeros the output buffer. */
    il = il_start[ix];
    Pval_prevprev = P[ix];
    Pval_prev = Pp1[ix];
    for (j = 0; j != nvecs; ++j) {
      y[ix * nvecs + j] = Pval_prevprev * a[il * nvecs + j];
    }
    ++il;
    for (j = 0; j != nvecs; ++j) {
      y[ix * nvecs + j] += Pval_prev * a[il * nvecs + j];
    }
    ++il;
    for (; il < nl; ++il) {
      Pval = c_inv[il - 1] * ((x_squared[ix] - d[il - 1]) * Pval_prev - c[il - 2] * Pval_prevprev);
      Pval_prevprev = Pval_prev;
      Pval_prev = Pval;
      for (j = 0; j != nvecs; ++j) {
        y[ix * nvecs + j] += Pval * a[il * nvecs + j];        
      }
    }
  }
}
