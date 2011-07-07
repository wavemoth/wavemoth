#undef NDEBUG
#include <assert.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdlib.h>
#include <stdio.h>

#include "legendre_transform.h"

typedef __m128d m128d;
typedef __m128 m128;

void _printreg(char *msg, m128d r) {
  double *pd = (double*)&r;
  printf("%s = [%f %f]\n", msg, pd[0], pd[1]);
}

#define printreg(x) _printreg(#x, x)

void fastsht_associated_legendre_transform_sse(size_t nx, size_t nl,
                                               size_t _nvecs,
                                               size_t *k_start, 
                                               double *a,
                                               double *y,
                                               double *x_squared, 
                                               double *c, double *d,
                                               double *c_inv,
                                               double *P, double *Pp1) {
  size_t ix, il, j, il_start_val, k;
  double Pval, Pval_prev, Pval_prevprev;
  m128d rp0, rp1, rp00, rp01, rp10, rp11, ry0, ry1, ra, tmp;
  #define nvecs 2
  assert(nl >= 2);
  assert(_nvecs == 2);
  assert(nx % 2 == 0);
  assert((size_t)a % 16 == 0);
  assert((size_t)y % 16 == 0);
  assert((size_t)c % 16 == 0);
  assert((size_t)d % 16 == 0);
  assert((size_t)c_inv % 16 == 0);
  assert((size_t)P % 16 == 0);
  assert((size_t)Pp1 % 16 == 0);
  for (ix = 0; ix != nx; ix += 2) {
    assert(k_start[ix] == k_start[ix + 1]);
    k = k_start[ix];

    m128d P_0i = _mm_load_pd(P + ix);
    m128d P_0il = _mm_unpacklo_pd(P_0i, P_0i);
    m128d P_0ih = _mm_unpackhi_pd(P_0i, P_0i);

    m128d a_0j = _mm_load_pd(a + k * nvecs);
    m128d y_ij = _mm_mul_pd(a_0j, P_0il);
    _mm_store_pd(y + ix * nvecs, y_ij);
    y_ij = _mm_mul_pd(a_0j, P_0ih);
    _mm_store_pd(y + (ix + 1) * nvecs, y_ij);

    for (int ixp = 0; ixp != 2; ++ixp) {
      k = k_start[ix] + 1;
      Pval_prevprev = P[ix + ixp];
      Pval_prev = Pp1[ix + ixp];
      for (j = 0; j != nvecs; ++j) {
        y[(ix + ixp) * nvecs + j] += Pval_prev * a[k * nvecs + j];
      }
      ++k;
      for (; k < nl; ++k) {
        Pval = c_inv[k - 1] * ((x_squared[ix + ixp] - d[k - 1]) * Pval_prev - c[k - 2] * Pval_prevprev);
        Pval_prevprev = Pval_prev;
        Pval_prev = Pval;
        for (j = 0; j != nvecs; ++j) {
          y[(ix + ixp) * nvecs + j] += Pval * a[k * nvecs + j];        
        }
      }
    }
  }
    /* Make rp[lx] contain [P_i(x), P_i(x)] */
    /*    m128d P_0i = _mm_load_pd(P + ix);
    m128d P_0il = _mm_unpacklo_pd(P_0i, P_0i);
    m128d P_0ih = _mm_unpackhi_pd(P_0i, P_0i);*/

    /* Multiply rp00 and rp01 with a_l's for all j */
    /*    m128d a_0j = _mm_load_pd(a + k * nvecs);
    m128d y_ij = _mm_mul_pd(a_0j, P_0il);
    _mm_store_pd(y + ix * nvecs, y_ij);
    y_ij = _mm_mul_pd(a_0j, P_0ih);
    _mm_store_pd(y + (ix + 1) * nvecs, y_ij);

    printreg(P_0i);
    printreg(P_0il);
    printreg(P_0ih);
    printreg(a_0j);
    printreg(y_ij);
    printf("%f\n", *(P + ix));*/
    //    abort();
    /*
    y[ix * nvecs] = P[ix] * a[k * nvecs];
    y[ix * nvecs + 1] = P[ix] * a[k * nvecs + 1];
    y[ix * nvecs + 2] = P[ix + 1] * a[k * nvecs];
    y[ix * nvecs + 3] = P[ix + 1] * a[k * nvecs + 1];

    */
    //    abort();

    /*    rp10 = _mm_load_pd(Pp1 + ix);
    rp11 = _mm_shuffle_pd(rp11, rp10, SHUFCONST(1, 1));
    rp11 = _mm_shuffle_pd(rp11, rp10, SHUFCONST(0, 0));*/
  #undef nvecs
}

void fastsht_associated_legendre_transform(size_t nx, size_t nl,
                                           size_t nvecs,
                                           size_t *k_start, 
                                           double *a,
                                           double *y,
                                           double *x_squared, 
                                           double *c, double *d,
                                           double *c_inv,
                                           double *P, double *Pp1) {
  size_t ix, k, j, k_start_val;
  double Pval, Pval_prev, Pval_prevprev;

  assert(nl >= 2);
  for (ix = 0; ix != nx; ++ix) {
    /* First get away with the precomputed values. This also zeros the output buffer. */
    k = k_start[ix];
    Pval_prevprev = P[ix];
    Pval_prev = Pp1[ix];
    for (j = 0; j != nvecs; ++j) {
      y[ix * nvecs + j] = Pval_prevprev * a[k * nvecs + j];
    }
    ++k;
    for (j = 0; j != nvecs; ++j) {
      y[ix * nvecs + j] += Pval_prev * a[k * nvecs + j];
    }
    ++k;
    for (; k < nl; ++k) {
      Pval = c_inv[k - 1] * ((x_squared[ix] - d[k - 1]) * Pval_prev - c[k - 2] * Pval_prevprev);
      Pval_prevprev = Pval_prev;
      Pval_prev = Pval;
      for (j = 0; j != nvecs; ++j) {
        y[ix * nvecs + j] += Pval * a[k * nvecs + j];        
      }
    }
  }
}
