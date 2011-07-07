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

#define IDX_C 0
#define IDX_CINV 1
#define IDX_D 2

void fastsht_associated_legendre_transform_sse(size_t nx, size_t nl,
                                               size_t _nvecs,
                                               size_t *k_start, 
                                               double *a,
                                               double *y,
                                               double *x_squared, 
                                               double *auxdata,
                                               double *P, double *Pp1) {
  size_t i, il, j, il_start_val, k;
  double Pval, Pval_prev, Pval_prevprev;
  m128d rp0, rp1, rp00, rp01, rp10, rp11, ry0, ry1, ra, tmp;
  #define nvecs 2
  assert(nl >= 2);
  assert(_nvecs == 2);
  assert(nx % 2 == 0);
  assert((size_t)a % 16 == 0);
  assert((size_t)y % 16 == 0);
  assert((size_t)auxdata % 16 == 0);
  assert((size_t)P % 16 == 0);
  assert((size_t)Pp1 % 16 == 0);
  for (i = 0; i != nx; i += 2) {
    assert(k_start[i] == k_start[i + 1]);
    /* In comments and variable names we will assume that k starts on
       0 to keep things brief. */
    k = k_start[i];

    /* We loop over k and compute y_ij = y[i, j] and y_ijp = y[i, j + 1]. */
    m128d y_ij, y_ijp;

    /* k=0 and k=1 needs special treatment as they are already computed (starting values) */
    /* y_ij = a_0i * P_0i for all j. */
    m128d P_0i = _mm_load_pd(P + i);
    m128d P_0i_l = _mm_unpacklo_pd(P_0i, P_0i);
    m128d P_0i_h = _mm_unpackhi_pd(P_0i, P_0i);
    m128d a_0j = _mm_load_pd(a + k * nvecs);
    y_ij = _mm_mul_pd(a_0j, P_0i_l);
    y_ijp = _mm_mul_pd(a_0j, P_0i_h);

    /* y_ij += a_1i * P_1i for all j */
    ++k;
    m128d P_1i = _mm_load_pd(Pp1 + i);
    m128d P_1i_l = _mm_unpacklo_pd(P_1i, P_1i);
    m128d P_1i_h = _mm_unpackhi_pd(P_1i, P_1i);
    m128d a_1j = _mm_load_pd(a + k * nvecs);
    y_ij = _mm_add_pd(y_ij, _mm_mul_pd(a_1j, P_1i_l));
    y_ijp = _mm_add_pd(y_ijp, _mm_mul_pd(a_1j, P_1i_h));


    /* Loop over k = 2..nl-1 */
    ++k;
    for (; k < nl; ++k) {
      /* Load auxiliary data */
    }

    _mm_store_pd(y + i * nvecs, y_ij);
    _mm_store_pd(y + (i + 1) * nvecs, y_ijp);

    for (int it = 0; it != 2; ++it) {
      k = k_start[i] + 1;
      Pval_prevprev = P[i + it];
      Pval_prev = Pp1[i + it];
      ++k;
      for (; k < nl; ++k) {
        double c_sub2 = auxdata[4 * (k - 2) + 0];
        double cinv_sub1 = auxdata[4 * (k - 2) + 1];
        double d_sub1 = auxdata[4 * (k - 2) + 2];
        Pval = cinv_sub1 * ((x_squared[i + it] - d_sub1) * Pval_prev - c_sub2 * Pval_prevprev);
        Pval_prevprev = Pval_prev;
        Pval_prev = Pval;
        for (j = 0; j != nvecs; ++j) {
          y[(i + it) * nvecs + j] += Pval * a[k * nvecs + j];        
        }
      }
    }
  }
  #undef nvecs
}

void fastsht_associated_legendre_transform(size_t nx, size_t nl,
                                           size_t nvecs,
                                           size_t *k_start, 
                                           double *a,
                                           double *y,
                                           double *x_squared, 
                                           double *auxdata,
                                           double *P, double *Pp1) {
  size_t i, k, j, k_start_val;
  double Pval, Pval_prev, Pval_prevprev;

  assert(nl >= 2);
  for (i = 0; i != nx; ++i) {
    /* First get away with the precomputed values. This also zeros the output buffer. */
    k = k_start[i];
    Pval_prevprev = P[i];
    Pval_prev = Pp1[i];
    for (j = 0; j != nvecs; ++j) {
      y[i * nvecs + j] = Pval_prevprev * a[k * nvecs + j];
    }
    ++k;
    for (j = 0; j != nvecs; ++j) {
      y[i * nvecs + j] += Pval_prev * a[k * nvecs + j];
    }
    ++k;
    for (; k < nl; ++k) {
      double c_sub2 = auxdata[4 * (k - 2) + 0];
      double cinv_sub1 = auxdata[4 * (k - 2) + 1];
      double d_sub1 = auxdata[4 * (k - 2) + 2];
      Pval = cinv_sub1 * ((x_squared[i] - d_sub1) * Pval_prev - c_sub2 * Pval_prevprev);
      Pval_prevprev = Pval_prev;
      Pval_prev = Pval;
      for (j = 0; j != nvecs; ++j) {
        y[i * nvecs + j] += Pval * a[k * nvecs + j];        
      }
    }
  }
}
