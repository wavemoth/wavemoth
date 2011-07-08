//#undef NDEBUG
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
                                               double *auxdata,
                                               double *P, double *Pp1) {
  #define nvecs 2
  #define NS 2

  size_t i, j, k, s;
  assert(nl >= 2);
  assert(_nvecs == 2);
  assert(nx % 2 == 0);
  assert((size_t)a % 16 == 0);
  assert((size_t)y % 16 == 0);
  assert((size_t)auxdata % 16 == 0);
  assert((size_t)P % 16 == 0);
  assert((size_t)Pp1 % 16 == 0);
  for (i = 0; i != nx; i += 2 * NS) {
    /* In comments and variable names we will assume that k starts on
       0 to keep things brief; however, this is a dynamic quantity to
       support ignoring near-zero parts of P. The precomputation should
       ensure that k_start is constant over the block-sizes we end up
       using. */
    for (s = 1; s != 2 * NS; ++s) assert(k_start[i] == k_start[i + s]);
    k = k_start[i];

    /* The xsq_i only needs to be loaded once per strip. */
    m128d xsq_i[NS];
    for (s = 0; s != NS; ++s) xsq_i[s] = _mm_load_pd(x_squared + i + 2 * s);

    /* We loop over k in the inner-most loop and fully compute y_ij
       before storing it. The registers store y_ij transposed of the
       final order, the transpose happens right before storing to
       memory. I.e., for each strip i,

       y_ij  = [y_{i,j}     y_{i+1, j}    ]
       y_ijp = [y_{i,j + 1} y_{i+1, j + 1}]      

    */
    m128d y_ij[NS], y_ijp[NS];
#define MULADD(z, a, b) _mm_add_pd(z, _mm_mul_pd(a, b))

    /* For each strip of two x's, we keep

       P_ki =   [P_{k,i} P_{k,i+1}],
       Pp_ki =  [Pp_{k-1,i} P_{k-1,i+1}]
       Ppp_ki = [Pp_{k-2,i} P_{k-2,i+1}]
     */
    m128d P_ki[NS], Pp_ki[NS], Ppp_ki[NS];

    /* a_kj is loaded and shuffled for each k -- a load brings
       in a_{k,j} and a_{k,j+1}, which we duplicate in registers
       so that they can be multiplied with P_ki.

       a_kj  = [ a_{k,j}  a_{k,j} ]
       a_kjp = [ a_{k,j+1}  a_{k,j+1} ]
    */
    m128d a_kj, a_kjp;
#define LOAD_A(k) { \
  a_kj = _mm_load_pd(a + k * nvecs); \
  a_kjp = _mm_unpackhi_pd(a_kj, a_kj); \
  a_kj = _mm_unpacklo_pd(a_kj, a_kj); \
  }

    /* First two values of k needs special treatment as they are
       already computed (starting values). For the first k we
       initialize y_ij, and after that we accumulate in y_ij.
    */

    LOAD_A(k);
    for (s = 0; s != NS; ++s) {
      Ppp_ki[s] = _mm_load_pd(P + i + 2 * s);
      y_ij[s] = _mm_mul_pd(Ppp_ki[s], a_kj);
      y_ijp[s] = _mm_mul_pd(Ppp_ki[s], a_kjp);
    }
    ++k;

    LOAD_A(k);
    for (s = 0; s != NS; ++s) {
      Pp_ki[s] = _mm_load_pd(Pp1 + i + 2 * s);
      y_ij[s] = MULADD(y_ij[s], Pp_ki[s], a_kj);
      y_ijp[s] = MULADD(y_ijp[s], Pp_ki[s], a_kjp);
    }
    ++k;

    /* Loop over k = 2..nl-1 */
    for (; k < nl; ++k) {
      /* The recurrence relation we compute is, for each x_i,

         P_{k} = (x^2 + [-d_{k-1}]) * [1/c_{k-1}] P_{k-1} + [-c_{k-2}/c_{k-1}] P_{k-2}

         which we write

         P_k = (x^2 + alpha) * beta * P_{k-2} + gamma * P_{k-2}

         The terms in []-brackets are precomputed and available packed
         in auxdata; they must be unpacked into registers. Storing
         c_{k-2}/c_{k-1} seperately removes one dependency in the chain
         to hopefully improve pipelining.
       */

      /* Load auxiliary data. */
      m128d aux1 = _mm_load_pd(auxdata + 4 * (k - 2));
      m128d aux2 = _mm_load_pd(auxdata + 4 * (k - 2) + 2);

      m128d alpha = _mm_unpacklo_pd(aux1, aux1);
      m128d beta = _mm_unpackhi_pd(aux1, aux1);
      m128d gamma = _mm_unpacklo_pd(aux2, aux2);

      /* Use the recurrence relation */
      m128d w[NS];
      for (s = 0; s != NS; ++s) {
        w[s] = _mm_add_pd(xsq_i[s], alpha);
        w[s] = _mm_mul_pd(w[s], beta);
        w[s] = _mm_mul_pd(w[s], Pp_ki[s]);

        P_ki[s] = _mm_mul_pd(Ppp_ki[s], gamma);
        P_ki[s] = _mm_add_pd(P_ki[s], w[s]);

        Ppp_ki[s] = Pp_ki[s];
        Pp_ki[s] = P_ki[s];
      }

      LOAD_A(k);

      for (s = 0; s != NS; ++s) {
        y_ij[s] = MULADD(y_ij[s], P_ki[s], a_kj);
        y_ijp[s] = MULADD(y_ijp[s], P_ki[s], a_kjp);
      }
    }

    /* Finally, transpose and store the computed y_ij's. */
    m128d ycol_i[s], ycol_ip[s];
    for (s = 0; s != NS; ++s) {
      ycol_i[s] = _mm_shuffle_pd(y_ij[s], y_ijp[s], _MM_SHUFFLE2(0, 0));
      ycol_ip[s] = _mm_shuffle_pd(y_ij[s], y_ijp[s], _MM_SHUFFLE2(1, 1));
      _mm_store_pd(y + (i + 2 * s) * nvecs, ycol_i[s]);
      _mm_store_pd(y + (i + 2 * s + 1) * nvecs, ycol_ip[s]);
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
      double alpha = auxdata[4 * (k - 2) + 0];
      double beta = auxdata[4 * (k - 2) + 1];
      double gamma = auxdata[4 * (k - 2) + 2];
      Pval = (x_squared[i] + alpha) * beta * Pval_prev + gamma * Pval_prevprev;
      Pval_prevprev = Pval_prev;
      Pval_prev = Pval;
      for (j = 0; j != nvecs; ++j) {
        y[i * nvecs + j] += Pval * a[k * nvecs + j];        
      }
    }
  }
}
