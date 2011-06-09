/*
Low-level API, primarily exposed for unit testing: It
is wrapped in Cython and used for unit-tests, but it should
probably

The API is unstable and may change at any point.
*/

#ifndef _FASTSHT_PRIVATE_H_
#define _FASTSHT_PRIVATE_H_

#include "fastsht.h"
#include "complex.h"
#include <fftw3.h>

typedef struct {
  /* To phase shift, we multiply with

         e^(i m phi_0) = cos(m phi_0) + i sin(m phi_0)

     for each ring. This is computed by
     
         cos(x + phi_0) = cos(x) - (alpha * cos(x) + beta * sin(x))
         sin(phi_0) = sin(x) - (alpha * sin(x) - beta * cos(x))

     and we precompute ...
   */
  double *phi0s;
  bfm_index_t *ring_offsets;
  bfm_index_t nrings, mid_ring;
  int has_equator;
} fastsht_grid_info;

struct _fastsht_plan {
  int type;
  int lmax, mmax;
  double *output, *input, *work;
  double complex *work_a_l, *work_g_m_roots, *work_g_m_even, *work_g_m_odd;
  fastsht_grid_info *grid;
  fftw_plan *fft_plans;
  int Nside;
};

/*
Goes from plan->input to plan->work_g_m_roots; that is, evaluate
g_m(theta) in the Ass. Legendre roots for the given m and odd.
*/
void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd);
void fastsht_perform_interpolation(fastsht_plan plan, bfm_index_t m, int odd);
void fastsht_merge_even_odd_and_transpose(fastsht_plan plan, bfm_index_t m);


void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end);

fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside);
void fastsht_free_grid_info(fastsht_grid_info *info);


#endif
