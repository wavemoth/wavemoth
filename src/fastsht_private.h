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

struct _precomputation_t;
typedef struct _precomputation_t precomputation_t;

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
  bfm_index_t npix;
  int has_equator;
} fastsht_grid_info;

struct _fastsht_plan {
  int type;
  int lmax, mmax;
  int nmaps;
  double *output, *input;
  fastsht_grid_info *grid;
  fftw_plan *fft_plans;
  precomputation_t *resources;
  int did_allocate_resources;
  int Nside;
};

void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd, double complex *work_a_l, double complex *output);
void fastsht_perform_interpolation(fastsht_plan plan, bfm_index_t m, int odd);
void fastsht_assemble_rings(fastsht_plan plan,
                            int ms_len, int *ms,
                            double complex **q_list);
void fastsht_assemble_rings_omp_worker(fastsht_plan plan,
                                       int ms_len, int *ms,
                                       double complex **q_list);

void fastsht_legendre_transform(fastsht_plan plan, int mstart, int mstop, int mstride);

void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end);

fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside);
void fastsht_free_grid_info(fastsht_grid_info *info);
void fastsht_disable_phase_shifting(fastsht_plan plan);

void fastsht_execute_out_of_core(fastsht_plan plan,
                                 double *out_compute_time,
                                 double *out_load_time);

#endif
