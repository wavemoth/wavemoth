/*
Low-level API, primarily exposed for unit testing: It
is wrapped in Cython and used for unit-tests, but it should
probably

The API is unstable and may change at any point.
*/

#ifndef _FASTSHT_PRIVATE_H_
#define _FASTSHT_PRIVATE_H_

#include "fastsht.h"

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


void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end);

fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside);
void fastsht_free_grid_info(fastsht_grid_info *info);


#endif
