/*
Data format of precomputed file:

  int64_t max_m
  int64_t offset[4 * (max_m + 1)]: Offsets, relative to start of file, of
      the data. The array is indexed by "4*m" for the start of even part,
      4*m + 1 for the end of even part, 4*m+2 and 4*m+4 for start and length
      of odd part.

The data is a compressed butterfly matrix, as documented in butterfly.h.

*/

#ifndef _FASTSHT_H_
#define _FASTSHT_H_

#include <stdint.h>
#include "butterfly.h"

typedef struct _fastsht_plan *fastsht_plan;

#define FASTSHT_MMAJOR 0x0

/*
Driver functions. Stable API.
*/

void fastsht_configure(char *resource_dir);

fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax, int nmaps,
                                     double *input, double *output, double *work,
                                     int ordering, char *resource_filename);

void fastsht_destroy_plan(fastsht_plan plan);
void fastsht_execute(fastsht_plan plan);

int64_t fastsht_get_legendre_flops(fastsht_plan plan, int m, int odd);

int fastsht_query_resourcefile(char *filename, int *out_Nside, int *out_lmax);

#endif
