/*
Data format of precomputed file:

  int64_t max_m
  int64_t offset[4 * (max_m + 1)]: Offsets, relative to start of file, of
      the data. The array is indexed by "4*m" for the start of even part,
      4*m + 1 for the end of even part, 4*m+2 and 4*m+4 for start and length
      of odd part.

The data is a compressed butterfly matrix, as documented in butterfly.h.

*/

#ifndef _WAVEMOTH_H_
#define _WAVEMOTH_H_

#include <stdint.h>
#include "butterfly.h"

typedef struct _wavemoth_plan *wavemoth_plan;

#define WAVEMOTH_MMAJOR 0x0

#define WAVEMOTH_ESTIMATE 0x0
#define WAVEMOTH_MEASURE 0x1

#define WAVEMOTH_NO_RESOURCE_COPY 0x10

/*
Driver functions. Stable API.
*/

void wavemoth_configure(char *resource_dir);

wavemoth_plan wavemoth_plan_to_healpix(int Nside, int lmax, int mmax, int nmaps,
                                     int nthreads,
                                     double *input, double *output,
                                     int ordering, unsigned flags,
                                     char *resource_filename);

void wavemoth_destroy_plan(wavemoth_plan plan);
void wavemoth_execute(wavemoth_plan plan);

int64_t wavemoth_get_legendre_flops(wavemoth_plan plan, int m, int odd);

int wavemoth_query_resourcefile(char *filename, int *out_Nside, int *out_lmax);

#endif
