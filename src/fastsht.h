#ifndef _FASTSHT_H_
#define _FASTSHT_H_

#include "butterfly.h"

typedef struct _fastsht_plan *fastsht_plan;

#define FASTSHT_MMAJOR 0x0

/*
Driver functions. Stable API.
*/

fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax, double *input,
                                     double *output, double *work, int ordering);

void fastsht_destroy_plan(fastsht_plan plan);
void fastsht_execute(fastsht_plan plan);

#endif
