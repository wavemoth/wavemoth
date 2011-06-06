#ifndef _FASTSHT_FMM1D_H_
#define _FASTSHT_FMM1D_H_

#include <stdlib.h>

struct _fmm1d_plan;
typedef struct fmm1d_plan *fmm1d_plan;

/*
Fast multipole method -- evaluates

output_x[j] = sum_{i = 1, i != j}^{nx} \frac{input_x[i]}{y_j - x_i}

using [...]


fmm1d_plan fmm1d_plan_simple(double *grid_x, size_t nx, double *grid_y, size_t ny);
void fmm1d_destroy_plan(fmm1d_plan plan);
void fmm1d_execute(fmm1d_plan plan, double *input_x, double *output_y);
*/

void fastsht_fmm1d(double *x_grid, double *input_x, size_t nx,
                   double *y_grid, double *output_y, size_t ny);


#endif
