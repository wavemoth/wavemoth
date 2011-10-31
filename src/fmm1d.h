#ifndef _WAVEMOTH_FMM1D_H_
#define _WAVEMOTH_FMM1D_H_

#include <stdlib.h>

struct _fmm1d_plan;
typedef struct fmm1d_plan *fmm1d_plan;

/*
Fast multipole method. Evaluates

\phi_{jk} = sum_{i = 0}^{n-1} (\omega_j \gamma_i q_{ik})/(y_j - x_i),

using [..TODO ref..].

BUGS:

The procedure currently aborts the program if a point y_j is too close
to x_i. Some mechanism for transitioning to a limiting expression
should be added.
*/

void wavemoth_fmm1d(const double *restrict x_grid, const double *restrict gamma,
                   const double *restrict q, size_t nx,
                   const double *restrict y_grid, const double *restrict omega,
                   double *restrict phi, size_t ny, size_t nvecs);


#endif
