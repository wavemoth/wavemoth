#include "fastsht_private.h"

#undef NDEBUG
#include "complex.h"
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#define PLANTYPE_HEALPIX 0x0


struct _fastsht_plan {
  int type;
  int lmax, mmax;
  double *output, *input, *work;
  fastsht_grid_info *grid;
  fftw_plan *fft_plans;
};

struct _fastsht_plan_healpix {
  struct _fastsht_plan base;
  int Nside;
};
typedef struct _fastsht_plan_healpix *fastsht_plan_healpix;

/*
Private
*/

static void phase_shift_ring_inplace(int mmax, complex double *g_m, double phi0) {
  int m = 0;
  for (m = 0; m != mmax; ++m) {
    g_m[m] *= (cos(m * phi0) + I * sin(m * phi0));
  }
}


/*
Public
*/

fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax, double *input,
                                     double *output, double *work, int ordering) {
  fastsht_plan_healpix plan = 
    (fastsht_plan_healpix)malloc(sizeof(struct _fastsht_plan_healpix));
  int iring;
  fastsht_grid_info *grid;
  int nrings = 4 * Nside - 1;
  bfm_index_t work_stride = 1 + mmax, start, stop;
  unsigned flags = FFTW_DESTROY_INPUT | FFTW_MEASURE;

  plan->base.type = PLANTYPE_HEALPIX;
  plan->base.input = input;
  plan->base.output = output;
  plan->base.work = work;
  plan->base.grid = grid = fastsht_create_healpix_grid_info(Nside);

  plan->Nside = Nside;
  plan->base.lmax = lmax;
  plan->base.mmax = mmax;
  plan->base.fft_plans = (fftw_plan*)malloc(nrings * sizeof(fftw_plan));
  for (iring = 0; iring != grid->nrings; ++iring) {
    start = grid->ring_offsets[iring];
    stop = grid->ring_offsets[iring + 1];
    plan->base.fft_plans[iring] = 
      fftw_plan_dft_c2r_1d(stop - start, (fftw_complex*)(work + 2 * work_stride * iring),
                           output + start, flags);
  }
  return (fastsht_plan)plan;
}

void fastsht_destroy_plan(fastsht_plan plan) {
  int iring;
  for (iring = 0; iring != plan->grid->nrings; ++iring) {
    fftw_destroy_plan(plan->fft_plans[iring]);
  }
  fastsht_free_grid_info(plan->grid);
  free(plan->fft_plans);
  free(plan);
}

void fastsht_execute(fastsht_plan plan) {

}

void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end) {
  int iring, mmax, j, N, offset;
  double *g_m;
  fastsht_grid_info *grid = plan->grid;
  /*
    TODO: Phase-shifting
  */
  mmax = plan->mmax;
  for (iring = ring_start; iring != ring_end; ++iring) {
    g_m = plan->work + 2 * (1 + mmax) * iring;
    N = grid->ring_offsets[iring + 1] - grid->ring_offsets[iring];
    /*    for (j = 0; j < 2 * (N/2 + 1); ++j) {
      g_m[j] = 0;
      }*/
    phase_shift_ring_inplace(mmax, (complex double*)g_m, grid->phi0s[iring]);
    fftw_execute(plan->fft_plans[iring]);
    /*    for (j = (plan->ring_offsets[iring] + plan->ring_offsets[iring + 1]) / 2;
         j < plan->ring_offsets[iring + 1]; ++j)
         plan->output[j] = 0;*/
  }
}

fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside) {
  int iring, ring_npix, ipix;
  int nrings = 4 * Nside - 1;
  /* Allocate all memory for the ring info in a single blob and just
     set up internal pointers. */
  char *buf = (char*)malloc(sizeof(fastsht_grid_info) + sizeof(double[nrings]) +
                            sizeof(bfm_index_t[nrings + 1]));
  fastsht_grid_info *result = (fastsht_grid_info*)buf;
  buf += sizeof(fastsht_grid_info);
  result->phi0s = (double*)buf;
  buf += sizeof(double[nrings]);
  result->ring_offsets = (bfm_index_t*)buf;

  result->nrings = nrings;
  ring_npix = 0;
  ipix = 0;
  for (iring = 0; iring != nrings; ++iring) {
    if (iring <= Nside - 1) {
      ring_npix += 4;
      result->phi0s[iring] = M_PI / (4.0 * (iring + 1));
    } else if (iring > 3 * Nside - 1) {
      ring_npix -= 4;
      result->phi0s[iring] = M_PI / (4.0 * (nrings - iring));
    } else {
      result->phi0s[iring] = (M_PI / (4.0 * Nside)) * (iring  % 2);
    }
    result->ring_offsets[iring] = ipix;
    ipix += ring_npix;
  }
  result->ring_offsets[nrings] = ipix;
  return result;
}

void fastsht_free_grid_info(fastsht_grid_info *info) {
  /* In the constructor we allocate the internal arrays as part of the
     same blob. */
  free((char*)info);
}
