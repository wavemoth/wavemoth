#include "fastsht.h"
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#define PLANTYPE_HEALPIX 0x0


struct _fastsht_plan {
  int type;
  bfm_index_t nrings;
  int lmax, mmax;
  double *output, *input, *work;

  /* To phase shift, we multiply with

         e^(i m phi_0) = cos(m phi_0) + i sin(m phi_0)

     for each ring. This is computed by
     
         cos(x + phi_0) = cos(x) - (alpha * cos(x) + beta * sin(x))
         sin(phi_0) = sin(x) - (alpha * sin(x) - beta * cos(x))

     and we precompute ...
   */
  double *phi0s;
  bfm_index_t *ring_offsets;
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

static void phase_shift_ring_inplace(int mmax, double *g_m, double phi0) {
  int m = 0;
  for (m = 0; m != 2 * mmax; m += 2) {
    g_m[m] *= cos(m * phi0);
    g_m[m + 1] *= sin(m * phi0);
  }
}


/*
Public
*/

fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax, double *input,
                                     double *output, double *work, int ordering) {
  fastsht_plan_healpix plan = 
    (fastsht_plan_healpix)malloc(sizeof(struct _fastsht_plan_healpix));
  int iring, ring_npix, ipix;
  int nrings = 4 * Nside - 1;
  bfm_index_t work_stride = 1 + mmax;
  unsigned flags = FFTW_DESTROY_INPUT | FFTW_MEASURE;

  plan->base.type = PLANTYPE_HEALPIX;
  plan->base.input = input;
  plan->base.output = output;
  plan->base.work = work;
  plan->base.nrings = nrings;
  plan->Nside = Nside;
  plan->base.lmax = lmax;
  plan->base.mmax = mmax;
  plan->base.fft_plans = (fftw_plan*)malloc(nrings * sizeof(fftw_plan));
  plan->base.phi0s = (double*)malloc(sizeof(double[nrings]));
  plan->base.ring_offsets = (bfm_index_t*)malloc(sizeof(bfm_index_t[nrings + 1]));
  ring_npix = 0;
  ipix = 0;
  for (iring = 0; iring != nrings; ++iring) {
    if (iring <= Nside - 1) {
      ring_npix += 4;
      plan->base.phi0s[iring] = M_PI / (4.0 * ring_npix);
    } else if (iring > 3 * Nside - 1) {
      ring_npix -= 4;
      plan->base.phi0s[iring] = M_PI / (4.0 * ring_npix);
    } else {
      plan->base.phi0s[iring] = (M_PI / (4.0 * ring_npix)) * (iring % 2);
    }
    plan->base.ring_offsets[iring] = ipix;
    plan->base.fft_plans[iring] = 
      fftw_plan_dft_c2r_1d(ring_npix, (fftw_complex*)(work + 2 * work_stride * iring),
                           output + ipix, flags);
    ipix += ring_npix;
  }
  plan->base.ring_offsets[nrings] = ipix;
  return (fastsht_plan)plan;

  /*
    counts = get_ring_pixel_counts(Nside)
    phi = np.zeros(4 * Nside - 1)
    ring_indices = np.arange(1, 4 * Nside)
    phi[:Nside - 1] = pi / (4 * ring_indices[:Nside - 1])
    phi[Nside - 1:3 * Nside] = pi / 4 / Nside
    phi[Nside:3 * Nside:2] = 0
    phi[3 * Nside:] = phi[Nside - 2::-1]
    return phi
  */
}

void fastsht_destroy_plan(fastsht_plan plan) {
  int iring;
  for (iring = 0; iring != plan->nrings; ++iring) {
    fftw_destroy_plan(plan->fft_plans[iring]);
  }
  free(plan->phi0s);
  free(plan->ring_offsets);
  free(plan);
}

void fastsht_execute(fastsht_plan plan) {

}

void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end) {
  int iring, mmax, j, N, offset;
  double *g_m;
  /*
    TODO: Phase-shifting
  */
  mmax = plan->mmax;
  for (iring = ring_start; iring != ring_end; ++iring) {
    g_m = plan->work + 2 * (1 + mmax) * iring;
    N = plan->ring_offsets[iring + 1] - plan->ring_offsets[iring];
    /*    for (j = 0; j < 2 * (N/2 + 1); ++j) {
      g_m[j] = 0;
      }*/
    /*    phase_shift_ring_inplace(mmax, g_m, plan->phi0s[iring]);*/
    fftw_execute(plan->fft_plans[iring]);
    /*    for (j = (plan->ring_offsets[iring] + plan->ring_offsets[iring + 1]) / 2;
         j < plan->ring_offsets[iring + 1]; ++j)
         plan->output[j] = 0;*/
  }
}
