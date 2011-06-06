#include "fastsht_private.h"
#include "fastsht_error.h"

#undef NDEBUG
#include "complex.h"
#include <assert.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

/* For memory mapping */
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>


#define PI 3.14159265358979323846

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
Global storage
*/

/*
The precomputed data, per m. For now we only support a single Nside
at the time, this will definitely change.
*/
typedef struct {
  char *even_matrix;
  char *odd_matrix;
} precomputation_t;


static char *mmapped_buffer;
static size_t mmap_len;
static precomputation_t *precomputed_data;



/*
Private
*/

static void phase_shift_ring_inplace(int mmax, complex double *g_m, double phi0) {
  int m = 0;
  for (m = 0; m != mmax; ++m) {
    g_m[m] *= (cos(m * phi0) + I * sin(m * phi0));
  }
}

static int read_int64(FILE *fd, int64_t *p_out, size_t n) {
  return fread(p_out, sizeof(int64_t), n, fd) == n;
}



static void print_array(char *msg, double* arr, bfm_index_t len) {
  bfm_index_t i;
  printf("%s ", msg);
  for (i = 0; i != len; ++i) {
    printf("%e ", arr[i]);
  }
  printf("\n");
}

/*
Public
*/


int fastsht_add_precomputation_file(char *filename) {
  int fd;
  struct stat fileinfo;
  int64_t mmax, m, len;
  int64_t *offsets = NULL;
  int retcode;
  char *head;
  /*
    Memory map buffer
   */
  mmapped_buffer = MAP_FAILED;
  fd = open(filename, O_RDONLY);
  if (fd == -1) goto ERROR;
  if (fstat(fd, &fileinfo) == -1) goto ERROR;
  mmap_len = fileinfo.st_size;
  mmapped_buffer = mmap(NULL, mmap_len, PROT_READ, MAP_SHARED,
                        fd, 0);
  if (mmapped_buffer == MAP_FAILED) goto ERROR;
  head = mmapped_buffer;
  /* Read mmax, allocate arrays, read offsets */
  mmax = ((int64_t*)head)[0];
  head += sizeof(int64_t);
  precomputed_data = malloc(sizeof(precomputation_t[mmax + 1]));
  memset(precomputed_data, 0, sizeof(precomputation_t[mmax + 1]));
  if (precomputed_data == NULL) goto ERROR;
  offsets = (int64_t*)head;
  /* Read compressed matrices */
  for (m = 0; m != mmax + 1; ++m) {
    precomputed_data[m].even_matrix = mmapped_buffer + offsets[4 * m];
    precomputed_data[m].odd_matrix = mmapped_buffer + offsets[4 * m + 2];
  }
  retcode = 0;
  goto FINALLY;
 ERROR:
  retcode = -1;
  free(precomputed_data);
  if (fd != -1) close(fd);
  if (mmapped_buffer == MAP_FAILED) {
    munmap(mmapped_buffer, mmap_len);
    mmapped_buffer = NULL;
  }
 FINALLY:
  return retcode;
}

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
  bfm_index_t m, lmax, mmax, nrows, nrings, ncols, iring, l, mid_ring, j;
  complex double *input_m;
  complex double *work_input, *work_even, *work_odd, *work;
  assert(plan->grid->has_equator);
  mmax = plan->mmax;
  lmax = plan->lmax;
  mid_ring = plan->grid->mid_ring;
  nrings = plan->grid->nrings;
  nrows = nrings - mid_ring;
  work_input = memalign(16, sizeof(complex double[2 * (lmax + 1)]));
  work_even = memalign(16, sizeof(complex double[mid_ring + 1]));
  work_odd = memalign(16, sizeof(complex double[mid_ring + 1]));
  work = (complex double*)plan->work;
  /* Compute g_m(theta_i), including transpose step for now */
  for (m = 0; m != mmax + 1; ++m) {
    input_m = (complex double*)plan->input + (m * (lmax + 1) - (m * (m - 1)) / 2);
    ncols = lmax - m + 1;

    /* Copy in even parts */
    ncols = 0;
    for (l = m + (m % 2); l <= lmax; l += 2) {
      work_input[ncols] = input_m[l];
      ++ncols;
    }
    /* Apply even matrix */
    bfm_apply_d(precomputed_data[m].even_matrix, (double*)work_input, (double*)work_even,
                nrows, ncols, 2);
    /* Odd part */
    ncols = 0;
    for (l = m + ((m + 1) % 2); l <= lmax; l += 2) {
      work_input[ncols] = input_m[l];
      ++ncols;
    }
    /* Apply odd matrix */
    bfm_apply_d(precomputed_data[m].odd_matrix, (double*)work_input, (double*)work_odd,
                nrows, ncols, 2);

    /* Add together parts and distribute/transpose to plan->work */
    /* Equator */
    work[mid_ring * (mmax + 1) + m] = work_even[0];
    /* Ring-pairs */
    for (iring = 1; iring < mid_ring + 1; ++iring) {
      /* Top ring -- switch odd sign */
      work[(mid_ring - iring) * (mmax + 1) + m] = 
        work_even[iring] - work_odd[iring]; /* sign! */
      /* Bottom ring */
      work[(mid_ring + iring) * (mmax + 1) + m] =
        work_even[iring] + work_odd[iring];
    }
  }
  free(work_input);
  free(work_even);
  free(work_odd);
  /* Backward FFTs from plan->work to plan->output */
  fastsht_perform_backward_ffts(plan, 0, nrings);
}

void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end) {
  int iring, mmax, j, N, offset;
  double *g_m;
  fastsht_grid_info *grid = plan->grid;
  mmax = plan->mmax;
  for (iring = ring_start; iring != ring_end; ++iring) {
    g_m = plan->work + 2 * (1 + mmax) * iring;
    N = grid->ring_offsets[iring + 1] - grid->ring_offsets[iring];
    phase_shift_ring_inplace(mmax, (complex double*)g_m, grid->phi0s[iring]);
    fftw_execute(plan->fft_plans[iring]);
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
  result->has_equator = 1;
  result->nrings = nrings;
  result->mid_ring = 2 * Nside - 1;
  ring_npix = 0;
  ipix = 0;
  for (iring = 0; iring != nrings; ++iring) {
    if (iring <= Nside - 1) {
      ring_npix += 4;
      result->phi0s[iring] = PI / (4.0 * (iring + 1));
    } else if (iring > 3 * Nside - 1) {
      ring_npix -= 4;
      result->phi0s[iring] = PI / (4.0 * (nrings - iring));
    } else {
      result->phi0s[iring] = (PI / (4.0 * Nside)) * (iring  % 2);
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
