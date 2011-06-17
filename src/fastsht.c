#include "fastsht_private.h"
#include "fastsht_error.h"
#include "fmm1d.h"

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


/*
Every time resource format changes, we increase this, so that
we can keep multiple resource files around and jump in git
history.
*/
#define RESOURCE_FORMAT_VERSION 0
/*#define RESOURCE_HEADER "butterfly-compressed matrix data"*/

#define PI 3.14159265358979323846

#define PLANTYPE_HEALPIX 0x0




/*
Global storage of precomputed data.

Precomputed data is stored as:
 - Array of precomputation_t indexed by resolution nside_level (Nside = 2**nside_level)
*/

#define MAX_NSIDE_LEVEL 15

/*
The precomputed data, per m. For now we only support a single Nside
at the time, this will definitely change.
*/
typedef struct {
  double *evaluation_grid_squared, *output_grid_squared, *gamma, *omega;
  char *matrix_data;
  int64_t combined_matrix_size;
} m_resource_t;

struct _precomputation_t {
  char *mmapped_buffer;
  size_t mmap_len;
  m_resource_t *P_matrices;  /* 2D array [ms, odd ? 1 : 0] */
  int lmax, mmax;
  int refcount;
}; /* typedef in fastsht_private.h */


/*
Precomputed
*/
static precomputation_t precomputed_data[MAX_NSIDE_LEVEL + 1];

#define MAX_RESOURCE_PATH 2048
static char global_resource_path[MAX_RESOURCE_PATH];

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

static int configured = 0;

static char *skip_padding(char *ptr) {
  size_t m = (size_t)ptr % 16;
  if (m == 0) {
    return ptr;
  } else { 
    return ptr + 16 - m;
  }
}

void fastsht_configure(char *resource_path) {
  int i;
  /*check(!configured, "Already configured");*/
  configured = 1;
  strncpy(global_resource_path, resource_path, MAX_RESOURCE_PATH);
  global_resource_path[MAX_RESOURCE_PATH - 1] = '\0';
  memset(precomputed_data, 0, sizeof(precomputed_data));
}

int fastsht_load_resource(int Nside, precomputation_t *data) {
  int fd;
  struct stat fileinfo;
  int64_t mmax, lmax, m, len, n, odd, should_interpolate;
  int64_t *offsets = NULL;
  m_resource_t *rec;
  int retcode;
  char *head;
  char filename[MAX_RESOURCE_PATH];

  /* 
   */
  snprintf(filename, MAX_RESOURCE_PATH, "%s/rev%d/%d.dat",
           global_resource_path, RESOURCE_FORMAT_VERSION,
           Nside);
  filename[MAX_RESOURCE_PATH - 1] = '\0';

  /*
    Memory map buffer
   */
  data->mmapped_buffer = MAP_FAILED;
  fd = open(filename, O_RDONLY);
  if (fd == -1) goto ERROR;
  if (fstat(fd, &fileinfo) == -1) goto ERROR;
  data->mmap_len = fileinfo.st_size;
  data->mmapped_buffer = mmap(NULL, data->mmap_len, PROT_READ, MAP_SHARED,
                              fd, 0);
  if (data->mmapped_buffer == MAP_FAILED) goto ERROR;
  close(fd);
  fd = -1;
  head = data->mmapped_buffer;
  /* Read mmax, allocate arrays, read offsets */
  lmax = ((int64_t*)head)[0];
  mmax = ((int64_t*)head)[1];
  checkf(Nside == ((int64_t*)head)[2], "Loading precomputation: Expected Nside=%d but got %d in %s",
         Nside, (int)((int64_t*)head)[2], filename);
  head += sizeof(int64_t[3]);
  data->P_matrices = malloc(sizeof(m_resource_t[2 * (mmax + 1)]));
  data->lmax = lmax;
  data->mmax = mmax;
  if (data->P_matrices == NULL) goto ERROR;
  offsets = (int64_t*)head;
  /* Read compressed matrices */
  for (m = 0; m != mmax + 1; ++m) {
    n = (mmax - m) / 2;
    for (odd = 0; odd != 2; ++odd) {
      rec = data->P_matrices + 2 * m + odd;
      head = data->mmapped_buffer + offsets[4 * m + 2 * odd];
      should_interpolate = ((int64_t*)head)[0];
      rec->combined_matrix_size = ((int64_t*)head)[1];
      head = skip_padding(head + sizeof(int64_t[2]));
      if (should_interpolate) {
        //      assert(n > 0);
        rec->evaluation_grid_squared = (double*)head;
        head = skip_padding(head + sizeof(double[n]));
        rec->gamma = (double*)head;
        head = skip_padding(head + sizeof(double[n]));
        rec->output_grid_squared = (double*)head;
        head = skip_padding(head + sizeof(double[2 * Nside]));
        rec->omega = (double*)head;
        head = skip_padding(head + sizeof(double[2 * Nside]));
      } else {
        rec->evaluation_grid_squared = rec->gamma = rec->output_grid_squared = rec->omega = NULL;
      }
      rec->matrix_data = head;
    }
  }
  retcode = 0;
  goto FINALLY;
 ERROR:
  retcode = -1;
  if (fd != -1) close(fd);
  if (data->mmapped_buffer == MAP_FAILED) {
    munmap(data->mmapped_buffer, data->mmap_len);
    data->mmapped_buffer = NULL;
  }
 FINALLY:
  return retcode;
}

precomputation_t* fastsht_fetch_resource(int Nside) {
  int Nside_level = 0, tmp;
  if (!configured) {
    return NULL;
  }
  tmp = Nside;
  while (tmp /= 2) ++Nside_level;
  checkf(Nside_level < MAX_NSIDE_LEVEL + 1, "Nside=2**%d but maximum value is 2**%d",
         Nside_level, MAX_NSIDE_LEVEL);
  if (precomputed_data[Nside_level].refcount == 0) {
    check(fastsht_load_resource(Nside, precomputed_data + Nside_level) == 0,
          "resource load failed");
  }
  ++precomputed_data[Nside_level].refcount;
  return &precomputed_data[Nside_level];
}

void fastsht_release_resource(precomputation_t *data) {
  --data->refcount;
  if (data->refcount == 0) {
    munmap(data->mmapped_buffer, data->mmap_len);
    data->mmapped_buffer = NULL;
    free(data->P_matrices);
  }
}

fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax, double *input,
                                     double *output, double *work, int ordering) {
  fastsht_plan plan = malloc(sizeof(struct _fastsht_plan));
  int iring;
  fastsht_grid_info *grid;
  int nrings = 4 * Nside - 1;
  bfm_index_t work_stride = 1 + mmax, start, stop;
  unsigned flags = FFTW_DESTROY_INPUT | FFTW_MEASURE;

  plan->type = PLANTYPE_HEALPIX;
  plan->input = input;
  plan->output = output;
  plan->work = work;
  plan->grid = grid = fastsht_create_healpix_grid_info(Nside);

  plan->work_a_l = memalign(16, sizeof(complex double[2 * (lmax + 1)])); /*TODO: 2 x here? */
  plan->work_g_m_roots = memalign(16, sizeof(complex double[(lmax + 1) / 2]));
  plan->work_g_m_even = memalign(16, sizeof(complex double[plan->grid->mid_ring + 1]));
  plan->work_g_m_odd = memalign(16, sizeof(complex double[plan->grid->mid_ring + 1]));

  plan->resources = fastsht_fetch_resource(Nside);

  plan->Nside = Nside;
  plan->lmax = lmax;
  plan->mmax = mmax;
  plan->fft_plans = (fftw_plan*)malloc(nrings * sizeof(fftw_plan));
  for (iring = 0; iring != grid->nrings; ++iring) {
    start = grid->ring_offsets[iring];
    stop = grid->ring_offsets[iring + 1];
    plan->fft_plans[iring] = 
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
  fastsht_release_resource(plan->resources);
  free(plan->fft_plans);
  free(plan->work_a_l);
  free(plan->work_g_m_roots);
  free(plan->work_g_m_even);
  free(plan->work_g_m_odd);
  free(plan);
}

int64_t fastsht_get_legendre_flops(fastsht_plan plan, int m, int odd) {
  int64_t N, nvecs;
  N = (plan->resources->P_matrices + 2 * m + odd)->combined_matrix_size;
  nvecs = 2;
  N *= nvecs;
  return N * 2; /* count mul and add seperately */
}

void fastsht_execute(fastsht_plan plan) {
  bfm_index_t m, mmax;
  int odd;
  mmax = plan->mmax;
  /* Compute g_m(theta_i), including transpose step for now */
  for (m = 0; m != mmax + 1 - 2; ++m) { /* TODO TODO TODO */
    for (odd = 0; odd != 2; ++odd) {
      fastsht_perform_matmul(plan, m, odd);
      if ((plan->resources->P_matrices + 2 * m + odd)->evaluation_grid_squared != NULL) {
        /* Interpolate with FMM */
        fastsht_perform_interpolation(plan, m, odd);
      }
    }
    fastsht_merge_even_odd_and_transpose(plan, m);
  }
  /* Backward FFTs from plan->work to plan->output */
  fastsht_perform_backward_ffts(plan, 0, plan->grid->nrings);
}

void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd) {
  bfm_index_t ncols, nrows, l, lmax = plan->lmax;
  double complex *input_m = (double complex*)plan->input + m * (2 * lmax - m + 3) / 2;
  double complex *target;
  m_resource_t *rec;
  rec = plan->resources->P_matrices + 2 * m + odd;
  ncols = 0;
  for (l = m + odd; l <= lmax; l += 2) {
    plan->work_a_l[ncols] = input_m[l - m];
    ++ncols;
  }
  if (rec->evaluation_grid_squared == NULL) {
    target = odd ? plan->work_g_m_odd : plan->work_g_m_even;
    nrows = plan->grid->nrings - plan->grid->mid_ring;
  } else {
    target = plan->work_g_m_roots;
    nrows = (lmax - m) / 2;
  }
  /* Apply even matrix to evaluate g_{odd,m}(theta) at n Ass. Legendre roots*/
  bfm_apply_d(rec->matrix_data, (double*)plan->work_a_l, (double*)target,
              nrows, ncols, 2);
}

void fastsht_perform_interpolation(fastsht_plan plan, bfm_index_t m, int odd) {
  bfm_index_t n;
  m_resource_t *rec;
  n = (plan->lmax - m) / 2;
  rec = plan->resources->P_matrices + 2 * m + odd;
  assert(rec->evaluation_grid_squared != NULL); /* should_interpolate flag */
  fastsht_fmm1d(rec->evaluation_grid_squared, rec->gamma, (double*)plan->work_g_m_roots, n,
                rec->output_grid_squared, rec->omega,
                (double*)(odd ? plan->work_g_m_odd : plan->work_g_m_even),
                2 * plan->Nside, 2);
}

void fastsht_merge_even_odd_and_transpose(fastsht_plan plan, bfm_index_t m) {
  bfm_index_t mmax, nrings, iring, mid_ring;
  double complex *work;
  assert(plan->grid->has_equator);
  mmax = plan->mmax;
  mid_ring = plan->grid->mid_ring;
  nrings = plan->grid->nrings;
  work = (complex double*)plan->work;
  /* Add together parts and distribute/transpose to plan->work */
  /* Equator */
  work[mid_ring * (mmax + 1) + m] = plan->work_g_m_even[0];
  /* Ring-pairs */
  for (iring = 1; iring < mid_ring + 1; ++iring) {
    /* Top ring */
    work[(mid_ring - iring) * (mmax + 1) + m] = 
      plan->work_g_m_even[iring] + plan->work_g_m_odd[iring];
    /* Bottom ring -- switch odd sign */
    work[(mid_ring + iring) * (mmax + 1) + m] =
      plan->work_g_m_even[iring] - plan->work_g_m_odd[iring]; /* sign! */
  }
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
