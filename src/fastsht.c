#include "fastsht_private.h"
#include "fastsht_error.h"
#include "fmm1d.h"
#include "blas.h"

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

/* Wall timer */
#include <sys/times.h>
#include <time.h>

/*
Every time resource format changes, we increase this, so that
we can keep multiple resource files around and jump in git
history.
*/
#define RESOURCE_FORMAT_VERSION 0
/*#define RESOURCE_HEADER "butterfly-compressed matrix data"*/

#define PI 3.14159265358979323846

#define PLANTYPE_HEALPIX 0x0

static INLINE int imin(int a, int b) {
  return (a < b) ? a : b;
}



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
  char *matrix_data;
  size_t matrix_len;
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
  /*check(!configured, "Already configured");*/
  configured = 1;
  strncpy(global_resource_path, resource_path, MAX_RESOURCE_PATH);
  global_resource_path[MAX_RESOURCE_PATH - 1] = '\0';
  memset(precomputed_data, 0, sizeof(precomputed_data));
}

static void fastsht_get_resources_filename(char *filename, size_t buflen, int Nside) {
  snprintf(filename, buflen, "%s/rev%d/%d.dat",
           global_resource_path, RESOURCE_FORMAT_VERSION,
           Nside);
  filename[buflen - 1] = '\0';
}

int fastsht_query_resourcefile(char *filename, int *out_Nside, int *out_lmax) {
  FILE *fd;
  int64_t fields[3];

  fd = fopen(filename, "r");
  if (fd == NULL) return -1;
  checkf(fread(fields, sizeof(int64_t[3]), 1, fd) == 1, "Could not read file header: %s", filename);
  fclose(fd);
  *out_lmax = fields[0];
  *out_Nside = fields[2];
  return 0;
}

int fastsht_mmap_resources(char *filename, precomputation_t *data, int *out_Nside) {
  int fd;
  struct stat fileinfo;
  int64_t mmax, lmax, m, n, odd, should_interpolate, Nside;
  int64_t *offsets = NULL;
  m_resource_t *rec;
  int retcode;
  char *head;

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
  *out_Nside = Nside = ((int64_t*)head)[2];
  head += sizeof(int64_t[3]);
  data->P_matrices = malloc(sizeof(m_resource_t[2 * (mmax + 1)]));
  data->lmax = lmax;
  data->mmax = mmax;
  if (data->P_matrices == NULL) goto ERROR;
  offsets = (int64_t*)head;
  /* Assign pointers to compressed matrices. This doesn't actually load it
     into memory, just set out pointers into virtual memory presently on
     disk. */
  for (m = 0; m != mmax + 1; ++m) {
    n = (mmax - m) / 2;
    for (odd = 0; odd != 2; ++odd) {
      rec = data->P_matrices + 2 * m + odd;
      head = data->mmapped_buffer + offsets[4 * m + 2 * odd];
      if (head == data->mmapped_buffer) {
        /* For debugging/benchmarking, sometimes matrices are missing.
           In those cases the offset is registered as 0. */
        rec->matrix_data = NULL;
        continue;
      }
      should_interpolate = ((int64_t*)head)[0];
      check(!should_interpolate, "Interpolation not supported");
      rec->combined_matrix_size = ((int64_t*)head)[1];
      head = skip_padding(head + sizeof(int64_t[2]));
      rec->matrix_data = head;
      rec->matrix_len = offsets[4 * m + 2 * odd + 1];
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

volatile char _fastsht_dummy; /* Export symbol to avoid optimizing out. */

static void fastsht_swap_in_resources(precomputation_t *resources, int m_start, int m_stop) {
  int m, odd;
  m_resource_t *rec;
  char acc = 0;
  char *ptr, *stop;
  size_t size = 0;
  /* Read a byte every 1024 bytes from all the matrix data in the given range
     of m's, to force loading all pages from disk. */
  for (m = m_start; m < m_stop; ++m) {
    for (odd = 0; odd != 1; ++odd) {
      rec = resources->P_matrices + 2 * m + odd;
      if (rec->matrix_data == NULL) continue;
      stop = rec->matrix_data + rec->matrix_len;
      size += rec->matrix_len;
      for (ptr = rec->matrix_data; ptr < stop; ptr += 1) {
        //        printf("%ld %ld\n", (size_t)ptr, (size_t)stop);
        acc += *ptr;
      }
    }
  }
  fprintf(stderr, "Swapped in %d MB\n", (int)(size / 1024 / 1024));
  _fastsht_dummy = acc;
}

static void fastsht_swap_out_resources(precomputation_t *resources, int m_start, int m_stop) {
  int m, odd;
  m_resource_t *rec;
  size_t size = 0;
  /* Use madvise to tell OS to drop pages; they'll be reread from disk
     when needed. */
  for (m = m_start; m < m_stop; ++m) {
    for (odd = 0; odd != 1; ++odd) {
      rec = resources->P_matrices + 2 * m + odd;
      if (rec->matrix_data == NULL) continue;
      size += rec->matrix_len;
      madvise(rec->matrix_data, rec->matrix_len, MADV_DONTNEED);
    }
  }
  fprintf(stderr, "Swapped out %d MB\n", (int)(size / 1024 / 1024));
}

precomputation_t* fastsht_fetch_resource(int Nside) {
  int Nside_level = 0, tmp, got_Nside;
  char filename[MAX_RESOURCE_PATH];

  if (!configured) {
    return NULL;
  }
  fastsht_get_resources_filename(filename, MAX_RESOURCE_PATH, Nside);
  tmp = Nside;
  while (tmp /= 2) ++Nside_level;
  checkf(Nside_level < MAX_NSIDE_LEVEL + 1, "Nside=2**%d but maximum value is 2**%d",
         Nside_level, MAX_NSIDE_LEVEL);
  if (precomputed_data[Nside_level].refcount == 0) {
    check(fastsht_mmap_resources(filename, precomputed_data + Nside_level, &got_Nside) == 0,
          "resource load failed");
    checkf(Nside == got_Nside, "Loading precomputation: Expected Nside=%d but got %d in %s",
           Nside, got_Nside, filename);
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

fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax, int nmaps,
                                     double *input, double *output, double *work,
                                     int ordering, char *resource_filename) {
  fastsht_plan plan = malloc(sizeof(struct _fastsht_plan));
  int nrings, iring;
  int out_Nside;
  fastsht_grid_info *grid;
  bfm_index_t work_stride = 1 + mmax, start, stop;
  unsigned flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE;

  if (resource_filename != NULL) {
    /* Used in debugging/benchmarking */
    plan->resources = malloc(sizeof(precomputation_t));
    plan->did_allocate_resources = 1;
    checkf(fastsht_mmap_resources(resource_filename, plan->resources, &out_Nside) == 0,
           "Error in loading resource %s", resource_filename);
    check(Nside < 0 || out_Nside == Nside, "Incompatible Nside");
    Nside = out_Nside;
  } else {
    check(Nside >= 0, "Invalid Nside");
    plan->did_allocate_resources = 0;
    plan->resources = fastsht_fetch_resource(Nside);
  }

  nrings = 4 * Nside - 1;

  plan->type = PLANTYPE_HEALPIX;
  plan->input = input;
  plan->output = output;
  plan->work = work;
  plan->grid = grid = fastsht_create_healpix_grid_info(Nside);
  plan->nmaps = nmaps;

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
  if (plan->did_allocate_resources) free(plan->resources);
  free(plan->fft_plans);
  free(plan);
}

int64_t fastsht_get_legendre_flops(fastsht_plan plan, int m, int odd) {
  int64_t N, nvecs;
  N = (plan->resources->P_matrices + 2 * m + odd)->combined_matrix_size;
  nvecs = 2;
  N *= nvecs;
  return N * 2; /* count mul and add seperately */
}

void fastsht_legendre_transform(fastsht_plan plan, int mstart, int mstop, int mstride) {
  int lmax = plan->lmax, nmaps = plan->nmaps, nrings_half = plan->grid->mid_ring + 1;

#pragma omp parallel
  {
    int m, odd;
    m_resource_t *rec;
    double complex *work_g_m_even, *work_g_m_odd, *work_a_l;

    work_a_l = memalign(16, sizeof(complex double[2 * nmaps * (lmax + 1)])); /*TODO: 2 x here? */
    work_g_m_even = memalign(16, sizeof(complex double[nmaps * nrings_half]));
    work_g_m_odd = memalign(16, sizeof(complex double[nmaps * nrings_half]));
  
    /* Compute g_m(theta_i), including transpose step for now */
#pragma omp for schedule(dynamic,1)
    for (m = mstart; m < mstop - 2; m += mstride) { /* TODO TODO TODO */
      for (odd = 0; odd != 2; ++odd) {
        rec = plan->resources->P_matrices + 2 * m + odd;
        check(rec->matrix_data != NULL, "matrix data not present, invalid mstride");
        fastsht_perform_matmul(plan, m, odd, work_a_l, odd ? work_g_m_odd : work_g_m_even);
      }
      fastsht_merge_even_odd_and_transpose(plan, m, work_g_m_even, work_g_m_odd);
    }
    free(work_a_l);
    free(work_g_m_even);
    free(work_g_m_odd);
  }
}

void fastsht_execute(fastsht_plan plan) {
  fastsht_legendre_transform(plan, 0, plan->mmax + 1, 1);
  /* Backward FFTs from plan->work to plan->output */
  fastsht_perform_backward_ffts(plan, 0, plan->grid->nrings);
}

void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd,
                            double complex *work_a_l, double complex *output) {
  bfm_index_t ncols, nrows, l, lmax = plan->lmax, j, nmaps = plan->nmaps;
  double complex *input_m = (double complex*)plan->input + nmaps * m * (2 * lmax - m + 3) / 2;
  double complex *target;
  m_resource_t *rec;
  rec = plan->resources->P_matrices + 2 * m + odd;
  ncols = 0;
  for (l = m + odd; l <= lmax; l += 2) {
    for (j = 0; j != nmaps; ++j) {
      work_a_l[ncols * nmaps + j] = input_m[(l - m) * nmaps + j];
    }
    ++ncols;
  }
  nrows = plan->grid->nrings - plan->grid->mid_ring;
  /* Apply even matrix to evaluate g_{odd,m}(theta) at n Ass. Legendre roots*/
  bfm_apply_d(rec->matrix_data, (double*)work_a_l, (double*)output,
              nrows, ncols, 2 * plan->nmaps);
}

void fastsht_merge_even_odd_and_transpose(fastsht_plan plan, int m,
                                          double complex *g_m_even,
                                          double complex *g_m_odd) {
  bfm_index_t mmax, nrings, iring, mid_ring, imap;
  double complex *work;
  int nmaps = plan->nmaps;
  bfm_index_t work_stride;
  assert(plan->grid->has_equator);
  mmax = plan->mmax;
  mid_ring = plan->grid->mid_ring;
  nrings = plan->grid->nrings;
  work = (complex double*)plan->work;
  /* Add together parts and distribute/transpose to plan->work */
  /* Equator */
  work_stride = (2 * mid_ring + 1) * (mmax + 1);
  for (imap = 0; imap != nmaps; ++imap) {
    work[imap * work_stride + mid_ring * (mmax + 1) + m] = g_m_even[imap];
  }
  /* Ring-pairs */
  for (iring = 1; iring < mid_ring + 1; ++iring) {
    for (imap = 0; imap != nmaps; ++imap) {
      /* Top ring */
      work[imap * work_stride + (mid_ring - iring) * (mmax + 1) + m] = 
        g_m_even[iring * nmaps + imap] + g_m_odd[iring * nmaps + imap];
      /* Bottom ring -- switch odd sign */
      work[imap * work_stride + (mid_ring + iring) * (mmax + 1) + m] =
        g_m_even[iring * nmaps + imap] - g_m_odd[iring * nmaps + imap];
    }
  }
}

void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end) {
  fastsht_grid_info *grid = plan->grid;
  int mmax = plan->mmax;
  size_t work_stride = (2 * plan->grid->mid_ring + 1) * (mmax + 1);
  size_t npix = plan->grid->npix;
#pragma omp parallel
  {
    double complex *g_m;
    double *output_ring;
    int iring, imap, j, k, N;
    imap = 0;
    for (imap = 0; imap < plan->nmaps; ++imap) {
#pragma omp for schedule(dynamic, 16) nowait
      for (iring = ring_start; iring < ring_end; ++iring) {
        g_m = (double complex*)plan->work + imap * work_stride + (1 + mmax) * iring;
        N = grid->ring_offsets[iring + 1] - grid->ring_offsets[iring];
        phase_shift_ring_inplace(mmax, g_m, grid->phi0s[iring]);
        /* Zero the rest of the array. TODO TODO: Get rid of the -2. */
        for (k = mmax - 2; k < N / 2 + 1; ++k) {
          g_m[k] = 0;
        }
        output_ring = plan->output + imap * npix + plan->grid->ring_offsets[iring];
        fftw_execute_dft_c2r(plan->fft_plans[iring], g_m, output_ring);
      }
    }
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
  result->npix = ipix;
  return result;
}

void fastsht_free_grid_info(fastsht_grid_info *info) {
  /* In the constructor we allocate the internal arrays as part of the
     same blob. */
  free((char*)info);
}
