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
  double *evaluation_grid_squared, *output_grid_squared, *gamma, *omega;
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

  plan->work_a_l = memalign(16, sizeof(complex double[2 * nmaps * (lmax + 1)])); /*TODO: 2 x here? */
  plan->work_g_m_roots = memalign(16, sizeof(complex double[nmaps * ((lmax + 1) / 2)]));
  plan->work_g_m_even = memalign(16, sizeof(complex double[nmaps * (plan->grid->mid_ring + 1)]));
  plan->work_g_m_odd = memalign(16, sizeof(complex double[nmaps * (plan->grid->mid_ring + 1)]));

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
  m_resource_t *rec;
  mmax = plan->mmax;
  /* Compute g_m(theta_i), including transpose step for now */
  for (m = 0; m != mmax + 1 - 2; ++m) { /* TODO TODO TODO */
    for (odd = 0; odd != 2; ++odd) {
      rec = plan->resources->P_matrices + 2 * m + odd;
      if (rec->matrix_data == NULL) {
        /* Only during benchmarks, so computed values are unimportant. */
        continue;
      }
      fastsht_perform_matmul(plan, m, odd);
      if (rec->evaluation_grid_squared != NULL) {
        /* Interpolate with FMM */
        fastsht_perform_interpolation(plan, m, odd);
      }
    }
    fastsht_merge_even_odd_and_transpose(plan, m);
  }
  /* Backward FFTs from plan->work to plan->output */
  fastsht_perform_backward_ffts(plan, 0, plan->grid->nrings);
}

typedef struct tms benchtime_t;
#define ZERO_TIMER {0, 0, 0, 0}

static benchtime_t walltime_fetch() {
  struct tms t;
  times(&t);
  return t;
  /*
  struct timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  return tv;*/
}

static benchtime_t walltime_add_elapsed(benchtime_t *target, benchtime_t t0) {
  struct tms t = {0, 0, 0, 0};
  times(&t);
  target->tms_utime += t.tms_utime - t0.tms_utime;
  return t;
  /*struct timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  target->tv_sec += tv.tv_sec - t0.tv_sec;
  target->tv_nsec += tv.tv_nsec - t0.tv_nsec;
  return tv;*/
}

double timer_to_double(benchtime_t t) {
  return (double)t.tms_utime / sysconf(_SC_CLK_TCK);
  //return tv.tv_sec + 1e-9 * tv.tv_nsec;
}

void fastsht_execute_out_of_core(fastsht_plan plan,
                                 double *out_compute_time,
                                 double *out_load_time) {
  /* Do an out-of-core computation where we start and stop the clock,
     to emulate a benchmark on a bigger-memory machine.

     We keep two buffers of 'chunksize' m's each; we take care
     to interleave loading and computation so that preloaded
     data should be evicted from cache before being used. Seperate
     clocks is used for loading and computation.
   */
  int m, mmax, m_chunk_start, m_chunk_stop;
  int odd;
  const int chunksize = 40;
  
  benchtime_t t0, t1;
  benchtime_t t_compute = ZERO_TIMER, t_load = ZERO_TIMER;

  mmax = plan->mmax;

  /* Swap in first chunk and enter loop */
  printf("Load\n");
  t0 = walltime_fetch();
  fastsht_swap_in_resources(plan->resources, 0, imin(chunksize, mmax + 1));
  t0 = walltime_add_elapsed(&t_load, t0);
  mmax -= 2; // TODO fix mmax!
  for (m_chunk_start = 0; m_chunk_start < mmax + 1; m_chunk_start += chunksize) {
    m_chunk_stop = imin(m_chunk_start + chunksize, mmax + 1);
    /* Swap in next chunk before computation */
    printf("Load m=%d\n", m_chunk_start);
    fastsht_swap_in_resources(plan->resources,
                              m_chunk_stop,
                              imin(m_chunk_stop + chunksize, mmax + 1));
    t0 = walltime_add_elapsed(&t_load, t0);
    /* Computation */
    printf("Compute\n");
    for (m = m_chunk_start; m != m_chunk_stop; ++m) {
      for (odd = 0; odd != 2; ++odd) {
        fastsht_perform_matmul(plan, m, odd);
      }
      fastsht_merge_even_odd_and_transpose(plan, m);    
    }
    t0 = walltime_add_elapsed(&t_compute, t0);
    fastsht_swap_out_resources(plan->resources, m_chunk_start, m_chunk_stop);
    t0 = walltime_add_elapsed(&t_load, t0);
  }
  /* Backward FFTs from plan->work to plan->output */
  printf("FFT\n");
  fastsht_perform_backward_ffts(plan, 0, plan->grid->nrings);
  walltime_add_elapsed(&t_compute, t0);
  *out_compute_time = timer_to_double(t_compute);
  *out_load_time = timer_to_double(t_load);
}

void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd) {
  bfm_index_t ncols, nrows, l, lmax = plan->lmax, j, nmaps = plan->nmaps;
  double complex *input_m = (double complex*)plan->input + nmaps * m * (2 * lmax - m + 3) / 2;
  double complex *target;
  m_resource_t *rec;
  rec = plan->resources->P_matrices + 2 * m + odd;
  ncols = 0;
  for (l = m + odd; l <= lmax; l += 2) {
    for (j = 0; j != nmaps; ++j) {
      plan->work_a_l[ncols * nmaps + j] = input_m[(l - m) * nmaps + j];
    }
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
              nrows, ncols, 2 * plan->nmaps);
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
    work[imap * work_stride + mid_ring * (mmax + 1) + m] = plan->work_g_m_even[imap];
  }
  /* Ring-pairs */
  for (iring = 1; iring < mid_ring + 1; ++iring) {
    for (imap = 0; imap != nmaps; ++imap) {
      /* Top ring */
      work[imap * work_stride + (mid_ring - iring) * (mmax + 1) + m] = 
        plan->work_g_m_even[iring * nmaps + imap] + plan->work_g_m_odd[iring * nmaps + imap];
      /* Bottom ring -- switch odd sign */
      work[imap * work_stride + (mid_ring + iring) * (mmax + 1) + m] =
        plan->work_g_m_even[iring * nmaps + imap] - plan->work_g_m_odd[iring * nmaps + imap];
    }
  }
}

void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end) {
  int iring, imap, mmax, j, N;
  double complex *g_m;
  double *output_ring;
  bfm_index_t work_stride, npix;
  fastsht_grid_info *grid = plan->grid;
  mmax = plan->mmax;
  work_stride = (2 * plan->grid->mid_ring + 1) * (mmax + 1);
  npix = plan->grid->npix;
  imap = 0;
  for (imap = 0; imap != plan->nmaps; ++imap) {
    for (iring = ring_start; iring != ring_end; ++iring) {
      g_m = (double complex*)plan->work + imap * work_stride + (1 + mmax) * iring;
      N = grid->ring_offsets[iring + 1] - grid->ring_offsets[iring];
      phase_shift_ring_inplace(mmax, g_m, grid->phi0s[iring]);
      output_ring = plan->output + imap * npix + plan->grid->ring_offsets[iring];
      fftw_execute_dft_c2r(plan->fft_plans[iring], g_m, output_ring);
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
