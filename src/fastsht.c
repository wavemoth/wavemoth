#undef NDEBUG

#define _GNU_SOURCE

/* libc */
#include <complex.h>
#include <assert.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* intrinsics */
#include <xmmintrin.h>
#include <emmintrin.h>

/* OS, Numa, pthreads, OpenMP */
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <numa.h>
#include <numaif.h>

#include <pthread.h>
#include <sched.h>

#include <sys/times.h>
#include <time.h>

#include <omp.h>

/* FFTW3 */
#include <fftw3.h>

/* Wavemoth */

#include "fastsht_private.h"
#include "fastsht_error.h"
#include "fmm1d.h"
#include "blas.h"
#include "butterfly_utils.h"
#include "legendre_transform.h"

typedef __m128d m128d;


/*
Every time resource format changes, we increase this, so that
we can keep multiple resource files around and jump in git
history.
*/
#define RESOURCE_FORMAT_VERSION 1
/*#define RESOURCE_HEADER "butterfly-compressed matrix data"*/

#define PI 3.14159265358979323846

#define PLANTYPE_HEALPIX 0x0

#define FFT_CHUNK_SIZE 4

static INLINE int imin(int a, int b) {
  return (a < b) ? a : b;
}

static INLINE size_t zmax(size_t a, size_t b) {
  return (a > b) ? a : b;
}

/** A more useful mod function; the result will have the same sign as
    the divisor rather than the dividend.
 */
static INLINE int imod_divisorsign(int a, int b) {
  int r;
  assert(-7 % 4 == -3); /* Behaviour of % is implementation-defined until C99 */
  r = a % b;
  r += ((r != 0) & ((r ^ b) < 0)) * b;
  return r;
}

/*
Global storage of precomputed data.

Precomputed data is stored as:
 - Array of precomputation_t indexed by resolution nside_level (Nside = 2**nside_level)
*/

#define MAX_NSIDE_LEVEL 15


/*
Precomputed
*/
static precomputation_t precomputed_data[MAX_NSIDE_LEVEL + 1];

#define MAX_RESOURCE_PATH 2048
static char global_resource_path[MAX_RESOURCE_PATH];

/*
Private
*/

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

/*
#define PAGESIZE 4096
#define WEAK __attribute__((weak))
#define __NR_move_pages 279
#define MPOL_MF_STRICT  (1<<0)
#define MPOL_MF_MOVE	(1<<1) 
#define MPOL_MF_MOVE_ALL (1<<2)
long WEAK move_pages(int pid, unsigned long count,
	void **pages, const int *nodes, int *status, int flags) {
  return syscall(__NR_move_pages, pid, count, pages, nodes, status, flags);
}


static void migrate_pages_numa(void *start, size_t len, int node) {
  start -= (size_t)start % PAGESIZE;
  len += (size_t)start % PAGESIZE;
  if (len % PAGESIZE > 0) {
    len += PAGESIZE - (len % PAGESIZE);
  }
  size_t pagecount = len / PAGESIZE;
  void *pages[1024];
  int nodes[1024];
  int status[1024];
  for (size_t i = 0; i < pagecount; i += 1024) {
    size_t stop = imin(1024, pagecount - i);
    for (size_t j = 0; j != stop; j++) {
      pages[j] = start + (i * 1024 + j) * PAGESIZE;
      nodes[j] = node;
      status[j] = 0;
    }
    int ret = move_pages(getpid(), stop, pages, nodes, status, MPOL_MF_MOVE);
    for (size_t j = 0; j != stop; j++) {
      if (status[j] < 0) {
	printf("status: %d of %d %d %d %d %d %d %d %d\n", status[j],
	EACCES, EINVAL, ENODEV, ENOENT, EPERM, ENOMEM, E2BIG, ESRCH 
	       );
	//-1= EPERM
      }
    }
    printf("%d\n", ret);
  }
  }*/

//int fastsht_copy_resources(precomputation_t *data


int fastsht_mmap_resources(char *filename, precomputation_t *data, int *out_Nside) {
  int fd;
  struct stat fileinfo;
  int64_t mmax, lmax, m, n, odd, Nside;
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
  lmax = read_int64(&head);
  mmax = read_int64(&head);
  *out_Nside = Nside = read_int64(&head);
  data->matrices = malloc(sizeof(m_resource_t[mmax + 1]));
  if (data->matrices == NULL) goto ERROR;
  data->lmax = lmax;
  data->mmax = mmax;
  offsets = (int64_t*)head;
  /* Assign pointers to compressed matrices. This doesn't actually load it
     into memory, just set out pointers into virtual memory presently on
     disk. */
  for (m = 0; m != mmax + 1; ++m) {
    n = (mmax - m) / 2;
    for (odd = 0; odd != 2; ++odd) {
      rec = &data->matrices[m];
      head = data->mmapped_buffer + offsets[4 * m + 2 * odd];
      if (head == data->mmapped_buffer) {
        /* For debugging/benchmarking, sometimes matrices are missing.
           In those cases the offset is registered as 0. */
        rec->data[odd] = NULL;
        continue;
      }
      rec->data[odd] = head;
      rec->len[odd] = offsets[4 * m + 2 * odd + 1];
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
  return;
  --data->refcount;
  if (data->refcount == 0) {
    munmap(data->mmapped_buffer, data->mmap_len);
    data->mmapped_buffer = NULL;
    free(data->matrices);
  }
}

static void bitmask_or(struct bitmask *a, struct bitmask *b, struct bitmask *out) {
  int n = numa_bitmask_nbytes(a) * 8;
  for (int i = 0; i != n; ++i) {
    if (numa_bitmask_isbitset(a, i) && numa_bitmask_isbitset(b, i)) {
      numa_bitmask_setbit(out, i);
    } else {
      numa_bitmask_clearbit(out, i);
    }
  }
}

typedef void (*thread_main_func_t)(fastsht_plan, int, void*);

typedef struct {
  void *ctx;
  thread_main_func_t func;
  fastsht_plan plan;
  int ithread;
} thread_ctx_t;

static void *thread_main_adaptor(void *ctx_) {
  thread_ctx_t *ctx = ctx_;

  /* Ensure that the thread only allocates memory locally. */
  int node = ctx->plan->threadlocal[ctx->ithread].node;
  struct bitmask *mask = numa_allocate_nodemask();
  numa_bitmask_clearall(mask);
  numa_bitmask_setbit(mask, node);
  numa_set_membind(mask);
  numa_free_nodemask(mask);

  ctx->func(ctx->plan, ctx->ithread, ctx->ctx);
  return NULL;
}

static void fastsht_run_in_threads(fastsht_plan plan, thread_main_func_t func, void *ctx) {
  /* Spawn pthreads on the CPUs designated in the plan, and wait for them
     to finish. Each thread will only allocate memory locally, because
     of numa_set_membind in thread_main_adaptor. */
  int n = plan->nthreads;
  thread_ctx_t adaptor_ctx[n];
  pthread_t threads[n];
  for (int ithread = 0; ithread != n; ++ithread) {
    cpu_set_t cpu_set;
    pthread_attr_t attr;
    CPU_ZERO(&cpu_set);
    CPU_SET(plan->threadlocal[ithread].cpu, &cpu_set);

    adaptor_ctx[ithread].ctx = ctx;
    adaptor_ctx[ithread].plan = plan;
    adaptor_ctx[ithread].func = func;
    adaptor_ctx[ithread].ithread = ithread;

    pthread_attr_init(&attr);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpu_set);
    pthread_create(&threads[ithread], &attr, thread_main_adaptor, &adaptor_ctx[ithread]);
    pthread_attr_destroy(&attr);
  }
  for (int ithread = 0; ithread != n; ++ithread) {
    pthread_join(threads[ithread], NULL);
  }
}

static void fastsht_create_plan_thread(fastsht_plan plan, int ithread, void *ctx); /* forward decl */

fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax, int nmaps,
                                     int nthreads, double *input, double *output,
                                     int ordering, char *resource_filename) {
  fastsht_plan plan = malloc(sizeof(struct _fastsht_plan));
  int nrings, iring;
  int out_Nside;
  fastsht_grid_info *grid;
  bfm_index_t start, stop;

  /* Simple attribute assignment */
  nrings = 4 * Nside - 1;
  plan->nthreads = nthreads;
  plan->type = PLANTYPE_HEALPIX;
  plan->input = input;
  plan->output = output;
  plan->grid = grid = fastsht_create_healpix_grid_info(Nside);
  plan->nmaps = nmaps;
  plan->Nside = Nside;
  plan->lmax = lmax;
  plan->mmax = mmax;

  /* Figure out how threads should be distributed. We query NUMA for
     the nodes we can run on (intersection of cpubind and membind),
     and then fill up node-by-node until we
     hit nthreads. */

  plan->threadlocal = malloc(sizeof(fastsht_plan_threadlocal[nthreads]));

  if (nthreads <= 0) {
    fprintf(stderr, "Require nthreads > 0\n");
    return NULL; /* TODO */
  }

  /* Find intersection of cpubind and membind */
  struct bitmask *nodemask = numa_allocate_nodemask(), *a, *b;
  a = numa_get_run_node_mask();
  b = numa_get_membind();
  bitmask_or(a, b, nodemask);
  numa_free_nodemask(a);
  numa_free_nodemask(b);

  struct bitmask *cpumask = numa_allocate_cpumask();
  int ithread = 0;
  int numnodes = numa_max_node() + 1;
  for (int node = 0; node != numnodes; ++node) {
    if (ithread == nthreads) break;
    if (numa_bitmask_isbitset(nodemask, node)) {
      int r = numa_node_to_cpus(node, cpumask);
      check(r >= 0, "numa_node_to_cpus failed");
      for (int cpu = 0; cpu < numa_bitmask_nbytes(cpumask) * 8; ++cpu) {
	if (ithread == nthreads) break;
	if (numa_bitmask_isbitset(cpumask, cpu)) {
	  plan->threadlocal[ithread].cpu = cpu;
	  plan->threadlocal[ithread].node = node;
	  ithread++;
	}
      }
    }
  }
  numa_free_nodemask(nodemask);
  numa_free_cpumask(cpumask);

  /* Figure out how work should be distributed. */
  /* First allocate information buffers */
  int ring_block_size = FFT_CHUNK_SIZE;
  size_t nm_bound = (mmax + 1) / nthreads + 1;
  size_t nring_bound = nrings / nthreads + ring_block_size;
  size_t nms[nthreads], nrings_list[nthreads];
  for (ithread = 0; ithread != nthreads; ++ithread) {
    nms[ithread] = nrings_list[ithread] = 0;
    fastsht_plan_threadlocal *td = &plan->threadlocal[ithread];
    td->buf_size = (sizeof(m_resource_t[nm_bound]) + sizeof(ring_pair_info_t[nring_bound]) +
                    sizeof(double*[mmax + 1]));
    char *buf = numa_alloc_onnode(td->buf_size, td->node);
    td->buf = buf;
    td->m_to_phase_ring = (double**)buf;
    buf += sizeof(double*[mmax + 1]);
    td->m_resources = (m_resource_t*)buf;
    buf += sizeof(m_resource_t[nm_bound]);
    td->ring_pairs = (ring_pair_info_t*)buf;
    buf += sizeof(ring_pair_info_t[nring_bound]);
  }
  /* Distribute m's */
  ithread = 0;
  for (size_t m = 0; m != mmax + 1; ++m) {
    ithread = (ithread + 1) % nthreads;
    int im = nms[ithread];
    plan->threadlocal[ithread].m_resources[im].m = m;
    nms[ithread]++;
  }
  /* Distribute rings */
  size_t mid_ring = plan->grid->mid_ring;
  size_t nrings_half = mid_ring + 1;
  ithread = 0;
  for (size_t iring = 0; iring != nrings_half; iring += ring_block_size) {
    size_t stop = imin(nrings_half, iring + ring_block_size);
    size_t rings_in_block = stop - iring;
    ithread = (ithread + 1) % nthreads;
    for (size_t j = 0; j != rings_in_block; ++j) {
      ring_pair_info_t *ri = &plan->threadlocal[ithread].ring_pairs[nrings_list[ithread] + j];
      ri->ring_number = iring + j;
      ri->phi0 = grid->phi0s[mid_ring + iring + j];
      ri->offset_top = grid->ring_offsets[mid_ring - ri->ring_number];
      ri->offset_bottom = grid->ring_offsets[mid_ring + ri->ring_number];
      ri->length = (grid->ring_offsets[mid_ring + + ri->ring_number + 1] - 
                    grid->ring_offsets[mid_ring + + ri->ring_number]);
    }
    nrings_list[ithread] += rings_in_block;
  }
  /* Copy nms and nrings to threads */
  for (int ithread = 0; ithread != nthreads; ++ithread) {
    plan->threadlocal[ithread].nm = nms[ithread];
    plan->threadlocal[ithread].nrings = nrings_list[ithread];
  }

  /* Load map of resources globally... */
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
  check(mmax == plan->resources->mmax, "Incompatible mmax");
  check(lmax == plan->resources->lmax, "Incompatible lmax");

  /* Spawn threads to do thread-local intialization:
     Copy over precomputed data, initialize butterfly & FFT plans */
  pthread_mutex_t mutex_fftw;
  pthread_mutex_init(&mutex_fftw, NULL);
  fastsht_run_in_threads(plan, &fastsht_create_plan_thread, &mutex_fftw);
  pthread_mutex_destroy(&mutex_fftw);

  /* Now that work has been allocated, set up m_to_phase_ring */
  double *m_to_phase_ring[mmax + 1];
  size_t work_stride = 2 * plan->nmaps * nrings_half;
  for (ithread = 0; ithread != nthreads; ++ithread) {
    fastsht_plan_threadlocal *lp = &plan->threadlocal[ithread];
    for (size_t im = 0; im != lp->nm; ++im) {
      m_to_phase_ring[lp->m_resources[im].m] = lp->work + (2 * im) * work_stride;
    }
  }
  /* And copy it to each thread */
  for (ithread = 0; ithread != nthreads; ++ithread) {
    memcpy(plan->threadlocal[ithread].m_to_phase_ring, m_to_phase_ring,
           sizeof(double[mmax + 1]));
  }
  return plan;
}

static void fastsht_create_plan_thread(fastsht_plan plan, int ithread, void *mutex_fftw_) {
  pthread_mutex_t *mutex_fftw = mutex_fftw_;
  fastsht_plan_threadlocal *localplan = &plan->threadlocal[ithread];
  size_t nm = localplan->nm;
  int nmaps = plan->nmaps;
  int node = localplan->node;
  size_t nrings_half = plan->grid->mid_ring + 1;
  unsigned flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE;

  /* Copy matrix data into threadlocal buffers */
  for (size_t im = 0; im != nm; ++im) {
    m_resource_t *localres = &localplan->m_resources[im];
    int m = localres->m;
    m_resource_t *fileres = &plan->resources->matrices[m];
    for (int odd = 0; odd != 2; ++odd) {
      localres->data[odd] = memalign(4096, fileres->len[odd]);
      checkf(localres->data[odd] != NULL, "No memory allocated of size %ld on node %d",
	     fileres->len[odd], node);
      localres->len[odd] = fileres->len[odd];
      memcpy(localres->data[odd], fileres->data[odd], fileres->len[odd]);
    }
  }

  /* Make Butterfly plans. Need to inspect precomputed data to figure
     out buffer sizes. */
  size_t k_max = 0, nblocks_max = 0;
  for (size_t im = 0; im != nm; ++im) {
    for (int odd = 0; odd != 2; ++odd) {
      bfm_matrix_data_info info;
      bfm_query_matrix_data(localplan->m_resources[im].data[odd], &info);
      k_max = zmax(k_max, info.k_max);
      nblocks_max = zmax(nblocks_max, info.nblocks_max);
    }
  }

  size_t legendre_work_size = fastsht_legendre_transform_sse_query_work(2 * nmaps);
  localplan->bfm = bfm_create_plan(k_max, nblocks_max, 2 * nmaps);
  localplan->legendre_transform_work = 
    (legendre_work_size == 0) ? NULL : memalign(4096, legendre_work_size);

  size_t nvecs = 2 * plan->nmaps;
  size_t nmats = 2 * nm;
  localplan->work = memalign(4096, sizeof(double[nmats * nvecs * nrings_half]));
  localplan->work_a_l = memalign(4096, sizeof(double[(nvecs * (plan->lmax + 1))]));

  /* For FFTs, we use inplace c2r/r2c. This means that the buffer per
     map must be as long as the longest ring + one complex
     coefficient: len(complex) = len(real) // 2 + 1. We allocate
     work space for FFT_CHUNK_SIZE rings in both the northern and southern
     hemisphere. */
  localplan->work_fft = memalign(4096, sizeof(double[2 * FFT_CHUNK_SIZE * nmaps * 
                                                     (4 * plan->Nside + 2)]));

  /* Make FFT plans. FFTW is *not* thread-safe in the fftw_plan_X functions,
     but we *do* want to run it in each local thread, to properly benchmark
     using local memory. So, we serialize access to FFTW. Note that the
     fftw_execute_... functions *are* thread-safe (as long as used with
     different plans).
  */
  pthread_mutex_lock(mutex_fftw);
  for (int i = 0; i != localplan->nrings; ++i) {
    ring_pair_info_t *ri = &localplan->ring_pairs[i];
    int ringlen = ri->length;
    ri->fft_plan = fftw_plan_many_dft_c2r(1, &ringlen, nmaps,
                                          (fftw_complex*)localplan->work_fft, NULL, nmaps, 1,
                                          localplan->work_fft, NULL, nmaps, 1,
                                          flags);
  }
  pthread_mutex_unlock(mutex_fftw);
}


void fastsht_destroy_plan(fastsht_plan plan) {
  int iring;

  /* Cleanup for all threads. Do not move to per-thread without a mutex,
     FFTW3 destructor access must be serialized!
   */
  for (int ithread = 0; ithread != plan->nthreads; ++ithread) {
    fastsht_plan_threadlocal *lp = &plan->threadlocal[ithread];
    for (size_t iring = 0; iring != lp->nrings; ++iring) {
      fftw_destroy_plan(lp->ring_pairs[iring].fft_plan);
    }
    for (size_t im = 0; im != lp->nm; ++im) {
      for (int odd = 0; odd != 2; ++odd) {
        free(lp->m_resources[im].data[odd]);
      }
    }
    bfm_destroy_plan(lp->bfm);
    free(lp->work);
    free(lp->work_a_l);
    free(lp->work_fft);
    /* Free buffer for ring_pairs, m_resources, and m_to_phase_ring,
       allocated by numa_alloc. */
    numa_free(lp->buf, lp->buf_size);
  }

  fastsht_free_grid_info(plan->grid);
  fastsht_release_resource(plan->resources);
  if (plan->did_allocate_resources) free(plan->resources);
  free(plan->threadlocal);
  free(plan);
}

int64_t fastsht_get_legendre_flops(fastsht_plan plan, int m, int odd) {
  int64_t N, nvecs;
  bfm_matrix_data_info info;
  bfm_query_matrix_data(plan->resources->matrices[m].data[odd],
                        &info);
  N = info.element_count;
  nvecs = 2;
  N *= nvecs;
  return N * 2; /* count mul and add seperately */
}


static void legendre_transforms_thread(fastsht_plan plan, int ithread, void *ctx) {
  fastsht_plan_threadlocal *localplan = &plan->threadlocal[ithread];
  size_t nrings_half = plan->grid->mid_ring + 1;
  double *work = localplan->work;
  int nvecs = 2 * plan->nmaps;
  size_t lmax = plan->lmax;
  size_t nm = localplan->nm;
    
  /* Compute into the buffer */
  for (int im = 0; im != nm; ++im) {
    m_resource_t *m_resource = &localplan->m_resources[im];
    size_t m = m_resource->m;
    
    for (int odd = 0; odd < 2; ++odd) {
      double *target = work + (2 * im + odd) * nvecs * nrings_half;
      fastsht_perform_matmul(plan, ithread, m, odd, nrings_half, target,
                             localplan->legendre_transform_work,
                             localplan->work_a_l);
    }
  }
}

void fastsht_perform_legendre_transforms(fastsht_plan plan) {
  fastsht_run_in_threads(plan, &legendre_transforms_thread, NULL);
}

void fastsht_execute(fastsht_plan plan) {
  fastsht_perform_legendre_transforms(plan);
  /* Backward FFTs from plan->work to plan->output */
  fastsht_perform_backward_ffts(plan);
}


typedef struct {
  double *input;
  char *work;
} transpose_apply_ctx_t;

void pull_a_through_legendre_block(double *buf, size_t start, size_t stop,
                                   size_t nvecs, char *payload, size_t payload_len,
                                   void *ctx_) {
  transpose_apply_ctx_t *ctx = ctx_;
  double *input = ctx->input;
  skip128(&payload);
  size_t row_start = read_int64(&payload);
  size_t row_stop = read_int64(&payload);
  size_t nk = row_stop - row_start;
  input += row_start * nvecs;
  if (nk <= 4 || start == stop) {
    double *A = read_aligned_array_d(&payload, (stop - start) * nk);
    dgemm_ccc(input, A, buf,
              nvecs, stop - start, nk, 0.0);
  } else {
    size_t nstrips = read_int64(&payload);
    double *auxdata = read_aligned_array_d(&payload, 3 * (nk - 2));
    size_t rstart, cstart, cstop;
    cstart = 0;
    for (size_t i = 0; i != nstrips; ++i) {
      rstart = read_int64(&payload);
      cstop = read_int64(&payload);
      size_t nx_strip = cstop - cstart, nk_strip = nk - rstart;
      if (nk - rstart <= 4) {
        double *A = read_aligned_array_d(&payload, nk_strip * nx_strip);
        dgemm_ccc(input + rstart * nvecs,
                  A,
                  buf + cstart * nvecs,
                  nvecs,
                  nx_strip,
                  nk_strip, 0.0);
      } else {
        double *x_squared = read_aligned_array_d(&payload, nx_strip);
        double *P0 = read_aligned_array_d(&payload, nx_strip);
        double *P1 = read_aligned_array_d(&payload, nx_strip);
        fastsht_legendre_transform_sse(nx_strip, nk_strip, nvecs,
                                       input + rstart * nvecs,
                                       buf + cstart * nvecs,
                                       x_squared,
                                       auxdata + 3 * rstart,
                                       P0, P1,
                                       ctx->work);
      }
      cstart = cstop;
    }
  }
}

void fastsht_perform_matmul(fastsht_plan plan, int ithread, bfm_index_t m, int odd, size_t ncols,
                            double *output, char *legendre_transform_work,
                            double *work_a_l) {
  bfm_index_t nrows, l, lmax = plan->lmax, j;
  size_t nvecs = 2 * plan->nmaps;
  double *input_m = plan->input + nvecs * m * (2 * lmax - m + 3) / 2;
  char *matrix_data = plan->resources->matrices[m].data[odd];
  nrows = 0;
  for (l = m + odd; l <= lmax; l += 2) {
    for (j = 0; j != nvecs; ++j) {
      work_a_l[nrows * nvecs + j] = input_m[(l - m) * nvecs + j];
    }
    ++nrows;
  }
  transpose_apply_ctx_t ctx = { work_a_l, legendre_transform_work };
  int ret = bfm_transpose_apply_d(plan->threadlocal[ithread].bfm,
                                  matrix_data,
                                  pull_a_through_legendre_block,
                                  output,
                                  ncols * nvecs,
                                  &ctx);
  checkf(ret == 0, "bfm_transpose_apply_d retcode %d", ret);
}

void fastsht_assemble_rings(fastsht_plan plan,
                            int ms_len, int *ms,
                            double complex **q_list) {
  /* The routine is made for being called by a team of threads;
     this wrapper is here so that one can call the worker directly
     if one is already within an parallel section. */
  //  #pragma omp parallel
  {
    fastsht_assemble_rings_omp_worker(plan, ms_len, ms, q_list);  
  }
}

void fastsht_assemble_rings_omp_worker(fastsht_plan plan,
                                       int ms_len, int *ms,
                                       double complex **q_list) {
  /* NOTE: This function can be called from within an OpenMP parallel section. */
  bfm_index_t mmax, nrings, mid_ring;
  double *output;
  int nmaps = plan->nmaps;
  bfm_index_t *ring_offsets = plan->grid->ring_offsets;
  double *phi0s = plan->grid->phi0s;


  int iring, i_m, m, imap;
  double cos_phi, sin_phi;
  double complex *q_even, *q_odd;
  size_t n, idx_top, idx_bottom;

  assert(plan->grid->has_equator);
  mmax = plan->mmax;
  mid_ring = plan->grid->mid_ring;
  nrings = plan->grid->nrings;
  output = plan->output;

  /* We take the even and odd parts for the given m, merge then, and
     distribute it to plan->output. Each ring in output is kept in the
     FFTW half-complex format:
    
         r_0, r_1, r_2, ..., r_{n//2}, i_{(n+1)//2 - 1}, ..., i_2, i_1

     For some rings, mmax < n/2, while for others, mmax > n/2. In the
     former case one must pad with zeros, whereas in the other case,
     the signal must be wrapped around. Let q_m be the phases, and let
     q_m = 0 for |m| > mmax. Since e^{ik} is periodic, we have

         \sum_{m=-\infty}^\infty q_m e^{i * k * 2 * \pi * m / n} =
         
         \sum_{j=0}^n
               [ \sum_{t = -\infty}^\infty q_{t * n + j} ]
               e^{i * k * 2 * \pi * j / n}

     So the proper coefficients in the Fourier transform is to sum up
     q_m for all m divisible by n. However, we need to treat q_m both
     for negative and positive m (with q_{-m} = q_m^*).  Since we only
     store one half of the coefficients in the end this complicates
     the logic somewhat, with an extra case for when m%n > n/2.

     We also process rings in pairs in order to exploit symmetry. For
     an equator ring we simply rely on all(g_m_odd == 0), no special
     treatment necesarry.
  */

  //  #pragma omp for schedule(dynamic, 4)
  for (iring = 0; iring < mid_ring + 1; ++iring) {
    n = ring_offsets[mid_ring - iring + 1] - ring_offsets[mid_ring - iring];

    for (i_m = 0; i_m < ms_len; ++i_m) {
      m = ms[i_m];
      q_even = q_list[2 * i_m];
      q_odd = q_list[2 * i_m + 1];
      if (q_even == NULL) continue;
        
      cos_phi = cos(m * phi0s[mid_ring + iring]);
      sin_phi = sin(m * phi0s[mid_ring + iring]);

      for (imap = 0; imap != nmaps; ++imap) {
        int j1, j2;
        double complex q_top_1, q_top_2, q_bottom_1, q_bottom_2;
        idx_top = nmaps * ring_offsets[mid_ring - iring];
        idx_bottom = nmaps * ring_offsets[mid_ring + iring];
        
        /* Merge even/odd, changing the sign of odd part on bottom half */
        q_top_1 = q_even[iring * nmaps + imap] + q_odd[iring * nmaps + imap];
        q_bottom_1 = q_even[iring * nmaps + imap] - q_odd[iring * nmaps + imap];

        /* Phase-shift the coefficients */
        q_top_1 *= cos_phi + I * sin_phi;
        q_bottom_1 *= cos_phi + I * sin_phi;

        q_top_2 = conj(q_top_1);
        q_bottom_2 = conj(q_bottom_1);

        j1 = m % n;
        j2 = imod_divisorsign(n - m, n);

        if (j1 <= n / 2) {
          output[idx_top + j1 * nmaps + imap] += creal(q_top_1);
          if (j1 > 0) output[idx_top + (n - j1) * nmaps + imap] += cimag(q_top_1);
          if (iring > 0) {
            output[idx_bottom + j1 * nmaps + imap] += creal(q_bottom_1);
            if (j1 > 0) output[idx_bottom + (n - j1) * nmaps + imap] += cimag(q_bottom_1);
          }
        }
        if (m != 0 && j2 <= n / 2) {
          output[idx_top + j2 * nmaps + imap] += creal(q_top_2);
          if (j2 > 0) output[idx_top + (n - j2) * nmaps + imap] += cimag(q_top_2);
          if (iring > 0) {
            output[idx_bottom + j2 * nmaps + imap] += creal(q_bottom_2);
            if (j2 > 0) output[idx_bottom + (n - j2) * nmaps + imap] += cimag(q_bottom_2);
          }
        }

      }
    }
  }
}

void fastsht_cossin(double *out, size_t n, double x0, double delta) {
  /* Computes cos(x0 + i * delta) and sin(x0 + i * delta) */
  double a = sin(.5 * delta);
  a = 2.0 * a * a;
  m128d alpha = (m128d){ -a, -a };
  double b = sin(delta);
  m128d beta = (m128d){ b, -b };
  m128d y = (m128d){ cos(x0), sin(x0) };
  _mm_store_pd(out, y);
  out += 2;
  n--;
  while (n--) {
    m128d t = _mm_mul_pd(alpha, y);
    m128d u = _mm_mul_pd(beta, y);
    u = _mm_shuffle_pd(u, u, _MM_SHUFFLE2(0, 1)); /* flip elements of u*/
    u = _mm_add_pd(t, u);
    y = _mm_add_pd(y, u);
    _mm_store_pd(out, y);
    out += 2;
  }
}

/*static void fetch_and_wrap_ring(fastsht_plan plan, ring_pair_info_t info) {
  size_t mmax = plan->mmax;
  int *m_to_phase_ring = plan->m_to_phase_ring;

  for (size_t m = 0; m != mmax + 1; ++m) {
    m_to_thread[m]
  }
} 
*/

static INLINE m128d complex_mul_pd(m128d a_b, m128d c_d) {
  /* Multiply (a + I*b) and (c + I*d) */
  m128d a_a = _mm_unpacklo_pd(a_b, a_b);
  m128d b_b = _mm_unpackhi_pd(a_b, a_b);
  m128d minusb_b = _mm_mul_pd(b_b, (m128d){-1.0, 1.0});
  m128d d_c = _mm_shuffle_pd(c_d, c_d, _MM_SHUFFLE2(0, 1));

  return _mm_add_pd(_mm_mul_pd(a_a, c_d), _mm_mul_pd(minusb_b, d_c));
}

static INLINE void inplace_add_pd(double *px, m128d r) {
  m128d x = _mm_load_pd(px);
  _mm_store_pd(px, _mm_add_pd(x, r));
}

static void _printreg(char *msg, m128d r) {
  double *pd = (double*)&r;
  printf("%s = [%.2f %.2f]\n", msg, pd[0], pd[1]);
}
#define printreg(x) _printreg(#x, x)

static void perform_backward_ffts_thread(fastsht_plan plan, int ithread, void *ctx) {
  /*
    a) Phase shift all coefficients according to their phi0
    b) Zero padding and wrap-around coefficients (contribution from +/- m)
    c) Fourier transforms
  */
  int nmaps = plan->nmaps;
  size_t mid_ring = plan->grid->mid_ring;
  size_t nrings_half = mid_ring + 1;
  int mmax = plan->mmax;
  size_t npix = plan->grid->npix;
  size_t n;
  int m;
  int imap, iring, ipix;
  double *output = plan->output;
  bfm_index_t *ring_offsets = plan->grid->ring_offsets;
  double *cos_and_sin = memalign(16, sizeof(double[2 * plan->mmax]));
  double phi0;
  double *map, *ring;

  fastsht_plan_threadlocal *threadplan = &plan->threadlocal[ithread];
  ring_pair_info_t *ring_pairs = threadplan->ring_pairs;

  double **m_to_phase_ring = threadplan->m_to_phase_ring;

  size_t idx, offset, length;

  double *work = threadplan->work_fft;
  size_t work_stride = nmaps * (4 * plan->Nside + 2);
  
  m128d conjugating_const = (m128d){ 1.0, -1.0 };

  assert((mid_ring + 1) % FFT_CHUNK_SIZE == 0);

  int s;
  for (size_t chunk_start = 0;
       chunk_start < threadplan->nrings;
       chunk_start += FFT_CHUNK_SIZE) {

    memset(work, 0, sizeof(double[2 * FFT_CHUNK_SIZE * work_stride]));
    for (size_t m = 0; m != mmax + 1; ++m) {
      double *q_m_even_array = m_to_phase_ring[m];
      double *q_m_odd_array = q_m_even_array + 2 * nmaps * nrings_half;

      for (size_t j = 0; j != FFT_CHUNK_SIZE; ++j) {
        double *work_top = work + 2 * j * work_stride;
        double *work_bottom = work + (2 * j + 1) * work_stride;
        size_t iring = chunk_start + j;
        ring_pair_info_t *ri = &ring_pairs[iring];

        double cos_phi = cos(m * ri->phi0);
        double sin_phi = sin(m * ri->phi0);
        m128d phase_shift = (m128d){cos_phi, sin_phi};

        int ringlen = ring_pairs[iring].length;
        int j1, j2;
        j1 = m % ringlen;
        j2 = imod_divisorsign(ringlen - m, ringlen);

        for (size_t k = 0; k != nmaps; ++k) {
          m128d q_even, q_odd, q_top_1, q_bottom_1, q_top_2, q_bottom_2;
          q_even = _mm_load_pd(q_m_even_array + 2 * (iring * nmaps + k));
          q_odd = _mm_load_pd(q_m_odd_array + 2 * (iring * nmaps + k));
          q_top_1 = _mm_add_pd(q_even, q_odd);
          q_bottom_1 = _mm_sub_pd(q_even, q_odd);

          q_top_1 = complex_mul_pd(q_top_1, phase_shift);
          q_bottom_1 = complex_mul_pd(q_bottom_1, phase_shift);

          q_top_2 = _mm_mul_pd(q_top_1, conjugating_const);
          q_bottom_2 = _mm_mul_pd(q_bottom_1, conjugating_const);

          if (j1 <= ringlen / 2) {
            inplace_add_pd(work_top + 2 * (j1 * nmaps + k), q_top_1);
            inplace_add_pd(work_bottom + 2 * (j1 * nmaps + k), q_bottom_1);
          }
          if (m != 0 && j2 <= ringlen / 2) {
            inplace_add_pd(work_top + 2 * (j2 * nmaps + k), q_top_2);
            inplace_add_pd(work_bottom + 2 * (j2 * nmaps + k), q_bottom_2);
          }
        }
      }
    }
    
    for (size_t j = 0; j != FFT_CHUNK_SIZE; ++j) {
      size_t iring = chunk_start + j;

      ring_pair_info_t *ri = &ring_pairs[chunk_start + j];
      printf("thread %d RING %d\n", ithread, ri->ring_number);
      double *work_top = work + 2 * j * work_stride;
      double *work_bottom = work + (2 * j + 1) * work_stride;
      fftw_plan fft_plan = ring_pairs[iring].fft_plan;
      fftw_execute_dft_c2r(fft_plan, (fftw_complex*)work_top, work_top);
      memcpy(output + nmaps * ri->offset_top, work_top,
             sizeof(double[ri->length * nmaps]));
      if (ri->offset_bottom != ri->offset_top) {
        fftw_execute_dft_c2r(fft_plan, (fftw_complex*)work_bottom, work_bottom);
        memcpy(output + nmaps * ri->offset_bottom, work_bottom,
               sizeof(double[ri->length * nmaps]));
      }
    }
  }
}

void fastsht_perform_backward_ffts(fastsht_plan plan) {
  fastsht_run_in_threads(plan, &perform_backward_ffts_thread, NULL);
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

void fastsht_disable_phase_shifting(fastsht_plan plan) {
  int iring;
  /* Used for debug purposes.
     TODO: If the grid starts to get shared across plans we need to think
     of something else here. */
  for (iring = 0; iring != plan->grid->nrings; ++iring) {
    plan->grid->phi0s[iring] = 0;
  }
}
