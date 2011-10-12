/*
Low-level API, primarily exposed for unit testing: It
is wrapped in Cython and used for unit-tests, but it should
probably

The API is unstable and may change at any point.
*/

#ifndef _FASTSHT_PRIVATE_H_
#define _FASTSHT_PRIVATE_H_

#include "butterfly.h"
#include "fastsht.h"
#include "complex.h"
#include <fftw3.h>

/*
The precomputed data, per m. Index to data/len is even=0, odd=1
*/
typedef struct {
  char *data[2];
  size_t len[2];
  size_t m;
} m_resource_t;

typedef struct {
  char *mmapped_buffer;
  size_t mmap_len;

  m_resource_t *matrices;  /* indexed by m */
  int lmax, mmax;
  int refcount;
} precomputation_t;


typedef struct {
  double phi0;
  /* ring_number is with respect to equator; it is implied that both
     negative and positive ring belongs to thread */
  size_t ring_number, offset_top, offset_bottom, length;
  fftw_plan fft_plan;
} ring_pair_info_t;

typedef struct {
  /* To phase shift, we multiply with

         e^(i m phi_0) = cos(m phi_0) + i sin(m phi_0)

     for each ring. This is computed by
     
         cos(x + phi_0) = cos(x) - (alpha * cos(x) + beta * sin(x))
         sin(phi_0) = sin(x) - (alpha * sin(x) - beta * cos(x))

     and we precompute ...
   */
  double *phi0s;
  bfm_index_t *ring_offsets;
  bfm_index_t nrings, mid_ring;
  bfm_index_t npix;
  int has_equator;
} fastsht_grid_info;

typedef struct {
  bfm_plan *bfm;
  char *legendre_transform_work;
  double *work_a_l;  
} fastsht_legendre_worker_t;

typedef struct {
  size_t buf_size;
  ring_pair_info_t *ring_pairs;
  double *work_fft;
  size_t nrings;
  int threadnum_on_node;
  int cpu_id;
  sem_t cpu_lock;
  fastsht_legendre_worker_t *legendre_workers;
} fastsht_cpu_plan_t;


typedef struct {
  double *work_q;
  size_t size_allocated;
  m_resource_t *m_resources;
  /* Set up a map of m -> phase ring. This is copied to all threads
     until it can be proven that sharing it for read-only access
     doesn't hurt... */
  size_t nm, im;
  sem_t memory_bus_semaphore;
  pthread_mutex_t queue_lock;
  size_t k_max, nblocks_max;
  fastsht_cpu_plan_t *cpu_plans;
  int ncpus;
  int node_id;
} fastsht_node_plan_t;


struct _fastsht_plan {
  double *output, *input;
  fastsht_grid_info *grid;
  fftw_plan *fft_plans;
  precomputation_t *resources;
  fastsht_node_plan_t *node_plans[8];
  double **m_to_phase_ring;

  size_t work_q_stride;

  int type;
  int lmax, mmax;
  int nmaps;
  int nnodes, ncpus_total;

  int did_allocate_resources;
  int Nside;
  unsigned fftw_flags;

  struct {
    double legendre_transform_start, legendre_transform_done, fft_done;
  } times;
};

void fastsht_perform_matmul(fastsht_plan plan, bfm_plan *bfm, char *matrix_data,
                            bfm_index_t m, int odd, size_t ncols, double *output,
                            char *legendre_transform_work, double *work_a_l);
void fastsht_perform_interpolation(fastsht_plan plan, bfm_index_t m, int odd);
void fastsht_perform_legendre_transforms(fastsht_plan plan);

void fastsht_perform_backward_ffts(fastsht_plan plan);

fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside);
void fastsht_free_grid_info(fastsht_grid_info *info);
void fastsht_disable_phase_shifting(fastsht_plan plan);

void fastsht_execute_out_of_core(fastsht_plan plan,
                                 double *out_compute_time,
                                 double *out_load_time);

#endif
