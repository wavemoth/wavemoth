/*
C program to benchmark spherical harmonic transforms
*/

#include "fastsht.h"
#include "fastsht_private.h"
#include "fastsht_error.h"
#include "blas.h"

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <fftw3.h>
#ifdef HAS_PPROF
#include <google/profiler.h>
#endif

#include <omp.h>

#include <stddef.h>
#include <psht.h>
#include <psht_geomhelpers.h>


#define PROFILE_TIME 10.0

int Nside, lmax;
char *sht_resourcefile;


int N_threads;

/*
Utils
*/
static double walltime() {
  struct timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  return tv.tv_sec + 1e-9 * tv.tv_nsec;
}

static double *zeros(size_t n) {
  double *buf;
  buf = memalign(16, n * sizeof(double));
  memset(buf, 0, n * sizeof(double));
  return buf;
}

static void printtime(char* msg, int n, double time) {
  char *units;
  time /= n;
  if (time < 1e-06) {
    units = "ns";
    time *= 1e9;
  } else if (time < 1e-03) {
    units = "us";
    time *= 1e6;
  } else if (time < 1) {
    units = "ms";
    time *= 1e3;
  } else {
    units = "s";
  }
  printf("%s%d times, avg: %.1f %s\n", msg, n, time, units);
}


/*
Benchmark memory throughput
*/
double *mem_buf;
size_t N_mem = 200 * 1024 * 1024 / sizeof(double);

void setup_memory_benchmark() {
  /* zero buffer, so that we won't hit denormal numbers in reading loop */
  mem_buf = zeros(N_threads * N_mem);
}

void finish_memory_benchmark(double dt) {
  printf("  Bandwidth: %.3f GB/s = %.3f giga-doubles/s\n ",
         N_threads * N_mem * sizeof(double) / dt / (1024 * 1024 * 1024),
         N_threads * N_mem / dt / (1024 * 1024 * 1024));
  free(mem_buf);
}

void execute_memory_benchmark(int threadnum) {
  /* Simply accumulate the buffer in a double -- we're going to be very
     IO-bound so we can basically ignore the addition put in in order to
     avoid optimizing things out.   */
  int j;
  size_t n = N_mem;
  double acc = 0;
  double *buf = mem_buf;
  for (j = 0; j != n; j += 2) {
    acc += buf[threadnum * n + j];
  }
  if (acc != 0.0) printf("dummy\n");
}

/*
FLOPS benchmark
*/
size_t N_dgemm = 200;
double **A, **B, **Y;

void execute_dgemm(int threadnum) {
  dgemm_rrr(A[threadnum], B[threadnum], Y[threadnum],
            N_dgemm, N_dgemm, N_dgemm, 1.0);
}

void setup_dgemm() {
  int i;
  A = malloc(sizeof(void*) * N_threads);
  B = malloc(sizeof(void*) * N_threads);
  Y = malloc(sizeof(void*) * N_threads);
  for (i = 0; i != N_threads; ++i) {
    A[i] = zeros(N_dgemm * N_dgemm);
    B[i] = zeros(N_dgemm * N_dgemm);
    Y[i] = zeros(N_dgemm * N_dgemm);
  }
}

void finish_dgemm(double dt) {
  int i;
  size_t flops = N_threads * N_dgemm * N_dgemm * N_dgemm * 2;
  printf("  Speed: %.3f GFLOPS\n", flops / dt / 1e9);
  for (i = 0; i != N_threads; ++i) {
    free(A[i]);
    free(B[i]);
    free(Y[i]);
  }
  free(A);
  free(B);
  free(Y);
}



/*
Butterfly SHT benchmark
*/

double *sht_input, *sht_output, *sht_work;
fastsht_plan sht_plan;
int sht_nmaps;
int sht_m_stride = 1;
size_t *sht_mstart;

void setup_sht_buffers() {
  /* Allocate input, output and work -- in m-major mode with some m's dropped */
  int m;
  size_t pos;
  pos = 0;
  sht_mstart = malloc(sizeof(size_t[lmax + 1]));
  for (m = 0; m < lmax + 1; m += 1) {
    if (m % sht_m_stride == 0) {
      sht_mstart[m] = pos - m;
      pos += m + 1;      
    } else {
      sht_mstart[m] = 0x7FFFFFFF;
    }
  }
  sht_input = zeros(pos * 2 * sht_nmaps);
  sht_output = zeros(12 * Nside * Nside * sht_nmaps);
  sht_work = zeros((lmax + 1) * (4 * Nside - 1) * 2 * sht_nmaps);
}

void free_sht_buffers() {
  free(sht_mstart);
  free(sht_input);
  free(sht_output);
  free(sht_work);
}

void execute_sht(int threadnum) {
  double t_compute, t_load;
  /*  fastsht_execute_out_of_core(sht_plan, &t_compute, &t_load);
      printf("compute:load: %f %f\n", t_compute, t_load);*/
  fastsht_execute(sht_plan);
}

void execute_legendre(int threadnum) {
  int m, odd;
  for (m = 0; m != sht_plan->mmax + 1; ++m) {
    for (odd = 0; odd != 2; ++odd) {
      fastsht_perform_matmul(sht_plan, m, odd);
    }
  }
}

void finish_legendre(double dt) {
  int64_t flops = 0;
  int m, odd;
  for (m = 0; m != sht_plan->mmax + 1; ++m) {
    for (odd = 0; odd != 2; ++odd) {
      flops += fastsht_get_legendre_flops(sht_plan, m, odd) * 2;
    }
  }
  printf("  Speed: %.3f GFLOPS\n", flops / dt / 1e9);
}

void _setup_sht(int nmaps) {
  FILE *fd;
  printf("  Initializing (incl. FFTW)\n");
  sht_nmaps = nmaps;
  /* Import FFTW plan if it exists */
  fd = fopen("fftw.wisdom", "r");
  if (fd != NULL) {
    fftw_import_wisdom_from_file(fd);
    fclose(fd);
  }

  sht_input = zeros((lmax + 1) * (lmax + 1) * 2 * nmaps);
  sht_output = zeros(12 * Nside * Nside * nmaps);
  sht_work = zeros((lmax + 1) * (4 * Nside - 1) * 2 * nmaps);
  sht_plan = fastsht_plan_to_healpix(Nside, lmax, lmax, nmaps, sht_input,
                                     sht_output, sht_work, FASTSHT_MMAJOR,
                                     sht_resourcefile);

  /* Export FFTW wisdom generated during planning */
  fd = fopen("fftw.wisdom", "w");
  if (fd != NULL) {
    fftw_export_wisdom_to_file(fd);
    fclose(fd);
  }
}

void setup_sht1() {
  _setup_sht(1);
}

void setup_sht2() {
  _setup_sht(2);
}

void setup_sht10() {
  _setup_sht(10);
}

void setup_sht20() {
  _setup_sht(20);
}

void finish_sht(double dt) {
  fastsht_destroy_plan(sht_plan);
  free_sht_buffers();
}


/*
PSHT
*/

/*
Inserted our own hack in PSHT...
*/
//void _psht_set_m_stride(int m_stride);

psht_alm_info *benchpsht_alm_info;
psht_geom_info *benchpsht_geom_info;
pshtd_joblist *benchpsht_joblist;

ptrdiff_t lm_to_idx_mmajor(ptrdiff_t l, ptrdiff_t m) {
  return m * (2 * lmax - m + 3) / 2 + (l - m);
}

void _setup_psht(int nmaps) {
  int m;
  int marr[lmax + 1];
  int j;
  ptrdiff_t npix = 12 * Nside * Nside;
  ptrdiff_t stride;
  for (m = 0; m < lmax + 1; m += 1) {
    marr[m] = m;
  }
  check(Nside >= 0, "Invalid Nside");
  sht_nmaps = nmaps;
  /* Setup m-major alm info */
  /* Input is strided/interleaved in m-major triangular order;
     output has one map after the other (non-interleaved). We support
     skipping some m's to speed benchmarks up.
  */
  setup_sht_buffers();
  //  _psht_set_m_stride(sht_m_stride);
  stride = nmaps;
  psht_make_general_alm_info(lmax, lmax + 1, stride, marr, sht_mstart, &benchpsht_alm_info);
  /* The rest is standard */
  psht_make_healpix_geom_info(Nside, 1, &benchpsht_geom_info);
  pshtd_make_joblist(&benchpsht_joblist);

  for (j = 0; j != nmaps; ++j) {
    pshtd_add_job_alm2map(benchpsht_joblist, (pshtd_cmplx*)sht_input + j,
                          sht_output + j * npix, 0);
  }
}

void setup_psht1() {
  _setup_psht(1);
}

void setup_psht2() {
  _setup_psht(2);
}

void setup_psht10() {
  _setup_psht(10);
}

void setup_psht20() {
  _setup_psht(20);
}

void finish_psht() {
  psht_destroy_alm_info(benchpsht_alm_info);
  psht_destroy_geom_info(benchpsht_geom_info);
  pshtd_clear_joblist(benchpsht_joblist);
  pshtd_destroy_joblist(benchpsht_joblist);
  free_sht_buffers();
}

void execute_psht(int threadnum) {
  pshtd_execute_jobs(benchpsht_joblist, benchpsht_geom_info, benchpsht_alm_info);
}



/*
Main
*/

typedef struct {
  char *name;
  void (*setup)(void); 
  void (*execute)(int threadnum);
  void (*finish)(double dt);
} benchmark_t;

benchmark_t benchmarks[] = {
  {"sht1", setup_sht1, execute_sht, finish_sht},
  {"sht2", setup_sht2, execute_sht, finish_sht},
  {"sht10", setup_sht10, execute_sht, finish_sht},
  {"sht20", setup_sht20, execute_sht, finish_sht},
  {"psht1", setup_psht1, execute_psht, finish_psht},
  {"psht2", setup_psht2, execute_psht, finish_psht},
  {"psht10", setup_psht10, execute_psht, finish_psht},
  {"psht20", setup_psht20, execute_psht, finish_psht},
  {"legendre", setup_sht1, execute_legendre, finish_legendre},
  {"memread", setup_memory_benchmark, execute_memory_benchmark, finish_memory_benchmark},
  {"dgemm", setup_dgemm, execute_dgemm, finish_dgemm},
  {"dgemm-single", setup_dgemm, execute_dgemm, finish_dgemm},
  {NULL, NULL, NULL}
};


char *usage[] = {
  "usage: shbench [-p] [-r resourcefile] [benchmarknames...]",
};

#define MAXPATH 2048

int main(int argc, char *argv[]) {
  double t0, t1;
  int n, i, j, should_run;
  benchmark_t *pbench;
  int nthreads;
  #ifdef HAS_PPROF
  char profilefile[MAXPATH];
  #endif
  char *resourcename;

  int c;

  int should_profile = 0;
  int got_threads;


  nthreads = omp_get_max_threads();
  sht_resourcefile = NULL;
  Nside = -1;

  opterr = 0;
  while ((c = getopt (argc, argv, "pr:N:j:")) != -1) {
    switch (c) {
    case 'p':
#ifndef HAS_PPROF
      fprintf(stderr, "ERROR: Profiling requested but not compiled with Google perftools!\n");
      return -1;
#else
      should_profile = 1;
      break;
#endif
    case 'r':
      sht_resourcefile = optarg;
      break;
    case 'N':
      Nside = atoi(optarg);
      break;
    case 'j':
      N_threads = atoi(optarg);
      break;
    }
  }
  argv += (optind - 1);
  argc -= (optind - 1);

  /* Resource configuration */
  fastsht_configure("/home/dagss/code/spherew/resources");
  if (sht_resourcefile != NULL) {
    fastsht_query_resourcefile(sht_resourcefile, &Nside, &lmax);
  } else {
    lmax = 2 * Nside;
  }

  omp_set_dynamic(0);
  omp_set_num_threads(N_threads);
#pragma omp parallel shared(got_threads)
  {
    got_threads = omp_get_num_threads();
  }
  if (got_threads < N_threads) {
    fprintf(stderr, "WARNING: Threads available less than requested.\n");
  }
  fprintf(stderr, "Using %d threads\n", got_threads);

  pbench = benchmarks;
  while (pbench->execute != NULL) {
    should_run = 0;
    if (argc > 1) {
      for (j = 1; j != argc; ++j) {
        if (strcmp(argv[j], pbench->name) == 0) should_run = 1;
      }
      if (!should_run) {
        pbench++;
        continue;
      }
    }
    printf("%s:\n", pbench->name);
    sht_nmaps = -1;
    if (pbench->setup != NULL) pbench->setup();
    #ifdef HAS_PPROF
    if (should_profile) {
      snprintf(profilefile, MAXPATH, "profiles/%s.prof", pbench->name);
      profilefile[MAXPATH - 1] = '\0';
      ProfilerStart(profilefile);
    }
    #endif
    t0 = walltime();
    n = 0;
    do {
      pbench->execute(i);
      t1 = walltime();
      n++;
    } while (t1 - t0 < PROFILE_TIME);
    #ifdef HAS_PPROF
    ProfilerStop();
    #endif
    printtime("  ", n, t1 - t0);
    if (sht_nmaps >= 1) {
      printtime("  Per map: ", n * sht_nmaps, t1 - t0);
    }
    if (pbench->finish) pbench->finish((t1 - t0) / n);
    pbench++;
  }

}
