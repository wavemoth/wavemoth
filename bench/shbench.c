/*
C program to benchmark spherical harmonic transforms
*/

#include "fastsht.h"
#include "blas.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <fftw3.h>
#ifdef USE_PPROF
#include <google/profiler.h>
#endif

#include <omp.h>

#define Nside 512
#define lmax 2 * Nside
#define mmax lmax
#define PROFILE_TIME 3.0

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
  printf("  Bandwidth: %.3f GB/s\n", N_threads * N_mem * sizeof(double) / dt / (1024 * 1024 * 1024));
  free(mem_buf);
}

void execute_memory_benchmark(int threadnum) {
  /* Simply accumulate the buffer in a double -- we're going to be very
     IO-bound so we can basically ignore the addition put in in order to
     avoid optimizing things out.   */
  int i, j;
  size_t n = N_mem;
  double acc = 0;
  double *buf = mem_buf;
#pragma omp parallel for schedule(static, 1) private(j) \
    firstprivate(threadnum) shared(n) reduction(+:acc)
  for (i = 0; i < N_threads; ++i) {
    for (j = 0; j != n; j += 2) {
      acc += buf[threadnum * n + j];
    }
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

void execute_sht(int threadnum) {
  fastsht_execute(sht_plan);
}

void setup_sht() {
  FILE *fd;
  printf("  Initializing (incl. FFTW)\n");

  /* Import FFTW plan if it exists */
  fd = fopen("fftw.wisdom", "r");
  if (fd != NULL) {
    fftw_import_wisdom_from_file(fd);
    fclose(fd);
  }

  fastsht_configure("/home/dagss/code/spherew/resources");
  sht_input = zeros((lmax + 1) * (lmax + 1) * 2);
  sht_output = zeros(12 * Nside * Nside);
  sht_work = zeros((lmax + 1) * (4 * Nside - 1) * 2);
  sht_plan = fastsht_plan_to_healpix(Nside, lmax, mmax, sht_input,
                                     sht_output, sht_work, FASTSHT_MMAJOR);

  /* Export FFTW wisdom generated during planning */
  fd = fopen("fftw.wisdom", "w");
  if (fd != NULL) {
    fftw_export_wisdom_to_file(fd);
    fclose(fd);
  }
}

void finish_sht(double dt) {
  fastsht_destroy_plan(sht_plan);
  free(sht_input);
  free(sht_output);
  free(sht_work);
}




/*
Main
*/

typedef struct {
  char *name;
  void (*setup)(void); 
  void (*execute)(int threadnum);
  void (*finish)(double dt);
  int multithreaded;
} benchmark_t;

benchmark_t benchmarks[] = {
  {"SHT", setup_sht, execute_sht, finish_sht, 0},
  {"memread", setup_memory_benchmark, execute_memory_benchmark, finish_memory_benchmark, 1},
  {"dgemm", setup_dgemm, execute_dgemm, finish_dgemm, 1},
  {"dgemm-single", setup_dgemm, execute_dgemm, finish_dgemm, 0},
  {NULL, NULL, NULL}
};



int main(int argc, char *argv[]) {
  fastsht_plan plan;
  double *input, *output, *work;
  double t0, t1, dt;
  int n, i;
  FILE *fd;
  benchmark_t *pbench;
  
  omp_set_dynamic(0);

  pbench = benchmarks;
  while (pbench->execute != NULL) {
    printf("%s:\n", pbench->name);
    N_threads = pbench->multithreaded ? omp_get_max_threads() : 1;
    if (pbench->setup != NULL) pbench->setup();
    t0 = walltime();
    n = 0;
    #pragma omp parallel for num_threads(N_threads) schedule(static, 1)
    for (i = 0; i < N_threads; ++i) {
      pbench->execute(i);
    }
    do {
    #pragma omp parallel for num_threads(N_threads) schedule(static, 1)
      for (i = 0; i < N_threads; ++i) {
        pbench->execute(i);
      }
      t1 = walltime();
      n++;
    } while (t1 - t0 < PROFILE_TIME);
    
    t1 = walltime();
    printtime("  ", n, t1 - t0);
    if (pbench->finish) pbench->finish((t1 - t0) / n);
    pbench++;
  }

}
