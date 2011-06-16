/*
C program to benchmark spherical harmonic transforms
*/

#include "fastsht.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <fftw3.h>

#ifdef USE_PPROF
#include <google/profiler.h>
#endif

#define Nside 512
#define lmax 2 * Nside
#define mmax lmax
#define PROFILE_TIME 2.0

#define NTHREADS 4

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
  printf("%s -- %d times, avg: %.1f %s\n", msg, n, time, units);
}


/*
Benchmark memory throughput
*/
double *mem_buf;
size_t N_mem = 200 * 1024 * 1024 / sizeof(double);

void setup_memory_benchmark() {
  /* zero buffer, so that we won't hit denormal numbers in reading loop */
  mem_buf = zeros(NTHREADS * N_mem);
}

void finish_memory_benchmark(double dt) {
  printf("  Bandwidth: %.3f GB/s\n", NTHREADS * N_mem * sizeof(double) / dt / (1024 * 1024 * 1024));
  free(mem_buf);
}

void execute_memory_benchmark() {
  /* Simply accumulate the buffer in a double -- we're going to be very
     IO-bound so we can basically ignore the addition put in in order to
     avoid optimizing things out.   */
  int i, j;
  size_t n = N_mem;
  double acc;
  double *buf = mem_buf;
  #pragma omp parallel for num_threads(NTHREADS) private(i, j) shared(n) reduction(+:acc)
  for (i = 0; i < NTHREADS; ++i) {
    for (j = 0; j != n; j += 2) {
      acc += buf[i * n + j];
    }
  }
  if (acc != 0.0) printf("dummy\n");
}

/*
Main
*/

typedef void (*voidfunc)(void);
typedef struct {
  char *name;
  voidfunc setup, execute;
  void (*finish)(double dt);
} benchmark_t;

benchmark_t benchmarks[] = {
  {"memread", setup_memory_benchmark, execute_memory_benchmark, finish_memory_benchmark},
  {NULL, NULL, NULL}
};



int main(int argc, char *argv[]) {
  fastsht_plan plan;
  double *input, *output, *work;
  double t0, t1, dt;
  int n;
  FILE *fd;
  benchmark_t *pbench;

  /* Import FFTW plan if it exists */
  printf("Initializing (incl. FFTW)\n");
  fd = fopen("fftw.wisdom", "r");
  if (fd != NULL) {
    fftw_import_wisdom_from_file(fd);
    fclose(fd);
  }

  fastsht_configure("/home/dagss/code/spherew/resources");

  input = zeros((lmax + 1) * (lmax + 1) * 2);
  output = zeros(12 * Nside * Nside);
  work = zeros((lmax + 1) * (4 * Nside - 1) * 2);

  plan = fastsht_plan_to_healpix(Nside, lmax, mmax, input, output, work, FASTSHT_MMAJOR);

  /* Export FFTW wisdom generated during planning */
  fd = fopen("fftw.wisdom", "w");
  if (fd != NULL) {
    fftw_export_wisdom_to_file(fd);
    fclose(fd);
  }

  fastsht_execute(plan); /* Ensure precomputed data is loaded */
  printf("Computing for > %.1f seconds, Nside=%d...\n", PROFILE_TIME, Nside);
  #ifdef USE_PPROF
  ProfilerStart("shbench.prof");
  #endif
  t0 = walltime();
  n = 0;
  do {
    fastsht_execute(plan);   
    n++;
    t1 = walltime();
  } while (t1 - t0 < PROFILE_TIME);
  t1 = walltime();
  #ifdef USE_PPROF
  ProfilerStop();
  #endif
  printf("Repeated %d times\n", n);
  printtime("Avg. time", n, t1 - t0);
  fastsht_destroy_plan(plan);

  pbench = benchmarks;
  while (pbench->execute != NULL) {
    printf("%s:\n", pbench->name);
    if (pbench->setup != NULL) pbench->setup();
    t0 = walltime();
    n = 0;
    do {
      pbench->execute();
      n++;
      t1 = walltime();
    } while (t1 - t0 < PROFILE_TIME);
    t1 = walltime();
    printtime("  ", n, t1 - t0);
    if (pbench->finish) pbench->finish((t1 - t0) / n);
    pbench++;
  }

}
