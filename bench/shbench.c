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

#define Nside 512
#define lmax 2 * Nside
#define mmax lmax
#define PROFILE_TIME 3.0

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

static void printtime(char* msg, double time) {
  char *units;
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
  printf("%s: %.1f %s\n", msg, time, units);
}

int main(int argc, char *argv[]) {
  fastsht_plan plan;
  double *input, *output, *work;
  double t0, t1, dt;
  int n;
  FILE *fd;

  /* Import FFTW plan if it exists */
  printf("Initializing (incl. FFTW)\n");
  fd = fopen("fftw.wisdom", "r");
  if (fd != NULL) {
    fftw_import_wisdom_from_file(fd);
    fclose(fd);
  }

  fastsht_add_precomputation_file("/home/dagss/code/spherew/precomputed.dat");

  input = zeros((lmax + 1) * (lmax + 1) * 2);
  output = zeros(12 * Nside * Nside);
  work = zeros((lmax + 1) * (4 * Nside - 1));

  plan = fastsht_plan_to_healpix(Nside, lmax, mmax, input, output, work, FASTSHT_MMAJOR);

  /* Export FFTW wisdom generated during planning */
  fd = fopen("fftw.wisdom", "w");
  if (fd != NULL) {
    fftw_export_wisdom_to_file(fd);
    fclose(fd);
  }

  
  printf("Computing for > %.1f seconds, Nside=%d...\n", PROFILE_TIME, Nside);
  t0 = walltime();
  n = 0;
  do {
    fastsht_execute(plan);   
    n++;
    t1 = walltime();
  } while (t1 - t0 < PROFILE_TIME);
  t1 = walltime();
  printf("Repeated %d times\n", n);
  printtime("Avg. time", (t1 - t0) / n);
  fastsht_destroy_plan(plan);

}
