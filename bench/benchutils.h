#ifndef _BENCHUTILS_H_
#define _BENCHUTILS_H_

#ifdef HAS_PPROF
#include <google/profiler.h>
#endif
#include <omp.h>

#include <stdlib.h>
#include <math.h>

static void print_array(char *msg, double* arr, bfm_index_t len) {
  bfm_index_t i;
  printf("%s ", msg);
  for (i = 0; i != len; ++i) {
    printf("%.2e ", arr[i]);
  }
  printf("\n");
}

static void fprint_array(FILE *f, double* arr, bfm_index_t len) {
  bfm_index_t i;
  for (i = 0; i != len; ++i) {
    fprintf(f, "%e ", arr[i]);
  }
  fprintf(f, "\n");
}

static double *zeros(size_t n) {
  double *buf;
  buf = memalign(16, n * sizeof(double));
  memset(buf, 0, n * sizeof(double));
  return buf;
}

static double gauss_rand() {
  static double V1, V2, S;
  static int phase = 0;
  double X;

  if (phase == 0) {
    do {
      double U1 = (double)rand() / RAND_MAX;
      double U2 = (double)rand() / RAND_MAX;

      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);
  } else
    X = V2 * sqrt(-2 * log(S) / S);
  
  phase = 1 - phase;
  
  return X;
}

static double *gauss_array(size_t n) {
  double *buf;
  buf = memalign(16, n * sizeof(double));
  for (size_t i = 0; i != n; ++i) {
    buf[i] = gauss_rand();
  }
  return buf;
}

static double walltime() {
  /*  struct timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  return tv.tv_sec + 1e-9 * tv.tv_nsec;*/
  return omp_get_wtime();
}

static void snftime(char *buf, size_t n, double time) {
  char *units;
  if (time < 1e-06) {
    units = "ns";
    time = 1e9;
  } else if (time < 1e-03) {
    units = "us";
    time *= 1e6;
  } else if (time < 10) {
    units = "ms";
    time *= 1e3;
  } else {
    units = "s";
  }
  snprintf(buf, n, "%.1f %s", time, units);
  buf[n - 1] = '\0';
}

static void load_wisdom(char *filename) {
  FILE *fd;
  /* Import FFTW plan if it exists */
  fd = fopen(filename, "r");
  if (fd != NULL) {
    fftw_import_wisdom_from_file(fd);
    fclose(fd);
  }
}


static void dump_wisdom(char *filename) {
  FILE *fd;
  fd = fopen(filename, "w");
  if (fd != NULL) {
    fftw_export_wisdom_to_file(fd);
    fclose(fd);
  }
}

static double benchmark(char *name, void (*func)(void *), void *ctx,
                        int atleast_iters, double atleast_time) {
  double dtmin, t0, tit0, tit1, t1, dtmean, dtmax;
  int n;
  char sbuf1[20], sbuf2[20], sbuf3[20];
  
  dtmin = 1e300;
  dtmax = 0;
  t0 = walltime();
  n = 0;
  do {
    tit0 = walltime();
    func(ctx);
    tit1 = t1 = walltime();
    if (tit1 - tit0 < dtmin) dtmin = tit1 - tit0;
    if (tit1 - tit0 > dtmax) dtmax = tit1 - tit0;
    n++;
  } while (n < atleast_iters || t1 - t0 < atleast_time);
  dtmean = (t1 - t0) / n;
  snftime(sbuf1, 20, dtmin);
  snftime(sbuf2, 20, dtmean);
  snftime(sbuf3, 20, dtmax);
  printf("%s (%d reps): %s (min), %s (mean), %s (max) \n", name, n, sbuf1, sbuf2, sbuf3);
  return dtmin;
}

#endif
