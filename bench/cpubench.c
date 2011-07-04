#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <omp.h>

#include "blas.h"

size_t n = 1000;
int repeats = 10;
double **A, **B, **Y;

static double *zeros(size_t n) {
  double *buf;
  buf = memalign(4096, n * sizeof(double));
  memset(buf, 0, n * sizeof(double));
  return buf;
}

int main() {
  int N_threads = omp_get_max_threads();
  int i;
  double t0, dt;
  size_t flops;

  A = malloc(sizeof(void*) * N_threads);
  B = malloc(sizeof(void*) * N_threads);
  Y = malloc(sizeof(void*) * N_threads);
  for (i = 0; i != N_threads; ++i) {
    A[i] = zeros(n * n);
    B[i] = zeros(n * n);
    Y[i] = zeros(n * n);
  }

  
  t0 = omp_get_wtime();
  #pragma omp parallel
  {
    int ii, jj;
    #pragma omp for
    for (ii = 0; ii < N_threads; ++ii) {
      for (jj = 0; jj < repeats; ++jj) {
        dgemm_rrr(A[ii], B[ii], Y[ii], n, n, n, 1.0);
      }
    }
  }
  dt = omp_get_wtime() - t0;
  
  flops = N_threads * n * n * n * repeats * 2;
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


