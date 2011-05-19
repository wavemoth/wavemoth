#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "blas.h"



double doit(int m, int k, int n) {
  clock_t t0, dt;
  int i, j, repeats;
  double *A, *x, *y;
  A = (double*)malloc(m * k * sizeof(double));
  x = (double*)malloc(k * n * sizeof(double));
  y = (double*)malloc(m * n * sizeof(double));
  for (i = 0; i != m * n; ++i) {
    A[i] = i;
  }
  for (i = 0; i != k * n; ++i) {
    x[i] = i;
  }
  for (i = 0; i != m * n; ++i) {
    y[i] = i;
  }
  dgemm('N', 'N', m, n, k, 1.0, A, m, x, k, 1.0, y, m);
  t0 = clock(); dt = 0;
  repeats = 0;
  while (dt < CLOCKS_PER_SEC) {
    dgemm('N', 'N', m, n, k, 1.0, A, m, x, k, 1.0, y, m);
    dt = clock() - t0;
    ++repeats;
  }
  free(A);
  free(x);
  free(y);
  return ((double)dt / CLOCKS_PER_SEC) / repeats;
}

void printtime(char* msg, double time) {
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

int main() {
  printtime("Call only", doit(1, 1, 1));
  printtime("60-by-60-by-2", doit(60, 60, 2));
}
