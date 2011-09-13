/*
C program to benchmark spherical harmonic transforms
*/

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <stddef.h>

#include <fftw3.h>
#ifdef HAS_PPROF
#include <google/profiler.h>
#endif
#include <omp.h>


fftw_plan plan_transpose(char storage_type,
                         int rows,
                         int cols,
                         double *in,
                         double *out) {
  const unsigned flags = FFTW_MEASURE;

  fftw_iodim howmany_dims[2];
  switch (storage_type) {
  case 'R':
    howmany_dims[0].n  = rows;
    howmany_dims[0].is = cols;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = cols;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = rows;
    break;
  case 'C':
    howmany_dims[0].n  = rows;
    howmany_dims[0].is = 1;
    howmany_dims[0].os = cols;
    howmany_dims[1].n  = cols;
    howmany_dims[1].is = rows;
    howmany_dims[1].os = 1;
    break;
  default:
    return NULL;
  }
  const int howmany_rank = sizeof(howmany_dims)/sizeof(howmany_dims[0]);
  
  return fftw_plan_guru_r2r(/*rank*/0, /*dims*/NULL,
                            howmany_rank, howmany_dims,
                            in, out, /*kind*/NULL, flags);
}

static double *zeros(size_t n) {
  double *buf;
  buf = memalign(16, n * sizeof(double));
  memset(buf, 0, n * sizeof(double));
  return buf;
}

/*int get_ring_offset(int iring) {
  int idx = 0, ringsize = 4;
  for (int i = 0; i != iring; ++i) {
    idx += ringsize;
    ringsize += 4;
  }
  }*/


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

int main(int argc, char *argv[]) {

  int Nside = 2048;
  int nmaps = 1;
  unsigned flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE;
  int n_ring_list[nmaps];
  fftw_r2r_kind kind_list[nmaps];
  fftw_plan plan;
  int i;
  FILE *fd;
  double *map = zeros(16 * Nside * Nside * nmaps * 2);
  int SBUF_SIZE = 20;
  char sbuf[SBUF_SIZE];

  int miniter = 2;
  double mintime = 1.0;

  printf("Loading wisdom\n");
  /* Import FFTW plan if it exists */
  fd = fopen("fftbench.wisdom", "r");
  if (fd != NULL) {
    fftw_import_wisdom_from_file(fd);
    fclose(fd);
  }

  /* Center rings, one by one */
  
  for (i = 0; i != nmaps; ++i) {
    n_ring_list[i] = 4 * Nside;
    kind_list[i] = FFTW_HC2R;
  }
  printf("Making plan...\n");
  plan = fftw_plan_many_r2r(1, n_ring_list, 1,
                            map, n_ring_list, 1, 1,
                            map, n_ring_list, 1, 1,
                            kind_list, flags);
  printf("done\n");

  double dtmin, t0, tit0, tit1, t1, dt;
  int n;

  dtmin = 1e300;
  t0 = walltime();
  n = 0;
  do {
    tit0 = walltime();

    double *ring = map;
    for (i  = 0; i != nmaps * 2 * Nside; ++i) {
      fftw_execute_r2r(plan, ring, ring);
      ring += 4 * Nside;
    }

    tit1 = t1 = walltime();
    if (tit1 - tit0 < dtmin) dtmin = tit1 - tit0;
    n++;
  } while (n < miniter || t1 - t0 < mintime);
  dt = (t1 - t0) / n;
  snftime(sbuf, SBUF_SIZE, dt);
  printf("Center, one by one: %s\n", sbuf);
  fftw_destroy_plan(plan);

  printf("Making plan\n");
  /*  plan = fftw_plan_many_r2r(1, n_ring_list, nmaps * 2 * Nside,
                            map, n_ring_list, 1, 4 * Nside,
                            map, n_ring_list, 1, 4 * Nside,
                            kind_list, flags);*/
  plan = plan_transpose('R', 4 * Nside, 3 * Nside, map, map + 16 * Nside * Nside * 0);
  printf("done\n");
  dtmin = 1e300;
  t0 = walltime();
  n = 0;
  do {
    tit0 = walltime();

    fftw_execute_r2r(plan, map, map + 0*16 * Nside * Nside);

    tit1 = t1 = walltime();
    if (tit1 - tit0 < dtmin) dtmin = tit1 - tit0;
    n++;
  } while (n < miniter || t1 - t0 < mintime);
  dt = (t1 - t0) / n;
  snftime(sbuf, SBUF_SIZE, dt);
  printf("Center, all at once: %s\n", sbuf);
  fftw_destroy_plan(plan);


  printf("Making plans for non-equatorial-band rings\n");
  fftw_plan fftplans[Nside];
  int ringlen = 4;
  double *head = map;
  for (i = 0; i != Nside; ++i) {
    fftplans[i] = fftw_plan_many_r2r(1, &ringlen, 1,
                                     head, NULL, 1, 0,
                                     head, NULL, 1, 0,
                                     kind_list, flags);
    head += ringlen;
    ringlen += 4;
    if (i % 100 == 0) printf("%d\n", i);
  }
  dtmin = 1e300;
  t0 = walltime();
  n = 0;
  do {
    tit0 = walltime();
    head = map;
    for (i = 0; i != Nside; ++i) {
      fftw_execute_r2r(fftplans[i], head, head);
      head += ringlen;
      ringlen += 4;
    }

    tit1 = t1 = walltime();
    if (tit1 - tit0 < dtmin) dtmin = tit1 - tit0;
    n++;
  } while (n < miniter || t1 - t0 < mintime);
  dt = (t1 - t0) / n;
  snftime(sbuf, SBUF_SIZE, dt);
  printf("Center, all at once: %s\n", sbuf);
  for (i = 0; i != Nside; ++i) fftw_destroy_plan(fftplans[i]);

  
/*     fftw_plan fftw_plan_many_r2r(int rank, const int *n, int howmany,
                                  double *in, const int *inembed,
                                  int istride, int idist,
                                  double *out, const int *onembed,
                                  int ostride, int odist,
                                  const fftw_r2r_kind *kind, unsigned flags);

     */


  printf("Saving wisdom\n");
  fd = fopen("fftbench.wisdom", "w");
  if (fd != NULL) {
    fftw_export_wisdom_to_file(fd);
    fclose(fd);
  }

  

}
