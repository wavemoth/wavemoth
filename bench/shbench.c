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

#define MAXPATH 2048
#define MAXTIME 50

int Nside, lmax;
char *sht_resourcefile;
int do_ffts;
int N_threads = 1;

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


/*
Butterfly SHT benchmark
*/

double *sht_input, *sht_output;
fastsht_plan sht_plan;
int sht_nmaps;
int sht_m_stride = 1;
size_t *sht_mstart;

void setup_sht_buffers() {
  /* Allocate input, output -- in m-major mode with some m's dropped */
  int m;
  size_t pos;
  pos = 0;
  sht_mstart = malloc(sizeof(size_t[lmax + 1]));
  for (m = 0; m < lmax + 1; m += 1) {
    if (m % sht_m_stride == 0) {
      sht_mstart[m] = pos - m;
      pos += lmax - m + 1;
    } else {
      sht_mstart[m] = 0x7FFFFFFF;
    }
  }
  sht_input = zeros(pos * 2 * sht_nmaps);
  sht_output = zeros(12 * Nside * Nside * sht_nmaps);
}

void free_sht_buffers() {
  free(sht_mstart);
  free(sht_input);
  free(sht_output);
}

void execute_sht() {
  double t_compute, t_load;
  if (do_ffts) {
    fastsht_execute(sht_plan); 
  } else {
    fastsht_perform_legendre_transforms(sht_plan, 0, lmax + 1, sht_m_stride);
  }
}

void setup_sht() {
  int nmaps = sht_nmaps;
  FILE *fd;
  printf("  Initializing (incl. FFTW plans)\n");
  /* Import FFTW plan if it exists */
  fd = fopen("fftw.wisdom", "r");
  if (fd != NULL) {
    fftw_import_wisdom_from_file(fd);
    fclose(fd);
  }

  sht_input = zeros((lmax + 1) * (lmax + 1) * 2 * nmaps);
  sht_output = zeros(12 * Nside * Nside * nmaps);
  sht_plan = fastsht_plan_to_healpix(Nside, lmax, lmax, nmaps, N_threads, sht_input,
                                     sht_output, FASTSHT_MMAJOR,
                                     sht_resourcefile);
  checkf(sht_plan, "plan not created, nthreads=%d", N_threads);

  /* Export FFTW wisdom generated during planning */
  fd = fopen("fftw.wisdom", "w");
  if (fd != NULL) {
    fftw_export_wisdom_to_file(fd);
    fclose(fd);
  }
}

void finish_sht(double dt) {
  int64_t flops = 0;
  int m, odd;
  double stridefudge;
  char timestr[MAXTIME];

  fastsht_destroy_plan(sht_plan);
  free_sht_buffers();
}


/*
PSHT
*/

/*
Inserted our own hack in PSHT...
*/

psht_alm_info *benchpsht_alm_info;
psht_geom_info *benchpsht_geom_info;
pshtd_joblist *benchpsht_joblist;

#ifdef PATCHED_LIBPSHT
void _psht_set_benchmark_parameters(int m_start, int m_stop, int m_stride,
                                    int run_ffts);
#else
void _psht_set_benchmark_parameters(int m_start, int m_stop, int m_stride,
                                    int run_ffts) {
  if (m_stride != 1 || !run_ffts) {
    fprintf(stderr, "Requested benchmark requires a patched version of libpsht\n");
    exit(1);
  }
}
#endif

ptrdiff_t lm_to_idx_mmajor(ptrdiff_t l, ptrdiff_t m) {
  return m * (2 * lmax - m + 3) / 2 + (l - m);
}

void setup_psht() {
  int m;
  int marr[lmax + 1];
  int j;
  ptrdiff_t npix = 12 * Nside * Nside;
  for (m = 0; m < lmax + 1; m += 1) {
    marr[m] = m;
  }

  check(Nside >= 0, "Invalid Nside");
  /* Setup m-major alm info */
  /* Input is strided/interleaved in m-major triangular order;
     output has one map after the other (non-interleaved). We support
     skipping some m's to speed benchmarks up.
  */
  setup_sht_buffers();
  _psht_set_benchmark_parameters(0, lmax + 1, sht_m_stride, do_ffts);
  psht_make_general_alm_info(lmax, lmax + 1, sht_nmaps, marr, sht_mstart, &benchpsht_alm_info);
  /* The rest is standard */
  psht_make_healpix_geom_info(Nside, 1, &benchpsht_geom_info);
  pshtd_make_joblist(&benchpsht_joblist);

  for (j = 0; j != sht_nmaps; ++j) {
    pshtd_add_job_alm2map(benchpsht_joblist, (pshtd_cmplx*)sht_input + j,
                          sht_output + j * npix, 0);
  }
}

void finish_psht() {
  psht_destroy_alm_info(benchpsht_alm_info);
  psht_destroy_geom_info(benchpsht_geom_info);
  pshtd_clear_joblist(benchpsht_joblist);
  pshtd_destroy_joblist(benchpsht_joblist);
  free_sht_buffers();
}

void execute_psht() {
  pshtd_execute_jobs(benchpsht_joblist, benchpsht_geom_info, benchpsht_alm_info);
}



/*
Main
*/

typedef struct {
  char *name;
  void (*setup)(void); 
  void (*execute)(void);
  void (*finish)(double dt);
} benchmark_t;

benchmark_t benchmarks[] = {
  {"sht", setup_sht, execute_sht, finish_sht},
  {"psht", setup_psht, execute_psht, finish_psht},
  {NULL, NULL, NULL}
};



int main(int argc, char *argv[]) {
  double t0, t1, dtmin, tit0, tit1, dt;
  int n, i, j, should_run;
  benchmark_t *pbench;
  #ifdef HAS_PPROF
  char profilefile[MAXPATH];
  #endif
  char *resourcename;
  char timestr1[MAXTIME], timestr2[MAXTIME];

  int c;
  int got_threads, miniter;
  double mintime;
  char *resource_path;

  /* Parse options */
  sht_resourcefile = NULL;
  Nside = -1;
  miniter = 1;
  mintime = 0;
  opterr = 0;
  sht_nmaps = 1;
  do_ffts = -1;

  while ((c = getopt (argc, argv, "r:N:j:n:t:S:k:F")) != -1) {
    switch (c) {
    case 'r': sht_resourcefile = optarg; break;
    case 'N': Nside = atoi(optarg);  break;
    case 'j': N_threads = atoi(optarg); break;
    case 'n': miniter = atoi(optarg); break;
    case 't': mintime = atof(optarg); break;
    case 'S': 
      sht_m_stride = atoi(optarg); 
      do_ffts = 0;
      break;
    case 'F':
      do_ffts = 0;
      break;
    case 'k': sht_nmaps = atoi(optarg); break;
    }
  }
  argv += (optind - 1);
  argc -= (optind - 1);

  if (do_ffts == -1) do_ffts = 1;

  /* Resource configuration */
  resource_path = getenv("SHTRESOURCES");
  if (sht_resourcefile != NULL) {
    fastsht_configure("");
    fastsht_query_resourcefile(sht_resourcefile, &Nside, &lmax);
  } else {
    check(resource_path != NULL, "Please define SHTRESOURCES or use -r switch");
    fastsht_configure(resource_path);
    lmax = 2 * Nside;
  }

  /* Set up multithreading asked for */
  omp_set_dynamic(0);
  omp_set_num_threads(N_threads);
#pragma omp parallel shared(got_threads)
  {
    got_threads = omp_get_num_threads();
  }
  if (got_threads < N_threads) {
    fprintf(stderr, "WARNING: Threads available less than requested.\n");
  }
  fprintf(stderr, "Using %d threads, %d maps, %s, m-thinning %d\n", got_threads, sht_nmaps,
          do_ffts ? "with FFTs" : "without FFTs", sht_m_stride);
  fprintf(stderr, "Nside=%d, lmax=%d\n", Nside, lmax);

  /* Run requested benchmarks */
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
    if (pbench->setup != NULL) pbench->setup();
    printf("  Warming up\n");
    pbench->execute();
    printf("  Executing\n");
    #ifdef HAS_PPROF
    snprintf(profilefile, MAXPATH, "profiles/%s.prof", pbench->name);
    profilefile[MAXPATH - 1] = '\0';
    ProfilerStart(profilefile);
    #endif
    dtmin = 1e300;
    t0 = walltime();
    n = 0;
    do {
      tit0 = walltime();
      pbench->execute();
      tit1 = t1 = walltime();
      if (tit1 - tit0 < dtmin) dtmin = tit1 - tit0;
      n++;
    } while (n < miniter || t1 - t0 < mintime);
    #ifdef HAS_PPROF
    ProfilerStop();
    #endif
    dt = (t1 - t0) / n;

    snftime(timestr1, MAXTIME, dtmin);
    snftime(timestr2, MAXTIME, dtmin * N_threads);
    printf("  Best of %d reps: %s wall, %s thread-wall\n", n, timestr1, timestr2);
    snftime(timestr1, MAXTIME, dt);
    snftime(timestr2, MAXTIME, dt * N_threads);
    printf("  Mean of %d reps: %s wall, %s thread-wall\n", n, timestr1, timestr2);
    if (pbench->finish) pbench->finish(dtmin);
    pbench++;
  }

}
