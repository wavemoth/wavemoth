/*
C program to benchmark spherical harmonic transforms
*/

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

#include "fastsht.h"
#include "fastsht_private.h"
#include "fastsht_error.h"
#include "blas.h"
#include "benchutils.h"


#define MAXPATH 2048
#define MAXTIME 50

int Nside, lmax;
char *sht_resourcefile;
int do_ffts;
int N_threads = 1;

int N_threads;

/*
Butterfly SHT benchmark
*/

double *sht_input, *sht_output, *psht_output;
fastsht_plan sht_plan;
int sht_nmaps;
int sht_m_stride = 1;

void execute_sht(void *ctx) {
  double t_compute, t_load;
  if (do_ffts) {
    fastsht_execute(sht_plan); 
  } else {
    fastsht_perform_legendre_transforms(sht_plan);
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

void finish_sht(void) {
  int64_t flops = 0;
  int m, odd;
  double stridefudge;
  char timestr[MAXTIME];

  fastsht_destroy_plan(sht_plan);
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
  size_t mstart[lmax + 1];

  size_t pos = 0;
  for (m = 0; m < lmax + 1; m += 1) {
    mstart[m] = (pos - m) * sht_nmaps;
    pos += lmax - m + 1; 
    marr[m] = m;
  }

  omp_set_dynamic(0);
  omp_set_num_threads(N_threads);

  check(Nside >= 0, "Invalid Nside");
  /* Setup m-major alm info */
  /* Input is strided/interleaved in m-major triangular order;
     output has one map after the other (non-interleaved). We support
     skipping some m's to speed benchmarks up.
  */

  /* Allocate input, output -- in m-major mode with some m's dropped */

  _psht_set_benchmark_parameters(0, lmax + 1, sht_m_stride, do_ffts);
  psht_make_general_alm_info(lmax, lmax + 1, sht_nmaps, marr, mstart,
                             &benchpsht_alm_info);
  /* The rest is standard */
  psht_make_healpix_geom_info(Nside, sht_nmaps, &benchpsht_geom_info);
  pshtd_make_joblist(&benchpsht_joblist);

  for (j = 0; j != sht_nmaps; ++j) {
    pshtd_add_job_alm2map(benchpsht_joblist, (pshtd_cmplx*)sht_input + j,
                          psht_output + j, 0);
  }
}

void finish_psht(void) {
  psht_destroy_alm_info(benchpsht_alm_info);
  psht_destroy_geom_info(benchpsht_geom_info);
  pshtd_clear_joblist(benchpsht_joblist);
  pshtd_destroy_joblist(benchpsht_joblist);
}

void execute_psht(void *ctx) {
  pshtd_execute_jobs(benchpsht_joblist, benchpsht_geom_info, benchpsht_alm_info);
}



/*
Main
*/

typedef struct {
  char *name;
  void (*setup)(void);
  void (*execute)(void *ctx);
  void (*finish)(void);
  int should_run;
  double min_time;
} benchmark_t;


double relative_error(double *a, double *b, size_t n) {
  size_t i;
  double diffnorm = 0.0, anorm = 0;
  for (i = 0; i != n; ++i) {
    diffnorm += (a[i] - b[i]) * (a[i] - b[i]);
    anorm += a[i] * a[i];
  }
  if (diffnorm == 0) {
    return 0.0;
  } else {
    return sqrt(diffnorm / anorm);
  }
}

int main(int argc, char *argv[]) {
  benchmark_t benchmarks[] = {
    (benchmark_t){"sht", setup_sht, execute_sht, finish_sht, 1, 0.0},
    (benchmark_t){"psht", setup_psht, execute_psht, finish_psht, 1, 0.0},
    {NULL, NULL, NULL, NULL, 0, 0.0}
  };

  benchmark_t *sht_benchmark = &benchmarks[0];
  benchmark_t *psht_benchmark = &benchmarks[1];

  double t0, t1, dtmin, tit0, tit1, dt;
  int n, i, j, should_run;
  benchmark_t *pbench;
  #ifdef HAS_PPROF
  char profilefile[MAXPATH];
  #endif
  char *resourcename;
  char timestr1[MAXTIME], timestr2[MAXTIME];

  char *stats_filename = NULL;
  char *stats_mode;

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

  while ((c = getopt (argc, argv, "r:N:j:n:t:S:k:a:o:F")) != -1) {
    switch (c) {
    case 'r': sht_resourcefile = optarg; break;
    case 'N': Nside = atoi(optarg);  break;
    case 'j': N_threads = atoi(optarg); break;
    case 'n': miniter = atoi(optarg); break;
    case 't': mintime = atof(optarg); break;
    case 'a':
      stats_filename = optarg;
      stats_mode = "a";
      break;
    case 'o':
      stats_filename = optarg;
      stats_mode = "w";
      break;
    case 'S': 
      sht_m_stride = atoi(optarg); 
      do_ffts = 0;
      break;
    case 'F':
      do_ffts = 0;
      break;
    case 'k': sht_nmaps = atoi(optarg); break;
    }  }
  argv += optind;
  argc -= optind;

  for (pbench = benchmarks; pbench->execute; ++pbench) {
    pbench->should_run = (argc == 0);
  }
  for (j = 0; j != argc; ++j) {
    for (pbench = benchmarks; pbench->execute; ++pbench) {
      if (strcmp(argv[j], pbench->name) == 0) {
        pbench->should_run = 1;
      }
    }
  }

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

  fprintf(stderr, "Using %d threads, %d maps, %s, m-thinning %d\n", N_threads, sht_nmaps,
          do_ffts ? "with FFTs" : "without FFTs", sht_m_stride);


  /* Set up input and output */
  size_t npix = 12 * Nside * Nside;
  sht_input = gauss_array(((lmax + 1) * (lmax + 2)) / 2 * 2 * sht_nmaps);
  sht_output = zeros(npix * sht_nmaps);
  psht_output = zeros(npix * sht_nmaps);

  /* Wavemoth benchmark */
  for (pbench = benchmarks; pbench->execute; ++pbench) {
    if (!pbench->should_run) continue;

    printf("%s:\n", pbench->name);
    pbench->setup();
    printf("  Warming up\n");
    pbench->execute(NULL);
    printf("  Executing\n");
    
#ifdef HAS_PPROF
    snprintf(profilefile, MAXPATH, "profiles/%s.prof", pbench->name);
    profilefile[MAXPATH - 1] = '\0';
    ProfilerStart(profilefile);
#endif

    printf("  ");
    pbench->min_time = benchmark(pbench->name, pbench->execute, NULL,
                                 miniter, mintime);

#ifdef HAS_PPROF
      ProfilerStop();
#endif

    pbench->finish();
  }

  printf("Runtime sht/psht: %f\n", sht_benchmark->min_time / psht_benchmark->min_time);
  printf("Speedup psht/sht: %f\n", psht_benchmark->min_time / sht_benchmark->min_time);
  double rho = relative_error(psht_output, sht_output, npix * sht_nmaps);
  printf("Relative error: %e\n", rho);

  if (stats_filename != NULL) {
    /* Write result to file as a line in the format
       nside lmax nmaps nthreads wavemoth_time psht_time relative_error
    */
    FILE *f = fopen(stats_filename, stats_mode);
    if (!f) {
      fprintf(stderr, "Could not open %s in mode %s\n", stats_filename, stats_mode);
    } else {
      fprintf(f, "%d %d %d %d %.15e %.15e %.15e\n",
              Nside, lmax, sht_nmaps, N_threads,
              sht_benchmark->min_time, psht_benchmark->min_time,
              rho);
      fclose(f);
    }
  }


  free(psht_output);
  free(sht_output);
  free(sht_input);
}
