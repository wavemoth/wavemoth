#define _GNU_SOURCE
#include <stdio.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <math.h>

#include <sys/types.h>
#include <sys/times.h>
#include <unistd.h>

#include <xmmintrin.h>
#include <emmintrin.h>

typedef __m128 m128;

void snprintsize(char *buf, size_t n, size_t size) {
  double dsize = size;
  if (dsize > 1024 * 1024 * 1024) {
    dsize /= 1024 * 1024 * 1024;
    snprintf(buf, n, "%.1f GB", dsize);
  } else {
    snprintf(buf, n, "%d bytes", size);
  }
}

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

#define SBUFLEN 40

#define KB 1024UL
#define MB KB * KB
#define GB KB * KB * KB

#define BENCH_BUF_SIZE GB / 4
#define NSAMPLES 5

typedef struct {
  double value, std;
} measurement_t;

typedef struct {
  int node;
  int thread;
  int threads_per_node;
  pthread_barrier_t *node_barrier;
} context_t;

double mean(double *array, size_t n) {
  double x = 0;
  for (size_t i = 0; i != n; ++i) {
    x += array[i];
  }
  return x / n;
}

double std(double *array, size_t n, double m) {
  double x = 0;
  for (size_t i = 0; i != n; ++i) {
    double z = array[i] - m;
    x += z * z;
  }
  return sqrt(x / n);
}

measurement_t stats(double *array, size_t n) {
  double m = mean(array, n);
  return (measurement_t){ m, std(array, n, m) };
}


int nodecount;
measurement_t *rates;
pthread_barrier_t barrier_main;
pthread_barrier_t barrier_iteration;
float **buffers;

void write_test(float *buf) {
  m128 two = (m128){2.0f, 2.0f, 2.0f, 2.0f};
  for (int i = 0; i != BENCH_BUF_SIZE / sizeof(float); i += 4) {
    _mm_store_ps(buf + i, two);
  }
}

void read_test(float *buf) {
  m128 acc = _mm_setzero_ps();
  for (int i = 0; i != BENCH_BUF_SIZE / sizeof(float); i += 4) {
    acc = _mm_add_ps(acc, _mm_load_ps(buf + i));
  }
  _mm_store_ps(buf, acc);
}

measurement_t benchmark_all_vs_all(void (*func)(float*), measurement_t *rates, int node,
				   context_t *ctx) {
  for (int shift = 0; shift != nodecount; shift++) {
    int col = (node + shift) % nodecount, row = node;

    float *buf = buffers[ctx->thread * nodecount + col];

    double times[NSAMPLES];
    int it;
    for (it = 0; it != NSAMPLES; it++) {
      pthread_barrier_wait(&barrier_iteration);
      double t0 = walltime();
      pthread_barrier_wait(ctx->node_barrier);
      (*func)(buf);
      pthread_barrier_wait(ctx->node_barrier);
      double dt = walltime() - t0;
      times[it] = dt;
    }
    if (ctx->thread == 0) {
      /* Only use the timings of the first thread on the node,
         they should be identical due to barrier */
      for (it = 0; it != NSAMPLES; it++) {
	times[it] = (double)(ctx->threads_per_node * BENCH_BUF_SIZE) / (GB * times[it]);
      }
      rates[row * nodecount + col] = stats(times, NSAMPLES);
    }
  }
}

int get_node_cpu(cpu_set_t *set, int node, int cpu_index) {
  unsigned long cpumask[32];
  int ret;
  ret = numa_node_to_cpus(node, cpumask, sizeof(cpumask));
  if (ret < 0) return ret;
  /* TODO: Support more than 64 CPUs */
  for (int cpuid = 0; cpuid < sizeof(long) * 8; ++cpuid) {
    if ((cpumask[0] >> cpuid) & 1) {
      if (cpu_index == 0) {
	CPU_SET(cpuid, set);
      }
      cpu_index--;
    }
  }
}

void print_table_row(measurement_t *rates, int row) {
  for (int i = 0; i != nodecount; ++i) {
    measurement_t m = rates[row * nodecount + i];
    printf("%6.1f  ", m.value);
  }
  printf("\n");
}

void print_table(char *title, int threads_per_node, measurement_t *rates) {
  printf("== %s (%d threads per node) ==\n", title, threads_per_node);
  for (int row = 0; row != nodecount; ++row) {
    print_table_row(rates, row);
  }
  double maxsd = 0;
  for (int i = 0; i != nodecount * nodecount; ++i) {
    maxsd = fmax(maxsd, rates[i].std);
  }
  printf("Max standard deviation: %f\n\n", maxsd);
}

void * thread_main(void *ctx_) {
  context_t *ctx = ctx_;
  int node = ctx->node;

  pthread_barrier_wait(&barrier_main);
  /* Warmup & initialization*/
  write_test(buffers[ctx->thread * nodecount + node]);

  /* Benchmark */
  benchmark_all_vs_all(write_test, rates, node, ctx);
  pthread_barrier_wait(&barrier_main);
  benchmark_all_vs_all(read_test, rates, node, ctx);
  pthread_barrier_wait(&barrier_main);
}


void execute(int threads_per_node) {
  int threadcount = threads_per_node * nodecount;

  /* Allocate one buffer per thread */
  float *bufferpointers_on_stack[threadcount];
  buffers = bufferpointers_on_stack;
  for (int node = 0; node != nodecount; ++node) {
    for (int thread = 0; thread != threads_per_node; ++thread) {
      buffers[thread * nodecount + node] = numa_alloc_onnode(BENCH_BUF_SIZE, node);
    }
  }


  pthread_t threads[threadcount];
  context_t contexts[threadcount];
  measurement_t rates_on_stack[nodecount * nodecount];
  pthread_barrier_t node_barriers[nodecount];
  rates = rates_on_stack;

  for (int i = 0; i != nodecount * nodecount; ++i) {
    rates[i] = (measurement_t){ 0.0, 0.0 };
  }

  /* Set up a barrier so that all threads cooperate. The current
     thread is part of barrier_main (for synchronizing tests and
     writing the result of each test to screen), while
     barrier_iteration is only for the threads (for synchronizing loop
     iterations within a test). */
  pthread_barrier_init(&barrier_main, NULL, threadcount + 1);
  pthread_barrier_init(&barrier_iteration, NULL, threadcount);

  /* Create threads */
  for (int node = 0; node != nodecount; ++node) {
    pthread_barrier_init(&node_barriers[node], NULL, threads_per_node);
    for (int ithread = 0; ithread != threads_per_node; ++ithread) {
      context_t *pctx = &contexts[node * threads_per_node + ithread];
      cpu_set_t cpu_set;
      pthread_attr_t attr;
      CPU_ZERO(&cpu_set);
      if (get_node_cpu(&cpu_set, node, ithread) < 0) {
	exit(1);
      }

      pctx->node = node;
      pctx->thread = ithread;
      pctx->threads_per_node = threads_per_node;
      pctx->node_barrier = &node_barriers[node];

      pthread_attr_init(&attr);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpu_set);
      pthread_create(&threads[node], &attr, &thread_main, pctx);
      pthread_attr_destroy(&attr);
    }
  }

  /* First benchmark */
  pthread_barrier_wait(&barrier_main); /* startup */
  pthread_barrier_wait(&barrier_main); /* write test */
  print_table("Write", threads_per_node, rates);
  pthread_barrier_wait(&barrier_main); /* read test */
  print_table("Read", threads_per_node, rates);


  /* Teardown: Join thread, destroy sync objects, free buffers */
  for (int i = 0; i != nodecount; ++i) {
    pthread_join(threads[i], NULL);
    pthread_barrier_destroy(&node_barriers[i]);
  }
  pthread_barrier_destroy(&barrier_main);
  pthread_barrier_destroy(&barrier_iteration);

  /* Free buffers */
  for (int i = 0; i != threadcount; ++i) {
    numa_free(buffers[i], BENCH_BUF_SIZE);
  }

}

int main() {
  char s1[SBUFLEN], s2[SBUFLEN], s3[SBUFLEN];
  if (numa_available() < 0) {
    printf("NUMA not available!\n");
    return 1;
  }
  nodecount = numa_max_node() + 1;
  printf("Number of nodes: %d\n", nodecount);
  int cpucount = 0;
  for (int i = 0; i != nodecount; ++i) {
    long memtotal, memfree;
    unsigned long cpumask[32]; /* TODO: >64 cores */
    memtotal = numa_node_size(i, &memfree);
    snprintsize(s1, SBUFLEN, memtotal);
    snprintsize(s2, SBUFLEN, memfree);
    if (numa_node_to_cpus(i, cpumask, sizeof(cpumask)) < 0) {
      printf("ERANGE\n");
      return 2;
    }
    char *cpus = s3;
    for (int cpu = 0; cpu < sizeof(long) * 8; ++cpu) {
      if ((cpumask[0] >> cpu) & 1) {
	cpucount++;
	cpus += snprintf(cpus, SBUFLEN - (cpus - s3), "%d ", cpu);
      }
    }
    printf("#%d: Mem: %s (%s free), CPUs: %s\n", i, s1, s2, s3);
  }


  //execute(1);
  execute(6);


}
