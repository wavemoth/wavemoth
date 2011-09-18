#include <stdio.h>
#include <numa.h>
#include <pthread.h>

void snprintsize(char *buf, size_t n, size_t size) {
  double dsize = size;
  if (dsize > 1024 * 1024 * 1024) {
    dsize /= 1024 * 1024 * 1024;
    snprintf(buf, n, "%.1f GB", dsize);
  } else {
    snprintf(buf, n, "%d bytes", size);
  }
}

#define SBUFLEN 40

int main() {
  char s1[SBUFLEN], s2[SBUFLEN], s3[SBUFLEN];
  if (numa_available() < 0) {
    printf("NUMA not available!\n");
    return 1;
  }
  int n = numa_max_node() + 1;
  printf("Number of nodes: %d\n", n);
  for (int i = 0; i != n; ++i) {
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
	cpus += snprintf(cpus, SBUFLEN - (cpus - s3), "%d ", cpu);
      }
    }
    printf("#%d: Mem: %s (%s free), CPUs: %s\n", i, s1, s2, s3);
  }

  
}
