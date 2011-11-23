#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include "wavemoth_cuda.h"

typedef struct {
  wavemoth_cuda_fft_plan_t *plan;
  int ithread;
} thread_ctx_t;

struct _wavemoth_cuda_fft_plan_t {
  pthread_t threads[NSTREAMS];
  thread_ctx_t ctx_list[NSTREAMS];
  pthread_barrier_t barrier;
  int nside;
  size_t input, output;
  volatile int terminate;
};

static void *thread_main(void *_ctx) {
  thread_ctx_t *ctx = _ctx;
  int ithread = ctx->ithread;
  wavemoth_cuda_fft_plan_t *plan = ctx->plan;

  int nffts;
  cufftHandle fft_plans[(plan->nside + NSTREAMS) / NSTREAMS];

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  while (1) {
    /* Construct plans */
    int n = 4;
    int planidx = 0;
    for (int iring = 0; iring < plan->nside; ++iring) {
      if (iring % NSTREAMS == ithread) {
        cufftPlan1d(&fft_plans[planidx], n, CUFFT_D2Z, 1);
        cufftSetStream(fft_plans[planidx], stream);
        planidx++;
      }
      n += 4;
    }
    nffts = planidx;

    /* Wait for next execute message or termination */
    pthread_barrier_wait(&plan->barrier);
    if (plan->terminate) {
      break;
    }

    /* Execute */
    printf("Executing\n");
    for (int idx = 0; idx != nffts; ++idx) {
      cufftExecD2Z(fft_plans[idx],
                   (cufftDoubleReal*)plan->input,
                   (cufftDoubleComplex*)plan->output);
    }
    printf("Done Executing\n");

  }
  /* Destroy plans */
  for (int idx = 0; idx != nffts; ++idx) {
    cufftDestroy(fft_plans[idx]);
  }
}

wavemoth_cuda_fft_plan_t *wavemoth_cuda_plan_healpix_fft(int nside, size_t input, size_t output) {
  wavemoth_cuda_fft_plan_t *plan = malloc(sizeof(wavemoth_cuda_fft_plan_t));
  plan->terminate = 0;
  plan->nside = nside;
  plan->input = input;
  plan->output = output;
  pthread_barrier_init(&plan->barrier, NULL, NSTREAMS + 1);
  for (int ithread = 0; ithread != NSTREAMS; ++ithread) {
    plan->ctx_list[ithread].ithread = ithread;
    plan->ctx_list[ithread].plan = plan;
    pthread_create(&plan->threads[ithread], NULL, thread_main, &plan->ctx_list[ithread]);
  }
  return plan;
}

void wavemoth_cuda_destroy(wavemoth_cuda_fft_plan_t *plan) {
  plan->terminate = 1;
  pthread_barrier_wait(&plan->barrier);
  void *ret;
  for (int ithread = 0; ithread != NSTREAMS; ++ithread) {
    pthread_join(plan->threads[ithread], &ret);
  }
  return;
  pthread_barrier_destroy(&plan->barrier);
}

void wavemoth_cuda_execute(wavemoth_cuda_fft_plan_t *plan) {
  pthread_barrier_wait(&plan->barrier);
}
