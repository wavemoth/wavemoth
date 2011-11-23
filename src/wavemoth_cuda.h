
#define NSTREAMS 10

struct _wavemoth_cuda_fft_plan_t;
typedef struct _wavemoth_cuda_fft_plan_t wavemoth_cuda_fft_plan_t;

wavemoth_cuda_fft_plan_t *wavemoth_cuda_plan_healpix_fft(int nside, size_t input, size_t output);
void wavemoth_cuda_destroy(wavemoth_cuda_fft_plan_t *plan);
void wavemoth_cuda_execute(wavemoth_cuda_fft_plan_t *plan);
