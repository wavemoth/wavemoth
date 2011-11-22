from libc.stdlib cimport malloc, free


cdef extern from "cuda_runtime_api.h":
    ctypedef unsigned int cudaStream_t
    int cudaStreamCreate(cudaStream_t *pStream)
    int cudaStreamDestroy(cudaStream_t pStream)
    int cudaDeviceSynchronize()

cdef extern from "cufft.h":
    ctypedef enum cufftResult:
        CUFFT_SUCCESS        = 0x0
        CUFFT_INVALID_PLAN   = 0x1
        CUFFT_ALLOC_FAILED   = 0x2
        CUFFT_INVALID_TYPE   = 0x3
        CUFFT_INVALID_VALUE  = 0x4
        CUFFT_INTERNAL_ERROR = 0x5
        CUFFT_EXEC_FAILED    = 0x6
        CUFFT_SETUP_FAILED   = 0x7
        CUFFT_INVALID_SIZE   = 0x8
        CUFFT_UNALIGNED_DATA = 0x9

    ctypedef enum cufftType:
        CUFFT_R2C = 0x2a
        CUFFT_C2R = 0x2c
        CUFFT_C2C = 0x29
        CUFFT_D2Z = 0x6a
        CUFFT_Z2D = 0x6c
        CUFFT_Z2Z = 0x69

    ctypedef unsigned int cufftHandle
    ctypedef struct cufftDoubleReal
    ctypedef struct cufftDoubleComplex

    cufftResult cufftPlan1d(cufftHandle *plan, 
                            int nx, 
                            cufftType type, 
                            int batch)
    
    cufftResult cufftDestroy(cufftHandle plan)

    cufftResult cufftSetStream(cufftHandle plan,
                               cudaStream_t stream)

    cufftResult cufftExecD2Z(cufftHandle plan, 
                             cufftDoubleReal *idata,
                             cufftDoubleComplex *odata)


cdef class HealpixCuFFTPlan:
    cdef cufftHandle *plans
    cdef cudaStream_t *streams
    cdef int nside, nstreams

    def __cinit__(self, int nside, int nstreams):
        self.nstreams = nstreams
        self.nside = nside

        self.plans = <cufftHandle*>malloc(sizeof(cufftHandle) * nside);
        self.streams = <cudaStream_t*>malloc(sizeof(cudaStream_t) * nstreams);

        for i in range(nstreams):
            print cudaStreamCreate(&self.streams[i])

        cdef int n = 4
        cdef cufftResult ret
        for i in range(nside):
            ret = cufftPlan1d(&self.plans[i], n, CUFFT_D2Z, 1)
            if ret != CUFFT_SUCCESS:
                raise Exception("cufft failed: %d" % ret)
            cufftSetStream(self.plans[i], self.streams[i % nstreams]);
            n += 4

    def __dealloc__(self):
        for i in range(self.nside):
            cufftDestroy(self.plans[i])
        for i in range(self.nstreams):
            print self.streams[i]
            #cudaStreamDestroy(self.streams[i])
        free(self.plans)
        free(self.streams)

    def execute(self, size_t indata, size_t outdata):
        cdef int ret
        n = 4
        for i in range(self.nside):
            ret |= cufftExecD2Z(self.plans[i],
                                <cufftDoubleReal*>indata,
                                <cufftDoubleComplex*>outdata)
            indata += n * 8
            outdata += (n // 2 + 1) * 16
            n += 4
            
        if ret != 0:
            raise Exception("Did not succeed: %d" % ret)
        #cudaDeviceSynchronize()

