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
                            int batch) nogil
    
    cufftResult cufftDestroy(cufftHandle plan)

    cufftResult cufftSetStream(cufftHandle plan,
                               cudaStream_t stream)

    cufftResult cufftExecD2Z(cufftHandle plan, 
                             cufftDoubleReal *idata,
                             cufftDoubleComplex *odata) nogil


## cdef extern from "pthread.h":
##     ctypedef struct pthread_t
##     int pthread_create(pthread_t * thread,
##                        pthread_attr_t * attr,
##                        void *(*start_routine)(void*),
##                        void * arg)
##     int pthread_join(pthread_t thread, void **value_ptr)
    

#from cython.parallel cimport prange, parallel, threadid
from openmp cimport omp_set_num_threads, omp_get_thread_num, omp_get_wtime

from concurrent.futures import ThreadPoolExecutor

cdef check(int retcode):
    if retcode != 0:
        raise Exception("nonzero return code: %d" % retcode)


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
            check(cudaStreamCreate(&self.streams[i]))
            

        cdef int n = 4
        cdef cufftResult ret
        for i in range(nside):
            check(cufftPlan1d(&self.plans[i], n, CUFFT_D2Z, 1))
            #check(cufftSetStream(self.plans[i], self.streams[i % nstreams]))
            n += 4

    def __dealloc__(self):
        for i in range(self.nside):
            cufftDestroy(self.plans[i])
        #for i in range(self.nstreams):
        #    cudaStreamDestroy(self.streams[i])
        free(self.plans)
        free(self.streams)

    def execute(self, size_t indata, size_t outdata):
        cdef int ret
        cdef int n = 4
        cdef int i
        print self.nstreams
        e = ThreadPoolExecutor(max_workers=self.nstreams)
        futures = []
        for i in range(self.nstreams):
            futures.append(e.submit(executor, self, i, indata, outdata))
            #print cufftExecD2Z(self.plans[i],
            #                   <cufftDoubleReal*>indata,
            #                   <cufftDoubleComplex*>outdata)

        print futures
        for fut in futures:
            fut.result()
        cudaDeviceSynchronize()
            
        

def executor(HealpixCuFFTPlan plan, int streamidx, size_t indata, size_t outdata):
    cdef size_t inp, outp
    cdef int i
    cdef int nside = plan.nside
    cdef int nstreams = plan.nstreams
    cdef cufftHandle h
    
    print 'inside executor', plan.plans[0]

    cdef double t0, dt0, dt1
    cdef int n = 8192 - 4
    with nogil:
        for i from streamidx <= i < nside by nstreams:
            inp = indata + i * n * 8
            outp = outdata + i * n * 16
            t0 = omp_get_wtime()
            cufftPlan1d(&h, n, CUFFT_D2Z, 1)
            dt0 += omp_get_wtime() - t0

            t0 = omp_get_wtime()
            cufftExecD2Z(h,
                         <cufftDoubleReal*>inp,
                         <cufftDoubleComplex*>outp)
            dt1 += omp_get_wtime() - t0
    print dt0, dt1
