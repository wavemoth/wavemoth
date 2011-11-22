

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
    

    cufftResult cufftPlan1d(cufftHandle *plan, 
                            int nx, 
                            cufftType type, 
                            int batch)
    
    cufftResult cufftDestroy(cufftHandle plan);

def make_plan(int nx):
    cdef cufftHandle plan
    cufftPlan1d(&plan, nx, CUFFT_R2C, 0)
    
    cufftDestroy(plan)
