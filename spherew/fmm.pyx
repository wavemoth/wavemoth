cimport numpy as np
cimport cython

cdef extern from "fmm1d.h":
    void fastsht_fmm1d(double *x_grid, double *input_x, size_t nx,
                       double *y_grid, double *output_y, size_t ny)

cdef extern from "math.h":
    double exp(double)

cdef extern from "mkl_vml.h":
    cdef enum:
        VML_HA
        VML_LA
        VML_EP
        VML_FTZDAZ_ON
        VML_ERRMODE_STDERR
    int vmlSetMode(int)
    void vdExp(Py_ssize_t n, double *a, double *y)
    

def fmm1d(np.ndarray[double, mode='c'] x_grid,
          np.ndarray[double, mode='c'] input_x,
          np.ndarray[double, mode='c'] y_grid,
          np.ndarray[double, mode='c'] output_y,
          int repeat=1):
    cdef int i
    if x_grid.shape[0] != input_x.shape[0] or y_grid.shape[0] != output_y.shape[0]:
        raise ValueError("Shapes do not conform")
    for i in range(repeat):
        fastsht_fmm1d(<double*>x_grid.data, <double*>input_x.data, x_grid.shape[0],
                      <double*>y_grid.data, <double*>output_y.data, y_grid.shape[0])

                  
@cython.boundscheck(False)
@cython.wraparound(False)
def bench_libc_exp(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    for i in range(repeat):
        for j in range(x.shape[0]):
            out[j] = exp(x[j])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bench_inv(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    for i in range(repeat):
        for j in range(x.shape[0]):
            out[j] = 1.0 / x[j]

@cython.boundscheck(False)
@cython.wraparound(False)
def bench_add(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    for i in range(repeat):
        for j in range(x.shape[0]):
            out[j] = x[j] + x[j]

@cython.boundscheck(False)
@cython.wraparound(False)
def bench_vml_exp(np.ndarray[double, mode='c'] x, np.ndarray[double, mode='c'] out, int repeat):
    cdef int i, j
    vmlSetMode(VML_LA | VML_FTZDAZ_ON | VML_ERRMODE_STDERR)
    for i in range(repeat):
        for j in range(x.shape[0]):
            vdExp(28, <double*>x.data, <double*>out.data)

