cimport numpy as np
import numpy as np

cdef extern from "complex.h":
    pass

cdef extern from "fastsht.h":
    cdef enum:
        FASTSHT_MMAJOR
    
    cdef struct _fastsht_plan:
        int type_ "type"
        int lmax, mmax
        double complex *output, *input, *work, \
                *work_a_l, *work_g_m_roots, *work_g_m_even, *work_g_m_odd
        int Nside

    ctypedef _fastsht_plan *fastsht_plan

    ctypedef int bfm_index_t
    
    fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax,
                                         double *input,
                                         double *output,
                                         double *work,
                                         int ordering)

    void fastsht_destroy_plan(fastsht_plan plan)
    void fastsht_execute(fastsht_plan plan)
    int fastsht_add_precomputation_file(char *filename)
    void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd)

cdef extern from "fastsht_private.h":
    ctypedef struct fastsht_grid_info:
        double *phi0s
        bfm_index_t *ring_offsets
        bfm_index_t nrings

    void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end)
    fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside)
    void fastsht_free_grid_info(fastsht_grid_info *info)
    void fastsht_interpolate_legendre(bfm_index_t n, bfm_index_t odd, bfm_index_t m,
                                      double *roots, double *weights,
                                      double *samples, bfm_index_t nvecs,
                                      double *target_grid, bfm_index_t target_grid_len,
                                      double *target_legendre, double *output)
    

_precomputation_loaded = False
_precomputation_file = b"/home/dagss/code/spherew/precomputed.dat"

cdef class ShtPlan:
    cdef fastsht_plan plan
    cdef readonly object input, output, work
    
    def __cinit__(self, int Nside, int lmax, int mmax,
                  np.ndarray[double complex, mode='c'] input,
                  np.ndarray[double, mode='c'] output,
                  np.ndarray[double, mode='c'] work,
                  ordering):
        global _precomputation_loaded
        cdef int flags
        if ordering == 'mmajor':
            flags = FASTSHT_MMAJOR
        else:
            raise ValueError("Invalid ordering: %s" % ordering)
        self.input = input
        self.output = output
        self.work = work

        if not _precomputation_loaded:
            if fastsht_add_precomputation_file(_precomputation_file) != 0:
                raise RuntimeError()
            _precomputation_loaded = True
        
        self.plan = fastsht_plan_to_healpix(Nside, lmax, mmax,
                                            <double*>input.data, <double*>output.data,
                                            <double*>work.data, flags)

    def __dealloc__(self):
        if self.plan != NULL:
            fastsht_destroy_plan(self.plan)

    def execute(self, int repeat=1):
        for i in range(repeat):
            fastsht_execute(self.plan)

    def perform_backward_ffts(self, int ring_start, int ring_end):
        fastsht_perform_backward_ffts(self.plan, ring_start, ring_end)

    def perform_matmul(self, bfm_index_t m, int odd):
        cdef int n = (self.plan.lmax - m) / 2, i
        cdef np.ndarray[double complex] out = np.zeros(n, np.complex128)
        fastsht_perform_matmul(self.plan, m, odd)
        for i in range(n):
            out[i] = self.plan.work_g_m_roots[i]
        return out

def _get_healpix_phi0s(Nside):
    " Expose fastsht_create_healpix_grid_info for unit tests. "

    cdef fastsht_grid_info *info = fastsht_create_healpix_grid_info(Nside)
    try:
        phi0s = np.zeros(info.nrings)
        for i in range(info.nrings):
            phi0s[i] = info.phi0s[i]
    finally:
        fastsht_free_grid_info(info)
    return phi0s
    
