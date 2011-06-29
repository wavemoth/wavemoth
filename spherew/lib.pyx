cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "complex.h":
    pass

cdef extern from "fastsht.h":
    cdef enum:
        FASTSHT_MMAJOR
    
    cdef struct _fastsht_plan:
        int type_ "type"
        int lmax, mmax
        double complex *output, *input
        int Nside

    ctypedef _fastsht_plan *fastsht_plan

    ctypedef int bfm_index_t
    
    fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax,
                                         int nmaps,
                                         double *input,
                                         double *output,
                                         int ordering,
                                         char *resourcename)

    void fastsht_destroy_plan(fastsht_plan plan)
    void fastsht_execute(fastsht_plan plan)
    void fastsht_configure(char *resource_dir)
    void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd)
    void fastsht_legendre_transform(fastsht_plan plan, int mstart, int mstop, int mstride)
    void fastsht_merge_even_odd_and_transpose(fastsht_plan plan, int m,
                                              double complex *g_m_even,
                                              double complex *g_m_odd)
    void fastsht_disable_phase_shifting(fastsht_plan plan)

cdef extern from "fastsht_private.h":
    ctypedef struct fastsht_grid_info:
        double *phi0s
        bfm_index_t *ring_offsets
        bfm_index_t nrings

    void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end)
    fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside)
    void fastsht_free_grid_info(fastsht_grid_info *info)
    

_configured = False
_resource_dir = b"/home/dagss/code/spherew/resources"

cdef class ShtPlan:
    cdef fastsht_plan plan
    cdef readonly object input, output
    cdef public int Nside, lmax
    
    def __cinit__(self, int Nside, int lmax, int mmax,
                  np.ndarray[double complex, ndim=2, mode='c'] input,
                  np.ndarray[double, ndim=2, mode='c'] output,
                  ordering, phase_shifts=True):
        global _configured
        cdef int flags
        if ordering == 'mmajor':
            flags = FASTSHT_MMAJOR
        else:
            raise ValueError("Invalid ordering: %s" % ordering)
        self.input = input
        self.output = output
        if not (input.shape[1] == output.shape[0]):
            raise ValueError("Nonconforming arrays")

        if not _configured:
            fastsht_configure(_resource_dir)
            _configured = True
        
        self.plan = fastsht_plan_to_healpix(Nside, lmax, mmax, input.shape[1],
                                            <double*>input.data, <double*>output.data,
                                            flags, NULL)
        self.Nside = Nside
        self.lmax = lmax
        if not phase_shifts:
            fastsht_disable_phase_shifting(self.plan)

    def __dealloc__(self):
        if self.plan != NULL:
            fastsht_destroy_plan(self.plan)

    def execute(self, int repeat=1):
        for i in range(repeat):
            fastsht_execute(self.plan)
        return self.output

    def perform_backward_ffts(self, int ring_start, int ring_end):
        fastsht_perform_backward_ffts(self.plan, ring_start, ring_end)

    def perform_legendre_transform(self, mstart, mstop, mstride, int repeat=1):
        cdef int k
        for k in range(repeat):
            fastsht_legendre_transform(self.plan, mstart, mstop, mstride)

    def merge_even_odd_and_transpose(self, int m,
                                     np.ndarray[double complex, ndim=2, mode='c'] g_m_even,
                                     np.ndarray[double complex, ndim=2, mode='c'] g_m_odd):
        if not (g_m_even.shape[0] == g_m_odd.shape[0] == 2 * self.Nside):
            raise ValueError("Invalid array length")
        if  not (g_m_even.shape[1] == g_m_odd.shape[1] == self.output.shape[0]):
            raise ValueError("Does not conform with nmaps")
        fastsht_merge_even_odd_and_transpose(self.plan, m, <double complex*>g_m_even.data,
                                             <double complex*>g_m_odd.data)

        

#    def perform_matmul(self, bfm_index_t m, int odd):
#        cdef int n = (self.plan.lmax - m) / 2, i
#        cdef np.ndarray[double complex] out = np.zeros(n, np.complex128)
#        fastsht_perform_matmul(self.plan, m, odd)
#        return out

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
    


