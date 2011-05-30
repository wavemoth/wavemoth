cimport numpy as np
import numpy as np

cdef extern from "fastsht.h":
    cdef enum:
        FASTSHT_MMAJOR
    
    cdef struct _fastsht_plan:
        pass
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

cdef extern from "fastsht_private.h":
    ctypedef struct fastsht_grid_info:
        double *phi0s
        bfm_index_t *ring_offsets
        bfm_index_t nrings

    void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end)
    fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside)
    void fastsht_free_grid_info(fastsht_grid_info *info)

_precomputation_loaded = False
_precomputation_file = b"/home/dagss/code/spherew/precomputed.dat"

cdef class ShtPlan:
    cdef fastsht_plan plan
    cdef object input, output, work
    
    def __cinit__(self, int Nside, int lmax, int mmax,
                  np.ndarray[double, mode='c'] input,
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
    
