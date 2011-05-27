cimport numpy as np

cdef extern from "fastsht.h":
    cdef enum:
        FASTSHT_MMAJOR
    
    cdef struct _fastsht_plan:
        pass
    ctypedef _fastsht_plan *fastsht_plan
    
    fastsht_plan fastsht_plan_to_healpix(int Nside, int lmax, int mmax,
                                         double *input,
                                         double *output,
                                         double *work,
                                         int ordering)

    void fastsht_destroy_plan(fastsht_plan plan)
    void fastsht_execute(fastsht_plan plan)
    void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end)



cdef class ShtPlan:
    cdef fastsht_plan plan
    cdef object input, output, work
    
    def __cinit__(self, int Nside, int lmax, int mmax,
                  np.ndarray[double, mode='c'] input,
                  np.ndarray[double, mode='c'] output,
                  np.ndarray[double, mode='c'] work,
                  ordering):
        cdef int flags
        if ordering == 'mmajor':
            flags = FASTSHT_MMAJOR
        else:
            raise ValueError("Invalid ordering: %s" % ordering)
        self.input = input
        self.output = output
        self.work = work
        
        self.plan = fastsht_plan_to_healpix(Nside, lmax, mmax,
                                            <double*>input.data, <double*>output.data,
                                            <double*>work.data, flags)

    def __dealloc__(self):
        if self.plan != NULL:
            fastsht_destroy_plan(self.plan)

    def execute(self):
        fastsht_execute(self.plan)

    def perform_backward_ffts(self, int ring_start, int ring_end):
        fastsht_perform_backward_ffts(self.plan, ring_start, ring_end)
