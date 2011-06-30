cdef extern from "omp.h":
    extern void omp_set_num_threads(int) nogil
    extern int omp_get_num_threads() nogil
    extern int omp_get_max_threads() nogil
    extern int omp_get_thread_num() nogil
    extern int omp_get_num_procs() nogil

    extern int omp_in_parallel() nogil

    extern void omp_set_dynamic(int) nogil
    extern int omp_get_dynamic() nogil

    extern void omp_set_nested(int) nogil
    extern int omp_get_nested() nogil

import contextlib

def set_dynamic(bint flag):
    omp_set_dynamic(flag)

def get_dynamic():
    return omp_get_dynamic()

def get_max_threads():
    return omp_get_max_threads()

def set_num_threads(int nthreads):
    omp_set_num_threads(nthreads)
