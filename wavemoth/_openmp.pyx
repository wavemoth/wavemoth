cdef extern from "omp.h":
    void omp_set_num_threads(int) nogil
    int omp_get_num_threads() nogil
    int omp_get_max_threads() nogil
    int omp_get_thread_num() nogil
    int omp_get_num_procs() nogil

    int omp_in_parallel() nogil

    void omp_set_dynamic(int) nogil
    int omp_get_dynamic() nogil

    void omp_set_nested(int) nogil
    int omp_get_nested() nogil

    double omp_get_wtime() nogil
    

import contextlib

def set_dynamic(bint flag):
    omp_set_dynamic(flag)

def get_dynamic():
    return omp_get_dynamic()

def get_max_threads():
    return omp_get_max_threads()

def set_num_threads(int nthreads):
    omp_set_num_threads(nthreads)

def get_wtime():
    return omp_get_wtime()
