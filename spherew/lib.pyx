cimport numpy as np

import os
import numpy as np

from butterfly import write_int64, pad128
from butterfly import butterfly_compress, serialize_butterfly_matrix

np.import_array()

cdef extern from "complex.h":
    pass

cdef extern from "fastsht.h":
    ctypedef int int64_t
    
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
                                         int nmaps, int nthreads,
                                         double *input,
                                         double *output,
                                         int ordering,
                                         char *resourcename)

    void fastsht_destroy_plan(fastsht_plan plan)
    void fastsht_execute(fastsht_plan plan)
    void fastsht_configure(char *resource_dir)
    void fastsht_perform_matmul(fastsht_plan plan, bfm_index_t m, int odd)
    void fastsht_legendre_transform(fastsht_plan plan, int mstart, int mstop, int mstride)
    void fastsht_assemble_rings(fastsht_plan plan,
                                int ms_len, int *ms,
                                double complex **q_list)
    void fastsht_disable_phase_shifting(fastsht_plan plan)

cdef extern from "fastsht_private.h":
    ctypedef struct fastsht_grid_info:
        double *phi0s
        bfm_index_t *ring_offsets
        bfm_index_t nrings

    void fastsht_perform_backward_ffts(fastsht_plan plan, int ring_start, int ring_end)
    fastsht_grid_info* fastsht_create_healpix_grid_info(int Nside)
    void fastsht_free_grid_info(fastsht_grid_info *info)

cdef extern from "legendre_transform.h":
    void fastsht_associated_legendre_transform(size_t nx, size_t nl,
                                               size_t nvecs,
                                               size_t *k_start, 
                                               double *a_l,
                                               double *y,
                                               double *x_squared, 
                                               double *c_and_cinv_and_d,
                                               double *P, double *Pp1)
    void fastsht_associated_legendre_transform_sse(size_t nx, size_t nl,
                                                   size_t nvecs,
                                                   size_t *k_start, 
                                                   double *a_l,
                                                   double *y,
                                                   double *x_squared, 
                                                   double *c_and_cinv_and_d,
                                                   double *P, double *Pp1)
    
    void fastsht_associated_legendre_transform_auxdata(
        size_t m, size_t lmin, size_t nk,
        double *auxdata)

_configured = False

cdef class ShtPlan:
    cdef fastsht_plan plan
    cdef readonly object input, output
    cdef public int Nside, lmax
    
    def __cinit__(self, int Nside, int lmax, int mmax,
                  np.ndarray[double complex, ndim=2, mode='c'] input,
                  np.ndarray[double, ndim=2, mode='c'] output,
                  ordering, phase_shifts=True, bytes matrix_data_filename=None,
                  nthreads=0):
        global _configured
        cdef int flags
        if ordering == 'mmajor':
            flags = FASTSHT_MMAJOR
        else:
            raise ValueError("Invalid ordering: %s" % ordering)
        self.input = input
        self.output = output
        if not (input.shape[1] == output.shape[1]):
            raise ValueError("Nonconforming arrays")
        if output.shape[0] != 12 * Nside * Nside:
            raise ValueError("Output must have shape (Npix, nmaps), has %r" %
                             (<object>output).shape)

        if not _configured:
            fastsht_configure(os.environ['SHTRESOURCES'])
            _configured = True
        
        self.plan = fastsht_plan_to_healpix(Nside, lmax, mmax, input.shape[1], nthreads,
                                            <double*>input.data, <double*>output.data,
                                            flags,
                                            NULL if matrix_data_filename is None
                                            else <char*>matrix_data_filename)
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

    def assemble_rings(self, int m,
                       np.ndarray[double complex, ndim=2, mode='c'] q_even,
                       np.ndarray[double complex, ndim=2, mode='c'] q_odd):
        cdef double complex *q_list[2]
        if not (q_even.shape[0] == q_odd.shape[0] == 2 * self.Nside):
            raise ValueError("Invalid array length")
        if  not (q_even.shape[1] == q_odd.shape[1] == self.output.shape[1]):
            raise ValueError("Does not conform with nmaps")
        q_list[0] = <double complex*>q_even.data
        q_list[1] = <double complex*>q_odd.data
        fastsht_assemble_rings(self.plan, 1, &m, q_list)

        

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
    


def associated_legendre_transform(int m, int lmin,
                                  np.ndarray[np.int64_t, ndim=1, mode='c'] k_start,
                                  np.ndarray[double, ndim=2, mode='c'] a,
                                  np.ndarray[double, ndim=2, mode='c'] y,
                                  np.ndarray[double, ndim=1, mode='c'] x_squared,
                                  np.ndarray[double, ndim=1, mode='c'] P,
                                  np.ndarray[double, ndim=1, mode='c'] Pp1,
                                  int repeat=1, use_sse=False):
    cdef size_t nx, nk, nvecs
    cdef Py_ssize_t i, k
    
    nx = x_squared.shape[0]
    if not nx == k_start.shape[0] == P.shape[0] == Pp1.shape[0]:
        raise ValueError("nonconforming arrays")
    nk = a.shape[0]
    nvecs = a.shape[1]
    if not nvecs == y.shape[1]:
        raise ValueError("nonconforming arrays")

    # Pack the auxiliary data here, just to keep testcases and benchmarks
    # from having to change when internals change.
    cdef np.ndarray[double, mode='c'] auxdata = np.empty(3 * (nk - 2))
    fastsht_associated_legendre_transform_auxdata(m, lmin, nk, <double*>auxdata.data)

    if use_sse:
        for i in range(repeat):
            fastsht_associated_legendre_transform_sse(
                nx, nk, nvecs,
                <size_t*>k_start.data,
                <double*>a.data,
                <double*>y.data,
                <double*>x_squared.data,
                <double*>auxdata.data,
                <double*>P.data,
                <double*>Pp1.data)
    else:
        for i in range(repeat):
            fastsht_associated_legendre_transform(
                nx, nk, nvecs,
                <size_t*>k_start.data,
                <double*>a.data,
                <double*>y.data,
                <double*>x_squared.data,
                <double*>auxdata.data,
                <double*>P.data,
                <double*>Pp1.data)
            

class NullLogger(object):
    def info(self, msg):
        pass

null_logger = NullLogger()

def compute_resources_for_m(stream, m, odd, lmax, Nside, chunk_size,
                            eps, logger=null_logger):
    """
    Writes the parts of the precomputed data that corresponds to
    the m given to stream. Present only to allow easy distribution
    of computing between processes. See compute_resources.
    """
    # TODO: Computes Legendre matrix twice, for even and odd case!
    from .healpix import get_ring_thetas
    from .legendre import compute_normalized_associated_legendre

    # Compute & compress matrix
    thetas = get_ring_thetas(Nside, positive_only=True)
    Lambda = compute_normalized_associated_legendre(m, thetas, lmax,
                                                    epsilon=1e-30)
    Lambda = Lambda.T
    Lambda_subset = Lambda[odd::2, :]
    tree = butterfly_compress(Lambda_subset, chunk_size=chunk_size, eps=eps)
    logger.info('Computed m=%d of %d: %s' % (m, lmax, tree.get_stats()))
    # Serialize the butterfly tree to the stream
    serialize_butterfly_matrix(tree, Lambda_subset, stream=stream)

def compute_resources(stream, lmax, mmax, Nside, chunk_size=64, eps=1e-13,
                      logger=null_logger, compute_matrix_func=compute_resources_for_m):
    write_int64(stream, lmax)
    write_int64(stream, mmax)
    write_int64(stream, Nside)
    header_pos = stream.tell()
    for i in range(4 * (mmax + 1)):
        write_int64(stream, 0)
    for m in range(0, mmax + 1):
        for odd in [0, 1]:
            pad128(stream)
            start_pos = stream.tell()
            compute_matrix_func(stream, m, odd, lmax, Nside, chunk_size, eps, logger)
            end_pos = stream.tell()
            stream.seek(header_pos + (4 * m + 2 * odd) * sizeof(int64_t))
            write_int64(stream, start_pos)
            write_int64(stream, end_pos - start_pos)
            stream.seek(end_pos)
        
    

