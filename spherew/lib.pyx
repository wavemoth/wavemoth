cimport numpy as np

import os
import numpy as np
from io import BytesIO

from concurrent.futures import ProcessPoolExecutor

from butterfly import write_int64, pad128, write_array, write_aligned_array
from butterfly import butterfly_compress, serialize_butterfly_matrix
from utils import FakeExecutor
from .legendre import compute_normalized_associated_legendre
from .healpix import get_ring_thetas


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
                                               double *a_l,
                                               double *y,
                                               double *x_squared, 
                                               double *c_and_cinv_and_d,
                                               double *P, double *Pp1)
    void fastsht_associated_legendre_transform_sse(size_t nx, size_t nl,
                                                   size_t nvecs,
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
                                  np.ndarray[double, ndim=2, mode='c'] a,
                                  np.ndarray[double, ndim=2, mode='c'] y,
                                  np.ndarray[double, ndim=1, mode='c'] x_squared,
                                  np.ndarray[double, ndim=1, mode='c'] P,
                                  np.ndarray[double, ndim=1, mode='c'] Pp1,
                                  int repeat=1, use_sse=False):
    cdef size_t nx, nk, nvecs
    cdef Py_ssize_t i, k
    
    nx = x_squared.shape[0]
    if not nx == P.shape[0] == Pp1.shape[0]:
        raise ValueError("nonconforming arrays")
    nk = a.shape[0]
    nvecs = a.shape[1]
    if not nvecs == y.shape[1]:
        raise ValueError("nonconforming arrays")

    # Pack the auxiliary data here, just to keep testcases and benchmarks
    # from having to change when internals change.
    cdef np.ndarray[double, mode='c'] auxdata = (
        associated_legendre_transform_auxdata(m, lmin, nk))

    if use_sse:
        for i in range(repeat):
            fastsht_associated_legendre_transform_sse(
                nx, nk, nvecs,
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
                <double*>a.data,
                <double*>y.data,
                <double*>x_squared.data,
                <double*>auxdata.data,
                <double*>P.data,
                <double*>Pp1.data)

def associated_legendre_transform_auxdata(size_t m, size_t lmin, size_t nk):
    cdef np.ndarray[double, mode='c'] out = np.empty(3 * (nk - 2))
    fastsht_associated_legendre_transform_auxdata(m, lmin, nk, <double*>out.data)
    return out

class NullLogger(object):
    def info(self, msg):
        pass

null_logger = NullLogger()

class LegendreMatrixProvider(object):
    def __init__(self, m, odd, Nside):
        self.m, self.odd = m, odd
        self.thetas = get_ring_thetas(Nside, positive_only=True)
        self.xs = np.cos(self.thetas)
        self.ncols_full_matrix = self.xs.shape[0]

    def row_to_l(self, rowidx):
        return self.m + 2 * rowidx + self.odd

    def get_block(self, row_start, row_stop, col_indices):
        if len(col_indices) > 0 and row_stop > row_start:
            thetas = self.thetas[col_indices]
            lmin = self.row_to_l(row_start)
            lmax = self.row_to_l(row_stop - 1)
            Lambda = compute_normalized_associated_legendre(self.m, thetas, lmax,
                                                            epsilon=1e-30)
            return Lambda.T[lmin - self.m::2, :]
        else:
            return np.zeros((row_stop - row_start, len(col_indices)))

    def serialize_block_payload(self, stream, row_start, row_stop, col_indices):
        pad128(stream)
        write_int64(stream, row_start)
        write_int64(stream, row_stop)
        if len(col_indices) == 0 or row_stop == row_start:
            return

        if row_stop - row_start > 2:
            # First compute using normal routine up to the first two rows, making
            # sure to not truncate results (eps=0)
            lmin = self.row_to_l(row_start)
            lmax = self.row_to_l(row_stop - 1)
            thetas = self.thetas[col_indices]
            Lambda = compute_normalized_associated_legendre(self.m, thetas, lmax,
                                                            epsilon=0).T[lmin - self.m::2, :]
            # Ensure that numbers are safely representable as floating point for
            # all thetas. The below would hopefully catch this but this is a more
            # friendly report.
            if np.any(np.abs(Lambda) < 1e-150):
                raise NotImplementedError("TODO: Compression routine permuted columns too much")
            # Use the Legendre-transform implementation to compute the last row
            # of Lambda from the first two, to proove things are numerically
            # stable for these starting values.
            P = Lambda[0, :].copy()
            Pp1 = Lambda[1, :].copy()
            a = np.zeros((Lambda.shape[0], 2))
            a[-1,:] = 1
            y = np.zeros((Lambda.shape[1], 2)) * np.nan
            x_squared = self.xs[col_indices]**2
            associated_legendre_transform(self.m, lmin, a, y, x_squared,
                                          P, Pp1, use_sse=True)
            if np.linalg.norm(y[:, 0] - Lambda[-1, :]) > 1e-14:
                raise Exception("Appears to have hit a numerically unstable case, should not happen")

            auxdata = associated_legendre_transform_auxdata(self.m, lmin, row_stop - row_start)

            write_aligned_array(stream, x_squared)
            write_aligned_array(stream, P)
            write_aligned_array(stream, Pp1)
            write_aligned_array(stream, auxdata)
        else:
            block = np.asfortranarray(self.get_block(row_start, row_stop, col_indices),
                                      dtype=np.double)
            write_array(stream, block)

def compute_resources_for_m(stream, m, odd, lmax, Nside, chunk_size,
                            eps, num_levels, logger=null_logger):
    """
    Writes the parts of the precomputed data that corresponds to
    the m given to stream. Present only to allow easy distribution
    of computing between processes. See compute_resources.
    """
    # TODO: Computes Legendre matrix twice, for even and odd case!

    # Compute & compress matrix
    provider = LegendreMatrixProvider(m, odd, Nside)
    nk = (lmax - m - odd) // 2 + 1
    tree = butterfly_compress(provider, shape=(nk, provider.ncols_full_matrix),
                              chunk_size=chunk_size, eps=eps)
    logger.info('Computed m=%d of %d: %s' % (m, lmax, tree.get_stats()))
    # Serialize the butterfly tree to the stream
    serialize_butterfly_matrix(tree, provider, num_levels=num_levels, stream=stream)
    return stream

def compute_resources(stream, lmax, mmax, Nside, chunk_size=64, eps=1e-13,
                      num_levels=None, max_workers=1,
                      logger=null_logger, compute_matrix_func=compute_resources_for_m):
    if max_workers == 1:
        proc = FakeExecutor()
    else:
        # This keeps it all in memory, which speeds up unit testing, but is
        # useless for large resolutions. TODO: Refactor precomputation into OO classes.
        proc = ProcessPoolExecutor(max_workers=max_workers)

    write_int64(stream, lmax)
    write_int64(stream, mmax)
    write_int64(stream, Nside)
    header_pos = stream.tell()
    for i in range(4 * (mmax + 1)):
        write_int64(stream, 0)
        
    futures = []
    header_slot_offsets = []
    for m in range(0, mmax + 1):
        for odd in [0, 1]:
            substream = BytesIO()
            fut = proc.submit(compute_matrix_func, substream, m, odd, lmax, Nside,
                              chunk_size, eps, num_levels, logger)
            futures.append(fut)
            header_slot_offsets.append(header_pos + (4 * m + 2 * odd) * sizeof(int64_t))

    for fut, slot in zip(futures, header_slot_offsets):
        pad128(stream)
        start_pos = stream.tell()
        stream.write(fut.result().getvalue())
        end_pos = stream.tell()
        stream.seek(slot)
        write_int64(stream, start_pos)
        write_int64(stream, end_pos - start_pos)
        stream.seek(end_pos)


        
        
    
