from __future__ import division
cimport numpy as np

import os
import numpy as np
from io import BytesIO

from concurrent.futures import ProcessPoolExecutor

from butterfly import butterfly_compress, serialize_butterfly_matrix
from utils import FakeExecutor
from .legendre import compute_normalized_associated_legendre
from .healpix import get_ring_thetas
from .streamutils import write_int64, pad128, write_array, write_aligned_array

np.import_array()

def assert_aligned(np.ndarray x):
    assert <size_t>x.data % 16 == 0

cdef extern from "complex.h":
    pass

cdef extern from "wavemoth_private.h":

    ctypedef struct m_resource_t:
        int m
    
    ctypedef struct wavemoth_node_plan_t:
        double *work_q
        int nm
        m_resource_t *m_resources
        

    cdef struct _wavemoth_plan:
        int type_ "type"
        int lmax, mmax
        int nthreads
        int nnodes
        int nmaps
        double complex *output, *input
        double *work
        int Nside
        wavemoth_node_plan_t **node_plans

    ctypedef _wavemoth_plan *wavemoth_plan

    ctypedef int bfm_index_t
    
    ctypedef struct wavemoth_grid_info:
        double *phi0s
        bfm_index_t *ring_offsets
        bfm_index_t nrings

    void wavemoth_perform_backward_ffts(wavemoth_plan plan)
    wavemoth_grid_info* wavemoth_create_healpix_grid_info(int Nside)
    void wavemoth_free_grid_info(wavemoth_grid_info *info)

cdef extern from "wavemoth.h":
    ctypedef int int64_t
    
    cdef enum:
        WAVEMOTH_MMAJOR
        WAVEMOTH_MEASURE
        WAVEMOTH_ESTIMATE
        

    wavemoth_plan wavemoth_plan_to_healpix(int Nside, int lmax, int mmax,
                                         int nmaps, int nthreads,
                                         double *input,
                                         double *output,
                                         int ordering,
                                         unsigned flags,
                                         char *resourcename)

    void wavemoth_destroy_plan(wavemoth_plan plan)
    void wavemoth_execute(wavemoth_plan plan)
    void wavemoth_configure(char *resource_dir)
    void wavemoth_perform_matmul(wavemoth_plan plan, bfm_index_t m, int odd)
    void wavemoth_perform_legendre_transforms(wavemoth_plan plan)
    void wavemoth_disable_phase_shifting(wavemoth_plan plan)

cdef extern from "legendre_transform.h":
    void wavemoth_legendre_transform(size_t nx, size_t nl,
                                    size_t nvecs,
                                    double *a_l,
                                    double *y,
                                    double *x_squared, 
                                    double *c_and_cinv_and_d,
                                    double *P, double *Pp1)
    void wavemoth_legendre_transform_sse(size_t nx, size_t nl,
                                        size_t nvecs,
                                        double *a_l,
                                        double *y,
                                        double *x_squared, 
                                        double *c_and_cinv_and_d,
                                        double *P, double *Pp1,
                                        char *work)
    
    size_t wavemoth_legendre_transform_sse_query_work(size_t nvecs)
    
    void wavemoth_legendre_transform_auxdata(
        size_t m, size_t lmin, size_t nk,
        double *auxdata)

    void wavemoth_legendre_transform_pack(size_t nk, size_t nvecs, double *input,
                                         double *output)

    cdef size_t LEGENDRE_TRANSFORM_WORK_SIZE

_configured = False

cdef class ShtPlan:
    cdef wavemoth_plan plan
    cdef readonly object input, output
    cdef public int Nside, lmax
    
    def __cinit__(self, int Nside, int lmax, int mmax,
                  np.ndarray[double complex, ndim=2, mode='c'] input,
                  np.ndarray[double, ndim=2, mode='c'] output,
                  ordering, phase_shifts=True, bytes matrix_data_filename=None,
                  nthreads=1):
        global _configured
        cdef int flags
        if ordering == 'mmajor':
            flags = WAVEMOTH_MMAJOR
        else:
            raise ValueError("Invalid ordering: %s" % ordering)
        self.input = input
        self.output = output
        if not (input.shape[1] == output.shape[1]):
            raise ValueError("Nonconforming arrays")
        if output.shape[0] != 12 * Nside * Nside:
            raise ValueError("Output must have shape (Npix, nmaps), has %r" %
                             (<object>output).shape)

        if not _configured and matrix_data_filename is None:
            wavemoth_configure(os.environ['SHTRESOURCES'])
            _configured = True
        
        self.plan = wavemoth_plan_to_healpix(Nside, lmax, mmax, input.shape[1], nthreads,
                                            <double*>input.data, <double*>output.data,
                                            flags,
                                            WAVEMOTH_ESTIMATE,
                                            NULL if matrix_data_filename is None
                                            else <char*>matrix_data_filename)
        if self.plan == NULL:
            raise Exception("Plan creation failed")
        self.Nside = Nside
        self.lmax = lmax
        if not phase_shifts:
            wavemoth_disable_phase_shifting(self.plan)

    def __dealloc__(self):
        if self.plan != NULL:
            wavemoth_destroy_plan(self.plan)

    def get_work(self):
        """
        For testing purposes, copy all thread-specific work buffers together into
        a single NumPy array.
        """
        cdef np.ndarray[double complex, ndim=4, mode='c'] work
        cdef double complex *work_slice
        cdef Py_ssize_t im, m, j, odd, re_i, imap, nmaps, Nside
        nmaps = self.plan.nmaps
        nrings = 2 * self.Nside
        work = np.empty((self.plan.mmax + 1, 2, nrings, nmaps), dtype=np.complex128)
        for inode in range(self.plan.nnodes):
            for im in range(self.plan.node_plans[inode].nm):
                work_slice = <double complex*>self.plan.node_plans[inode].work_q
                m = self.plan.node_plans[inode].m_resources[im].m
                for odd in range(2):
                    for j in range(nrings):
                        for imap in range(nmaps):
                            work[m, odd, j, imap] = \
                                    work_slice[(2 * im + odd) * (nrings * nmaps) +
                                               j * nmaps + imap]
        return work
            

    def execute(self, int repeat=1):
        for i in range(repeat):
            wavemoth_execute(self.plan)
        return self.output

    def perform_backward_ffts(self):
        wavemoth_perform_backward_ffts(self.plan)

    def perform_legendre_transform(self, repeat=1):
        cdef int k
        for k in range(repeat):
            wavemoth_perform_legendre_transforms(self.plan)

        

#    def perform_matmul(self, bfm_index_t m, int odd):
#        cdef int n = (self.plan.lmax - m) / 2, i
#        cdef np.ndarray[double complex] out = np.zeros(n, np.complex128)
#        wavemoth_perform_matmul(self.plan, m, odd)
#        return out

def _get_healpix_phi0s(Nside):
    " Expose wavemoth_create_healpix_grid_info for unit tests. "

    cdef wavemoth_grid_info *info = wavemoth_create_healpix_grid_info(Nside)
    try:
        phi0s = np.zeros(info.nrings)
        for i in range(info.nrings):
            phi0s[i] = info.phi0s[i]
    finally:
        wavemoth_free_grid_info(info)
    return phi0s
    



_LEGENDRE_TRANSFORM_WORK_SIZE = LEGENDRE_TRANSFORM_WORK_SIZE # for test use

def legendre_transform(int m, int lmin,
                       np.ndarray[double, ndim=2, mode='c'] a,
                       np.ndarray[double, ndim=2, mode='c'] y,
                       np.ndarray[double, ndim=1, mode='c'] x_squared,
                       np.ndarray[double, ndim=1, mode='c'] P,
                       np.ndarray[double, ndim=1, mode='c'] Pp1,
                       int repeat=1, use_sse=False,
                       np.ndarray[double, ndim=1, mode='c'] auxdata=None):
    cdef size_t nx, nk, nvecs
    cdef Py_ssize_t i, k
    nx = x_squared.shape[0]
    if not nx == P.shape[0] == Pp1.shape[0]:
        raise ValueError("nonconforming arrays")
    nk = a.shape[0]
    nvecs = a.shape[1]
    if not nvecs == y.shape[1]:
        raise ValueError("nonconforming arrays")

    # Copy a into buffer twice as big
    cdef np.ndarray abig = np.zeros((2 * a.shape[0], a.shape[1]))
    abig[0::2, :] = a

    if auxdata is None:
        auxdata = legendre_transform_auxdata(m, lmin, nk)
    elif auxdata.shape[0] != 3 * (nk - 2):
        raise ValueError("auxdata.shape[0] != 3 * (nk - 2)")

    cdef char *work = NULL
    cdef np.ndarray work_array, a_buf

    cdef size_t work_size

    if nvecs % 2 != 0:
        raise ValueError("nvecs not divisible by 2")
    if use_sse:
        a_buf = np.zeros((a.shape[0], a.shape[1])) * np.nan
        wavemoth_legendre_transform_pack(nk, nvecs, <double*>abig.data, <double*>a_buf.data)

        work_size = wavemoth_legendre_transform_sse_query_work(nvecs);
        work_array = np.ones(work_size, dtype=np.int8) * np.nan
        work = work_array.data
        for i in range(repeat):
            wavemoth_legendre_transform_sse(
                nx, nk, nvecs,
                <double*>a_buf.data,
                <double*>y.data,
                <double*>x_squared.data,
                <double*>auxdata.data,
                <double*>P.data,
                <double*>Pp1.data,
                work)
    else:
        for i in range(repeat):
            wavemoth_legendre_transform(
                nx, nk, nvecs,
                <double*>abig.data,
                <double*>y.data,
                <double*>x_squared.data,
                <double*>auxdata.data,
                <double*>P.data,
                <double*>Pp1.data)

def legendre_transform_auxdata(size_t m, size_t lmin, size_t nk):
    cdef np.ndarray[double, mode='c'] out
    if nk < 3:
        return np.zeros(0)
    else:
        out = np.empty(3 * (nk - 2))
        wavemoth_legendre_transform_auxdata(m, lmin, nk, <double*>out.data)
        return out

def first_true(x):
    """ Given a 1D array of booleans x, return the first index that
        is True, or if all is False, the length of the array.
    """
    nz = x.nonzero()[0]
    if nz.shape[0] == 0:
        return x.shape[0]
    else:
        return nz[0]

def stripify(A, include_above=1e-30, exclude_below=1e-80,
             jump_treshold=10, col_divisor=6):
    """
    Partitions the elements of a matrix intro strips. Strips are made
    so that elements smaller than exclude_below are excluded and
    elements larger than include_above are included.

    In addition, a dumb greedy algorithm is made to try to make the
    number of elements as small as possible, without creating too many
    strips. Concretely, when more than `jump_treshold` rows can be
    dropped, a new strip is created, but only so that the number of
    columns in the resulting strip is a multiple of divisor.

    Assumption made: Each column is increasing "fast enough" in magnitude
    (i.e. can start on zero on top but don't decrease towards zero again,
    the bottom is included in all strips).

    Column coordinates will be divisible by col_divisor on best-effort,
    although the other constraints comes first.
    
    Returns
    -------

    List of tuples (row_start, row_stop, col_start, col_stop) describing
    each stripe.
    """
    # Assumption: Each column is increasing in magnitude.
    include_above = np.log2(include_above)
    exclude_below = np.log2(exclude_below)

    M = A.copy('F')
    mask = (M == 0)
    M[mask] = 1
    M = np.log2(np.abs(M))
    M[mask] = exclude_below - 1

    # First: For each column, find the index where it is first above
    # exclude_below (=a), and then where it it is first above
    # include_above (=b)
    col_starts = []
    for col in range(M.shape[1]):
        a = first_true(M[:, col] >= exclude_below)
        b = first_true(M[:, col] >= include_above)
        if np.any(M[a:, col] < exclude_below):
            raise ValueError("Magnitude of column %d not increasing regularly "
                             "enough in magnitude" % col)
        col_starts.append((a, b))

    # Then, take a greedy approach in creating as wide strips as
    # possible. This is not optimal for all inputs; TODO: use dynamic
    # programming + permutations to minimize a cost function
    inf = M.shape[0] + 1
    max_a = -1
    min_b = inf
    start_col = 0
    strips = []
    col_starts.append((inf, inf)) # sentinel
    for col, (a, b) in enumerate(col_starts):
        new_strip_needed = (a > min_b or b < max_a)
        new_strip_wanted = col > 0 and (b - min_b) > jump_treshold
        if new_strip_needed or (new_strip_wanted and col % col_divisor == 0):
            # Emit strip
            strips.append((min_b, M.shape[0], start_col, col))
            start_col = col
            max_a, min_b = a, b
        max_a = max(a, max_a)
        min_b = min(b, min_b)
    return strips

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
        if len(col_indices) == 0 or row_start == row_stop:
            # Zero case
            pad128(stream)
            write_int64(stream, 0)
            write_int64(stream, 0)
            return

        # Use Legendre transform code
        
        # Our task is to find rows that are suitable to start the recursion
        # with -- since some of the blocks will contain data that is not representable
        # in the exponent of double precision floats.

        # First compute using normal routine up to the first two rows, making
        # sure to not truncate results this time around (eps=0)
        lmin = self.row_to_l(row_start)
        lmax = self.row_to_l(row_stop - 1)
        thetas = self.thetas[col_indices]
        x_squared = self.xs[col_indices]**2
        Lambda = compute_normalized_associated_legendre(self.m, thetas, lmax,
                                                        epsilon=1e-300).T[lmin - self.m::2, :]
        strips = stripify(Lambda, include_above=1e-30, exclude_below=1e-250,
                          jump_treshold=10, col_divisor=6)
        # Adjust row_start in order to avoid saving unecesarry aux_data
        min_rstart = 2**63
        for rstart, rstop, cstart, cstop in strips:
            min_rstart = min(rstart, min_rstart)
        row_start += min_rstart
        lmin += 2 * min_rstart
        Lambda = Lambda[min_rstart:, :]
        strips = [(rstart - min_rstart, rstop - min_rstart, cstart, cstop)
                  for rstart, rstop, cstart, cstop in strips]

        # Save data shared for all strips
        assert row_stop >= row_start
        pad128(stream)
        write_int64(stream, row_start)
        write_int64(stream, row_stop)
        if row_stop - row_start <= 4:
            # Early return -- use dgemm (since auxdata etc. is not defined if
            # we don't have enough rows, this was simpler).
            write_aligned_array(stream, Lambda.copy('F'))
            return
            
        write_int64(stream, len(strips))
        auxdata = legendre_transform_auxdata(self.m, lmin, row_stop - row_start)
        assert auxdata.shape[0] == 3 * (row_stop - row_start - 2)
        write_aligned_array(stream, auxdata)

        first = True
        for rstart, rstop, cstart, cstop in strips:
            assert rstop == Lambda.shape[0]
            if first:
                assert cstart == 0
                first = False
            else:
                assert cstart == prev_cstop
            prev_cstop = cstop

            write_int64(stream, rstart)
            write_int64(stream, cstop)
            
            if (rstop - rstart <= 4):
                # Use dgemm for this single chunk
                write_aligned_array(stream, Lambda[rstart:rstop, cstart:cstop].copy('F'))
            else:
                L0 = Lambda[rstart, cstart:cstop].copy()
                L2 = Lambda[rstart + 1, cstart:cstop].copy()
                write_aligned_array(stream, x_squared[cstart:cstop])
                write_aligned_array(stream, L0)
                write_aligned_array(stream, L2)

                # Check:
                # Use the Legendre-transform implementation to compute the last row
                # of Lambda from the first two, to proove things are numerically
                # stable for these starting values.
                a = np.zeros((rstop - rstart, 2))
                a[-1,:] = 1
                y = np.zeros((cstop - cstart, 2)) * np.nan
                legendre_transform(self.m, lmin + 2 * rstart, a, y,
                                   x_squared[cstart:cstop],
                                   L0, L2, use_sse=True)
                y = y[:, 0]
                err = np.linalg.norm(y - Lambda[-1, cstart:cstop]) / np.linalg.norm(y)
                if err > 1e-9:
                    print err
                    raise Exception("Appears to have hit a numerically unstable case, "
                                    "should not happen")
        assert cstop == Lambda.shape[1]


def residual_flop_func(m, n):
    return m * n * (5/2 + 2) * 0.05

def _compute_matrix(resource_computer, m, odd, termination_filename):
    stream = BytesIO()
    if os.path.exists(termination_filename):
        return
    try:
        resource_computer.compute_matrix(stream, m, odd)
    except KeyboardInterrupt:
        with file(termination_filename, 'w') as f:
            f.write(os.getpid())
        return
    except:
        # Print unpickled traceback so that we can see what really
        # went wrong...
        import traceback
        traceback.print_exc()
        with file(termination_filename, 'w') as f:
            f.write(str(os.getpid()))
        raise
    return stream.getvalue()

class ResourceComputer:
    def __init__(self, Nside, lmax, mmax, chunk_size, eps, memop_cost, logger=null_logger):
        self.Nside, self.lmax, self.mmax, self.chunk_size, self.eps, self.memop_cost, self.logger = (
            Nside, lmax, mmax, chunk_size, eps, memop_cost, logger)
        assert lmax == mmax, 'Other cases not tested yet'

    def residual_cost(self, m, n):
        return m * n * (5. / 2. + 1) / self.memop_cost

    def compute_matrix(self, stream, m, odd):
        """
        Writes the parts of the precomputed data that corresponds to
        the m given to stream.
        """
        # Compute & compress matrix
        provider = LegendreMatrixProvider(m, odd, self.Nside)
        nk = (self.lmax - m - odd) // 2 + 1
        tree = butterfly_compress(provider, shape=(nk, provider.ncols_full_matrix),
                                  chunk_size=self.chunk_size, eps=self.eps)
        # Drop levels of compression until residual size is 70% or more
        depth = tree.get_max_depth()
        costs = np.zeros(depth + 1)
        for level in range(depth + 1):
            total, ip, res = tree.get_stats(level, self.residual_cost)
            costs[level] = ip + res
        best_level = costs.argmin()
        self.logger.info('Computed m=%d of %d, level=%d: %s' % (m, self.lmax, best_level,
                                                                tree.format_stats(
                                                                    best_level)))
        # Serialize the butterfly tree to the stream
        serialize_butterfly_matrix(tree, provider, num_levels=best_level, stream=stream)
        return stream

    def init_scheduler(self, max_workers):
        if max_workers == 1:
            return FakeExecutor()
        else:
            # This keeps it all in memory, which speeds up unit testing, but
            # requires lots of memory. Subclasses can override this.
            return ProcessPoolExecutor(max_workers=max_workers)

    def compute(self, stream, max_workers=1):
        proc = self.init_scheduler(max_workers)
        write_int64(stream, self.lmax)
        write_int64(stream, self.mmax)
        write_int64(stream, self.Nside)
        header_pos = stream.tell()
        for i in range(4 * (self.mmax + 1)):
            write_int64(stream, 0)

        import tempfile
        fd, termination_filename = tempfile.mkstemp()
        os.close(fd)
        os.unlink(termination_filename)
        try:
            futures = []
            header_slot_offsets = []
            for m in range(0, self.mmax + 1):
                for odd in [0, 1]:
                    fut = proc.submit(_compute_matrix, self, m, odd, termination_filename)
                    futures.append(fut)
                    header_slot_offsets.append(header_pos + (4 * m + 2 * odd) * sizeof(int64_t))

            for fut, slot in zip(futures, header_slot_offsets):
                pad128(stream)
                start_pos = stream.tell()
                value = fut.result()
                if value is None:
                    # termination, continue until we get to the exception future
                    proc.shutdown()
                    continue
                assert isinstance(value, bytes)
                stream.write(value)
                end_pos = stream.tell()
                stream.seek(slot)
                write_int64(stream, start_pos)
                write_int64(stream, end_pos - start_pos)
                stream.seek(end_pos)
        finally:
            if os.path.exists(termination_filename):
                os.unlink(termination_filename)
        
