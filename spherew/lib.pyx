from __future__ import division
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
             jump_treshold=10, row_divisor=2, col_divisor=6):
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

    Coordinates will be divisible by row_divisor and col_divisor, except
    at the matrix boundaries.
    
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
        new_strip_wanted = ((b - min_b) > jump_treshold)
        opportunity = (col > 0 and col % col_divisor == 0) or col == M.shape[1]
        if opportunity and (new_strip_needed or new_strip_wanted):
            # Emit strip
            if min_b < M.shape[0]:
                # Ensure that row start is divisible by row_divisor
                min_b -= min_b % row_divisor
            # Check that function is smooth enough to allow this to happen --
            # if row_divisor == col_divisor == 1, this is guaranteed.
            coords = (min_b, M.shape[0], start_col, col)
            r1, r2, c1, c2 = coords
            T = M[:r1, c1:c2]
            B = M[r1:r2:, c1:c2]
            if np.any(T >= include_above) or np.any(B < exclude_below):
                raise ValueError("Funtion too rapidly changing to be able to "
                                 "satisfy row_divisor/col_divisor constraints")
            strips.append(coords)

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
        Lambda = compute_normalized_associated_legendre(self.m, thetas, lmax,
                                                        epsilon=0).T[lmin - self.m::2, :]
        # Then, remove rows on top until we find a row containing an
        # element larger than 1e-30
        i = 0
        while i < Lambda.shape[0] and np.all(np.abs(Lambda[i, :]) < 1e-30):
            i += 1
            lmin += 2
        row_start += i
        Lambda = Lambda[i:, :]

        if row_stop - row_start <= 4:
            # The size of the auxiliary and initialization data is the same
            # as just saving the matrix, so we fall back to matrix multiplication.
            block = np.asfortranarray(Lambda, dtype=np.double)
            pad128(stream)
            write_int64(stream, row_start)
            write_int64(stream, row_stop)
            write_array(stream, block)
            return

        # Now, we just hope that no other elements on the same row are hopelessly
        # small. This appears to be true. If this assumption is violated at some
        # point, we must break up into smaller Legendre transforms, potentially
        # with extra permutations.
        if np.any(np.abs(Lambda) < 1e-150):
            raise NotImplementedError("TODO: Break up legendre transforms into smaller blocks."
                                      "NOTE!: Please make a note of parameters used to trigger this "
                                      "case!!!")
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
        y = y[:, 0]
        err = np.linalg.norm(y - Lambda[-1, :]) / np.linalg.norm(y)
        if err > 1e-8:
            raise Exception("Appears to have hit a numerically unstable case, should not happen")

        auxdata = associated_legendre_transform_auxdata(self.m, lmin, row_stop - row_start)

        pad128(stream)
        write_int64(stream, row_start)
        write_int64(stream, row_stop)
        write_aligned_array(stream, x_squared)
        write_aligned_array(stream, P)
        write_aligned_array(stream, Pp1)
        write_aligned_array(stream, auxdata)

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
        
