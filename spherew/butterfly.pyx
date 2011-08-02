from __future__ import division
from cpython cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free
from libc.string cimport memcpy

import sys
from io import BytesIO
import numpy as np
cimport numpy as np

from interpolative_decomposition import sparse_interpolative_decomposition
from collections import namedtuple

cdef extern from "malloc.h":
    void *memalign(size_t boundary, size_t size)

cdef extern from "butterfly.h":
    ctypedef int bfm_index_t
    ctypedef int int32_t
    ctypedef int int64_t


    ctypedef void (*push_func_t)(double *buf, size_t start, size_t stop,
                                 size_t nvecs, int should_add, void *ctx)
    ctypedef void (*pull_func_t)(double *buf, size_t start, size_t stop,
                                 size_t nvecs, void *ctx)

    int bfm_apply_d(char *matrixdata, double *x, double *y,
                    bfm_index_t nrows, bfm_index_t ncols, bfm_index_t nvecs)

    ctypedef struct bfm_plan

    bfm_plan* bfm_create_plan(size_t k_max, size_t nblocks_max, size_t nvecs)
    void bfm_destroy_plan(bfm_plan *plan)

    int bfm_transpose_apply_d(bfm_plan *plan,
                              char *matrix_data,
                              size_t nrows, 
                              size_t ncols,
                              pull_func_t pull_func,
                              push_func_t push_func,
                              void *caller_ctx)

    

if sizeof(bfm_index_t) == 4:
    index_dtype = np.int32
elif sizeof(bfm_index_t) == 8:
    index_dtype = np.int64
else:
    assert False

class ButterflyMatrixError(Exception):
    pass

cdef class PlanApplicationContext:
    cdef object input_array, output_array

cdef void pull_input_callback(double *buf, size_t start, size_t stop, size_t nvecs,
                              void *_ctx):
    cdef PlanApplicationContext ctx = <PlanApplicationContext>_ctx
    cdef np.ndarray[double, ndim=2] input = ctx.input_array
    cdef size_t i, j, idx = 0
    for i in range(start, stop):
        for j in range(nvecs):
            buf[idx] = input[i, j]
            idx += 1

cdef void push_output_callback(double *buf, size_t start, size_t stop, size_t nvecs,
                               int should_add, void *_ctx):
    cdef PlanApplicationContext ctx = <PlanApplicationContext>_ctx
    cdef np.ndarray[double, ndim=2] output = ctx.output_array
    cdef size_t i, j, idx = 0
    if should_add:
        for i in range(start, stop):
            for j in range(nvecs):
                output[i, j] += buf[idx]
                idx += 1
    else:
        for i in range(start, stop):
            for j in range(nvecs):
                output[i, j] = buf[idx]
                idx += 1

cdef class ButterflyPlan:
    cdef bfm_plan *plan
    cdef size_t nvecs
    
    def __cinit__(self, k_max, nblocks_max, nvecs):
        self.plan = bfm_create_plan(k_max, nblocks_max, nvecs)
        self.nvecs = nvecs

    def __dealloc__(self):
        pass
        #bfm_destroy_plan(self.plan)

    def transpose_apply(self, bytes matrix_data, ncols, x):
        cdef PlanApplicationContext ctx = PlanApplicationContext()
        ctx.input_array = np.asarray(x, dtype=np.double)
        ctx.output_array = np.zeros((ncols, self.nvecs))

        cdef char *buf
        cdef bint need_realign
        need_realign = <size_t><char*>matrix_data % 16 != 0
        if need_realign:
            buf = <char*>memalign(16, len(matrix_data))
            memcpy(buf, <char*>matrix_data, len(matrix_data))
        else:
            buf = <char*>matrix_data
        ret = bfm_transpose_apply_d(self.plan, buf, x.shape[0], ncols,
                                    &pull_input_callback, &push_output_callback,
                                    <void*>ctx)
        if need_realign:
            free(buf)
        if ret != 0:
            raise Exception("bfm_transpose_apply_d returned %d" % ret)
        return ctx.output_array
                              

cdef class SerializedMatrix:
    cdef char *buf
    cdef bint owns_data
    cdef object matrixdata
    cdef public int nrows, ncols
    cdef size_t buflen
    
    def __cinit__(self, bytes matrixdata, bfm_index_t nrows, bfm_index_t ncols):
        self.owns_data = <Py_ssize_t><char*>matrixdata % 16 != 0
        if self.owns_data:
            self.buf = <char*>memalign(16, len(matrixdata))
            memcpy(self.buf, <char*>matrixdata, len(matrixdata))
            self.matrixdata = None
        else:
            self.buf = <char*>matrixdata
            self.matrixdata = matrixdata
        self.buflen = len(matrixdata)
        self.nrows = nrows
        self.ncols = ncols

    def __dealloc__(self):
        if self.owns_data:
            free(self.buf)
            self.owns_data = False

    def __reduce__(self):
        matrixdata = PyBytes_FromStringAndSize(self.buf, self.buflen)
        return (SerializedMatrix, (matrixdata, self.nrows, self.ncols))

    def apply(self, vec, out=None, repeats=1):
        vec = np.ascontiguousarray(vec, dtype=np.double)
        orig_ndim = vec.ndim
        if vec.ndim == 1:
            vec = vec.reshape((vec.shape[0], 1))
        elif vec.ndim > 2:
            raise ValueError()
        if vec.shape[0] != self.ncols:
            raise ValueError("Matrix does not conform to vector")
        if out is None:
            out = np.zeros((self.nrows, vec.shape[1]), dtype=np.double)

        cdef np.ndarray[double, ndim=2, mode='c'] vec_ = vec
        cdef np.ndarray[double, ndim=2, mode='c'] out_ = out

        cdef int i
        for i in range(repeats):
            ret = bfm_apply_d(self.buf, <double*>vec_.data,
                              <double*>out_.data, self.nrows, self.ncols, vec.shape[1])
        if ret != 0:
            raise ButterflyMatrixError('Error code %d' % ret)
        if orig_ndim == 1:
            out = out[:, 0]
        return out

    def digest(self):
        import hashlib
        d = hashlib.sha1()
        d.update(self.get_data())
        return d.hexdigest()

    def get_data(self):
        return PyBytes_FromStringAndSize(self.buf, self.buflen)


cdef write_bin(stream, char *buf, Py_ssize_t size):
    stream.write(PyBytes_FromStringAndSize(buf, size))

def write_int32(stream, int32_t i):
    write_bin(stream, <char*>&i, sizeof(i))

def write_int64(stream, int64_t i):
    write_bin(stream, <char*>&i, sizeof(i))

def write_index_t(stream, bfm_index_t i):
    write_bin(stream, <char*>&i, sizeof(i))

def write_array(stream, arr):
    n = stream.write(bytes(arr.data))
    #assert n == np.prod(arr.shape) * arr.itemsize

def pad128(stream):
    i = stream.tell()
    m = i % 16
    if m != 0:
        stream.write(b'\0' * (16 - m))

class IdentityNode(object):
    def __init__(self, n, remainder_blocks=None):
        self.ncols = self.nrows = n
        self.block_heights = [n]
        self.remainder_blocks = remainder_blocks
        self.children = []

    def __repr__(self):
        return '<IdentityNode %dx%d>' % (self.ncols, self.ncols)

    def print_tree(self, indent='', stream=sys.stdout):
        stream.write('%s%r\n' % (indent, self))
                                         
    def write_to_stream(self, stream):
        write_index_t(stream, 0)
        write_index_t(stream, self.ncols)

    def apply(self, x):
        return x

    def size(self):
        return 0

    def count_blocks(self):
        return 0

    def get_multilevel_stats(self):
        raise NotImplementedError()
        return [0], [np.prod(self.remainder_blocks[0].shape)]

def unroll_pairs(pairs):
    result = []
    for a, b in pairs:
        result.append(a)
        result.append(b)
    return result

def format_numbytes(x):
    if x < 1024**2:
        units = 'KB'
        x /= 1024
    elif x < 1024**3:
        units = 'MB'
        x /= 1024**2
    else:
        units = 'GB'
        x /= 1024**3
    return '%.1f %s' % (x, units)

class RootNode(object):
    def __init__(self, D_blocks, S_node):
        self.D_blocks = [np.ascontiguousarray(D, dtype=np.double)
                         for D in D_blocks]
        if len(self.D_blocks) != 2 * len(S_node.blocks):
            raise ValueError("Wrong number of diagonal blocks w.r.t. S_node")
        self.S_node = S_node
        self.nrows = sum(block.shape[0] for block in self.D_blocks)
        self.ncols = S_node.ncols
        for D, ip in zip(self.D_blocks, unroll_pairs(self.S_node.blocks)):
            if D.shape[1] != ip.shape[0]:
                raise ValueError('Nonconforming matrices')

    def write_to_stream(self, stream):
        # We write both D_blocks and the first S_node interleaved
        # HEADER
        write_index_t(stream, len(self.D_blocks) // 2) # order field
        write_index_t(stream, self.nrows)
        write_index_t(stream, self.ncols)
        # ROOT NODE
        write_index_t(stream, len(self.D_blocks))
        block_heights = np.asarray(
            [D.shape[0] for D in self.D_blocks], dtype=index_dtype)
        write_array(stream, block_heights)
        L, R = self.S_node.children
        write_index_t(stream, L.nrows)
        write_index_t(stream, R.nrows)
        write_index_t(stream, L.ncols)
        L.write_to_stream(stream)
        R.write_to_stream(stream)
        
        for D, ip_block in zip(self.D_blocks, unroll_pairs(self.S_node.blocks)):
            write_index_t(stream, D.shape[1])
            ip_block.write_to_stream(stream)
            pad128(stream)
            write_array(stream, D)

    def apply(self, x):
        assert x.ndim == 2
        y = self.S_node.apply(x)
        out = np.empty((self.nrows, x.shape[1]), np.double)
        i_out = 0
        i_y = 0
        for block in self.D_blocks:
            m, n = block.shape
            out[i_out:i_out + m, :] = np.dot(block, y[i_y:i_y + n, :])
            i_out += m
            i_y += n
        return out

    def size(self):
        return (self.S_node.size() +
                sum(np.prod(block.shape) for block in self.D_blocks))

    def get_multilevel_stats(self):
        """
        Returns two arrays:
         - The interpolation block sizes at each level (in # of floats).
         - The size of remainder matrices, if compression was stopped
           at this level.

        The first element in each list corresponds to the level of self.
        """
        return self.S_node.get_multilevel_stats()        

    def get_stats(self):
        # Todo: rename to __repr__ or something...
        if self.nrows == 0 or self.ncols == 0:
            return "empty"
        Plms_size = sum(np.prod(block.shape) for block in self.D_blocks)
        size = self.size()
        dense = self.nrows * self.ncols
        return "%.2f -> %s (%.2f -> %s Plms), blocks=%d+%d" % (
            size / dense,
            format_numbytes(size * 8),
            Plms_size / size,
            format_numbytes(Plms_size * 8),
            len(self.D_blocks),
            self.S_node.count_blocks()
            )

class InterpolationBlock(object):
    def __init__(self, filter, interpolant):
        self.filter = np.ascontiguousarray(filter, dtype=np.int8)
        self.interpolant = np.ascontiguousarray(interpolant, dtype=np.double)
        n = self.filter.shape[0]
        k = n - np.sum(self.filter)
        self.shape = (self.interpolant.shape[0], n)
        if self.interpolant.shape[1] != n - k:
            raise ValueError("interpolant.shape[1] != n - k")
        if self.interpolant.shape[0] != k:
            raise ValueError("interpolant.shape[0] != k")
        
    def write_to_stream(self, stream):
        write_array(stream, self.filter)
        pad128(stream)
        write_array(stream, self.interpolant)

    def apply(self, x):
        if len(self.filter) == 0:
            assert x.shape[0] == 0
            return x
        y = x[self.filter == 0, :]
        y += np.dot(self.interpolant, x[self.filter == 1, :])
        return y

    def as_array(self, out=None):
        "Convert to dense array, for use in tests"
        if out is None:
            out = np.zeros(self.shape)
        elif out.shape != self.shape:
            print out.shape
            print self.shape
            raise ValueError('out has wrong shape')
        k, n = self.shape
        out[:, self.filter == 0] = np.eye(k)
        out[:, self.filter == 1] = self.interpolant
        return out

    def size(self):
        return np.prod(self.interpolant.shape)

class RemainderBlockProvider(object):
    """
    Override this class to plug in matrices that can be
    generated on the fly to butterfly compression.
    """
    def get_block(self, row_start, row_stop, col_indices):
        raise NotImplementedError()

class DenseArrayBlockProvider(RemainderBlockProvider):
    def __init__(self, array):
        self.array = array
        
    def get_block(self, row_start, row_stop, col_indices):
        return self.array[row_start:row_stop, col_indices]

RemainderBlockInfo = namedtuple('RemainderBlockInfo',
                                'row_start row_stop col_indices')

class InnerNode(object):
    # ncols - "virtual" number of columns
    # node_ncols - number of columns of only this node (=sum(nrows of children))
    # nrows - number of rows = sum(block_heights)
    
    def __init__(self, blocks, children, remainder_blocks=None):
        if 2**int(np.log2(len(blocks))) != len(blocks):
            raise ValueError("len(blocks) not a power of 2")
        if len(children) != 2:
            raise ValueError("len(children) != 2")
        L, R = children
        for (T_ip, B_ip), lh, rh in zip(blocks, L.block_heights, R.block_heights):
            if not (T_ip.shape[1] == B_ip.shape[1] == lh + rh):
                print T_ip.shape, B_ip.shape, lh, rh
                raise ValueError("Nonconforming matrices")
        self.blocks = blocks
        self.ncols = sum(child.ncols for child in children)
        self.node_ncols = sum(child.nrows for child in children)
        self.children = children
        self.block_heights = sum([[T_ip.shape[0], B_ip.shape[0]]
                                 for T_ip, B_ip in blocks], [])
        self.nrows = sum(self.block_heights)

        self.remainder_blocks = remainder_blocks
        if remainder_blocks is not None:
            if not isinstance(remainder_blocks, list):
                raise TypeError("expected list for remainder_blocks")
            if len(remainder_blocks) != len(self.block_heights):
                raise ValueError("remainder_blocks has wrong length")
            for rinfo, bh in zip(
                remainder_blocks, self.block_heights):
                if len(rinfo.col_indices) != bh:
                    raise ValueError("nonconforming remainder block specification")

    def __repr__(self):
        return '<InnerNode %dx%d(%d) block_heights=%r>' % (
            self.nrows, self.ncols,
            self.node_ncols, self.block_heights)

    def print_tree(self, indent='', stream=sys.stdout):
        stream.write('%s%r\n' % (indent, self))
        for child in self.children:
            child.print_tree(indent + '  ', stream=stream)

    def write_to_stream(self, stream):
        L, R = self.children
        write_index_t(stream, len(self.block_heights))
        write_array(stream, np.asarray(self.block_heights, dtype=index_dtype))
        write_index_t(stream, L.nrows)
        write_index_t(stream, R.nrows)
        write_index_t(stream, L.ncols)
        L.write_to_stream(stream)
        R.write_to_stream(stream)
        for T_ip, B_ip in self.blocks:
            T_ip.write_to_stream(stream)
            B_ip.write_to_stream(stream)

    def as_array(self, out=None):
        "Convert to dense array, for use in tests"
        if out is None:
            out = np.zeros((self.nrows, self.node_ncols))
        elif out.shape != (self.nrows, self.node_ncols):
            raise ValueError('out has wrong shape')
        i = j = 0
        for T_ip, B_ip in self.blocks:
            # Stack T_ip and B_ip vertically
            for X in [T_ip, B_ip]:
                X.as_array(out[i:i + X.shape[0], j:j + X.shape[1]])
                i += X.shape[0]
            j += T_ip.shape[1]
        return out

    def apply(self, x):
        # z is the vector containing the contiguous result of the 2 children
        # The permutation happens in reading from z in a permuted way; so each
        # butterfly application permutes its input
        # Recurse to children butterflies
        LS, RS = self.children
        assert x.shape[0] == self.ncols
        assert x.shape[0] == LS.ncols + RS.ncols
        z_left = LS.apply(x[:LS.ncols, :])
        z_right = RS.apply(x[LS.ncols:, :])

        # Apply this butterfly, permuting the input as we go
        y = np.empty((self.nrows, x.shape[1]), np.double)
        i_y = 0
        i_l = i_r = 0
        for i_block, (T_ip, B_ip) in enumerate(self.blocks):
            # Merge together input
            lw = LS.block_heights[i_block]
            rw = RS.block_heights[i_block]
            assert T_ip.shape[1] == B_ip.shape[1] == lw + rw
            buf = np.empty((lw + rw, x.shape[1]))
            buf[:lw, :] = z_left[i_l:i_l + lw, :]
            i_l += lw
            buf[lw:, :] = z_right[i_r:i_r + rw, :]
            i_r += rw
            # Do computation
            th = self.block_heights[2 * i_block]
            bh = self.block_heights[2 * i_block + 1]
            assert T_ip.shape[0] == th and B_ip.shape[0] == bh
            y[i_y:i_y + th, :] = T_ip.apply(buf)
            i_y += th
            y[i_y:i_y + bh, :] = B_ip.apply(buf)
            i_y += bh
        return y
        
    def size(self):
        return sum([child.size() for child in self.children] +
                   [A_ip.size() + B_ip.size()
                    for A_ip, B_ip in self.blocks])

    def count_blocks(self):
        return 2 * len(self.blocks) + sum([child.count_blocks() for child in self.children])

    def get_multilevel_stats(self):
        ip_sizes, remainder_sizes = self.children[0].get_multilevel_stats()
        for c in self.children[1:]:
            ip_sizes_2, remainder_sizes_2 = c.get_multilevel_stats()
            for j in range(len(ip_sizes)):
                ip_sizes[j] += ip_sizes_2[j]
                remainder_sizes[j] += remainder_sizes_2[j]

        ip_sizes.append(sum([A_ip.size() + B_ip.size() for A_ip, B_ip in self.blocks]) +
                        ip_sizes[-1] if len(ip_sizes) > 0 else 0)

        rsize = 0
        for b in self.remainder_blocks:
            rsize += b.shape[0] * b.shape[1]
        remainder_sizes.append(rsize)
        return ip_sizes, remainder_sizes

    def remainder_as_array(self, array, out=None):
        m = n = 0
        for row_start, row_stop, col_indices in self.remainder_blocks:
            m += row_stop - row_start
            n += len(col_indices)
        if out is None:
            out = np.zeros((m, n))
        elif out.shape != (m, n):
            raise ValueError()
        i = 0
        j = 0
        for row_start, row_stop, col_indices in self.remainder_blocks:
            R = array[row_start:row_stop, col_indices]
            out[i:i + R.shape[0], j:j + R.shape[1]] = R
            i += R.shape[0]
            j += R.shape[1]
        return out

    def uncompress(self, array):
        """
        Returns a dense array corresponding to the matrix represented by the tree.
        Ironically, requires the array as input in order to reconstruct
        the remainder blocks.

        This is only useful for testing purposes.
        """
        # Construct block-diagonal matrix of remainder blocks
        A = self.remainder_as_array(array)
        matrices = tree_to_matrices(self)
        for M in matrices:
            A = np.dot(A, M)
        return A


def permutations_to_filter(alst, blst):
    filter = np.zeros(len(alst) + len(blst), dtype=np.int8)
    filter[blst] = 1
    return filter

def matrix_interpolative_decomposition(A, eps):
    iden_list, ipol_list, A_ip = sparse_interpolative_decomposition(A, eps)
    filter = permutations_to_filter(iden_list, ipol_list)
    return iden_list, InterpolationBlock(filter, A_ip)

def pick_array_columns(A, indices):
    return A[:, indices] if len(indices) > 0 else A[:, 0:0]

def pick_remainder_block(A, rinfo):
    row_start, row_stop, indices = rinfo
    if len(indices) > 0:
        return A[row_start:row_stop, indices]
    else:
        return np.zeros((row_stop - row_start, 0))

def butterfly_core(A_k_blocks, eps, A, icol=0):
    if len(A_k_blocks) == 1:
        # No compression achieved anyway at this stage when split
        # into odd l/even l
        block = A_k_blocks[0]
        return IdentityNode(block.shape[1], remainder_blocks=[
            RemainderBlockInfo(row_start=0, row_stop=block.shape[0],
                               col_indices=np.arange(icol, icol + block.shape[1]))])

    mid = len(A_k_blocks) // 2
    L_node = butterfly_core(A_k_blocks[:mid], eps, A, icol)
    R_node = butterfly_core(A_k_blocks[mid:], eps, A, icol + L_node.ncols)

    # Compress further
    out_remainders = []
    out_interpolants = []
    for L_remainder, R_remainder in zip(L_node.remainder_blocks,
                                        R_node.remainder_blocks):
        row_start = L_remainder.row_start
        row_stop = L_remainder.row_stop
        assert row_start == R_remainder.row_start
        assert row_stop == R_remainder.row_stop
        L = pick_remainder_block(A, L_remainder)
        R = pick_remainder_block(A, R_remainder)
        LR_col_indices = np.hstack([L_remainder.col_indices,
                                    R_remainder.col_indices])
        # Horizontal join
        LR = np.hstack([L, R])
        # Vertical split & compress
        vmid = LR.shape[0] // 2
        T = LR[:vmid, :]
        B = LR[vmid:, :]

        interpolant_pair = []
        i = row_start
        for X in [T, B]:
            col_indices, interpolant = matrix_interpolative_decomposition(X, eps)
            col_indices = LR_col_indices[col_indices]
            interpolant_pair.append(interpolant)
            out_remainders.append(RemainderBlockInfo(row_start=i,
                                                     row_stop=i + X.shape[0],
                                                     col_indices=col_indices))
            i += X.shape[0]
        out_interpolants.append(interpolant_pair)

    return InnerNode(out_interpolants, (L_node, R_node),
                     out_remainders)

def get_number_of_levels(n, min):
    levels = 0
    while True:
        n //= 2
        if n < min:
            break
        levels += 1
    return levels

def partition_columns(X, numlevels):
    if numlevels == 0:
        return [X]
    else:
        hmid = X.shape[1] // 2
        return (partition_columns(X[:, :hmid], numlevels - 1) +
                partition_columns(X[:, hmid:], numlevels - 1))

def partition_columns_given_width(A, column_width):
    # In addition to splitting up the array, we also need to insert
    # empty columns to match a power of 2. We start by spliting
    # the matrix; divide the residual columns over the last two blocs.
    blocks = []

def pad_with_empty_columns(blocks):
    n = len(blocks)
    target_n = 1
    while n > target_n:
        target_n *= 2
    to_add = target_n - n
    result = []
    for block in blocks:
        result.append(block)
        if to_add > 0:
            result.append(block[:, -1:-1])
            to_add -= 1
    return result

def partition(P, chunk_size):
    n = P.shape[1]
    result = []
    for idx in range(0, n, chunk_size):
        result.append(P[:, idx:idx + chunk_size])
    return result

def butterfly_compress(A, chunk_size, eps=1e-10):
    blocks = partition(A, chunk_size)
    blocks = pad_with_empty_columns(blocks)
    root = butterfly_core(blocks, eps, A)
    return root

def serialize_butterfly_matrix(M):
    data = BytesIO()
    M.write_to_stream(data)
    return SerializedMatrix(data.getvalue(), M.nrows, M.ncols)


def find_max_depth(node):
    if len(node.children) == 0:
        return 0
    else:
        return 1 + max([find_max_depth(child) for child in node.children])

def find_heap_size(node):
    K = find_max_depth(node)
    return 2**(K + 1) - 1

def heapify(node, first_idx=1, idx=1, heap=None):
    if heap is None:
        heap = [None] * find_heap_size(node)
        
    heap[idx - first_idx] = node
    if len(node.children) == 0:
        return heap
    else:
        heapify(node.children[0], first_idx, 2 * idx, heap)
        if len(node.children) == 1:
            heap[2 * idx + 1 - first_idx] = None
        else:
            heapify(node.children[1], first_idx, 2 * idx + 1, heap)
        if len(node.children) > 2:
            raise ValueError('Not a binary tree')

    return heap

def serialize_node(stream, node):
    def serialize_interpolation_block(block):
        write_array(stream, block.filter)
        pad128(stream)
        write_array(stream, block.interpolant)

    if isinstance(node, InnerNode):
        write_index_t(stream, len(node.block_heights))
        write_array(stream, np.asarray(node.block_heights, dtype=index_dtype))
        for T_ip, B_ip in node.blocks:
            serialize_interpolation_block(T_ip)
            serialize_interpolation_block(B_ip)
        
    elif isinstance(node, IdentityNode):
        write_index_t(stream, 0)
        write_index_t(stream, node.nrows)
    else:
        raise AssertionError()
    

def refactored_serializer(root, out=None):
    if out is None:
        out = BytesIO()

    # Only support one root node for now
    first_level_size = 1
    heap_first_index = 1

    heap_size = find_heap_size(root)
    heap = [None] * heap_size
    heapify(root, heap_first_index, 1, heap)
    
    write_int32(out, root.nrows)
    write_int32(out, root.ncols)
    write_int32(out, first_level_size)
    write_int32(out, heap_size)
    write_int32(out, heap_first_index)
    write_int32(out, 0) # padding

    # Output placeholder heap table
    heap_pos = out.tell()
    for node in heap:
        write_int64(out, 0)
    node_offsets = [0] * heap_size

    # Write nodes and record offsets
    for i, node in enumerate(heap):
        pad128(out)
        node_offsets[i] = out.tell()
        serialize_node(out, node)

    # Output actual offsets to heap table
    end_pos = out.tell()
    out.seek(heap_pos)
    for offset in node_offsets:
        write_int64(out, offset)
    out.seek(end_pos)

    return out

def tree_to_matrices(tree):
    """
    For testing and visualization purposes.
    
    Converts a compressed tree representation to a list of
    block-diagonal and permutation matrices stored as dense NumPy
    arrays.
    """
    assert isinstance(tree, InnerNode)
    matrices = []

    # Convert depth-first to breadth-first.
    bfs_tree = []
    heap = heapify(tree)
    heap = heap[:len(heap) // 2] # Drop identity leaf nodes
    idx = 0
    n = 1
    while idx + n <= len(heap):
        bfs_tree.append(heap[idx:idx + n])
        idx += n
        n *= 2

    # Make two matrices out of each level: An interpolation matrix
    # and a permutation matrix.
    for nodes_on_level in bfs_tree:
        nrows = ncols = 0
        children = []
        for node in nodes_on_level:
            nrows += node.nrows
            ncols += node.node_ncols
            children.extend(node.children)
        nrows_children = sum([child.nrows for child in children])

        # Interpolation matrix            
        M = np.zeros((nrows, ncols))
        i = j = 0
        for node in nodes_on_level:
            node.as_array(out=M[i:i + node.nrows, j:j+node.node_ncols])
            i += node.nrows
            j += node.node_ncols
        matrices.append(M)

        # Permutation matrix
        i = jl = jr = 0
        P = np.zeros((ncols, nrows_children))
        for node in nodes_on_level:
            L, R = node.children
            i = i
            jl = jr
            jr += L.nrows
            for lh, rh in zip(L.block_heights, R.block_heights):
                P[i:i + lh, jl:jl + lh] = np.eye(lh)
                i += lh
                jl += lh
                P[i:i + rh, jr:jr + rh] = np.eye(rh)
                i += rh
                jr += rh
        matrices.append(P)
    return matrices

