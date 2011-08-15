from __future__ import division
from cpython cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free
from libc.string cimport memcpy

import sys
from io import BytesIO
import numpy as np
cimport numpy as np

from spherew cimport blas

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
                                 size_t nvecs, char *payload, size_t payload_len, void *ctx)

    int bfm_apply_d(char *matrixdata, double *x, double *y,
                    bfm_index_t nrows, bfm_index_t ncols, bfm_index_t nvecs)

    ctypedef struct bfm_plan

    bfm_plan* bfm_create_plan(size_t k_max, size_t nblocks_max, size_t nvecs)
    void bfm_destroy_plan(bfm_plan *plan)

    int bfm_transpose_apply_d(bfm_plan *plan,
                              char *matrix_data,
                              pull_func_t pull_func,
                              double *target,
                              size_t target_len,
                              void *caller_ctx)
    
    ctypedef struct bfm_matrix_data_info:
        size_t nrows, ncols

    char *bfm_query_matrix_data(char *head, bfm_matrix_data_info *info)

    

if sizeof(bfm_index_t) == 4:
    index_dtype = np.int32
elif sizeof(bfm_index_t) == 8:
    index_dtype = np.int64
else:
    assert False

class ButterflyMatrixError(Exception):
    pass

cdef void pull_input_callback(double *buf, size_t start, size_t stop, size_t nvecs,
                              char *payload, size_t payload_len, void *_ctx):
    (<ButterflyPlan>_ctx).transpose_pull_input(buf, start, stop, nvecs,
                                               payload, payload_len)

cdef class ButterflyPlan:
    cdef bfm_plan *plan
    cdef size_t nvecs
    cdef np.ndarray input_array, output_array
    
    def __cinit__(self, k_max, nblocks_max, nvecs):
        self.plan = bfm_create_plan(k_max, nblocks_max, nvecs)
        self.nvecs = nvecs

    def __dealloc__(self):
        pass
        #bfm_destroy_plan(self.plan)

    def transpose_apply(self, bytes matrix_data, x):
        cdef char *buf
        cdef bint need_realign

        # Read ncols from matrix_data
        cdef bfm_matrix_data_info info
        bfm_query_matrix_data(<char*>matrix_data, &info)
        cdef size_t ncols = info.ncols

        need_realign = False
        try:
            self.input_array = np.asarray(x, dtype=np.double)
            if self.input_array.shape[0] != info.nrows:
                raise ValueError("Invalid shape for x, got %s, wanted %s" % (
                    self.input_array.shape[0], info.nrows))
            output_array = self.output_array = np.zeros((ncols, self.nvecs))
            need_realign = <size_t><char*>matrix_data % 16 != 0
            if need_realign:
                buf = <char*>memalign(16, len(matrix_data))
                memcpy(buf, <char*>matrix_data, len(matrix_data))
            else:
                buf = <char*>matrix_data
            ret = bfm_transpose_apply_d(self.plan, buf,
                                        &pull_input_callback,
                                        <double*>self.output_array.data,
                                        self.output_array.shape[0] * self.output_array.shape[1],
                                        <void*>self)
            if ret != 0:
                raise Exception("bfm_transpose_apply_d returned %d" % ret)
        finally:
            self.input_array = self.output_array = None
            if need_realign:
                free(buf)
        return output_array

    cdef transpose_pull_input(self, double *buf, size_t start, size_t stop, size_t nvecs,
                              char *payload, size_t payload_len):
        cdef np.ndarray[double, ndim=2] input = self.input_array
        cdef size_t i, j, idx = 0
        for i in range(start, stop):
            for j in range(nvecs):
                buf[idx] = input[i, j]
                idx += 1
    

cdef class DenseResidualButterfly(ButterflyPlan):
    cdef transpose_pull_input(self, double *buf, size_t start, size_t stop, size_t nvecs,
                              char *payload, size_t payload_len):
        cdef np.ndarray[double, ndim=2, mode='c'] input = self.input_array
        cdef size_t i, j, v, k, idx = 0
        cdef double s
        if <size_t>payload % 16 != 0:
            payload += 16 - <size_t>payload % 16
        cdef size_t row_start = (<int64_t*>payload)[0]
        cdef size_t row_stop = (<int64_t*>payload)[1]
        cdef size_t ncols = stop - start
        payload += sizeof(int64_t) * 2        
        cdef double *A = <double*>payload
        # buf = dot(input.T, A)
        blas.dgemm_ccc_(<double*>input.data + row_start * nvecs,
                        A,
                        buf,
                        nvecs, stop - start, row_stop - row_start, 0.0)

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

def write_aligned_array(stream, arr):
    pad128(stream)
    n = stream.write(bytes(arr.data))

def pad128(stream):
    i = stream.tell()
    m = i % 16
    if m != 0:
        stream.write(b'\0' * (16 - m))

class Node(object):
    def format_stats(self, level=None, residual_size_func=int.__mul__):
        uncompressed_size, interpolative_matrices_size, residual_size = self.get_stats(level)
        compressed_size = interpolative_matrices_size + residual_size
        return "%s->%s=%s + %s (compression=%.2f, residual=%.2f)" % (
            format_numbytes(uncompressed_size * 8),
            format_numbytes(compressed_size * 8),
            format_numbytes(interpolative_matrices_size * 8),
            format_numbytes(residual_size * 8),
            compressed_size / uncompressed_size if uncompressed_size != 0 else 1,
            residual_size / compressed_size if uncompressed_size != 0 else 1)

class IdentityNode(Node):
    def __init__(self, n, remainder_blocks=None):
        self.ncols = self.node_ncols = self.node_nrows = n
        self.block_heights = [n]
        if remainder_blocks is None:
            remainder_blocks = [RemainderBlockInfo(0, n, np.asarray([]))]
        self.nrows = remainder_blocks[0].row_stop - remainder_blocks[0].row_start
        self.remainder_blocks = remainder_blocks
        self.children = []

    def get_k_max(self):
        return self.ncols

    def get_nodes_at_level(self, level):
        raise ValueError("level too high")

    def get_nblocks_max(self):
        return 1

    def get_max_depth(self):
        return 0

    def __repr__(self):
        return '<IdentityNode %dx%d>' % (self.ncols, self.ncols)

    def print_tree(self, indent='', stream=sys.stdout):
        stream.write('%s%r\n' % (indent, self))
                                         
    def apply(self, x):
        return x

    def size(self):
        return 0

    def count_blocks(self):
        return 0

    def get_multilevel_stats(self):
        R = self.remainder_blocks[0]
        return [0], [(R.row_stop - R.row_start) * len(R.col_indices)]

    def transpose_apply_interpolations(self, x):
        return x

    def get_stats(self, level=None, residual_size_func=int.__mul__):
        if level not in (0, None):
            raise ValueError("level to high")
        size = residual_size_func(self.nrows, self.ncols)
        return (size, 0, size)

    

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
        
    def apply(self, x):
        if len(self.filter) == 0:
            assert x.shape[0] == 0
            return x
        y = x[self.filter == 0, :]
        y += np.dot(self.interpolant, x[self.filter == 1, :])
        return y

    def transpose_apply(self, x):
        if len(self.filter) == 0:
            assert x.shape[0] == 0
            return x
        k, n = self.shape
        y = np.empty((n, x.shape[1]))
        y[self.filter == 0, :] = x
        y[self.filter == 1, :] = np.dot(self.interpolant.T, x)
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
        return np.prod(self.interpolant.shape) + len(self.filter)

RemainderBlockInfo = namedtuple('RemainderBlockInfo',
                                'row_start row_stop col_indices')

class InnerNode(Node):
    
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
        self.node_ncols = sum(child.node_nrows for child in children)
        self.children = children
        self.block_heights = sum([[T_ip.shape[0], B_ip.shape[0]]
                                 for T_ip, B_ip in blocks], [])
        self.node_nrows = sum(self.block_heights)

        self.nrows = 0
        if remainder_blocks is not None:
            if not isinstance(remainder_blocks, list):
                raise TypeError("expected list for remainder_blocks")
            if len(remainder_blocks) != len(self.block_heights):
                raise ValueError("remainder_blocks has wrong length")
            for rinfo, bh in zip(
                remainder_blocks, self.block_heights):
                if len(rinfo.col_indices) != bh:
                    raise ValueError("nonconforming remainder block specification")
                self.nrows += rinfo.row_stop - rinfo.row_start
        else:
            self.nrows = self.node_nrows
            remainder_blocks = [RemainderBlockInfo(0, n, np.asarray([])) for n in
                                self.block_heights]
        self.remainder_blocks = remainder_blocks

    def get_max_depth(self):
        return max([x.get_max_depth() for x in self.children]) + 1
            
    def get_k_max(self):
        my_max = max(self.block_heights)
        return max([my_max] + [c.get_k_max() for c in self.children])

    def get_nblocks_max(self):
        return len(self.block_heights)

    def get_nodes_at_level(self, level):
        if level == 0:
            return [self]
        else:
            return sum([child.get_nodes_at_level(level - 1) for child in self.children], [])

    def __repr__(self):
        return '<InnerNode %dx%d|%dx%d block_heights=%r>' % (
            self.nrows, self.ncols, self.node_nrows, self.node_ncols, self.block_heights)

    def print_tree(self, indent='', stream=sys.stdout):
        stream.write('%s%r\n' % (indent, self))
        for child in self.children:
            child.print_tree(indent + '  ', stream=stream)

    def as_array(self, out=None):
        "Convert to dense array, for use in tests"
        if out is None:
            out = np.zeros((self.node_nrows, self.node_ncols))
        elif out.shape != (self.node_nrows, self.node_ncols):
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

    def transpose_apply_interpolations(self, x):
        L, R = self.children
        z_left = np.empty((L.node_nrows, x.shape[1]))
        z_right = np.empty((R.node_nrows, x.shape[1]))
        i_left = i_right = i_x = 0
        for lw, rw, (T_ip, B_ip) in zip(L.block_heights, R.block_heights, self.blocks):
            assert T_ip.shape[1] == B_ip.shape[1] == lw + rw
            tmp = T_ip.transpose_apply(x[i_x:i_x + T_ip.shape[0], :])
            i_x += T_ip.shape[0]
            tmp += B_ip.transpose_apply(x[i_x:i_x + B_ip.shape[0], :])
            i_x += B_ip.shape[0]
            z_left[i_left:i_left + lw, :] = tmp[:lw, :]
            i_left += lw
            z_right[i_right:i_right + rw, :] = tmp[lw:, :]
            i_right += rw
        out_left = L.transpose_apply_interpolations(z_left)
        out_right = R.transpose_apply_interpolations(z_right)
        return np.vstack([out_left, out_right])

    def transpose_apply(self, x, matrix_provider):
        matrix_provider = as_matrix_provider(matrix_provider)
        # Apply remainder blocks
        y = np.empty((self.nrows, x.shape[1]))
        in_idx = y_idx = 0
        for (row_start, row_stop, col_indices), bh in zip(self.remainder_blocks,
                                                          self.block_heights):
            R = matrix_provider.get_block(row_start, row_stop, col_indices)
            y[y_idx:y_idx + R.shape[1]] = np.dot(R.T, x[in_idx:in_idx + R.shape[0], :])
            in_idx += R.shape[0]
            y_idx += R.shape[1]
        # Apply interpolation node tree
        return self.transpose_apply_interpolations(y)
            
        
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
        for R in self.remainder_blocks:
            rsize += (R.row_stop - R.row_start) * len(R.col_indices)
        remainder_sizes.append(rsize)
        return ip_sizes, remainder_sizes

    def remainder_as_array(self, matrix_provider, out=None):
        matrix_provider = as_matrix_provider(matrix_provider)
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
            R = matrix_provider.get_block(row_start, row_stop, col_indices)
            out[i:i + R.shape[0], j:j + R.shape[1]] = R
            i += R.shape[0]
            j += R.shape[1]
        return out

    def uncompress(self, matrix_provider):
        """
        Returns a dense array corresponding to the matrix represented by the tree.
        matrix_provider will only be queried for parts corresponding to
        remainder blocks.

        This is only useful for testing purposes.
        """
        # Construct block-diagonal matrix of remainder blocks
        matrix_provider = as_matrix_provider(matrix_provider)
        A = self.remainder_as_array(matrix_provider)
        matrices = tree_to_matrices(self)
        for M in matrices:
            A = np.dot(A, M)
        return A

    def get_stats(self, level=None, residual_size_func=int.__mul__):
        """
        Returns (uncompressed_size, interpolative_matrices_size, residual_size),
        all in number of elements.
        """
        depth = self.get_max_depth()
        if level is None:
            level = depth
        skip_levels = depth - level
        # Todo: rename to __repr__ or something...
        if self.nrows == 0 or self.ncols == 0:
            return (0, 0, 0)
        elif level == 0:
            size = residual_size_func(self.nrows, self.ncols)
            return (size, 0, size)
        nodes = self.get_nodes_at_level(skip_levels)
        uncompressed_size = self.nrows * self.ncols
        residual_size = 0
        interpolative_matrices_size = 0
        for node in nodes:
            residual_size += sum([residual_size_func(stop - start, len(cols))
                                  for start, stop, cols in node.remainder_blocks])
            interpolative_matrices_size += node.size()
        return (uncompressed_size, interpolative_matrices_size, residual_size)


def permutations_to_filter(alst, blst):
    filter = np.zeros(len(alst) + len(blst), dtype=np.int8)
    filter[blst] = 1
    return filter

def matrix_interpolative_decomposition(A, eps):
    iden_list, ipol_list, A_ip = sparse_interpolative_decomposition(A, eps)
    filter = permutations_to_filter(iden_list, ipol_list)
    return iden_list, InterpolationBlock(filter, A_ip)

def butterfly_core(col, row, nrows, partition, matrix_provider, eps):
    # partition: list of end-column-index of each column block;
    # partition[-1] == ncols.
    if len(partition) == 1:
        stop = partition[0]
        col_indices = np.arange(col, stop)
        columns_array = matrix_provider.get_block(row, nrows, col_indices)
        return [columns_array], IdentityNode(stop - col, remainder_blocks=[
            RemainderBlockInfo(row_start=row, row_stop=nrows,
                               col_indices=col_indices)])

    mid = len(partition) // 2
    L_k_list, L_node = butterfly_core(col, row, nrows, partition[:mid],
                                      matrix_provider, eps)
    R_k_list, R_node = butterfly_core(col + L_node.ncols, row, nrows, partition[mid:],
                                      matrix_provider, eps)

    # Compress further
    out_remainders = []
    out_interpolants = []
    residual_array_list = []
    for L_remainder, R_remainder, L, R in zip(L_node.remainder_blocks,
                                              R_node.remainder_blocks,
                                              L_k_list,
                                              R_k_list):
        row_start = L_remainder.row_start
        row_stop = L_remainder.row_stop
        assert row_start == R_remainder.row_start
        assert row_stop == R_remainder.row_stop
        # Horizontal join
        LR_col_indices = np.hstack([L_remainder.col_indices,
                                    R_remainder.col_indices])
        LR = np.hstack([L, R])
        # Vertical split & compress
        vmid = LR.shape[0] // 2
        T = LR[:vmid, :]
        B = LR[vmid:, :]

        interpolant_pair = []
        i = row_start
        for X in [T, B]:
            relative_col_indices, interpolant = matrix_interpolative_decomposition(X, eps)
            absolute_col_indices = LR_col_indices[relative_col_indices]
            interpolant_pair.append(interpolant)
            out_remainders.append(RemainderBlockInfo(row_start=i,
                                                     row_stop=i + X.shape[0],
                                                     col_indices=absolute_col_indices))
            if len(relative_col_indices) > 0:
                X_k = X[:, relative_col_indices]
            else:
                X_k = X[:, 0:0]
            X_k = X_k.copy('F') # Avoid holding a reference to the data we're eliminating
            residual_array_list.append(X_k)
            i += X.shape[0]
        out_interpolants.append(interpolant_pair)

    return residual_array_list, InnerNode(out_interpolants, (L_node, R_node),
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

def pad_with_empty_columns(partition):
    n = len(partition)
    target_n = 1
    while n > target_n:
        target_n *= 2
    to_add = target_n - n
    result = []
    for p in partition:
        result.append(p)
        if to_add > 0:
            result.append(p)
            to_add -= 1
    return result

def make_partition(start, stop, chunk_size):
    n = stop - start
    result = []
    for idx in range(chunk_size, n, chunk_size):
        result.append(idx)
    if len(result) == 0 or result[-1] != stop:
        result.append(stop)
    return result

class ArrayProvider(object):
    def __init__(self, array):
        self.array = array

    def get_block(self, row_start, row_stop, col_indices):
        if len(col_indices) > 0:
            return self.array[row_start:row_stop, col_indices]
        else:
            return np.zeros((row_stop - row_start, 0))

    def serialize_block_payload(self, stream, row_start, row_stop, col_indices):
        pad128(stream)
        block = np.asfortranarray(self.get_block(row_start, row_stop, col_indices),
                                  dtype=np.double)
        write_int64(stream, row_start)
        write_int64(stream, row_stop)
        write_array(stream, block)
        
def as_matrix_provider(x):
    if isinstance(x, np.ndarray):
        return ArrayProvider(x)
    elif hasattr(x, 'get_block'):
        return x
    else:
        raise TypeError('Not a matrix provider instance')

def butterfly_compress(matrix_provider, chunk_size, shape=None, eps=1e-10, 
                       col=0, row=0):
    if isinstance(matrix_provider, np.ndarray):
        if shape is None:
            shape = matrix_provider.shape
        matrix_provider = ArrayProvider(matrix_provider)
    elif shape is None:
        raise TypeError('shape must be provided')
        
    partition = make_partition(col, shape[1], chunk_size)
    partition = pad_with_empty_columns(partition)
    residual, root = butterfly_core(col, row, shape[0], partition, matrix_provider, eps)
    return root

def find_heap_size(node, skip_levels=0):
    K = node.get_max_depth()
    return 2**(K + 1) - 2**skip_levels

def _heapify(node, first_idx, idx, heap):
    heap[idx - first_idx] = node
    if len(node.children) == 0:
        return heap
    else:
        _heapify(node.children[0], first_idx, 2 * idx, heap)
        if len(node.children) == 1:
            heap[2 * idx + 1 - first_idx] = None
        else:
            _heapify(node.children[1], first_idx, 2 * idx + 1, heap)
        if len(node.children) > 2:
            raise ValueError('Not a binary tree')

    return heap

def fetch_forest(node, skip_levels):
    if (skip_levels == 0):
        return [node]
    return sum([fetch_forest(child, skip_levels - 1) for child in node.children], [])

def heapify(root, skip_levels=0):
    heap = [None] * find_heap_size(root, skip_levels=skip_levels)
    num_roots = 2**skip_levels
    heap[:num_roots] = fetch_forest(root, skip_levels)
    for i, root in enumerate(heap[:num_roots]):
        if root is not None:
            # The first index is always the same as the number of roots
            _heapify(root, num_roots, num_roots + i, heap)
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
        write_index_t(stream, node.ncols)
    else:
        raise AssertionError()
    

def serialize_butterfly_matrix(root, matrix_provider, num_levels=None, stream=None):
    if stream is None:
        stream = BytesIO()
    start_pos = stream.tell()
    if start_pos % 16 != 0:
        raise ValueError('Please align the stream on a 128-bit boundary')
    matrix_provider = as_matrix_provider(matrix_provider)

    tree_depth = root.get_max_depth()
    if num_levels is None:
        num_levels = tree_depth
    elif num_levels == 0:
        # Construct new noop tree
        nrows, ncols = root.nrows, root.ncols
        root = IdentityNode(ncols, remainder_blocks=[
            RemainderBlockInfo(0, nrows, np.arange(ncols))])
        tree_depth = num_levels = 0

    skip_levels = max(0, tree_depth - num_levels)

    # Only support one root node for now
    root_count = 2**skip_levels
    heap = heapify(root, skip_levels)
    forest = heap[:root_count]

    for r in forest:
        if r.nrows != forest[0].nrows:
            assert False
    k_max = 0 if tree_depth == 0 else max([r.get_k_max() for r in forest])
    nblocks_max = max([r.get_nblocks_max() for r in forest])
    
    write_int32(stream, forest[0].nrows)
    write_int32(stream, root.ncols)
    write_int32(stream, k_max)
    write_int32(stream, nblocks_max)
    write_int64(stream, root.size())
    write_int32(stream, root_count)
    write_int32(stream, len(heap))
    write_int32(stream, root_count)
    write_int32(stream, 0)
    # Output placeholder residual matrix payload table of size first_level_size
    residual_pos = stream.tell()
    for i in range(root_count):
        write_int64(stream, 0)

    # Output placeholder heap table
    heap_pos = stream.tell()
    for node in heap:
        write_int64(stream, 0)
    node_offsets = [0] * len(heap)

    # Now follows residual matrix payloads
    for i, node in enumerate(forest):
        # patch pointer
        pos = stream.tell()
        stream.seek(residual_pos + 8 * i)
        write_int64(stream, pos - start_pos)
        stream.seek(pos)

        # Table of offsets to each R block, of len remainder_blocks + 1(!). The +1
        # is here to allow computing sizes...
        write_int64(stream, len(node.remainder_blocks))
        offsets_pos = stream.tell()
        for _ in range(len(node.remainder_blocks) + 1):
            write_int64(stream, 0)
            R_offsets = [0] * (len(node.remainder_blocks) + 1)
            for idx, (row_start, row_stop, col_indices) in enumerate(node.remainder_blocks):
                R_offsets[idx] = stream.tell()
                matrix_provider.serialize_block_payload(stream, row_start, row_stop, col_indices)
        R_offsets[idx + 1] = stream.tell()
        
        end_pos = stream.tell()
        stream.seek(offsets_pos)
        for offset in R_offsets:
            write_int64(stream, offset - start_pos)
        stream.seek(end_pos)

    # Write nodes and record offsets
    for i, node in enumerate(heap):
        pad128(stream)
        node_offsets[i] = stream.tell()
        serialize_node(stream, node)

    # Output actual offsets to heap table
    end_pos = stream.tell()
    stream.seek(heap_pos)
    for offset in node_offsets:
        write_int64(stream, offset - start_pos)
    stream.seek(end_pos)

    return stream

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
            nrows += node.node_nrows
            ncols += node.node_ncols
            children.extend(node.children)
        nrows_children = sum([child.node_nrows for child in children])

        # Interpolation matrix            
        M = np.zeros((nrows, ncols))
        i = j = 0
        for node in nodes_on_level:
            node.as_array(out=M[i:i + node.node_nrows, j:j+node.node_ncols])
            i += node.node_nrows
            j += node.node_ncols
        matrices.append(M)

        # Permutation matrix
        i = jl = jr = 0
        P = np.zeros((ncols, nrows_children))
        for node in nodes_on_level:
            L, R = node.children
            i = i
            jl = jr
            jr += L.node_nrows
            for lh, rh in zip(L.block_heights, R.block_heights):
                P[i:i + lh, jl:jl + lh] = np.eye(lh)
                i += lh
                jl += lh
                P[i:i + rh, jr:jr + rh] = np.eye(rh)
                i += rh
                jr += rh
        matrices.append(P)
    return matrices

