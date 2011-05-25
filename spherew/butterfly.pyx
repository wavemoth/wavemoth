from cpython cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free
from libc.string cimport memcpy

cdef extern from "stdlib.h":
    void *memalign(size_t boundary, size_t size)

cdef extern from "butterfly.h":
    ctypedef int bfm_index_t
    ctypedef int int32_t

    int bfm_apply_d(char *matrixdata, double *x, double *y,
                    bfm_index_t nrows, bfm_index_t ncols, bfm_index_t nvecs)

import numpy as np
cimport numpy as np
from interpolative_decomposition import sparse_interpolative_decomposition

if sizeof(bfm_index_t) == 4:
    index_dtype = np.int32
elif sizeof(bfm_index_t) == 8:
    index_dtype = np.int64
else:
    assert False

class ButterflyMatrixError(Exception):
    pass

cdef class SerializedMatrix:
    cdef char *buf
    cdef bint owns_data
    cdef object matrixdata
    cdef int nrow, ncol
    
    def __cinit__(self, bytes matrixdata, bfm_index_t nrow, bfm_index_t ncol):
        self.owns_data = <Py_ssize_t><char*>matrixdata % 16 != 0
        if self.owns_data:
            self.buf = <char*>memalign(16, len(matrixdata))
            memcpy(self.buf, <char*>matrixdata, len(matrixdata))
            self.matrixdata = None
        else:
            self.buf = <char*>matrixdata
            self.matrixdata = matrixdata
        self.nrow = nrow
        self.ncol = ncol

    def __dealloc__(self):
        if self.owns_data:
            free(self.buf)
            self.owns_data = False

    def apply(self, vec, out=None):
        vec = np.ascontiguousarray(vec)
        if vec.ndim == 1:
            vec = vec[:, None]
        elif vec.ndim > 2:
            raise ValueError()
        if vec.shape[0] != self.ncol:
            raise ValueError()
        if out is None:
            out = np.zeros((self.nrow, vec.shape[1]), dtype=np.double)

        cdef np.ndarray[double, ndim=2, mode='c'] vec_ = vec
        cdef np.ndarray[double, ndim=2, mode='c'] out_ = out
            
        ret = bfm_apply_d(self.buf, <double*>vec_.data,
                          <double*>out_.data, self.nrow, self.ncol, vec.shape[1])
        if ret != 0:
            raise ButterflyMatrixError('Error code %d' % ret)
        return out


cdef write_bin(stream, char *buf, Py_ssize_t size):
    n = stream.write(PyBytes_FromStringAndSize(buf, size))
    assert n == size

cdef write_int32(stream, int32_t i):
    write_bin(stream, <char*>&i, sizeof(i))

cdef write_index_t(stream, bfm_index_t i):
    write_bin(stream, <char*>&i, sizeof(i))

cdef write_array(stream, arr):
    n = stream.write(bytes(arr.data))
    assert n == np.prod(arr.shape) * arr.itemsize

cdef pad128(stream):
    i = stream.tell()
    m = i % 16
    if m != 0:
        stream.write(b'\0' * (16 - m))

class IdentityNode(object):
    def __init__(self, n):
        self.ncols = self.nrows = n
        self.block_heights = [n]
        
    def write_to_stream(self, stream):
        write_index_t(stream, self.ncols)

    def apply(self, x):
        return x

    def size(self):
        return 0

class RootNode(object):
    def __init__(self, D_blocks, S_node):
        self.D_blocks = D_blocks
        self.S_node = S_node
        self.nrows = sum(block.shape[0] for block in self.D_blocks)

    def write_to_stream(self, stream):
        # We write both D_blocks and the first S_node interleaved
        1/0

    def apply(self, x):
        y = self.S_node.apply(x)
        out = np.empty(self.nrows, np.double)
        i_out = 0
        i_y = 0
        for block in self.D_blocks:
            m, n = block.shape
            out[i_out:i_out + m] = np.dot(block, y[i_y:i_y + n])
            i_out += m
            i_y += n
        return out

    def size(self):
        return (self.S_node.size() +
                sum(np.prod(block.shape) for block in self.D_blocks))

class InterpolationBlock(object):
    def __init__(self, filter, interpolant):
        self.filter = np.ascontiguousarray(filter, dtype=np.int8)
        self.interpolant = np.asfortranarray(interpolant, dtype=np.double)
        self.shape = (self.interpolant.shape[0], self.filter.shape[0])
        
    def write_to_stream(self, stream):
        write_array(stream, self.filter)
        pad128(stream)
        write_array(stream, self.interpolant)

    def apply(self, x):
        y = x[self.filter == 0, :]
        y += np.dot(self.interpolant, x[self.filter == 1, :])
        return y

    def size(self):
        return np.prod(self.interpolant.shape)
    
class InnerNode(object):
    def __init__(self, blocks, children):
        if 2**int(np.log2(len(blocks))) != len(blocks):
            raise ValueError("len(blocks) not a power of 2")
        if len(children) != 2:
            raise ValueError("len(children) != 2")
        self.blocks = blocks
        self.ncols = sum(child.ncols for child in children)
        self.children = children
        self.block_heights = sum([[T_ip.shape[0], B_ip.shape[0]]
                                 for T_ip, B_ip in blocks], [])
        self.nrows = sum(self.block_heights)

    def write_to_stream(self, stream):
        L, R = self.children
        write_array(stream, np.asarray(self.block_heights, dtype=index_dtype))
        write_index_t(stream, L.nrows)
        write_index_t(stream, R.nrows)
        write_index_t(stream, L.ncols)
        L.write_to_stream(stream)
        R.write_to_stream(stream)
        for T_ip, B_ip in self.blocks:
            1/0

    def apply(self, x):
        # z is the vector containing the contiguous result of the 2 children
        # The permutation happens in reading from z in a permuted way; so each
        # butterfly application permutes its input
        # Recurse to children butterflies
        LS, RS = self.children
        assert x.shape[0] == self.ncols
        assert x.shape[0] == LS.ncols + RS.ncols
        z_left = LS.apply(x[:LS.ncols])
        z_right = RS.apply(x[LS.ncols:])

        # Apply this butterfly, permuting the input as we go
        y = np.empty(self.nrows, np.double)
        i_y = 0
        i_l = i_r = 0
        for i_block, (T_ip, B_ip) in enumerate(self.blocks):
            # Merge together input
            lw = LS.block_heights[i_block]
            rw = RS.block_heights[i_block]
            assert T_ip.shape[1] == B_ip.shape[1] == lw + rw
            buf = np.empty(lw + rw)
            buf[:lw] = z_left[i_l:i_l + lw]
            i_l += lw
            buf[lw:] = z_right[i_r:i_r + rw]
            i_r += rw
            # Do computation
            th = self.block_heights[2 * i_block]
            bh = self.block_heights[2 * i_block + 1]
            assert T_ip.shape[0] == th and B_ip.shape[0] == bh
            y[i_y:i_y + th] = T_ip.apply(buf)
            i_y += th
            y[i_y:i_y + bh] = B_ip.apply(buf)
            i_y += bh
        return y
        
    def size(self):
        return sum([child.size() for child in self.children] +
                   [A_ip.size() + B_ip.size()
                    for A_ip, B_ip in self.blocks])
        

def permutations_to_filter(alst, blst):
    filter = np.zeros(len(alst) + len(blst), dtype=np.int8)
    filter[blst] = 1
    return filter

def matrix_interpolative_decomposition(A, eps):
    iden_list, ipol_list, A_k, A_ip = sparse_interpolative_decomposition(A, eps)
    filter = permutations_to_filter(iden_list, ipol_list)
    return A_k, InterpolationBlock(filter, A_ip)

def butterfly_core(A_k_blocks, eps):
    if len(A_k_blocks) == 1:
        # No compression achieved anyway at this stage when split
        # into odd l/even l
        return A_k_blocks, IdentityNode(A_k_blocks[0].shape[1])
    mid = len(A_k_blocks) // 2
    left_blocks, left_node = butterfly_core(A_k_blocks[:mid], eps)
    right_blocks, right_node = butterfly_core(A_k_blocks[mid:], eps)
    out_blocks = []
    out_interpolants = []
    for L, R in zip(left_blocks, right_blocks):
        # Horizontal join
        LR = np.hstack([L, R])
        # Vertical split & compress
        vmid = LR.shape[0] // 2
        T = LR[:vmid, :]
        B = LR[vmid:, :]
        T_k, T_ip = matrix_interpolative_decomposition(T, eps)
        B_k, B_ip = matrix_interpolative_decomposition(B, eps)
        assert T_ip.shape[1] == B_ip.shape[1] == LR.shape[1]       
        out_interpolants.append((T_ip, B_ip))
        out_blocks.append(T_k)
        out_blocks.append(B_k)
    return out_blocks, InnerNode(out_interpolants, (left_node, right_node))

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

def butterfly_compress(A, min_rows=32, min_cols=32, eps=1e-10):
    numlevels = min(get_number_of_levels(A.shape[0], min_rows),
                    get_number_of_levels(A.shape[1], min_cols))
    B_list = partition_columns(A, numlevels)
    diagonal_blocks, S_tree = butterfly_core(B_list, eps)
    result = RootNode(diagonal_blocks, S_tree)
    return result
