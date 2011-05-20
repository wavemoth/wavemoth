from cpython cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free
from libc.string cimport memcpy

cdef extern from "stdlib.h":
    void *memalign(size_t boundary, size_t size)

cdef extern from "butterfly.h":
    ctypedef int bfm_index_t
    ctypedef int int32_t

    ctypedef enum BFM_MatrixBlockType:
        BFM_BLOCK_ZERO
        BFM_BLOCK_DENSE_ROWMAJOR
        BFM_BLOCK_HSTACK
        BFM_BLOCK_BUTTERFLY

    ctypedef struct BFM_ButterflyHeader:
        int32_t type_id
        int32_t k_L, n_L, k_R
    
    int bfm_apply_right_d(char *matrixdata, double *x, double *y,
                          bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec)

import numpy as np
cimport numpy as np
from interpolative_decomposition import sparse_interpolative_decomposition

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
            
        ret = bfm_apply_right_d(self.buf, <double*>vec_.data,
                                <double*>out_.data, self.nrow, self.ncol, vec.shape[1])
        if ret != 0:
            raise ButterflyMatrixError('Error code %d' % ret)
        return out


cdef write_bin(stream, char *buf, Py_ssize_t size):
    n = stream.write(PyBytes_FromStringAndSize(buf, size))
    assert n == size

cdef write_int32(stream, int32_t i):
    write_bin(stream, <char*>&i, sizeof(i))

cdef write_array(stream, arr):
    n = stream.write(bytes(arr.data))
    assert n == np.prod(arr.shape) * arr.itemsize

cdef pad128(stream):
    i = stream.tell()
    m = i % 16
    if m != 0:
        stream.write(b'\0' * (16 - m))

class DenseMatrix:
    def __init__(self, A):
        self.A = np.ascontiguousarray(A)
        self.shape = self.A.shape

    def serialize(self, stream):
        write_int32(stream, BFM_BLOCK_DENSE_ROWMAJOR)
        pad128(stream)
        write_array(stream, self.A)

    def apply(self, x):
        return np.dot(self.A, x)

class InterpolationMatrix:
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

class ButterflyMatrix:
    def __init__(self, shape, L_ip, R_ip):
        assert isinstance(L_ip, InterpolationMatrix)
        assert isinstance(R_ip, InterpolationMatrix)
        self.L_ip = L_ip
        self.R_ip = R_ip
        self.shape = shape #(L_ip.shape[0], L_k.shape[1] + L_ip.shape[1])

    def serialize(self, stream):
        cdef BFM_ButterflyHeader header
        header.type_id = BFM_BLOCK_BUTTERFLY
        header.k_L = self.L_ip.shape[0]
        header.n_L = self.L_ip.shape[1]
        header.k_R = self.R_ip.shape[0]
        write_bin(stream, <char*>&header, sizeof(header))
        self.L_ip.write_to_stream(stream)
        self.R_ip.write_to_stream(stream)

    def apply(self, x):
        u = self.L_ip.apply(x[:self.L_ip.shape[1]])
        v = self.R_ip.apply(x[:self.R_ip.shape[1]])
        return np.vstack([u, v])

def permutations_to_filter(alst, blst):
    n_a = len(alst)
    n_b = len(blst)
    filter = np.empty(n_a + n_b, dtype=np.int8)
    a = alst[0] if n_a > 0 else None
    b = blst[0] if n_b > 0 else None
    idx = 0
    i_a = i_b = 1
    while not (a == b == None):
        while a is not None and (b is None or a < b):
            filter[idx] = 0
            prev_a = a
            a = alst[i_a] if i_a < n_a else None
            if prev_a >= a != None:
                raise ValueError("Input not strictly ordered")
            idx += 1
            i_a += 1
        while b is not None and (a is None or b < a):
            filter[idx] = 1
            prev_b = b
            b = blst[i_b] if i_b < n_b else None
            if prev_b >= b != None:
                raise ValueError("Input not strictly ordered")
            idx += 1
            i_b += 1
        if a == b != None:
            raise ValueError("An element is present in both lists")
    return filter

def matrix_interpolative_decomposition(A, eps):
    iden_list, ipol_list, A_k, A_ip = sparse_interpolative_decomposition(A, eps)
    filter = permutations_to_filter(iden_list, ipol_list)
    return A_k, InterpolationMatrix(filter, A_ip)

def butterfly_compress(A, min_rows=20, eps=1e-10):
    if A.shape[0] <= min_rows:
        return DenseMatrix(A)        
    hmid = A.shape[1] // 2
    L, R = A[:, :hmid], A[:, hmid:]
    L_k, L_ip = matrix_interpolative_decomposition(L, eps)
    R_k, R_ip = matrix_interpolative_decomposition(R, eps)
    return ButterflyMatrix(A.shape, L_ip, R_ip)
        
    
    
