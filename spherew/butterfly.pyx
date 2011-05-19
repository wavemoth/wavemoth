from cpython cimport PyBytes_FromStringAndSize

cdef extern from "butterfly.h":
    ctypedef enum BFM_MatrixBlockType:
        BFM_BLOCK_ZERO
        BFM_BLOCK_DENSE_ROWMAJOR
        BFM_BLOCK_HSTACK
        BFM_BLOCK_BUTTERFLY

    ctypedef int bfm_index_t
    ctypedef int int32_t
    
    int bfm_apply_right_d(char *matrixdata, double *x, double *y,
                          bfm_index_t nrow, bfm_index_t ncol, bfm_index_t nvec)

import numpy as np
cimport numpy as np

class ButterflyMatrixError(Exception):
    pass

class SerializedMatrix:
    def __init__(self, bytes matrixdata, bfm_index_t nrow, bfm_index_t ncol):
        self.matrixdata = matrixdata
        self.nrow = nrow
        self.ncol = ncol

    def apply(self, vec, out=None):
        vec = np.ascontiguousarray(vec)
        if vec.ndim == 1:
            vec = vec[:, None]
        elif vec.ndim > 2:
            raise ValueError()
        if vec.shape[0] != self.ncol:
            raise ValueError()
        if out is None:
            out = np.empty((self.nrow, vec.shape[1]), dtype=np.double)

        cdef np.ndarray[double, ndim=2, mode='c'] vec_ = vec
        cdef np.ndarray[double, ndim=2, mode='c'] out_ = out
            
        ret = bfm_apply_right_d(<char*>self.matrixdata, <double*>vec_.data,
                                <double*>out_.data, self.nrow, self.ncol, vec.shape[1])
        if ret != 0:
            raise ButterflyMatrixError('Error code %d' % ret)
        return out


def put_int32(stream, int32_t i):
    stream.write(PyBytes_FromStringAndSize(<char*>&i, 4))

def put_padding(stream, n):
    stream.write(b'\0' * n)

class DenseMatrix:
    def __init__(self, A):
        self.A = np.ascontiguousarray(A)

    def serialize(self, stream):
        put_int32(stream, BFM_BLOCK_DENSE_ROWMAJOR)
        put_padding(stream, 12)
        stream.write(bytes(self.A.data))

        
