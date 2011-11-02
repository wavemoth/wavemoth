import os
import tempita
import numpy as np
import pyopencl as cl

from . import core

def check_arrays(args):
    for array, ndim in args:
        if not isinstance(array, cl.array.Array):
            raise TypeError('pyopencl.array.Array expected')
        if len(array.shape) != ndim:
            raise ValueError('array has wrong number of dimensions')
        if array.dtype != np.double:
            raise ValueError('array has dtype != np.double')


class ClLegendreKernel(object):
    """

    major_col_chunk -- How many columns of Lambda to process between
        each round through global memory
    
    """
    def __init__(self, ctx, nvecs, **args):
        code = core.instantiate_template('legendre_transform.cl.in',
                                         nvecs=nvecs,
                                         local_size=2 * 128,
                                         **args)
        self.nvecs = nvecs
        self.prg = cl.Program(ctx, code).build()
        self._transpose_legendre_transform = self.prg.transpose_legendre_transform
        self._transpose_legendre_transform.set_scalar_arg_dtypes(
            [np.int32, np.int32, np.int32, np.int32,
             None, None, None, None, None])

    def transpose_legendre_transform(self, queue, m, lmin, x_squared, Lambda_0, Lambda_1,
                                     q, out):
        check_arrays([(x_squared, 1), (Lambda_0, 1), (Lambda_1, 1), (q, 2), (out, 2)])
        if not (q.shape[1] == out.shape[1] == self.nvecs):
            raise ValueError('q and out arrays do not conform to self.nvecs')
        if not (q.strides[0] == out.strides[0] == 8):
            raise ValueError('q and/or out has non-unit stride on axis 0')
        nx = q.shape[0]
        nk = out.shape[0]
        if not ((nx,) == Lambda_0.shape == Lambda_1.shape == x_squared.shape):
            raise ValueError('Lambda_0 and/or Lambda_1 and/or x_squared has wrong shape')
        if nx != 2 * 128:
            raise NotImplementedError()
        return self._transpose_legendre_transform(
            queue, (nx,), (nx,),
            m, lmin, nk, nx, x_squared.data,
            Lambda_0.data, Lambda_1.data, q.data, out.data)
        
