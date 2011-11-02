import os
import tempita
import numpy as np
import pyopencl as cl

from . import core

def check_on_device(args):
    for arg in args:
        if not isinstance(arg, cl.array.Array):
            raise TypeError('pyopencl.array.Array expected')


class ClLegendreKernel(object):
    def __init__(self, ctx, **args):
        code = core.instantiate_template('legendre_transform.cl.in', **args)
        self.prg = cl.Program(ctx, code).build()
        self._transpose_legendre_transform = self.prg.transpose_legendre_transform
        self._transpose_legendre_transform.set_scalar_arg_dtypes(
            [np.int32, np.int32, np.int32, np.int32,
             None, None, None, None, None])

    def transpose_legendre_transform(self, queue, m, lmin, x_squared, Lambda_0, Lambda_1,
                                     a, out):
        check_on_device([x_squared, Lambda_0, Lambda_1, a, out])
        nk = a.shape[0]
        nx = Lambda_0.shape[0]
        if not ((nx,) == Lambda_0.shape == Lambda_1.shape == x_squared.shape):
            raise ValueError('Lambda_0 and/or Lambda_1 and/or x_squared has wrong shape')
        return self._transpose_legendre_transform(
            queue, (nx,), None,
            m, lmin, nk, nx, x_squared.data,
            Lambda_0.data, Lambda_1.data, a.data, out.data)
        
