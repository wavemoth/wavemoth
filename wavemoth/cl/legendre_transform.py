import os
import tempita
import numpy as np
from . import flatpyopencl as cl

from . import core

def check_arrays(args):
    for array, ndim in args:
        if not isinstance(array, cl.array.Array):
            raise TypeError('pyopencl.array.Array expected')
        if len(array.shape) != ndim:
            raise ValueError('array has wrong number of dimensions')
        if array.dtype != np.double:
            raise ValueError('array has dtype != np.double')


# Decorator to use on functions so that any NumPy array passed
# in is properly transferred to device memory and back. Does
# not take into account that one does not wish to transfer *all*
# data both ways...primarily useful for testing. It is assumed
# that the first two arguments to the function are 'self' and 'queue'.
#
# Also, Python 'int' is turned automatically into np.int32.
def convertargs():
    def dec(real_func):
        def repl_func(self, queue, *args, **kw):
            transferred = []
            def convert(arg):
                if isinstance(arg, np.ndarray):
                    arr_d = cl.to_device(queue, arg)
                    transferred.append((arg, arr_d))
                    return arr_d
                elif isinstance(arg, int):
                    return np.int32(arg)
                else:
                    return arg
            new_args = [convert(arg) for arg in args]
            new_kw = dict([(name, convert(arg)) for name, arg in kw.iteritems()])
            try:
                return real_func(self, queue, *new_args, **new_kw)
            finally:
                for arr, arr_d in transferred:
                    arr[...] = arr_d.get()
        return repl_func
    return dec

class ClLegendreKernel(object):
    """

    major_col_chunk -- How many columns of Lambda to process between
        each round through global memory
    
    """
    def __init__(self, ctx, nvecs, nthreads, **args):
        self.nthreads = nthreads
        self.nvecs = nvecs

        code = core.instantiate_template('legendre_transform.cl.in',
                                         nvecs=nvecs,
                                         local_size=nthreads,
                                         warp_size=32,
                                         **args)
        self.prg = cl.Program(ctx, code).build()
        self._transpose_legendre_transform = self.prg.transpose_legendre_transform
        self._transpose_legendre_transform.set_scalar_arg_dtypes(
            [np.int32, np.int32, np.int32, np.int32,
             None, None, None, None, None])

    def transpose_legendre_transform(self, queue, m, lmin,
                                     x_squared, Lambda_0, Lambda_1, q, out):
        nblocks = q.shape[2]
        assert nblocks == out.shape[2] == Lambda_0.shape[1] == Lambda_1.shape[1]
        check_arrays([(x_squared, 2), (Lambda_0, 2), (Lambda_1, 2), (q, 3), (out, 3)])
        self.nvecs = q.shape[1]
        if not (q.shape[1] == out.shape[1] == self.nvecs):
            raise ValueError('q and out arrays do not conform to self.nvecs')
        if not (q.strides[0] == out.strides[0] == 8):
            raise ValueError('q and/or out has non-unit stride on axis 0')
        nx = q.shape[0]
        nk = out.shape[0]
        if not (nx == Lambda_0.shape[0] == Lambda_1.shape[0] == x_squared.shape[0]):
            raise ValueError('Lambda_0 and/or Lambda_1 and/or x_squared has wrong shape')

        assert self.nx == self.nk == self.nthreads

        return self._transpose_legendre_transform(
            queue, (nblocks * self.nthreads,), (self.nthreads,),
            m, lmin, nk, nx, x_squared.data,
            Lambda_0.data, Lambda_1.data, q.data, out.data)

    @convertargs()
    def dot_and_copy(self, queue, P, q, P_local, work_sum):
        self.prg.dot_and_copy_kernel(queue, (self.nthreads,), (self.nthreads,),
                                     P.data, q.data, P_local.data, work_sum.data,
                                     np.int32(P.shape[0]))
