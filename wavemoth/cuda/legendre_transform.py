import os
import tempita
import numpy as np
from numpy import int32

from . import flatcuda as cuda
from .flatcuda import InOut, In, Out

from . import core


def check_arrays(args):
    for array, ndim in args:
        #if not isinstance(array, cl.Array):
        #    raise TypeError('pyopencl.array.Array expected')
        if len(array.shape) != ndim:
            raise ValueError('array has wrong number of dimensions')
        if array.dtype != np.double:
            raise ValueError('array has dtype != np.double')

class CudaLegendreKernel(object):
    
    kernel_names = ['transpose_legendre_transform', 'test_reduce_kernel']
    def __init__(self, nvecs, nthreads, max_ni, i_chunk, warp_size=32, **args):
        self.nthreads = nthreads
        self.nvecs = nvecs
        self.warp_size = 32
        self.max_ni = max_ni
        self.i_chunk = i_chunk

        code = core.instantiate_template('legendre_transform.cu.in',
                                         nvecs=nvecs,
                                         nthreads=nthreads,
                                         warp_size=self.warp_size,
                                         max_ni=max_ni,
                                         i_chunk=i_chunk,
                                         **args)
        options = ['-ftz=true',
                   '-prec-div=true', '-prec-sqrt=true']
        #options += ['-Xptxas', '-v']
        self.module = cuda.SourceModule(code, options=options, keep=True)
        for name in self.kernel_names:
            setattr(self, '_' + name, self.module.get_function(name))

    def transpose_legendre_transform(self, m, lmin,
                                     x_squared, Lambda_0, Lambda_1, i_stops, q, a):
        # a: (nblocks, 2 * nk, nvecs)
        nblocks = q.shape[2]
        assert nblocks == a.shape[0] == Lambda_0.shape[1] == Lambda_1.shape[1]
        check_arrays([(x_squared, 1), (Lambda_0, 2), (Lambda_1, 2), (q, 3), (a, 3)])
        self.nvecs = q.shape[1]
        if not (q.shape[1] == a.shape[2] == self.nvecs):
            raise ValueError('q and a arrays do not conform to self.nvecs')
        if not (q.strides[0] == a.strides[2] == 8):
            raise ValueError('q and/or a has non-unit stride on axis 0')
        ni = q.shape[0]
        nk = a.shape[1] // 2
        if not (ni == Lambda_0.shape[0] == Lambda_1.shape[0] == x_squared.shape[0]):
            raise ValueError('Lambda_0 and/or Lambda_1 and/or x_squared has wrong shape')

        if i_stops.dtype != np.uint16:
            raise ValueError('i_stops.dtype != np.uint16')

        return self._transpose_legendre_transform(
            int32(m), int32(lmin), int32(nk), int32(ni), In(x_squared),
            In(Lambda_0), In(Lambda_1), In(i_stops), In(q), Out(a),
            int32(0),
            block=(self.nthreads, 1, 1), grid=(nblocks, 1))

    def test_reduce_kernel(self, output, repeat=1, nblocks=1):
        self._test_reduce_kernel(InOut(output), int32(repeat),
                                 block=(self.nthreads, 1, 1),
                                 grid=(nblocks, 1))
                       

