import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import ok_
from numpy import all, any

import pycuda.autoinit

from ..legendre_transform import *

import socket

WS = 32

k_chunk = 32
kopts = dict(max_ni=256, k_chunk=k_chunk, i_chunk=4)

def ndrange(shape, dtype=np.double):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    
def test_parallel_reduction():
    def test(nwarps):
        nthreads = 32 * nwarps
        nvecs = 2
        repeat = 2
        kernel = CudaLegendreKernel(nvecs=nvecs, nthreads=nthreads,
                                    skip_kernels=['transpose_legendre_transform',
                                                  'all_transpose_legendre_transforms'],
                                    **kopts)

        j, k, i = np.ogrid[:2, :16, :nthreads]
        output0 = repeat * ((j + 1) * k * i)

        output = np.zeros((nvecs, 16, nthreads // WS))
        kernel.test_reduce_kernel(output, repeat=repeat)

        if 0:
            print np.vstack([output.sum(axis=2), output0.sum(axis=2)]).T
            print output.T

        ok_(np.all(output.sum(axis=2) == output0.sum(axis=2)))
        
    yield test, 1
    yield test, 3
