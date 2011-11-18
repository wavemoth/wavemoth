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

def test_dot_and_copy():
    def test(nthreads, nvecs, nx):
        P = ndrange((nx,))

        q = ndrange((nvecs, nx))
        P_copy = np.ones(nx) * np.nan
        work_sum = np.ones((nvecs, nthreads)) * np.nan

        kernel = ClLegendreKernel(ctx, nvecs=nvecs, nthreads=nthreads, **kopts)
        kernel.dot_and_copy(queue, P, q, P_copy, work_sum)

        # Check that P was copied into P_local
        ok_(all(P == P_copy))
        # Check that work_sum contains the per-thread sum contribution for P[None, :] * q
        P_by_q = P[None, :] * q
        threadsum = np.zeros((nvecs, nthreads))
        for i in range(0, nx, nthreads):
            sub = P_by_q[:, i:i + nthreads]
            threadsum[:, :sub.shape[1]] += sub
        ok_(all(work_sum == threadsum))

    yield test, 128, 2, 20
    yield test, 128, 2, 128
    yield test, 128, 2, 129
    yield test, 128, 2, 1000
    yield test, 128, 4, 1000
    

def test_warp_sum_reduce():
    nvecs = 2
    nthreads = 128
    
    thread_sum = ndrange((nvecs, nthreads))
    warp_sum = np.ones((nthreads // WARP_SIZE, k_chunk, nvecs))
    
    kernel = ClLegendreKernel(ctx, nvecs=nvecs, nthreads=nthreads, **kopts)
    kernel.warp_sum_reduce(queue, 0, thread_sum.copy(), warp_sum)

    for iwarp in range(nthreads // WARP_SIZE):
        warp_sum0 = thread_sum[:, iwarp * WARP_SIZE:(iwarp + 1) * WARP_SIZE].sum(axis=1)
        warp_sum0 += 1 # as all(warp_sum == 1) prior to the call
        assert_almost_equal(warp_sum[iwarp, 0, :], warp_sum0)

def test_inter_warp_sum():
    k_chunk = 32

    def test(nthreads, nvecs, nk):
        opts = dict(kopts)
        opts.update(nvecs=nvecs,
                    nthreads=nthreads,
                    k_chunk=k_chunk)
        nwarps = nthreads // WARP_SIZE

        work_local_sum = ndrange((nwarps, k_chunk, nvecs))
        out = np.ones((nvecs, nk + 10)) * np.nan

        kernel = ClLegendreKernel(ctx, **opts)
        kernel.inter_warp_sum(queue, 0, nk, work_local_sum, out)
        expected = work_local_sum.sum(axis=0).T[:, :nk]
        assert_almost_equal(expected, out[:, :nk])
        ok_(np.isnan(out[0, nk]))

    yield test, 128, 2, 32
    yield test, 64, 4, 3
    yield test, 32, 2, 31
    
    
def test_parallel_reduction():
    def test(nwarps):
        nthreads = 32 * nwarps
        nvecs = 2
        repeat = 2
        kernel = CudaLegendreKernel(nvecs=nvecs, nthreads=nthreads, **kopts)

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
