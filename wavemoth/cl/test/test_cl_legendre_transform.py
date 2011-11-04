import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import ok_
from numpy import all, any

from ..legendre_transform import *

from .. import flatpyopencl as cl

import socket

WARP_SIZE = 32

for platform in cl.get_platforms():
    if socket.gethostname() == 'dagss-laptop':
        wanted = 'Intel'
        nblocks = 5
        has_warps = False
    else:
        wanted = 'NVIDIA'
        nblocks = 5000
        has_warps = True

    if wanted in platform.name:
        ctx = cl.Context(platform.get_devices())

queue = cl.CommandQueue(ctx, 
                        properties=cl.command_queue_properties.PROFILING_ENABLE)

def ndrange(shape, dtype=np.double):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

def test_dot_and_copy():
    def test(nthreads, nvecs, nx):
        P = ndrange((nx,))

        q = ndrange((nvecs, nx))
        P_local = np.ones(nx) * np.nan
        work_sum = np.ones((nvecs, nthreads)) * np.nan

        kernel = ClLegendreKernel(ctx, nvecs=nvecs, nthreads=nthreads, has_warps=has_warps)
        kernel.dot_and_copy(queue, P, q, P_local, work_sum)

        # Check that P was copied into P_local
        ok_(all(P == P_local))
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
    warp_sum = np.ones((nvecs, nthreads // WARP_SIZE)) * np.nan
    
    kernel = ClLegendreKernel(ctx, nvecs=nvecs, nthreads=nthreads, has_warps=has_warps)
    kernel.warp_sum_reduce(queue, thread_sum.copy(), warp_sum)

    for iwarp in range(nthreads // WARP_SIZE):
        warp_sum0 = thread_sum[:, iwarp * WARP_SIZE:(iwarp + 1) * WARP_SIZE].sum(axis=1)
        assert_almost_equal(warp_sum[:, iwarp], warp_sum0)
