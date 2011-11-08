# http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/


import numpy as np
import numpy.linalg as la

from matplotlib import pyplot as plt

from wavemoth import *
from wavemoth import healpix
from wavemoth.cl import ClLegendreKernel
import wavemoth.cl.flatpyopencl as cl

import socket

for platform in cl.get_platforms():
    if socket.gethostname() == 'dagss-laptop':
        wanted = 'Intel'
        nblocks = 10
        has_warps = False
        nside = 32
    else:
        wanted = 'NVIDIA'
        nblocks = 1000
        has_warps = True
        nside = 512

    if wanted in platform.name:
        ctx = cl.Context(platform.get_devices())

queue = cl.CommandQueue(ctx, 
                        properties=cl.command_queue_properties.PROFILING_ENABLE)


# Compute Lambda
k_chunk = 4
nvecs = 2

m = 0
lmax = 2 * nside
odd = 0
repeat = 1

thetas = healpix.get_ring_thetas(nside, positive_only=True)
Lambda = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=1e-100)
Lambda = Lambda[:, odd::2].T

for nwarps in [1]:#range(1, 6):
    nthreads = 32 * nwarps

    print
    print
    print '=== nthreads=%d, nvecs=%d ===' % (nthreads, nvecs)

    def hrepeat(x, n):
        return np.repeat(x[:, None], n, axis=1).copy('F')

    # Mock input vector
    nk, nx = Lambda.shape
    
    q = hrepeat(np.sin(np.arange(nx) * 0.4), nvecs * nblocks).reshape(
        (Lambda.shape[1], nvecs, nblocks), order='F')
    q_cl = cl.to_device(queue, q)
    Lambda_0_cl = cl.to_device(queue, hrepeat(Lambda[0, :], nblocks))
    Lambda_1_cl = cl.to_device(queue, hrepeat(Lambda[1, :], nblocks))
    x_squared_cl = cl.to_device(queue,
                                hrepeat(np.cos(thetas[:nx])**2, nblocks).copy('F'))
    out_cl = cl.zeros(queue, (nk, nvecs, nblocks), dtype=np.double, order='F')

    times = []
    for rep in range(repeat):
        kernel = ClLegendreKernel(ctx, max_ni=nx,
                                  nthreads=nthreads, nvecs=nvecs,
                                  has_warps=has_warps,
                                  k_chunk=k_chunk)
        e = kernel.transpose_legendre_transform(queue, m, m + odd,
                                                x_squared_cl, Lambda_0_cl, Lambda_1_cl,
                                                q_cl, out_cl)
        e.wait()
        times.append((e.profile.end - e.profile.start) * 1e-9)

    dt = min(times)

    nk, nx = Lambda.shape
    matrix_elements = nblocks * nx * nk
    UOP = matrix_elements * (6 + 2 * nvecs)
    print '%.2e +/- %.2e sec = %.2f GUOP/sec' % (dt, np.std(times), UOP / dt / 1e9)

a = out_cl.get()
if not np.all(a[:, :, 0:1] == a):
    print 'NOT ALL j EQUAL!:', np.linalg.norm(a[:, :, 0:1] - a)
a = a[:, :, 0]

a0 = np.dot(Lambda, q[:, :, 0])

print np.hstack([a, a0])
print la.norm(a - a0)

#plt.clf()
