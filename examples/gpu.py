# http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/


import os
import sys
import socket

if socket.gethostname() != 'dagss-laptop':
    sys.path.append('/home/dagss/wavemoth')

import numpy as np
import numpy.linalg as la

from matplotlib import pyplot as plt

import pycuda.autoinit
import wavemoth.cuda.flatcuda as cuda

from wavemoth import *
from wavemoth import healpix
from wavemoth.cuda import CudaLegendreKernel

def hrepeat(x, n):
    return np.repeat(x[:, None], n, axis=1).copy('F')

nblocks = 500
has_warps = True
nside = 32

# Compute Lambda
k_chunk = 64
i_chunk = 2
nvecs = 2

m = 0
lmax = 2 * nside
odd = 0
repeat = 1

thetas = healpix.get_ring_thetas(nside, positive_only=True)
Lambda = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=1e-100)
Lambda = Lambda[:, odd::2].T

# Mock input vector
nk, ni = Lambda.shape
    

q = hrepeat(np.sin(np.arange(ni) * 0.4), nvecs * nblocks).reshape(
    (Lambda.shape[1], nvecs, nblocks), order='F')

a0 = np.dot(Lambda, q[:, :, 0])

for nwarps in [1]:
    nthreads = 32 * nwarps

    print
    print
    print '=== nthreads=%d, nvecs=%d ===' % (nthreads, nvecs)

    Lambda_0 = hrepeat(Lambda[0, :], nblocks)
    Lambda_1 = hrepeat(Lambda[1, :], nblocks)
    x_squared = hrepeat(np.cos(thetas[:ni])**2, nblocks).copy('F')
    out = np.zeros((nk, nvecs, nblocks), dtype=np.double, order='F')

    kernel = CudaLegendreKernel(max_ni=ni,
                                nthreads=nthreads, nvecs=nvecs,
                                has_warps=has_warps,
                                k_chunk=k_chunk,
                                i_chunk=i_chunk)

    times = []
    for rep in range(repeat):
        kernel.transpose_legendre_transform(m, m + odd,
                                            x_squared, Lambda_0, Lambda_1,
                                            q, out)
        #e.wait()
        #times.append((e.profile.end - e.profile.start) * 1e-9)

    #dt = min(times)

    nk, ni = Lambda.shape
    matrix_elements = nblocks * ni * nk
    UOP = matrix_elements * (6 + 2 * nvecs)
    #print '%.2e +/- %.2e sec = %.2f GUOP/sec' % (dt, np.std(times), UOP / dt / 1e9)

    a = out
    if not np.all(a[:, :, 0:1] == a):
        print 'NOT ALL j EQUAL!:', np.linalg.norm(a[:, :, 0:1] - a)
    a = a[:, :, 0]

    print la.norm(a - a0)

print np.hstack([a, a0])


#plt.clf()
