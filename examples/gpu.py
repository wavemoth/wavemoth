from __future__ import division

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

from wavemoth import *
from wavemoth import healpix
from wavemoth.cuda import CudaLegendreKernel

from wavemoth.cuda.profile import cuda_profile
import wavemoth.cuda.flatcuda as cuda

def hrepeat(x, n):
    return np.repeat(x[:, None], n, axis=1).copy('F')



nblocks = 500
has_warps = True
nside = 512

# Compute Lambda
nvecs = 2

m = 0
lmax = 2 * nside
odd = 0
repeat = 3

thetas = healpix.get_ring_thetas(nside, positive_only=True)
Lambda = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=1e-100)
Lambda = Lambda[:, odd::2].T


nk, ni = Lambda.shape

Lambda_0 = hrepeat(Lambda[0, :], nblocks)
Lambda_1 = hrepeat(Lambda[1, :], nblocks)
x_squared = hrepeat(np.cos(thetas[:ni])**2, nblocks).copy('F')

# Mock input vector
    

q = hrepeat(np.sin(np.arange(ni) * 0.4), nvecs * nblocks).reshape(
    (Lambda.shape[1], nvecs, nblocks), order='F')

a0 = np.dot(Lambda, q[:, :, 0])

#cuda.initialize_profiler('/home/dagss/cuda_profile_config', "profiles/test", cuda.CSV)
#cuda.initialize_profiler('/home/dagss/cuda_profile_config', "profiles/test", cuda.CSV)

check = False

def doit(nvecs, nwarps, i_chunk, k_chunk):
    nthreads = 32 * nwarps

    print
    print
    print '=== nvecs={nvecs}, nthreads={nthreads}, i_chunk={i_chunk}, k_chunk={k_chunk}'.format(**locals())

    out = np.zeros((nk, nvecs, nblocks), dtype=np.double, order='F')

    kernel = CudaLegendreKernel(max_ni=ni,
                                nthreads=nthreads, nvecs=nvecs,
                                has_warps=has_warps,
                                k_chunk=k_chunk,
                                i_chunk=i_chunk)

    times = []
    #cuda.start_profiler()
    with cuda_profile() as prof:
        for rep in range(repeat):
            kernel.transpose_legendre_transform(m, m + odd,
                                                x_squared, Lambda_0, Lambda_1,
                                                q, out)
    times = np.asarray(prof.transpose_legendre_transform.times) * 1e-6
    dt = np.min(times)

    matrix_elements = nblocks * ni * nk
    UOP = matrix_elements * (6 + 2 * nvecs)
    print '%.2e +/- %.2e sec = %.2f GUOP/sec' % (dt, np.std(times), UOP / dt / 1e9)
    occupancy_fraction = prof.transpose_legendre_transform.occupancy[0]
    nblocks_per_sm = occupancy_fraction * 48. / (nwarps)
    print 'Occupancy: %.2f (%.2f warps, %.2f blocks)' % (
        occupancy_fraction, occupancy_fraction * 48, nblocks_per_sm) 

    a = out
    if check:
        if not np.all(a[:, :, 0:1] == a):
            print 'NOT ALL j EQUAL!'

    a = a[:, :, 0]
    print 'Error', la.norm(a - a0) / la.norm(a0)
    sys.stdout.flush()
    return a
    

for nwarps in [2]:
    for i_chunk in [4]:
        for k_chunk in [32]:
            a = doit(nvecs=nvecs, nwarps=nwarps, i_chunk=i_chunk, k_chunk=k_chunk)

print np.hstack([a, a0])


#plt.clf()


# TODO:
# k_chunk must be 32 (because of aux computations)
# i_chunk must be %2 (because of ni)
# nthreads = 96 does not work either (?)
