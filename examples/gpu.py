from __future__ import division

import os
import sys
import socket

# Problematic configurations:
# - Nside=64
# - Nside=m=2048, odd=0
#
#

if socket.gethostname() != 'dagss-laptop':
    sys.path.append('/home/dagss/wavemoth')

import numpy as np
import numpy.linalg as la
import logging

from matplotlib import pyplot as plt

import pycuda.autoinit

from wavemoth import *
from wavemoth import healpix
from wavemoth.cuda import CudaShtPlan, CudaLegendreKernel

from wavemoth.cuda.profile import cuda_profile
import wavemoth.cuda.flatcuda as cuda

def hrepeat(x, n):
    return np.repeat(x[:, None], n, axis=1).copy('F')

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

nblocks = 1000
nside = 2048

# Compute Lambda
nvecs = 2

m = 200
lmax = 2 * nside
odd = 0
repeat = 1

plan = CudaShtPlan(nside=nside, lmax=lmax)
    
ni = plan.ni
nk = (lmax + 1 - m - odd + 1) // 2

Lambda_0, Lambda_1, i_stops, nnz = plan.precompute_single(m, odd)

Lambda_0 = hrepeat(Lambda_0, nblocks)
Lambda_1 = hrepeat(Lambda_1, nblocks)
i_stops = hrepeat(i_stops, nblocks)

#print Lambda_0
#1/0

# Mock input vector 
q = hrepeat(np.sin(np.arange(ni) * 0.4), nvecs * nblocks).reshape(
    (ni, nvecs, nblocks), order='F')
q[:, 1] *= 2

Lambda = plan.get_Lambda(m, odd)
a0 = np.dot(Lambda, q[:, :, 0])

check = False

def doit(nvecs, nwarps, i_chunk, k_chunk):
    nthreads = 32 * nwarps

    print
    print
    print '=== nvecs={nvecs}, nthreads={nthreads}, i_chunk={i_chunk}, k_chunk={k_chunk}'.format(**locals())

    out = np.zeros((nblocks, 2 * nk, nvecs), dtype=np.double, order='C')

    kernel = CudaLegendreKernel(max_ni=ni,
                                nthreads=nthreads, nvecs=nvecs,
                                k_chunk=k_chunk,
                                i_chunk=i_chunk)

    if 0:
        print '======== Reduction '
        with cuda_profile() as prof:
            for rep in range(repeat):
                output = np.zeros((nblocks, 2, 16, nwarps))
                kernel.test_reduce_kernel(output, repeat=1000, nblocks=nblocks)
        print prof.format('test_reduce_kernel',
                          nflops=nblocks * 2 * 16 * nthreads * 1000,
                          nwarps=nwarps)

    print '======== Legendre transform '
    with cuda_profile() as prof:
        for rep in range(repeat):
            kernel.transpose_legendre_transform(m, m + odd,
                                                plan.x_squared, Lambda_0, Lambda_1,
                                                i_stops, q, out)
    print prof.format('transpose_legendre_transform',
                      nflops=nblocks * nnz * (5 + 2 * nvecs),
                      nwarps=nwarps)

    # Output is stored in strided format
    a = out[:, ::2, :]
    if check:
        if not np.all(a[:, :, 0:1] == a):
            print 'NOT ALL j EQUAL!'

    a = a[1, :, :]
    print 'Error', la.norm(a - a0) / la.norm(a0)
    sys.stdout.flush()
    return a
    

for nwarps in [2]:
    for i_chunk in [4]:
        for k_chunk in [64]:
            a = doit(nvecs=nvecs, nwarps=nwarps, i_chunk=i_chunk, k_chunk=k_chunk)
            
print np.hstack([a, a0])
#print np.isnan(a).nonzero()

print i_stops[:-300]
#plt.clf()


# TODO:
# k_chunk must be 32 (because of aux computations)
# i_chunk must be %2 (because of ni)
# nthreads = 96 does not work either (?)

