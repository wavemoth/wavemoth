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


epsilon_legendre = 1e-30

def get_edge(Lambda):
    zero_mask = Lambda == 0
    i_stops = np.zeros(Lambda.shape[0], dtype=np.uint16)
    Lambda_0 = np.zeros(Lambda.shape[1])
    Lambda_1 = np.zeros(Lambda.shape[1])
    cur_i = 0
    for k in range(Lambda.shape[0]):
        ilst, = zero_mask[k, :].nonzero()
        i_stops[k] = next_i = ilst[0] if len(ilst) > 0 else Lambda.shape[1]
        Lambda_0[cur_i:next_i] = Lambda[k, cur_i:next_i]
        if k + 1 < Lambda.shape[0]:
            Lambda_1[cur_i:next_i] = Lambda[k + 1, cur_i:next_i]
        cur_i = next_i
    return i_stops, Lambda_0, Lambda_1

nblocks = 500
has_warps = True
nside = 2048

# Compute Lambda
nvecs = 2

m = 200
lmax = 2 * nside
odd = 0
repeat = 1

def downto(x, mod):
    if x % mod != 0:
        x -= x % mod
    return x


thetas = healpix.get_ring_thetas(nside, positive_only=True)
Lambda = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=epsilon_legendre)
Lambda = Lambda[:, odd::2].T

def plot_matrix(M):
    ax = plt.gca()
    ax.imshow(M, interpolation='nearest')

#plot_matrix(Lambda)

nk, ni = Lambda.shape

nnz = np.sum(Lambda != 0)

x_squared = hrepeat(np.cos(thetas[:ni])**2, nblocks).copy('F')

i_stops, Lambda_0, Lambda_1 = get_edge(Lambda)

Lambda_0 = hrepeat(Lambda_0, nblocks)
Lambda_1 = hrepeat(Lambda_1, nblocks)
i_stops = hrepeat(i_stops, nblocks)

#print Lambda_0
#1/0

# Mock input vector 
q = hrepeat(np.sin(np.arange(ni) * 0.4), nvecs * nblocks).reshape(
    (Lambda.shape[1], nvecs, nblocks), order='F')
q[:, 1] *= 2

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
                                                x_squared, Lambda_0, Lambda_1,
                                                i_stops, q, out)
    print prof.format('transpose_legendre_transform',
                      nflops=nblocks * nnz * (6 + 2 * nvecs),
                      nwarps=nwarps)

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
        for k_chunk in [64]:
            a = doit(nvecs=nvecs, nwarps=nwarps, i_chunk=i_chunk, k_chunk=k_chunk)

print np.hstack([a, a0])


#plt.clf()


# TODO:
# k_chunk must be 32 (because of aux computations)
# i_chunk must be %2 (because of ni)
# nthreads = 96 does not work either (?)
