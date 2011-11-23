from __future__ import division

import os
import sys
from time import time

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

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

force = False
nside = 2048
nmaps = 1
lmax = 2 * nside
odd = 0

mmin = 0
mmax = lmax

test_ms = [mmin, 1, (mmax - mmin) // 2, mmax]

ni = 2 * nside

resource_path = '/home/dagss/wavemoth/resources/gpu/%d.dat' % nside

if not os.path.exists(resource_path) or force:
    plan = CudaShtPlan(nside=nside, lmax=lmax)
    with file(resource_path, 'w') as f:
        plan.precompute_to_stream(f, logger)
    
q = cuda.pagelocked_zeros((lmax + 1, 2, 2 * nmaps, ni), np.float64)
a = cuda.pagelocked_zeros(((lmax + 1)**2, nmaps), np.complex128)
q_gpu = cuda.mem_alloc(q.nbytes)
a_gpu = cuda.mem_alloc(a.nbytes)

plan = CudaShtPlan(nside=nside, lmax=lmax, mmin=mmin, mmax=mmax,
                   resource_path=resource_path)



# Fill with mock data
for m in test_ms:
    for odd in [0, 1]:
        for j in range(2 * nmaps):
            q[m, odd, j, :] = (1 + m) * (2 + odd) * (3 + j) * np.sin(np.arange(ni) * 0.4)

# Copy to device
stream = cuda.Stream()

t0 = time()
#with cuda_profile() as prof:
if 1:
    print time() - t0
    cuda.memcpy_htod_async(q_gpu, q, stream=stream)
    print time() - t0
    plan.execute_transpose_legendre(q_gpu, a_gpu, stream=stream)
    print time() - t0
    cuda.memcpy_dtoh_async(a, a_gpu, stream=stream)
    print time() - t0
#print stream.is_done()
stream.synchronize()
print stream.is_done()
print 'hoh', time() - t0
#print prof.format('all_transpose_legendre_transforms',
#                  nflops=plan.get_flops(),
#                  nwarps=2)
#print plan.get_in_transfer_bytes(), q.nbytes
#print plan.get_out_transfer_bytes(), a.nbytes

#print prof.format('memcpyHtoD', nflops=q.nbytes)
#print prof.format('memcpyDtoH', nflops=a.nbytes)
#print prof.kernels

# Do it with np.dot and compare for selected m's
def lm_to_idx_mmajor(l, m, lmax):
    # broadcasts
    return m * (2 * lmax - m + 3) // 2 + (l - m)

for m in test_ms:
    Lambda = compute_normalized_associated_legendre(m, plan.thetas, lmax,
                                                    epsilon=plan.epsilon_legendre)
    for odd in [0, 1]:
        Lambda_odd = Lambda[:, odd::2].T
        a0_slice = np.dot(Lambda_odd, q[m, odd, :, :].T)
        a0_slice = np.ascontiguousarray(a0_slice).view(np.complex128)

        a_slice = a[lm_to_idx_mmajor(m + odd, m, lmax):lm_to_idx_mmajor(m + 1, m + 1, lmax):2, :]
        if (m, odd) != (mmax, 1):
            print m, odd, np.linalg.norm(a_slice - a0_slice) / np.linalg.norm(a0_slice)


print 'GFLOPS performed', plan.get_flops() / 1e9
