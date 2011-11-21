from __future__ import division

import os
import sys
import socket

if socket.gethostname() != 'dagss-laptop':
    sys.path.append('/home/dagss/wavemoth')

import numpy as np
import numpy.linalg as la
import logging

from matplotlib import pyplot as plt

print 'Trying to get CUDA'
import pycuda.autoinit
print 'got CUDA'

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

if 0:
    mmin = 0
    mmax = lmax // 2
else:
    mmin = lmax // 2 + 1
    mmax = lmax
if 1:
    mmin = 0
    mmax = lmax

test_ms = [mmin, 1, (mmax - mmin) // 2, mmax]

resource_path = '/home/dagss/wavemoth/resources/gpu/%d.dat' % nside

if not os.path.exists(resource_path) or force:
    plan = CudaShtPlan(nside=nside, lmax=lmax)
    with file(resource_path, 'w') as f:
        plan.precompute_to_stream(f, logger)
    
plan = CudaShtPlan(nside=nside, lmax=lmax, mmin=mmin, mmax=mmax, resource_path=resource_path)
ni = plan.ni

q = np.zeros((lmax + 1, 2, 2 * nmaps, ni))
a = np.zeros(((lmax + 1)**2, nmaps), dtype=np.complex128)

# Fill with mock data
for m in test_ms:
    for odd in [0, 1]:
        for j in range(2 * nmaps):
            q[m, odd, j, :] = (1 + m) * (2 + odd) * (3 + j) * np.sin(np.arange(ni) * 0.4)

with cuda_profile() as prof:
    plan.execute_transpose_legendre(q, a)

print prof.format('all_transpose_legendre_transforms',
                  nflops=plan.get_flops(),
                  nwarps=2)

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


print 'FLOPS', plan.get_flops()
