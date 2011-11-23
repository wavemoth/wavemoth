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
ntransforms = 20

mmin = 0
mmax = lmax

test_ms = [mmin, 1, (mmax - mmin) // 2, mmax]

ni = 2 * nside

# Make precomputed data if it doesn't exist, and construct plan
resource_path = '/home/dagss/wavemoth/resources/gpu/%d.dat' % nside
if not os.path.exists(resource_path) or force:
    plan = CudaShtPlan(nside=nside, lmax=lmax)
    with file(resource_path, 'w') as f:
        plan.precompute_to_stream(f, logger)
plan = CudaShtPlan(nside=nside, lmax=lmax, mmin=mmin, mmax=mmax,
                   resource_path=resource_path)

# Set up mock data for all launches
q = cuda.pagelocked_zeros((ntransforms, lmax + 1, 2, 2 * nmaps, ni), np.float64)
a = cuda.pagelocked_zeros((ntransforms, (lmax + 1)**2, nmaps), np.complex128)
for itransform in range(ntransforms):
    for m in test_ms:
        for odd in [0, 1]:
            for j in range(2 * nmaps):
                q[itransform, m, odd, j, :] = (itransform +
                    (1 + m) * (2 + odd) * (3 + j) * np.sin(np.arange(ni) * 0.4))

# Set up buffers and streams for double-buffering. These only need to
# be large enough for one transform
nstreams = 2
stream_objects = [
    (cuda.Stream(), cuda.mem_alloc(q[0, ...].nbytes), cuda.mem_alloc(a[0, ...].nbytes))
    for i in range(nstreams)]

# Map transform data and streams/buffers together...
transform_objects = [stream_objects[itransform % nstreams] +
                     (q[itransform, ...], a[itransform, ...])
                     for itransform in range(ntransforms)]

# Set up all the jobs, interleaving them on available streams.
print '== Multi-buffered run with %d transforms at nside=%d' % (ntransforms, nside)
t0 = time()
for stream, q_gpu, a_gpu, q_slice, a_slice in transform_objects:
    cuda.memcpy_htod_async(q_gpu, q_slice, stream=stream)
    plan.execute_transpose_legendre(q_gpu, a_gpu, stream=stream)
    cuda.memcpy_dtoh_async(a_slice, a_gpu, stream=stream)
    
print 'Wall-time taken to set up instruction streams ("Python overhead"): %e' % (time() - t0)
for stream, q_gpu, a_gpu in stream_objects:
    stream.synchronize()
dt = time() - t0
print 'Wall-time taken to end of execution: %f total, %f per transform' % (dt, dt / ntransforms)
print 'Host-to-host compute rate: %f GFLOP/s' % (ntransforms * plan.get_flops() / dt / 1e9)


# Profiled, synchronous run
print
print '== Profiled run at nside=%d' % nside
with cuda_profile() as prof:
    for rep in range(3):
        stream, q_gpu, a_gpu = stream_objects[0]
        cuda.memcpy_htod(q_gpu, q[0, ...])
        plan.execute_transpose_legendre(q_gpu, a_gpu)    
        cuda.memcpy_dtoh(a[0, ...], a_gpu)

print 'Transfer in:  ', prof.format('memcpyHtoD', nflops=q.nbytes)
print 'Compute:      ', prof.format('all_transpose_legendre_transforms',
                                    nflops=plan.get_flops(),
                                    nwarps=2)
print 'Transfer out: ', prof.format('memcpyDtoH', nflops=a.nbytes)

# Check result
print
print '== Accuracy table (m, odd, relative error)'

# Do it with np.dot and compare for selected m's
def lm_to_idx_mmajor(l, m, lmax):
    # broadcasts
    return m * (2 * lmax - m + 3) // 2 + (l - m)

for m in test_ms:
    Lambda = compute_normalized_associated_legendre(m, plan.thetas, lmax,
                                                    epsilon=plan.epsilon_legendre)
    for odd in [0, 1]:
        Lambda_odd = Lambda[:, odd::2].T

        maxerr = 0
        for itransform in range(ntransforms):
            a0_slice = np.dot(Lambda_odd, q[itransform, m, odd, :, :].T)
            a0_slice = np.ascontiguousarray(a0_slice).view(np.complex128)

            if (m, odd) == (mmax, 1):
                continue # length=0, no data to check
        
            start = lm_to_idx_mmajor(m + odd, m, lmax)
            stop = lm_to_idx_mmajor(m + 1, m + 1, lmax)
            a_slice = a[itransform, start:stop:2, :]
            maxerr = max(maxerr, np.linalg.norm(a_slice - a0_slice) / np.linalg.norm(a0_slice))
        print m, odd, maxerr

