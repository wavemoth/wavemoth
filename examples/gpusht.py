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

nside = 64
nmaps = 1
lmax = 2 * nside
odd = 0

resource_path = '/home/dagss/wavemoth/resources/gpu/%d.dat' % nside

if not os.path.exists(resource_path) or True:
    plan = CudaShtPlan(nside=nside, lmax=lmax)
    with file(resource_path, 'w') as f:
        plan.precompute_to_stream(f, logger)
    
plan = CudaShtPlan(nside=nside, lmax=lmax, resource_path=resource_path)
ni = plan.ni

q = np.zeros((lmax + 1, 2, 2 * nmaps, ni))
a = np.zeros(((lmax + 1)**2, nmaps), dtype=np.complex128)

with cuda_profile() as prof:
    plan.execute_transpose_legendre(q, a)

print prof.format('all_transpose_legendre_transforms',
                  nflops=1e9,
                  nwarps=2)
    
print a.shape, q.shape
print np.std(a)
print a[0,0]
