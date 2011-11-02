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
    else:
        wanted = 'NVIDIA'
    if wanted in platform.name:
        ctx = cl.Context(platform.get_devices())

queue = cl.CommandQueue(ctx, 
                        properties=cl.command_queue_properties.PROFILING_ENABLE)


# Compute Lambda
nside = 128
m = 0
lmax = 2 * nside
odd = 0
nvecs = 2
nblocks = 1000
repeat = 5

thetas = healpix.get_ring_thetas(nside, positive_only=True)
Lambda = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=1e-100)
Lambda = Lambda[:, odd::2].T


def hrepeat(x, n):
    return np.repeat(x[:, None], n, axis=1).copy('F')

# Mock input vector
q = hrepeat(np.sin(np.arange(Lambda.shape[1]) * 0.4), nvecs * nblocks).reshape(
    (Lambda.shape[1], nvecs, nblocks), order='F')
q_cl = cl.to_device(queue, q)
Lambda_0_cl = cl.to_device(queue, hrepeat(Lambda[0, :], nblocks))
Lambda_1_cl = cl.to_device(queue, hrepeat(Lambda[1, :], nblocks))
x_squared_cl = cl.to_device(queue, hrepeat(np.cos(thetas)**2, nblocks))
out_cl = cl.zeros(queue, (Lambda.shape[0], nvecs, nblocks), dtype=np.double, order='F')

times = []
for rep in range(repeat):
    kernel = ClLegendreKernel(ctx, nvecs=nvecs)
    e = kernel.transpose_legendre_transform(queue, m, m + odd, x_squared_cl, 
                                            Lambda_0_cl, Lambda_1_cl, q_cl, out_cl)
    e.wait()
    times.append((e.profile.end - e.profile.start) * 1e-9)
dt = min(times)
nk, nx = Lambda.shape
matrix_elements = nblocks * nx * nk
UOP = matrix_elements * (6 + 2 * nvecs)
print UOP / dt / 1e9, 'GUOP/s'

a = out_cl.get()
assert np.all(a[:, :, 0:1] == a)
a = a[:, :, 0]

a0 = np.dot(Lambda, q[:, :, 0])

#print np.hstack([a, a0])

print la.norm(a - a0)

#plt.clf()
