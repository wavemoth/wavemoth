# http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/


import numpy as np
import numpy.linalg as la

from matplotlib import pyplot as plt

from wavemoth import *
from wavemoth import healpix
from wavemoth.cl import ClLegendreKernel
import wavemoth.cl.flatpyopencl as cl


for platform in cl.get_platforms():
    if 'Intel' in platform.name:
        ctx = cl.Context(platform.get_devices())

queue = cl.CommandQueue(ctx, 
                        properties=cl.command_queue_properties.PROFILING_ENABLE)


# Compute Lambda
nside = 128
m = 0
lmax = 2 * nside
odd = 0
nvecs = 2


thetas = healpix.get_ring_thetas(nside, positive_only=True)
Lambda = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=1e-100)
Lambda = Lambda[:, odd::2].T

# Mock input vector
q = np.sin(np.arange(Lambda.shape[1]) * 0.4)
q = np.vstack([q] * nvecs).T
q_cl = cl.to_device(queue, q)
Lambda_0_cl = cl.to_device(queue, Lambda[0, :].copy())
Lambda_1_cl = cl.to_device(queue, Lambda[1, :].copy())
x_squared_cl = cl.to_device(queue, np.cos(thetas)**2)
out_cl = cl.zeros(queue, (Lambda.shape[0], nvecs), dtype=np.double, order='F')

print q_cl.strides

kernel = ClLegendreKernel(ctx, nvecs=nvecs)
e = kernel.transpose_legendre_transform(queue, m, m + odd, x_squared_cl, 
                                        Lambda_0_cl, Lambda_1_cl, q_cl, out_cl)
e.wait()
dt = (e.profile.end - e.profile.start) * 1e-9
nk, nx = Lambda.shape
print (nx * nk * 6) / dt / 1e9, 'GUOP'

a = out_cl.get()
a0 = np.dot(Lambda, q)

print np.hstack([a, a0])

print la.norm(a - a0)

plt.clf()
