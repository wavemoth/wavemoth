# http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/


import pyopencl as cl
import numpy as np
import numpy.linalg as la
mf = cl.mem_flags
READ_ONLY = mf.READ_ONLY
COPY_HOST_PTR = mf.COPY_HOST_PTR
WRITE_ONLY = mf.WRITE_ONLY

from matplotlib import pyplot as plt

from wavemoth import *
from wavemoth import healpix



for platform in cl.get_platforms():
    if 'Intel' in platform.name:
        ctx = cl.Context(platform.get_devices())

queue = cl.CommandQueue(ctx, 
                        properties=cl.command_queue_properties.PROFILING_ENABLE)


# Compute Lambda
nside = 2048
m = 0
lmax = 2 * nside
odd = 0

thetas = healpix.get_ring_thetas(nside, positive_only=True)
Lambda = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=1e-100)
Lambda = Lambda[:, odd::2].T

# Mock input vector
a = np.sin(np.arange(Lambda.shape[0]) * 0.4)

prg = cl.Program(ctx, """
    double get_c(int m, int l) {
       double num = (long)(l - m + 1) * (long)(l - m + 2) *
                    (long)(l + m + 1) * (long)(l + m + 2);
       double den = (long)(2 * l + 1) * (long)(2 * l + 3) *
                    (long)(2 * l + 3) * (long)(2 * l + 5);
       return sqrt(num / den);
    }

    double get_d(int m, int l) {
        double num = 2 * l * (l + 1) - 2 * m * m - 1;
        double den = (2 * l - 1) * (2 * l + 3);
        return num / den;
    }

    __kernel void transpose_legendre_transform(int m, int lmin, int nk, int nx,
                                               __global const double *x_squared,
                                               __global const double *Lambda_0,
                                               __global const double *Lambda_1,
                                               __global const double *a,
                                               __global double *out) {
      int ix = get_global_id(0);
      double c, cp, cpp, d, dp, x, y;
      double Pval, Pval_prev, Pval_prevprev;
      double xsq_val = x_squared[ix];
      
      cpp = get_c(m, lmin);
      cp = get_c(m, lmin + 2);
      dp = get_d(m, lmin + 2);

      Pval_prevprev = Lambda_0[ix];
      Pval_prev = Lambda_1[ix];
      Pval = Pval_prev;
      
      for (int k = 2; k != nk; ++k) {
        /* Compute auxiliary scalars */
        c = get_c(m, lmin + 2 * k);
        d = get_d(m, lmin + 2 * k);

        double alpha = -dp;
        double beta = 1 / cp;
        double gamma = -cpp / cp;

        cpp = cp;
        cp = c;
        dp = d;

        /* Do recurrence */
        Pval = (xsq_val + alpha) * beta * Pval_prev + gamma * Pval_prevprev;
        Pval_prevprev = Pval_prev;
        Pval_prev = Pval;
      }
      out[ix] = Pval;
    }
    """).build()
transpose_legendre_transform = prg.transpose_legendre_transform
transpose_legendre_transform.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.int32,
                                                    None, None, None, None, None])

# cl buffers
Lambda_0_cl = cl.Buffer(ctx, READ_ONLY | COPY_HOST_PTR, hostbuf=Lambda[0, :].copy())
Lambda_1_cl = cl.Buffer(ctx, READ_ONLY | COPY_HOST_PTR, hostbuf=Lambda[1, :].copy())
x_squared_cl = cl.Buffer(ctx, READ_ONLY | COPY_HOST_PTR, hostbuf=np.cos(thetas)**2)
a_cl = cl.Buffer(ctx, READ_ONLY | COPY_HOST_PTR, hostbuf=a)
out_cl = cl.Buffer(ctx, WRITE_ONLY, 8 * Lambda.shape[1])

nk, nx = Lambda.shape
e = transpose_legendre_transform(queue, (nx,), None,
                                 m, m + odd, nk, nx, x_squared_cl,
                                 Lambda_0_cl, Lambda_1_cl, a_cl, out_cl)
e.wait()
dt = (e.profile.end - e.profile.start) * 1e-9
print (nx * nk * 6) / dt / 1e9, 'GFLOP'

Lambda_last = np.zeros(nx)
cl.enqueue_copy(queue, Lambda_last, out_cl)

print Lambda_last

print la.norm(Lambda_last - Lambda[-1, :])

plt.clf()
plt.plot(Lambda_last)
plt.plot(Lambda[-1, :])
