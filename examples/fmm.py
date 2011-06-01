from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

import numpy as np
from matplotlib import pyplot as plt
np.seterr(all='warn')

N = 200

x_N = 1
x_0 = 0

q = np.sin(np.arange(N) / 200)
x_grid = np.linspace(x_0, x_N, N)

# Compute correct answer in O(N^2)
iarr = np.arange(x_grid.shape[0])
inv_K = np.subtract.outer(x_grid, x_grid)
inv_K[iarr, iarr] = 1
K = 1 / inv_K
K[iarr, iarr] = 0
phi0 = np.dot(K, q)

# Try FMM

from gaussexp import gauss_exp_x, gauss_exp_w
M = gauss_exp_x.shape[0]

# Find rescale factor s and rescale quadrature
a, b = 1, 500
dx = x_grid[1:] - x_grid[:-1]
low = a / dx.min()
hi = b / dx.max()

s = low

assert dx.min() >= a / s
assert (x_grid[-1] - x_grid[0]) <= b / s
print 'dx in (%e, %e); a=%s, b=%s' % (dx.min(), x_grid[-1] - x_grid[0], a/s, b/s)

t_arr = gauss_exp_x * s
w_arr = gauss_exp_w * s
r = a / s # minimum range of expansions

# Exponential expansion at first point
phi = np.zeros(N)

# Pass rightwards
alpha = np.zeros(M)
alpha[:] = q[0]
for i in range(1, N):
    # Translate expansion from i-1 to i
    dx = x_grid[i - 1] - x_grid[i]
    alpha *= np.exp(dx * t_arr)
    # Evaluate potential at point i
    phi[i] += np.dot(w_arr, alpha)
    # Add charge i to expansion
    alpha += q[i]

# Pass leftwards
t_arr *= -1
w_arr *= -1
alpha[:] = q[-1]
for i in range(N - 2, -1, -1):
    # Translate expansion from i+1 to i
    dx = x_grid[i + 1] - x_grid[i]
    alpha *= np.exp(dx * t_arr)
    # Evaluate potential at point i
    phi[i] += np.dot(w_arr, alpha)
    # Add charge i to expansion
    alpha += q[i]

plt.clf()
plt.plot(x_grid, phi0)
plt.plot(x_grid, phi)
print np.max(np.abs(phi0 - phi) / np.max(np.abs(phi0)))
