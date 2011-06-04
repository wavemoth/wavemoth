from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

import numpy as np
from matplotlib import pyplot as plt
np.seterr(all='raise')
np.seterr(under='warn')
Nx = 2000
Ny = 1000

x_N = 1
x_0 = 0

qx = np.sin(np.arange(Nx) / 2)
x_grid = np.linspace(x_0, x_N, Nx)
y_grid = np.linspace(x_0 + 0.01, x_N - 0.01, Ny)

# Copmute another direct answer
inv_K = np.subtract.outer(y_grid, x_grid)
K = 1 / inv_K
phi_y0 = np.dot(K, qx)

# Try FMM

from gaussexp import gauss_exp_x, gauss_exp_w
M = gauss_exp_x.shape[0]

# Find rescale factor s and rescale quadrature
a, b = 1, 500
#dx = both_grid[1:] - both_grid[:-1]

both_grid = np.hstack([x_grid, y_grid])
inv_K_both = np.subtract.outer(both_grid, both_grid)
hi = b / inv_K_both.max()

s = hi

assert (both_grid[-1] - both_grid[0]) <= b / s

t_arr = gauss_exp_x * s
w_arr = gauss_exp_w * s
r = a / s # minimum range of expansions

# Exponential expansion at first point
phi = np.zeros(Ny)

# Pass rightwards
alpha = np.zeros(M)
alpha[:] = 0
i_far = 0
x_far = None
for i_y in range(Ny):
    # Translate expansion from i-1 to i
    while i_far < Nx and x_grid[i_far] < y_grid[i_y] - r:
        if i_far > 0:
            dx = x_grid[i_far - 1] - x_grid[i_far]
            alpha *= np.exp(dx * t_arr)
            # Add charge i to expansion
        alpha += qx[i_far]
        x_far = x_grid[i_far]
        i_far += 1

    if x_far is not None:
        dx = x_far - y_grid[i_y]
        phi[i_y] += np.dot(w_arr, alpha * np.exp(dx * t_arr))

    # Brute force
    i_x = i_far
    while i_x < Nx and x_grid[i_x] <= y_grid[i_y] + r:
        phi[i_y] += qx[i_x] / (y_grid[i_y] - x_grid[i_x])
        i_x += 1
        

# Pass leftwards
alpha[:] = 0
i_far = Nx - 1
x_far = None
for i_y in range(Ny - 1, -1, -1):
    # Translate expansion from i-1 to i
    while i_far >= 0 and x_grid[i_far] > y_grid[i_y] + r:
        if i_far < Nx - 1:
            dx = x_grid[i_far] - x_grid[i_far + 1]
            alpha *= np.exp(dx * t_arr)
            # Add charge i to expansion
        alpha += qx[i_far]
        x_far = x_grid[i_far]
        i_far -= 1

    if x_far is not None:
        dx = y_grid[i_y] - x_far
        phi[i_y] -= np.dot(w_arr, alpha * np.exp(dx * t_arr))

plt.clf()
#plt.plot(both_grid, phi0)
#plt.plot(both_grid, phi)
plt.plot(y_grid[:100], phi[:100])
plt.plot(y_grid[:100], phi_y0[:100])
plt.gca().set_ylim((-.1e5, .1e5))
#print np.max(np.abs(phi0 - phi) / np.max(np.abs(phi0)))
print np.max(np.abs(phi_y0 - phi) / np.max(np.abs(phi_y0)))
