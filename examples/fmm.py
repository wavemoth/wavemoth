from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import lpmn

N = 100
y_grid = np.linspace(0, 1, N)

# Set up data
from spherew.roots import associated_legendre_roots
from spherew.legendre import compute_normalized_associated_legendre
m = 5
n = 10
f_n = np.asarray([0.2, 0.4, 0.5, -0.3, 1, 1.3, 0.8, -.3, .3, .4])
x_n = associated_legendre_roots(m + 2 * n, m)

y_grid = x_n + 0.01
y_grid = y_grid[:-1]

P_m_2n_sub_2_at_x_n = compute_normalized_associated_legendre(
    m, np.arccos(x_n), m + 2 * n - 2)[:, -1]
P_m_2n_y = compute_normalized_associated_legendre(
    m, np.arccos(y_grid), m + 2 * n - 2)[:, -1]

# Compute rho
dP_m_2n = np.zeros_like(x_n)
l = m + 2 * n
fact = 1
for mm in range(-abs(m), abs(m) + 1):
    fact /= (l + mm)
normalization = np.sqrt((2 * l + 1) / 2 * fact)
for i, x in enumerate(x_n):
    w, dw = lpmn(m, l, x)
    dP_m_2n[i] = dw[-1, -1]
    dP_m_2n[i] *= normalization
rho_n = 2 * (2 * m + 4 * n + 1) / ((1 - x_n**2) * (dP_m_2n)**2)

#

l = m + 2 * n - 2
c = np.sqrt((2 * n + 1) * 2 * n * (2 * n + 2 * m - 1) * (2 * n + 2 * m) /
            ((2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)))


K = 1 / np.subtract.outer(y_grid**2, x_n**2)
f_y = np.dot(K, P_m_2n_sub_2_at_x_n * rho_n * f_n)
f_y *= c * P_m_2n_y

plt.plot(y_grid, f_y)
plt.plot(x_n, f_n, 'ro')
#for x in x_n:
#    plt.axvline(x)
