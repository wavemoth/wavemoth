from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

import numpy as np
from matplotlib import pyplot as plt

N = 20
y_grid = np.linspace(0, 1, N)

z_grid = np.linspace(0, 1, 200)

# Set up data
from spherew.roots import associated_legendre_roots
from spherew.legendre import compute_normalized_associated_legendre
m = 5
n = 6
#f_n = np.asarray([0.2, 0.4, 0.5, -0.3, 1, 1.3])#, 0.8, -.3, .3, .4])
x_grid = associated_legendre_roots(m + 2 * n, m)

a_l = np.asarray([0.2, 0.4, 0.5, -0.3, 1, 1.3])
P_z = compute_normalized_associated_legendre(
          m, np.arccos(z_grid), m + 2 * n - 2)
f_z = np.dot(P_z[:, ::2], a_l)

P_y = compute_normalized_associated_legendre(
          m, np.arccos(y_grid), m + 2 * n - 2)
f_y = np.dot(P_y[:, ::2], a_l)

P_x = compute_normalized_associated_legendre(
        m, np.arccos(x_grid), m + 2 * n)
f_x = np.dot(P_x[:, :-2:2], a_l)
# The derivative
l = m + 2 * n
refudge = np.sqrt((2 * l + 1) / (2 * l - 1) * (l - m) / (l + m))
dP_x = l * x_grid * P_x[:, l - m] - (l + m) * refudge * P_x[:, l - m - 1]
dP_x /= x_grid**2 - 1

P_m_2n_sub_2_at_x_n = compute_normalized_associated_legendre(
    m, np.arccos(x_grid), m + 2 * n - 2)[:, -1]
P_m_2n_y = compute_normalized_associated_legendre(
    m, np.arccos(y_grid), m + 2 * n)[:, -1]

rho_n = 2 * (2 * m + 4 * n + 1) / ((1 - x_grid**2) * (dP_x)**2)

#

def get_c(l, m):
    n = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    d = (2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)
    return np.sqrt(n / d)

l = m + 2 * n - 2
c = get_c(m + 2 * n - 2, m)


K = 1 / np.subtract.outer(y_grid**2, x_grid**2)

#K *= P_m_2n_y[:, None]

f_ip = np.dot(K, P_m_2n_sub_2_at_x_n * rho_n * f_x)
f_ip *= c * P_m_2n_y

plt.clf()
#plt.plot(y_grid, 5*P_m_2n_y)
plt.plot(z_grid, f_z, 'b-')
plt.plot(y_grid, f_ip, 'gx')
plt.plot(y_grid, f_y, 'g.')
plt.plot(x_grid, f_x, 'ro')
#plt.plot(y_grid, 1 / (y_grid - x_n[3]), 'r-')
plt.axhline(0, color='black')
plt.gca().set_ylim((-3, 3))
#for x in x_n:
#    plt.axvline(x)