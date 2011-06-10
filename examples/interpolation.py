from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from matplotlib import pyplot as plt
np.seterr(all='warn')

# Set up data
from spherew.roots import associated_legendre_roots
from spherew.legendre import compute_normalized_associated_legendre, Plm_and_dPlm
from spherew.fmm import fmm1d
from spherew import healpix

Nside = 128
lmax = 256

N = 20

m = 0
n = (lmax - m) // 2
odd = 1
x_grid = associated_legendre_roots(m + 2 * n + odd, m)

y_grid = np.cos(healpix.get_ring_thetas(Nside, positive_only=True))
#y_grid = x_grid - (x_grid[1] - x_grid[0]) / 2
#y_grid = np.linspace(0, 1, N)
a_l = np.zeros(lmax)
a_l[0] = 10
a_l[1] = 10

z_grid = np.linspace(0, .9999, 1000)

# Make a colliding point
#y_grid[2] = x_grid[0] + 1e-10

P_z = compute_normalized_associated_legendre(
          m, np.arccos(z_grid), lmax)
f_z = np.dot(P_z[:, odd::2], a_l[odd::2])

P_y = compute_normalized_associated_legendre(
          m, np.arccos(y_grid), lmax)
f_y = np.dot(P_y[:, odd::2], a_l[odd::2])

P_x = compute_normalized_associated_legendre(
        m, np.arccos(x_grid), lmax)
f_x = np.dot(P_x[:, odd::2], a_l[odd::2])

print P_x.shape, f_x.shape
print f_x

# The derivative
l = m + 2 * n + odd
_, dP_x = Plm_and_dPlm(l, m, x_grid)

P_m_2n_sub_2_at_x_n, _ = Plm_and_dPlm(m + 2 * n + odd - 2, m, x_grid)
P_m_2n_y, _ = Plm_and_dPlm(m + 2 * n + odd, m, y_grid)

rho_n = 2 * (2 * m + 4 * n + 1 + 2 * odd) / ((1 - x_grid**2) * (dP_x)**2)

def get_c(l, m):
    n = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    d = (2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)
    return np.sqrt(n / d)

l = m + 2 * n - 2
c = get_c(m + 2 * n - 2, m)

# The FMM operation
x = x_grid[None, :]
y = y_grid[:, None]
if 0:
    K = 1 / (2 * x * (y - x) + (y - x)**2)
else:
    K = 1 / (y**2 - x**2)
f_ip = c * P_m_2n_y * np.dot(K, P_m_2n_sub_2_at_x_n * rho_n * f_x)

#print 'Diff problem spot y vs x: %e' % (f_y[2] - f_x[0])
#print 'Diff problem spot ip vs x: %e' % (f_ip[2] - f_x[0])
#print 'Diff problem spot ip vs y: %e' % (f_ip[2] - f_y[2])
print 'Diff: %e' % np.linalg.norm(f_ip - f_y)


plt.clf()
plt.plot(x_grid, f_x, 'ro')
plt.plot(z_grid, f_z, 'b-')
plt.plot(y_grid, f_ip, 'gx')
plt.plot(y_grid, f_y, 'g.')
plt.axhline(0, color='black')
plt.gca().set_ylim((-3, 30))
