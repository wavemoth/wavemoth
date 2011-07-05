from __future__ import division

# Investigate whether the P_l-2, P_l -> P_l+2-scheme
# listed in Tygert (2008) (SHT II) gives the same answers up
# to numerical accuracy as when libpsht steps in l in increments of 1.
#
# The answer is yes, at least for Nside=2048

# Stick .. in PYTHONPATH
import sys
import os
import itertools

from spherew import *
from spherew.healpix import *
from spherew.roots import *
from cmb import as_matrix
from matplotlib import pyplot as plt

np.seterr(all='raise')

Nside = 2048
lmax = 2 * Nside
epsilon_legendre = 1e-30
odd = 0
m = 0

nodes = get_ring_thetas(Nside, positive_only=True)
x_list = np.cos(nodes)


P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
P = P.T
P = P[odd::2, :]

ls = np.arange(m + odd, lmax + 1, 2)

def get_c_l(l):
    num = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    den = (2 * l + 1) * (2 * l + 3) * (2 * l + 3) * (2 * l + 5)
    return np.sqrt(num / den)

def get_d_l(l):
    num = 2 * l * (l + 1) - 2 * m * m - 1
    den = (2 * l - 1) * (2 * l + 3)
    return num / den

c_l = get_c_l(ls)
d_l = get_d_l(ls)

Ptilde = P.copy()
Ptilde[2:, :] = 0
for idx in range(2, ls.shape[0]):
    l = ls[idx]
    assert (l - m) % 2 == odd
    # Term for l-4
    a = c_l[idx - 2] * P[idx - 2, :]
    # Term for l-2
    b = (d_l[idx - 1] - x_list**2) * P[idx - 1, :]
#    print a + b
    # Term for l
    Ptilde[idx, :] = - (a + b) / c_l[idx - 1]
    


print np.max(np.abs(P - Ptilde) / np.abs(P))
print np.max(np.abs(P - Ptilde))
print np.linalg.norm(P - Ptilde) / np.linalg.norm(P)

#plt.clf()
#plt.semilogy(ls, np.max(np.abs(P - Ptilde), axis=1))
#plt.plot(ls, P[:, [0]])
#plt.plot(ls, Ptilde[:, [0]])
