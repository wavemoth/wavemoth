from __future__ import division

import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

from time import clock
from wavemoth.fastsht import ShtPlan
from wavemoth.psht import PshtMmajorHealpix
from wavemoth.legendre import *
from wavemoth.healpix import *

import hashlib

def plot_map(m, title=None):
    from cmb.maps import pixel_sphere_map
    pixel_sphere_map(m[0, :]).plot(title=title)

Nside = 256
lmax = 2 * Nside
nmaps = 1

ring_counts = get_ring_pixel_counts(Nside)
ring_offsets = np.cumsum(np.r_[0, ring_counts])[2*Nside-1:]


input = np.zeros(((lmax + 1) * (lmax + 2) // 2, nmaps), dtype=np.complex128)
sht_output = np.zeros((12 * Nside**2, nmaps))
psht_output = np.zeros((12 * Nside**2, nmaps), order='F')

sht_plan = ShtPlan(Nside, lmax, lmax, input, sht_output, 'mmajor')
psht_plan = PshtMmajorHealpix(lmax=lmax, Nside=Nside, nmaps=nmaps)

def hash_array(x):
    h = hashlib.sha1()
    h.update(x)
    return h.hexdigest()

def do_comparison():
    psht_plan.alm2map(input, psht_output)
    sht_plan.execute()
    d = norm(sht_output - psht_output)
    f = norm(psht_output)
#    print d
#    print f
    return norm(sht_output - psht_output) / norm(psht_output)

def lm_to_idx(l, m):
    return m * (2 * lmax - m + 3) // 2 + (l - m)

residuals = []
ms = []
ls = []

misscount = 0

#l = 3
#m = 20
nodes = get_ring_thetas(Nside, positive_only=True)
#print compute_normalized_associated_legendre(m, nodes, l,
#                                             epsilon=1e-300)[4, :]


def compute(l, m):
    input[...] = 0
    idx = lm_to_idx(l, m)
    input[idx, :] = 2 + 1j if m > 0 else 1
    return do_comparison()

count = 0
def doit(l, m):
    global count
    z = compute(l, m)
    return z


def compute_single_point(l, m, iring):
    print l, m
    phi0 = get_ring_phi0(Nside)[2*Nside-1+iring]
    Plm = compute_normalized_associated_legendre(m, nodes[iring:iring + 1], l,
                                                 epsilon=1e-30)
    print Plm, np.exp(1j * m * phi0), np.exp(-1j * m * phi0)
    return Plm * np.exp(1j * m * phi0) + (-1)**m * Plm.conjugate() * np.exp(-1j * m * phi0)
    


residuals = []
for l in range(0, lmax + 1, 200):
    for m in range(0, l + 1, 200):
        print l, m
        residuals.append(doit(l, m))

plt.semilogy(residuals)
