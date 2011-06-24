from __future__ import division

import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

from time import clock
from spherew.fastsht import ShtPlan
from spherew.psht import PshtMmajorHealpix
from spherew.healpix import get_ring_thetas
from spherew.legendre import *
from spherew.butterfly import *

def plot_map(m):
    from cmb.maps import pixel_sphere_map
    pixel_sphere_map(m[0, :]).plot()

Nsides = [64, 128, 256]
m_fractions = [0, 0.25, 0.45, 0.9, 1]

plt.clf()

min_rows = 128
eps = 1e-15

for Nside in Nsides:
    lmax = 2 * Nside
    odd = 1

    for p in m_fractions:
        m = int(p * lmax)
        
        nodes = get_ring_thetas(Nside, positive_only=True)
        P = compute_normalized_associated_legendre(m, nodes, lmax)[:, odd::2]
        print P.shape
        C = butterfly_compress(P, min_rows=min_rows, eps=eps)
        print C.get_stats()

        residuals = []

        a_l = np.zeros(P.shape[1])
        for l in range(0, P.shape[1]):
            a_l[l] = 1
            x = C.apply(a_l[:, None])[:, 0]
            d = np.dot(P, a_l)
            a_l[l] = 0
            residuals.append(norm(x - d) / norm(d))

        if np.all(np.asarray(residuals) == 0):
            print 'all 0'
        else:
            plt.semilogy(residuals)
plt.gca().set_ylim((1e-20, 1))
