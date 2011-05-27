from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))
                
from spherew import *
from spherew.healpix import *
from spherew.butterfly import butterfly_compress, serialize_butterfly_matrix
import numpy as np
from numpy import pi, prod
from cmb.oomatrix import as_matrix
from matplotlib import pyplot as plt

#
# Some parameters
#
eps = 1e-15
limit = 32


#
# The butterfly algorithm
#

class DenseMatrix(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape

    def apply(self, x):
        return np.dot(self.A, x)

    def apply_left(self, x):
        return np.dot(x, self.A)

    def size(self):
        return prod(self.shape)

#
# Spherical harmonic transform for a single l,
# down to HEALPix grid
#

class InnerSumPerM:
    def __init__(self, m, x_grid, lmax, compress=True):
        P = compute_normalized_associated_legendre(m, np.arccos(x_grid), lmax, epsilon=1e-30)
        assert x_grid[-1] == 0

        self.P_even_arr = P_even_arr = P[:-1, ::2] # drop equator
        self.P_odd_arr = P_odd_arr = P[:-1, 1::2]
        if compress:
            self.P_even = butterfly_compress(P_even_arr, min_rows=limit)
            self.P_odd = butterfly_compress(P_odd_arr, min_rows=limit)
            print 'Compression:', ((self.P_even.size() + self.P_odd.size()) / 
                (prod(P_even_arr.shape) + prod(P_odd_arr.shape)))
            print 'Ratio in final blocks:', (
                (self.P_even.S_node.size() + self.P_odd.S_node.size()) /
                (self.P_even.size() + self.P_odd.size()))
            self.P_even = serialize_butterfly_matrix(self.P_even)
            self.P_odd = serialize_butterfly_matrix(self.P_odd)
        else:
            self.P_even = DenseMatrix(P_even_arr)
            self.P_odd = DenseMatrix(P_odd_arr)
        # Treat equator seperately, as we cannot interpolate to it from
        # samples in (0, 1). Only need even part, as odd part will be 0.
        self.P_equator = DenseMatrix(P[-1:, ::2])

    def compute(self, a_l):
        a_l_even = a_l[::2]
        a_l_odd = a_l[1::2]
        g_even = self.P_even.apply(a_l_even.real) + 1j * self.P_even.apply(a_l_even.imag)
        g_odd = self.P_odd.apply(a_l_odd.real) + 1j * self.P_odd.apply(a_l_odd.imag)
        g_equator = self.P_equator.apply(a_l_even.real) + 1j * self.P_equator.apply(a_l_even.imag)
        # We have now computed for all cos(theta) >= 0. Retrieve the
        # opposite hemisphere by symmetry.
        g_even = np.hstack([g_even, g_equator, g_even[::-1]])
        g_odd = np.hstack([g_odd, 0, -g_odd[::-1]])
        return g_even + g_odd
        

def al2gmtheta(m, a_l, theta_arr):
    lmax = a_l.shape[0] - 1 + m
    x = np.cos(theta_arr)
    x[np.abs(x) < 1e-10] = 0
    xneg = x[x < 0]
    xpos = x[x > 0]
    assert np.allclose(-xneg[::-1], xpos)
    return InnerSumPerM(m, x[x >= 0], lmax).compute(a_l)

def alm2map(m, a_l, Nside):
    theta = get_ring_thetas(Nside)
    g = al2gmtheta(m, a_l, theta)
    Npix = 12 * Nside**2
    map = np.zeros(Npix)
    g_m_theta = np.zeros((4 * Nside - 1, lmax + 1), dtype=np.complex)
    print g_m_theta.shape, g.shape
#    plt.clf()
#    plt.plot(g.real)
    g_m_theta[:, m] = g

    from spherew.fastsht import ShtPlan
    fake_input = np.zeros(1, dtype=np.double)
    g_m_theta = g_m_theta.reshape((4 * Nside - 1) * (lmax + 1)).view(np.double)
    plan = ShtPlan(Nside, lmax, lmax, fake_input, map, g_m_theta, 'mmajor')
    plan.perform_backward_ffts(0, 4 * Nside - 1)
    return map

##     else:
##         idx = 0
##         phi0_arr = get_ring_phi0(Nside)
##         for i, (rn, phi0) in enumerate(zip(get_ring_pixel_counts(Nside), phi0_arr)):
##             g_m = g_m_theta[i, :rn // 2 + 1]
##             # Phase-shift to phi_0
## #            g_m = g_m * np.exp(1j * m * phi0)
##             ring = np.fft.irfft(g_m, rn)
##             ring *= rn # see np.fft convention
##         #    print ring
##             map[idx:idx + rn] = ring
##             idx += rn

##     return map


#
# Parameters
#

# 8000/4096: 0.0628


Nside = 8
lmax = 2 * Nside#2000

#lmax = 200
#Nside = 64
#m = 2
#lmax = 2000
#Nside = 1024
#lmax = 1000
#Nside = 512
m = 1
a_l = np.zeros(lmax + 1 - m)
a_l[1 - m] = 1
#a_l[4 - m] = -1
#a_l[15] = 0.1
#a_l = (-1) ** np.zeros(lmax + 1)

from joblib import Memory
memory = Memory('joblib')

@memory.cache
def getroots(l, m):
    return associated_legendre_roots(lmax + 1, m)
    

if 0:
#    roots = getroots(lmax + 1, m)
    roots = get_ring_thetas(Nside)[2*Nside-1:]
    P = compute_normalized_associated_legendre(m, roots, lmax)
    #SPeven = butterfly_horz(P[::2])
    #SPodd = butterfly_horz(P[1::2])
    Peven = P[:, ::2]
#    as_matrix(np.log(np.abs(Peven))).plot()
    SPeven = butterfly(P[:, ::2])
    SPodd = butterfly(P[:, 1::2])
    print 'Compression', SPeven.size() / DenseMatrix(P[:, ::2]).size()
    print 'Compression', SPodd.size() / DenseMatrix(P[:, 1::2]).size()

if 0:
    x = np.cos(get_ring_thetas(Nside))
    x[np.abs(x) < 1e-10] = 0
    xneg = x[x < 0]
    xpos = x[x > 0]
    assert np.allclose(-xneg[::-1], xpos)
    X = InnerSumPerM(m, x[x >= 0], lmax)
    P_even_arr = X.P_even_arr.copy('F')
    P_even = X.P_even
    a_l_even = a_l[::2].copy()

    
if 1:
    map = alm2map(m, a_l, Nside)

    from cmb.maps import pixel_sphere_map, harmonic_sphere_map
    pixel_sphere_map(map).plot(title='fast')

    alm_fid = harmonic_sphere_map(0, lmin=0, lmax=lmax, is_complex=False)
    assert m != 0
    for l in range(m, lmax + 1):
        alm_fid[l**2 + l + m] = np.sqrt(2) * a_l[l - m] # real is repacked
    print 'Diff', np.linalg.norm(map - alm_fid.to_pixel(Nside))
    alm_fid.to_pixel(Nside).plot(title='fiducial')
    #(map - alm_fid.to_pixel(Nside)).plot(title='diff')
