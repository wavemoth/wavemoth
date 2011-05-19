from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))
                
from spherew import *
from spherew.healpix import *
import numpy as np
from numpy import pi
from cmb.oomatrix import as_matrix
from matplotlib import pyplot as plt

#
# Some parameters
#
C = 60*2
eps = 1e-10

#
# Tiny object-oriented matrix library; the compressed matrix
# is represented as a tree of these matrices of various types.
# 
# In C, the matrix is rather represented as a stream of contiguous
# data with flags specifying what type the data is.
#

class Dense(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape

    def apply(self, x):
        return np.dot(self.A, x)

    def size(self):
        return np.prod(self.A.shape)

class Butterfly(object):
    def __init__(self, T_p, B_p, L_p, R_p, T_obj, B_obj):
        self.T_p = T_p
        self.B_p = B_p
        self.L_p = L_p
        self.R_p = R_p
        self.T_obj = T_obj
        self.B_obj = B_obj
        self.hmid = L_p.shape[1]
        self.shape = (T_obj.shape[0] + B_obj.shape[0], L_p.shape[1] + R_p.shape[1])

    def apply(self, x):
        a = np.dot(self.L_p, x[:self.hmid])
        b = np.dot(self.R_p, x[self.hmid:])
        x = np.hstack([a, b])

        a = np.dot(self.T_p, x)
        a = self.T_obj.apply(a)
        
        b = np.dot(self.B_p, x)
        b = self.B_obj.apply(b)
        return np.hstack([a, b])

    def size(self):
        return (sum(np.prod(x.shape) for x in
                   [self.T_p, self.B_p, self.L_p, self.R_p]) +
                self.T_obj.size() + self.B_obj.size())

class HStack:
    def __init__(self, lst):
        self.lst = lst
        self.m = lst[0].shape[0]
    def apply(self, x):
        y = np.zeros(self.m)
        ix = 0
        for A in self.lst:
            y += A.apply(x[ix:ix + A.shape[1]])
            ix += A.shape[1]
        return y
    
    def size(self):
        return sum(x.size() for x in self.lst)


#
# The butterfly algorithm
#

def decomp(A, eps=eps):
    n = A.shape[1]
    A_tmp, k, ilist, rnorms = interpolative_decomposition(eps, A.copy('F'))
    ilist -= 1
    A_tilde = np.zeros((k, n))
    A_tilde[:, ilist[:k]] = np.eye(k)
    A_tilde[:, ilist[k:]] = A_tmp
    print 'n=%4d k=%4d' % (n, k)
    return A[:, ilist[:k]], A_tilde
       
def butterfly(A):
    hmid = A.shape[1] // 2
    if hmid <= 4:
        return Dense(A)        
    L = A[:, :hmid]
    L_subset, L_p = decomp(L)
    
    R = A[:, hmid:]
    R_subset, R_p = decomp(R)
        
    S = np.hstack([L_subset, R_subset])
    del L_subset
    del R_subset
    
    vmid = S.shape[0] // 2
    T_subset, T_p = decomp(S[:vmid, :])
    B_subset, B_p = decomp(S[vmid:, :])

    T_obj = butterfly(T_subset)
    B_obj = butterfly(B_subset)
    
    return Butterfly(T_p, B_p, L_p, R_p, T_obj, B_obj)

#
# Spherical harmonic transform for a single l,
# down to HEALPix grid
#

class InnerSumPerM:
    def __init__(self, m, x_grid, lmax):
        P = compute_normalized_associated_legendre(m, np.arccos(x_grid), lmax)
        assert x_grid[-1] == 0
        self.P_even = Dense(P[:-1, ::2])
        self.P_odd = Dense(P[:-1, 1::2])
        # Treat equator seperately, as we cannot interpolate to it from
        # samples in (0, 1). Only need even part, as odd part will be 0.
        self.P_equator = Dense(P[-1:, ::2])

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
    g_m_theta = np.zeros((4 * Nside - 1, 4 * Npix), dtype=np.complex)
    print g_m_theta.shape, g.shape
#    plt.clf()
#    plt.plot(g.real)
    g_m_theta[:, m] = g

    idx = 0

    phi0_arr = get_ring_phi0(Nside)

    for i, (rn, phi0) in enumerate(zip(get_ring_pixel_counts(Nside), phi0_arr)):
        g_m = g_m_theta[i, :rn // 2 + 1]
        # Phase-shift to phi_0
        g_m = g_m * np.exp(1j * m * phi0)
        ring = np.fft.irfft(g_m, rn)
        ring *= rn # see np.fft convention
    #    print ring
        map[idx:idx + rn] = ring
        idx += rn

    return map


#
# Parameters
#

lmax = 16
Nside = 16
m = 2
a_l = np.zeros(lmax + 1 - m)
a_l[3 - m] = 1
a_l[4 - m] = -1
#a_l[15] = 0.1
#a_l = (-1) ** np.zeros(lmax + 1)

    
if 1:
    map = alm2map(m, a_l, Nside)

    from cmb.maps import pixel_sphere_map, harmonic_sphere_map
#    pixel_sphere_map(map).plot(title='fast')

    alm_fid = harmonic_sphere_map(0, lmin=0, lmax=lmax, is_complex=False)
    assert m != 0
    for l in range(m, lmax + 1):
        alm_fid[l**2 + l + m] = np.sqrt(2) * a_l[l - m] # real is repacked

#    alm_fid.to_pixel(Nside).plot(title='fiducial')
    (map - alm_fid.to_pixel(Nside)).plot(title='diff')
