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

lmax = 32
Nside = 16
eps = 1e-10
m = 2
lsig = 30

theta = get_ring_thetas(Nside)

def bench():
    compute_normalized_associated_legendre(0, theta, lmax, out=P)

P = compute_normalized_associated_legendre(m, theta, lmax)
#print np.std(P)
#as_matrix(P).plot()

alm = np.zeros(lmax + 1, dtype=np.complex)
alm[lsig] = 1

def decomp(A, eps=eps):
    n = A.shape[1]
    A_tmp, k, ilist, rnorms = interpolative_decomposition(eps, A.copy('F'))
    ilist -= 1
    A_tilde = np.zeros((k, n))
    A_tilde[:, ilist[:k]] = np.eye(k)
    A_tilde[:, ilist[k:]] = A_tmp
    print 'n=%4d k=%4d' % (n, k)
    return A[:, ilist[:k]], A_tilde
    
#A_k, A_tilde = decomp(P[:, :split])
#A_tilde[A_tilde > 1] = 1
#as_matrix(A_tilde).plot()
#B = np.dot(A_k, A_tilde)
#as_matrix(B - P[:, :split]).plot()

#sizes = []
#horzbutterfly(P, sizes)
#print sum(sizes)


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
        return (sum(np.prod(x.shape) for x in self.lst))
    
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

if 0:
    plt.figure()
    plt.plot(get_ring_phi0(Nside))
    

if 1:
    C = 60 * 2
    lst = []
    for i in range(0, P.shape[1], C):
        X = P[:, i:i + C] 
        lst.append(butterfly(X))
    BP = HStack(lst)
    DP = Dense(P)
    g = BP.apply(alm.real) + 1j * BP.apply(alm.imag)
    g = DP.apply(alm.real) + 1j * DP.apply(alm.imag)

    Npix = 12 * Nside**2
    map = np.zeros(Npix)
    g_m_theta = np.zeros((4 * Nside - 1, 4 * Npix), dtype=np.complex)
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

    from cmb.maps import pixel_sphere_map, harmonic_sphere_map
    pixel_sphere_map(map).plot(title='fast')

    alm_fid = harmonic_sphere_map(0, lmin=0, lmax=lmax, is_complex=False)
#    alm_fid[1**2 + m] = 1

    alm_fid[lsig**2 + lsig + m] = np.sqrt(2) # real is repacked
#    alm_fid[1**2 - m] = -1
    alm_fid.to_pixel(Nside).plot(title='fiducial')

## plt.clf()


## from sympy.mpmath import legenp
## from scipy.special import gamma, sph_harm

## l = lmax

## if 0:
##     l = 2
    
##     plt.plot(theta, P[:, l])
##     Lambda_lm = np.sqrt((2 * l + 1) / np.pi / 4 * gamma(l -m + 1) / gamma(l + m  + 1))
##     Plm = np.array([float(legenp(l, m, ct)) for ct in np.cos(theta)])
##     plt.plot(theta, Lambda_lm * Plm)

##     sp = sph_harm(m, l, 0, theta)
##     plt.plot(theta, sp)
    

## #e2 = DP.apply(alm)
## #print BP.size() / DP.size()
