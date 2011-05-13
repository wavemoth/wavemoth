from __future__ import division
from spherew import *
import numpy as np
from numpy import pi
from cmb.oomatrix import as_matrix
from matplotlib import pyplot as plt

lmax = 1000
Nside = 2048
eps = 1e-10
m = 0

theta = np.linspace(0.1, pi/2-0.1, 1000)

def bench():
    computed_normalized_associated_legendre(0, theta, lmax, out=P)

P = computed_normalized_associated_legendre(0, theta, lmax)
#print np.std(P)
#as_matrix(P).plot()

alm = np.random.normal(size=lmax + 1)

#P = P.T
split = 200#1024


def decomp(A, eps=eps):
    n = A.shape[1]
    A_tmp, k, ilist, rnorms = interpolative_decomposition(eps, A.copy('F'))
#    as_matrix(A_tmp).plot()
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


def butterfly(A):
    hmid = A.shape[1] // 2
    if hmid < 32:
        return #1/0
    L = A[:, :hmid]
    L_subset, L_interpolate = decomp(L)
    
    R = A[:, hmid:]
    R_subset, R_interpolate = decomp(R)

    LR = np.hstack([L_subset, R_subset])
    vmid = LR.shape[0] // 2
    T = LR[:vmid, :]
    butterfly(T)
    B = LR[vmid:, :]
    butterfly(B)

butterfly(P[:, :])

#A1, krank, ilist, rnorms = interpolative_decomposition(eps, P[:, :split].copy('F'))
#print krank
#A2, krank, ilist, rnorms = interpolative_decomposition(eps, A1[:split//2].copy('F'))
#print krank
#A3, krank, ilist, rnorms = interpolative_decomposition(eps, A1[:split//4].copy('F'))
#print krank





#plt.plot(P[-1, :100])
