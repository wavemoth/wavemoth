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


def decomp(eps, A):
    n = A.shape[1]
    A_tmp, k, ilist, rnorms = interpolative_decomposition(eps, A.copy('F'))
    ilist -= 1
    A_tilde = np.zeros((k, n))
    A_tilde[:, ilist[:k]] = np.eye(k)
    A_tilde[:, ilist[k:]] = A_tmp
    return A[:, ilist[:k]], A_tilde
    
A_k, A_tilde = decomp(eps, P[:, :split])
#A_tilde[A_tilde > 1] = 1
#as_matrix(A_tilde).plot()
B = np.dot(A_k, A_tilde)
B[np.abs(B) > 1] = 0
as_matrix(B).plot()
#as_matrix(P[:, :split]).plot()


#A1, krank, ilist, rnorms = interpolative_decomposition(eps, P[:, :split].copy('F'))
#print krank
#A2, krank, ilist, rnorms = interpolative_decomposition(eps, A1[:split//2].copy('F'))
#print krank
#A3, krank, ilist, rnorms = interpolative_decomposition(eps, A1[:split//4].copy('F'))
#print krank





#plt.plot(P[-1, :100])
