from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))
                
from spherew import *
import numpy as np
from numpy import pi
from cmb.oomatrix import as_matrix
from matplotlib import pyplot as plt

lmax = 1000
#Nside = 2048
eps = 1e-10
m = 0

theta = np.linspace(0.1, pi/2-0.1, lmax + 1)

def bench():
    computed_normalized_associated_legendre(0, theta, lmax, out=P)

P = computed_normalized_associated_legendre(0, theta, lmax)
#print np.std(P)
#as_matrix(P).plot()

alm = np.random.normal(size=lmax + 1)

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
    if hmid < 64:
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

C = 128 * 2
lst = []
for i in range(0, P.shape[1], C):
    X = P[:, i:i + C] 
    lst.append(butterfly(X))
BP = HStack(lst)
DP = Dense(P)
e1 = BP.apply(alm)
e2 = DP.apply(alm)

