import os
import tempita
import numpy as np
from numpy import int32

#from . import flatcuda as cuda
#from .flatcuda import InOut, In, Out

#from .legendre_transform import CudaLegendreKernel

from .. import healpix, compute_normalized_associated_legendre


def plot_matrix(M):
    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.imshow(M, interpolation='nearest')

def get_edge(Lambda):
    zero_mask = Lambda == 0
    i_stops = np.zeros(Lambda.shape[0], dtype=np.uint16)
    Lambda_0 = np.zeros(Lambda.shape[1])
    Lambda_1 = np.zeros(Lambda.shape[1])
    cur_i = 0
    for k in range(Lambda.shape[0]):
        ilst, = zero_mask[k, :].nonzero()
        i_stops[k] = next_i = ilst[0] if len(ilst) > 0 else Lambda.shape[1]
        Lambda_0[cur_i:next_i] = Lambda[k, cur_i:next_i]
        if k + 1 < Lambda.shape[0]:
            Lambda_1[cur_i:next_i] = Lambda[k + 1, cur_i:next_i]
        cur_i = next_i
    return Lambda_0, Lambda_1, i_stops

class CudaShtPlan(object):

    def __init__(self, nside, lmax, epsilon_legendre=1e-30):
        self.nside, self.lmax, self.epsilon_legendre = nside, lmax, epsilon_legendre
        self.thetas = healpix.get_ring_thetas(nside, positive_only=True)
        self.x_squared = np.cos(self.thetas)**2
        self.ni = self.x_squared.shape[0]
        assert self.ni == 2 * nside

    def get_Lambda(self, m, odd):
        Lambda = compute_normalized_associated_legendre(m, self.thetas,
                                                        self.lmax,
                                                        epsilon=self.epsilon_legendre)
        Lambda = Lambda[:, odd::2].T
        return Lambda

    def precompute(self, m, odd):
        """
        Precomputes Lambda_0, Lambda_1, i_stops, nnz for a given
        m and odd.
        """
        Lambda = self.get_Lambda(m, odd)
        Lambda_0, Lambda_1, i_stops = get_edge(Lambda)
        nnz = np.sum(Lambda != 0)
        return Lambda_0, Lambda_1, i_stops, nnz
        
        
