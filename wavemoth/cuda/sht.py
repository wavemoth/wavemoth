import os
import tempita
import numpy as np
from numpy import int32

from . import flatcuda as cuda
#from .flatcuda import InOut, In, Out, 

from .legendre_transform import CudaLegendreKernel

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

def write_array(stream, arr):
    stream.write(bytes(arr.data))

class CudaShtPlan(object):

    def __init__(self, nside, lmax, resource_path=None, epsilon_legendre=1e-30,
                 nthreads=64, i_chunk=4):
        self.nside, self.lmax, self.epsilon_legendre = nside, lmax, epsilon_legendre
        self.thetas = healpix.get_ring_thetas(nside, positive_only=True)
        self.x_squared = np.cos(self.thetas)**2
        self.ni = self.x_squared.shape[0]
        assert self.ni == 2 * nside
        self.resource_path = resource_path
        if resource_path:
            self.load_resources()

        self.legendre_kernel = CudaLegendreKernel(nvecs=2, nthreads=nthreads,
                                                  max_ni=self.ni, i_chunk=i_chunk,
                                                  skip_kernels=['transpose_legendre_transform',
                                                                'test_reduce_kernel'])

    def execute_transpose_legendre(self, q, a):
        self.legendre_kernel.all_transpose_legendre_transforms(self.lmax,
                                                               self.resources_gpu,
                                                               q, a)
    
    def get_Lambda(self, m, odd):
        Lambda = compute_normalized_associated_legendre(m, self.thetas,
                                                        self.lmax,
                                                        epsilon=self.epsilon_legendre)
        Lambda = Lambda[:, odd::2].T
        return Lambda

    def precompute_single(self, m, odd):
        """
        Precomputes Lambda_0, Lambda_1, i_stops, nnz for a given
        m and odd.
        """
        Lambda = self.get_Lambda(m, odd)
        Lambda_0, Lambda_1, i_stops = get_edge(Lambda)
        nnz = np.sum(Lambda != 0)
        return Lambda_0, Lambda_1, i_stops, nnz
        
        
    def precompute_to_stream(self, stream, logger):
        write_array(stream, self.x_squared)
        print stream.tell()
        for m in range(self.lmax + 1):
            for odd in [0, 1]:
                logger.info('Precomputing %s m=%d' % (['even', 'odd'][odd], m))
                Lambda_0, Lambda_1, i_stops, nnz = self.precompute_single(m, odd)
                if m == 1 and odd == 0:
                    print i_stops
                    print stream.tell()
                for arr in [Lambda_0, Lambda_1, i_stops]:
                    write_array(stream, arr)

    def load_resources(self):
        data = np.memmap(self.resource_path)
        print data.nbytes
        self.resources_gpu = cuda.to_device(data)


    #def unload_resources():
