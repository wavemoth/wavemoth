import os
import tempita
import numpy as np
from numpy import int32

from . import flatcuda as cuda
#from .flatcuda import InOut, In, Out, 

from .legendre_transform import CudaLegendreKernel

from .. import healpix, compute_normalized_associated_legendre
from ..streamutils import write_int64, write_array


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

def precompute_single(thetas, lmax, epsilon, m, odd):
    Lambda = compute_normalized_associated_legendre(m, thetas,
                                                    lmax,
                                                    epsilon=epsilon)
    Lambda = Lambda[:, odd::2].T
    Lambda_0, Lambda_1, i_stops = get_edge(Lambda)
    nnz = np.sum(Lambda != 0)
    return Lambda_0, Lambda_1, i_stops, nnz

class CudaShtPlan(object):

    def __init__(self, nside, lmax, mmin=0, mmax=None,
                 resource_path=None, epsilon_legendre=1e-30,
                 nthreads=64, i_chunk=4):
        if mmax is None:
            mmax = lmax
        self.nside, self.lmax, self.epsilon_legendre = nside, lmax, epsilon_legendre
        self.mmin, self.mmax = mmin, mmax
        self.thetas = healpix.get_ring_thetas(nside, positive_only=True)
        self.x_squared = np.cos(self.thetas)**2
        self.ni = self.x_squared.shape[0]
        assert self.ni == 2 * nside
        self.resource_path = resource_path
        self.nmaps = 1
        if resource_path:
            self.load_resources()

        self.legendre_kernel = CudaLegendreKernel(nvecs=2 * self.nmaps, nthreads=nthreads,
                                                  max_ni=self.ni, i_chunk=i_chunk,
                                                  skip_kernels=['transpose_legendre_transform',
                                                                'test_reduce_kernel'])

    def execute_transpose_legendre(self, q, a):
        self.legendre_kernel.all_transpose_legendre_transforms(self.lmax,
                                                               self.mmin,
                                                               self.mmax,
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
        return precompute_single(self.thetas, self.lmax, self.epsilon_legendre,
                                 m, odd)
        
    def precompute_to_stream(self, stream, logger):
        """
        File format:
          int64: nnz in total
          padding to 128 bytes
          double[ni]: x_squared
          double[(lmax + 1) * ni]: Lambda_0
          double[(lmax + 1) * ni]: Lambda_1
          ushort[(lmax + 1)**2]: i_stops
          Format of i_stops is m-major ordering, but with, additionally, even coefficents
          all coming before the odd ones.
        """
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        executor = ProcessPoolExecutor(max_workers=8)

        start_pos = stream.tell()
        for i in range(2 * (self.lmax + 1)):
            write_int64(stream, 0)
        write_array(stream, self.x_squared)

        futures = []
        for m in range(self.lmax + 1):
            for odd in [0, 1]:
                futures.append(executor.submit(precompute_single, self.thetas, self.lmax,
                                               self.epsilon_legendre, m, odd))
        
        nnz_total = 0
        Lambda_1_list = []
        i_stops_list = []
        nnz_list = []
        it = iter(futures)
        for m in range(self.lmax + 1):
            for odd in [0, 1]:
                Lambda_0, Lambda_1, i_stops, nnz = it.next().result()
                logger.info('Got %s m=%d' % (['even', 'odd'][odd], m))
                write_array(stream, Lambda_0)
                Lambda_1_list.append(Lambda_1)
                i_stops_list.append(i_stops)
                nnz_list.append(nnz)
                nnz_total += nnz
        for arr in Lambda_1_list:
            write_array(stream, arr)
        for arr in i_stops_list:
            write_array(stream, arr)
        end_pos = stream.tell()
        stream.seek(start_pos)
        for nnz in nnz_list:
            write_int64(stream, nnz)
        stream.seek(end_pos)
        return nnz_total

    def load_resources(self):
        data = np.memmap(self.resource_path)
        self.nnz = data[:2 * (self.lmax + 1) * 8].view(np.uint64)
        self.resources_gpu = cuda.to_device(data)

    def get_flops(self):
        nnz = 0
        for m in range(2 * self.mmin, 2 * self.mmax + 1):
            nnz += self.nnz[m]
        return nnz * (6 + 2 * 2 * self.nmaps)

    #def unload_resources():
