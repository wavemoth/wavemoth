import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

from nose.tools import eq_, ok_
from numpy.testing import assert_almost_equal

from ..interpolative_decomposition import *

from cmb import as_matrix

def test_rank_1():
    x, y = np.ogrid[1:2:100j, 0:2:50j]
    A = x**2 * np.sin(y) # trivially rank-deficient
    A_k, A_ip = interpolative_decomposition(A)
    yield eq_, A_k.shape[1], 1
    B = np.dot(A_k, A_ip)
    yield assert_almost_equal, A, B

def test_rank_3():
    x, y = np.ogrid[1:2:5j, 0:2:20j]
    # Here I played around until the matrix managed to produce
    # reordering permutations that were not their own inverse.
    A = x**2 * np.sin(y) # first rank
    A[:, 8] = 4 # second rank
    A[:, 3] = 1
    A[3:, 3] = 2 # third rank
    A_k, A_ip = interpolative_decomposition(A)
    yield eq_, A_k.shape[1], 3
    B = np.dot(A_k[:, :], A_ip)
    yield assert_almost_equal, A, B

def test_full_rank():
    A = np.diagflat(np.arange(1, 11, dtype=np.double))
    A_k, A_ip = interpolative_decomposition(A)
    yield eq_, A_k.shape[0], 10
    B = np.dot(A_k, A_ip)
    yield assert_almost_equal, A, B

def test_nocols():
    A = np.zeros((100, 0), np.double)
    A_k, A_ip = interpolative_decomposition(A)
    yield eq_, A_k.shape, (100, 0)
    yield eq_, A_ip.shape, (0, 0)

def test_norows():
    A = np.zeros((0, 100), np.double)
    A_k, A_ip = interpolative_decomposition(A)
    yield eq_, A_k.shape, (0, 0)
    yield eq_, A_ip.shape, (0, 100)

def test_nomatrix():
    A = np.zeros((0, 0), np.double)
    A_k, A_ip = interpolative_decomposition(A)
    yield eq_, A_k.shape, (0, 0)
    yield eq_, A_ip.shape, (0, 0)

def test_zeromatrix():
    A = np.zeros((10, 20), np.double)
    A_k, A_ip = interpolative_decomposition(A)
    yield eq_, A_k.shape, (10, 0)
    yield eq_, A_ip.shape, (0, 20)    
                                 
def test_sparse_repr():
    x, y = np.ogrid[1:2:100j, 0:2:50j]
    A = x**2 * np.sin(y)
    A[:, 2] = 1
    iden_list, ipol_list, A_k, A_ip = sparse_interpolative_decomposition(A)
    yield eq_, sorted(list(iden_list)), list(iden_list)
    yield eq_, sorted(list(ipol_list)), list(ipol_list)
    yield eq_, len(iden_list), A_k.shape[1]
    yield eq_, len(ipol_list), A_ip.shape[1]
    yield eq_, A_k.shape[1], A_ip.shape[0]
    yield eq_, A_ip.shape[1], A.shape[1] - A_k.shape[1]
    yield eq_, A_k.shape, (100, 2)

    
