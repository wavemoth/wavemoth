import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from ..butterfly import *
from ..interpolative_decomposition import interpolative_decomposition
from io import BytesIO

def serialize(M):
    data = BytesIO()
    M.serialize(data)
    return SerializedMatrix(data.getvalue(), M.shape[0], M.shape[1])

def test_permutations_to_filter():
    yield eq_, list(permutations_to_filter([2, 3, 5], [4, 7])), [0, 0, 1, 0, 1]
    yield eq_, list(permutations_to_filter([2, 3, 5], [])), [0, 0, 0]
    yield eq_, list(permutations_to_filter([], [1, 2])), [1, 1]
    yield eq_, list(permutations_to_filter([], [])), []
    
    yield assert_raises, ValueError, permutations_to_filter, [0, 1], [1, 2]
    yield assert_raises, ValueError, permutations_to_filter, [], [1, 0]
    yield assert_raises, ValueError, permutations_to_filter, [1, 0], []
    yield assert_raises, ValueError, permutations_to_filter, [1, 1], [3, 4]

def test_dense_rowmajor():
    A = np.arange(100).reshape(5, 20).astype(np.double)
    x = np.ones((20, 3), np.double)
    x[:, 1] = 2

    M = serialize(DenseMatrix(A))
    C = M.apply(x)
    yield assert_almost_equal, C, np.dot(A, x)
  
def test_butterfly_matrix():
    x, y = np.ogrid[1:2:5j, 0:2:20j]
    A = x**2 * np.sin(y)
    A[:, 2] = 1

    # A is 5-by-20, rank 2
    
    x = np.ones((20, 3), np.double)
    x[:, 1] = 2
    M = serialize(butterfly_compress(A))
    print M.apply(x)

    A_k, A_ip = interpolative_decomposition(A, eps=1e-15)
    iden, ipol, A_ks, A_ips = sparse_interpolative_decomposition(A, eps=1e-15)
    print iden
    print ipol
#    print A_ip
    print np.dot(A_ip, x)
#    as_matrix(A_ips).plot()
    plt.show()
#    print A_k
#    print A_ip
#    as_matrix(np.dot(A_k, A_ip)).plot()
#    as_matrix(A).plot()
#    plt.show()

    
