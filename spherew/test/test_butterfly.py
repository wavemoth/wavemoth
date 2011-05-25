import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from ..butterfly import *
from ..interpolative_decomposition import interpolative_decomposition
from ..healpix import get_ring_thetas
from ..legendre import compute_normalized_associated_legendre
from io import BytesIO

def get_test_matrix():
    m = 3
    lmax = 200
    Nside = 128
    thetas = get_ring_thetas(Nside)[2*Nside:]
    P = compute_normalized_associated_legendre(m, thetas, lmax)
    return P

def serialize(M):
    data = BytesIO()
    M.serialize(data)
    return SerializedMatrix(data.getvalue(), M.shape[0], M.shape[1])


def test_permutations_to_filter():
    yield eq_, list(permutations_to_filter([2, 3, 5], [0, 4])), [1, 0, 0, 0, 1]
    yield eq_, list(permutations_to_filter([0, 1, 2], [])), [0, 0, 0]
    yield eq_, list(permutations_to_filter([], [0, 1])), [1, 1]
    yield eq_, list(permutations_to_filter([], [])), []
    yield eq_, list(permutations_to_filter([0, 1], [])), [0, 0]

def test_butterfly_apply():
    P = get_test_matrix()
    a_l = ((-1)**np.arange(P.shape[1])).astype(np.double)

    M = butterfly_compress(P)
    y1 = M.apply(a_l)
    y2 = np.dot(P, a_l)
    yield assert_almost_equal, y1, y2
    
#    print M_c.apply(x)

#    A_k, A_ip = interpolative_decomposition(A, eps=1e-15)
#    iden, ipol, A_ks, A_ips = sparse_interpolative_decomposition(A, eps=1e-15)
#    print iden
#    print ipol
#    print A_ip
#    print np.dot(A_ip, x)
#    as_matrix(A_ips).plot()
#    plt.show()
#    print A_k
#    print A_ip
#    as_matrix(np.dot(A_k, A_ip)).plot()
#    as_matrix(A).plot()
#    plt.show()

    
