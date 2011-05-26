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
    M.write_to_stream(data)
    return SerializedMatrix(data.getvalue(), M.nrows, M.ncols)


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
    
def test_compressed_application():
    "Manually constructed data, no interpolation matrix"
    I = IdentityNode(2)
    IP = InterpolationBlock([0, 0, 0, 0], np.zeros((4, 0)))
    S = InnerNode([(IP, IP)], (I, I))
    R = RootNode([np.eye(4) * 2, np.eye(4) * 3], S)
    C = serialize(R)
    y = C.apply(np.arange(4))
    yield assert_almost_equal, y[:, 0], [0, 2, 4, 6, 0, 3, 6, 9]

def test_compressed_application2():
    "Manually constructed data, one interpolation matrix"
    # k = 3, n = 4
    I = IdentityNode(2)
    s = np.asarray([[2], [3], [4]])
    filter = np.array([False, False, True, False])
    IP = InterpolationBlock(filter, s)
    S = InnerNode([(IP, IP)], (I, I))
    d = (np.eye(4) * 2)[:, :3]
    R = RootNode([d, d], S)
    C = serialize(R)
    x = np.ones(4)
    y = C.apply(x)
    y2 = np.dot(np.dot(d, s), x[filter][:, None]) + np.dot(d, x[~filter][:, None])
    yield assert_almost_equal, np.vstack([y2, y2]), y


def test_butterfly_compressed():
    P = get_test_matrix()
    a_l = ((-1)**np.arange(P.shape[1])).astype(np.double)

    M = butterfly_compress(P)
    print M.S_node.children[0].children[0].__dict__
    MC = serialize(M)
    y1 = MC.apply(a_l)
    y2 = M.apply(a_l)
    y3 = np.dot(P, a_l)
    print y1[:10]
    print y2[:10]
#    yield assert_almost_equal, y1, y2
#    print y1[:10], y1[-y1.shape[0] // 2:]
    

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

    
