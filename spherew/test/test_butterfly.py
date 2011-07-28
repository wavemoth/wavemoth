import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from cPickle import dumps, loads

from ..butterfly import *
from ..interpolative_decomposition import interpolative_decomposition
from ..healpix import get_ring_thetas
from ..legendre import compute_normalized_associated_legendre

def get_test_data():
    m = 3
    lmax = 200
    Nside = 512
    thetas = get_ring_thetas(Nside)[2*Nside:]
    P = compute_normalized_associated_legendre(m, thetas, lmax)
    a_l = ((-1)**np.arange(2 * P.shape[1])).reshape(P.shape[1], 2).astype(np.double)
    return P, a_l

def test_pickle_compressed():
    P, a_l = get_test_data()
    M = butterfly_compress(P, C=10)
    C = serialize_butterfly_matrix(M) 
    yield assert_almost_equal, C.apply(a_l), loads(dumps(C)).apply(a_l)
    

def test_permutations_to_filter():
    yield eq_, list(permutations_to_filter([2, 3, 5], [0, 4])), [1, 0, 0, 0, 1]
    yield eq_, list(permutations_to_filter([0, 1, 2], [])), [0, 0, 0]
    yield eq_, list(permutations_to_filter([], [0, 1])), [1, 1]
    yield eq_, list(permutations_to_filter([], [])), []
    yield eq_, list(permutations_to_filter([0, 1], [])), [0, 0]

def test_butterfly_apply():
    P, a_l = get_test_data()

    M = butterfly_compress(P, C=10)
    y1 = M.apply(a_l)
    y2 = np.dot(P, a_l)
    yield assert_almost_equal, y1, y2
    
def test_compressed_application():
    "Manually constructed data, no interpolation matrix"
    I = IdentityNode(2)
    IP = InterpolationBlock([0, 0, 0, 0], np.zeros((4, 0)))
    S = InnerNode([(IP, IP)], (I, I))
    R = RootNode([np.eye(4) * 2, np.eye(4) * 3], S)
    C = serialize_butterfly_matrix(R)
    X = np.vstack([np.arange(4), np.arange(10, 14)]).T.copy()
    y = C.apply(X)
    yield assert_almost_equal, y, [[  0.,  20.],
                                   [  2.,  22.],
                                   [  4.,  24.],
                                   [  6.,  26.],
                                   [  0.,  30.],
                                   [  3.,  33.],
                                   [  6.,  36.],
                                   [  9.,  39.]]

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
    C = serialize_butterfly_matrix(R)
    x = np.ones(4)
    x = np.vstack([x, x]).T.copy()
    y = C.apply(x)
    y2 = np.dot(np.dot(d, s), x[filter]) + np.dot(d, x[~filter])
    y3 = R.apply(x)
    yield assert_almost_equal, np.vstack([y2, y2]), y
    yield assert_almost_equal, np.vstack([y2, y2]), y3

def test_compressed_application3():
    "Deeper tree"

    # Level 1
    # k = 3, n = 4
    I = IdentityNode(2)
    s = np.asarray([[2], [3], [4]])
    filter = np.array([False, False, True, False])
    IP1 = InterpolationBlock(filter, s)
    S1 = InnerNode([(IP1, IP1)], (I, I))
    # Level 2
    # k = 4, n = 6
    filter = np.array([False, False, True, False, True, False])
    s = np.arange((4 * 2)).reshape(4, 2)
    IP2 = InterpolationBlock(filter, s)
    S2 = InnerNode([(IP2, IP2), (IP2, IP2)], (S1, S1))
    # Root
    d = (np.eye(4) * 2)
    R = RootNode([d, 2 * d, 3 * d, 4 * d], S2)
    # Do computation both in Python and C and compare
    C = serialize_butterfly_matrix(R)
    x = np.arange(8)
    x = np.vstack([x, x]).T.copy()
    y = C.apply(x)
    y2 = R.apply(x)
    yield assert_almost_equal, y, y2


def test_butterfly_compressed():
    "Test with a real, big matrix"
    P, a_l = get_test_data()

    M = butterfly_compress(P, C=10)
    MC = serialize_butterfly_matrix(M)
    y1 = MC.apply(a_l)
    y2 = M.apply(a_l)
    y3 = np.dot(P, a_l)
    yield assert_almost_equal, y1, y2




#
# Tests for refactored application
#

def test_heap():
    class Node(object):
        def __init__(self, value, children=[]):
            self.value = value
            self.children = children
        def __eq__(self, other):
            return self.value == int(other)
        def __index__(self):
            return self.value
        def __repr__(self):
            return '<%d>' % self.value

    heap = heapify(Node(0, [
        Node(1, [Node(3), Node(4)]),
        Node(2, [Node(5)])
        ]))
    eq_(heap, [0, 1, 2, 3, 4, 5, None])
    heap = heapify(Node(10))
    eq_(heap, [10])
    heap = heapify(Node(1, [Node(3, [Node(4)])]))
    eq_(heap, [1, 3, None, 4, None, None, None])

