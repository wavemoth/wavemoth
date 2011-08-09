import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from numpy import all
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal
from nose import SkipTest

from cPickle import dumps, loads

from ..butterfly import *
from ..interpolative_decomposition import interpolative_decomposition
from ..healpix import get_ring_thetas
from ..legendre import compute_normalized_associated_legendre

def arreq_(A, B):
    ok_(all(A == B))

def get_test_data():
    m = 3
    lmax = 200
    Nside = 512
    thetas = get_ring_thetas(Nside)[2*Nside:]
    P = compute_normalized_associated_legendre(m, thetas, lmax)
    a_l = ((-1)**np.arange(2 * P.shape[1])).reshape(P.shape[1], 2).astype(np.double)
    return P, a_l

def test_pickle_compressed():
    raise SkipTest()
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
    raise SkipTest()
    P, a_l = get_test_data()

    M = butterfly_compress(P, C=10)
    y1 = M.apply(a_l)
    y2 = np.dot(P, a_l)
    yield assert_almost_equal, y1, y2
    
def test_compressed_application():
    "Manually constructed data, no interpolation matrix"
    raise SkipTest()
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
    raise SkipTest()
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
    raise SkipTest()

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
    raise SkipTest()
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

def test_heap_size():
    def make_tree(d):
        if d == 0:
            return 1, Node(1)
        else:
            lc, l = make_tree(d - 1)
            rc, r = make_tree(d - 1)
            return lc + rc + 1, Node(1, [l, r])

    for nlevels in range(5):
        count, root = make_tree(nlevels)
        ok_(count == find_heap_size(root))

def test_heapify():

    heap = heapify(Node(0, [
        Node(1, [Node(3), Node(4)]),
        Node(2, [Node(5)])
        ]))
    eq_(heap, [0, 1, 2, 3, 4, 5, None])
    heap = heapify(Node(10))
    eq_(heap, [10])
    heap = heapify(Node(1, [Node(3, [Node(4)])]))
    eq_(heap, [1, 3, None, 4, None, None, None])

def ndrange(shape, start=0, dtype=np.double):
    return np.arange(start, np.prod(shape) + start, dtype=dtype).reshape(shape)

def vectors(m, n):
    x = np.arange(1, m + 1)
    result = np.zeros((m, n))
    for i in range(n):
        result[:, i] = x * 10**i
    return result

class NoPayload(object):
    def get_block(self, row_start, row_stop, col_indices):
        assert False
        
    def serialize_block_payload(self, stream, row_start, row_stop, col_indices):
        return

def test_transpose_apply_leaf():
    "butterfly.c.in: Transpose application of identity matrix"
    nvecs = 2
    nrows = ncols = 7
    plan = ButterflyPlan(k_max=nrows, nblocks_max=1, nvecs=nvecs)
    node = IdentityNode(nrows); assert nrows == ncols
    matrix_data = refactored_serializer(node, NoPayload()).getvalue()
    x = ndrange((ncols, nvecs))
    y = plan.transpose_apply(matrix_data, x)
    ok_(all(x == y))

def test_transpose_apply_small_fullrank():
    "butterfly.c.in: Transpose application of single S-matrix with all blocks of full rank"
    x = vectors(4, 2)
    A = np.array([
        [ 1,   0],
        [ 0,   1],
        [ 1,   0],
        [ 0,   1]], dtype=np.double)
    S1 = InnerNode([(InterpolationBlock([0, 0], [[], []]),
                     InterpolationBlock([0, 0], [[], []]))],
                   [IdentityNode(1), IdentityNode(1)])
    plan = ButterflyPlan(k_max=2, nblocks_max=2, nvecs=2)
    matrix_data = refactored_serializer(S1, NoPayload()).getvalue()
    y = plan.transpose_apply(matrix_data, x)
    ok_(all(y == np.dot(A.T, x)))

def test_transpose_apply_small_zerorank():
    "butterfly.c.in: Transpose application of single S-matrix with all blocks of zero rank"
    S1 = InnerNode([(InterpolationBlock([1, 1], np.zeros((0, 2))),
                     InterpolationBlock([1, 1], np.zeros((0, 2))))],
                   [IdentityNode(1), IdentityNode(1)])
    plan = ButterflyPlan(k_max=0, nblocks_max=2, nvecs=2)
    matrix_data = refactored_serializer(S1, NoPayload()).getvalue()
    y = plan.transpose_apply(matrix_data, np.ones((0, 2)))
    ok_(all(y == 0))
    eq_(y.shape, (2, 2))


def test_transpose_apply_small_ip():
    "butterfly.c.in: Application of single S-matrix with interpolation"
    x = vectors(4, 2)
    A = np.array([
        [ 1,   0,  10],
        [ 0,   1, 100],
        [ 1,  -1,   0],
        [ 0,  -2,   1]], dtype=np.double)
    S1 = InnerNode([(InterpolationBlock([0, 0, 1], [[10], [100]]),
                     InterpolationBlock([0, 1, 0], [[-1], [-2]]))],
                   [IdentityNode(1), IdentityNode(2)])
    plan = ButterflyPlan(k_max=2, nblocks_max=2, nvecs=2)
    matrix_data = refactored_serializer(S1, NoPayload()).getvalue()
    y = plan.transpose_apply(matrix_data, x)
    ok_(all(y == np.dot(A.T, x)))
                    
def test_transpose_apply_tree_generated():
    "butterfly.c.in: Transpose application of deep tree"
    nextk = [1]
    full_rank = False
    kmax = [0]
    ipidx = [0]
    ipset = range(100)#[0, 4]
    
    def make_tree(n):
        if n == 0:
            k = nextk[0]
            nextk[0] += 1
            kmax[0] = max([k, kmax[0]])
            return IdentityNode(k)
        else:
            L = make_tree(n // 2)
            R = make_tree(n // 2)
            blocks = []
            for lh, rh in zip(L.block_heights, R.block_heights):
                n = lh + rh
                if full_rank:
                    k1 = k2 = n
                else:
                    k1 = max(1, n - 4)
                    k2 = max(1, n - 3)
                TB = []
                for k in (k1, k2):
                    kmax[0] = max([k, kmax[0]])
                    filter = np.ones(n, dtype=np.int8)
                    filter[:k] = 0
                    interpolant = ndrange((k, n - k), start=3)
                    #if ipidx[0] not in ipset:
                    #    interpolant *= 0
                    #ipidx[0] += 1
                    block = InterpolationBlock(filter, interpolant)
                    TB.append(block)
                blocks.append(TB)
            node = InnerNode(blocks, [L, R])

            return node

    def compute_direct(matrices, y):
        for M in matrices:
            y = np.dot(M.T, y)
        return y

    nblocks = 2**3
    root = make_tree(nblocks // 2)
    #root.print_tree()
    
    matrices = tree_to_matrices(root)
    #for M in matrices:
    #    print M

    nvecs = 2
    plan = ButterflyPlan(k_max=kmax[0], nblocks_max=nblocks, nvecs=nvecs)

    matrix_data = refactored_serializer(root, NoPayload()).getvalue()
    x = ndrange((root.nrows, nvecs))

    y0 = compute_direct(matrices, x)
    y = plan.transpose_apply(matrix_data, x)
    ok_(all(y0 == y))
    

#
# Butterfly compression
#

def test_tree_to_matrices():
    S1 = InnerNode([(InterpolationBlock([0, 0, 1], [[10], [100]]),
                     InterpolationBlock([0, 1, 0], [[-1], [-2]]))],
                   [IdentityNode(1), IdentityNode(2)])
    A0 = np.array([
        [ 1,   0,  10],
        [ 0,   1, 100],
        [ 1,  -1,   0],
        [ 0,  -2,   1]], dtype=np.double)

    A, I = tree_to_matrices(S1)
    arreq_(A0, A)
    arreq_(np.eye(3), I)
            

def test_compress():
    "Compress and uncompress simple matrix"
    A = ndrange((10, 10), start = 1)
    row_indices_matrix = (A - 1) // 10 # make use of feature of ndrange

    def test(chunk_size):
        A_compressed = butterfly_compress(A, chunk_size)
        # A_compressed.print_tree()

        # Check that we can round-trip
        A_un  = A_compressed.uncompress(A)
        assert_almost_equal(A, A_un)

        # Check that remainder blocks preserve row index
        R = A_compressed.remainder_as_array(A)
        mask = R != 0
        arreq_((R[mask] - 1) // 10, row_indices_matrix[mask])

    for chunk_size in range(1, 10):
        yield test, chunk_size

    
def test_compress_generated():
    "Compress and uncompress matrix provided by an expression"

    class TestMatrix:
        def __init__(self, shape):
            self.shape = shape
            self.elements_fetched = 0
            
        def get_block(self, row_start, row_stop, col_indices):
            assert isinstance(row_start, int) and row_start >= 0
            assert isinstance(row_stop, int) and row_stop <= self.shape[0]
            assert row_start <= row_stop
            assert isinstance(col_indices, np.ndarray)
            for i in col_indices:
                assert 0 <= i < self.shape[1]
            
            axi = np.arange(row_start, row_stop)[:, None]
            axj = col_indices[None, :]
            block = (axi * axj).astype(np.double)
            self.elements_fetched += np.prod(block.shape)
            return block
        
    i, j = np.ogrid[:10, :10]
    A0 = i * j

    A_compressed = butterfly_compress(TestMatrix((10, 10)),
                                      chunk_size=3, shape=(10, 10))
    provider = TestMatrix((10, 10))
    # Check that matrix data can be acquired solely through
    # get_block method of TestMatrix, and that uncompressed
    # result matches A0
    A_un  = A_compressed.uncompress(provider)
    assert_almost_equal(A0, A_un)
    # Check that during uncompression, the number of elements
    # fetched is 10 (not 100)
    eq_(provider.elements_fetched, 10)
    # One check on compression stats
    ip_sizes, remainder_sizes = A_compressed.get_multilevel_stats()
    eq_(remainder_sizes[-1], provider.elements_fetched)
    

#
# Butterfly application
#

def test_transpose_apply_python():
    i, j = np.ogrid[:10, :10]
    A = (i * j).astype(np.double)
    A_compressed = butterfly_compress(A, chunk_size=3)
    x = ndrange((10, 2))
    assert_almost_equal(np.dot(A.T, x), A_compressed.transpose_apply(x, A))

def test_transpose_apply_c():
    plan = DenseResidualButterfly(k_max=10, nblocks_max=10, nvecs=2)

    i, j = np.ogrid[:20, :10]
    A = (i * j).astype(np.double)
    A_compressed = butterfly_compress(A, chunk_size=3)
    matrix_data = refactored_serializer(A_compressed, A).getvalue()
    x = ndrange((20, 2))
    y = plan.transpose_apply(matrix_data, x)
    assert_almost_equal(np.dot(A.T, x), y)


#
# Utils
#

def test_make_partition():
    yield eq_, [10, 20, 22], make_partition(0, 22, 10)
    yield eq_, [10, 12], make_partition(0, 12, 10)
    yield eq_, [10], make_partition(0, 10, 10)
    yield eq_, [9], make_partition(0, 9, 10)
    yield eq_, [0], make_partition(0, 0, 10)
