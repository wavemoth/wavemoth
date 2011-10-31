import numpy as np
from numpy import all
from numpy.testing import assert_almost_equal
from nose.tools import ok_
from spherew.blas import *

def ndrange(shape, dtype=np.double, order='C'):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape).copy(order)

def assert_dgemm(dgemm_func, A_order, B_order, C_order):
    def test(m, n, k):
        A = ndrange((m, k), order=A_order)
        B = ndrange((k, n), order=B_order)
        C = np.zeros((m, n), order=C_order)
        dgemm_func(A, B, C)
        assert_almost_equal(C, np.dot(A, B))
    test(2, 3, 4)
    test(0, 3, 4)
    test(2, 0, 4)
    test(2, 3, 0)
    test(0, 0, 2)
    test(0, 2, 0)
    test(0, 0, 2)
    test(0, 0, 0)

def test_dgemm():
    yield assert_dgemm, dgemm_crc, 'F', 'C', 'F'
    yield assert_dgemm, dgemm_ccc, 'F', 'F', 'F'

    
