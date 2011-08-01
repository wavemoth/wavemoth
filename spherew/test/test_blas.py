import numpy as np
from numpy import all
from nose.tools import ok_
from spherew.blas import *

def ndrange(shape, dtype=np.double, order='C'):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape).copy(order)

def test_dgemm_crc():
    def test(m, n, k):
        A = ndrange((m, k), order='F')
        B = ndrange((k, n))
        C = np.zeros((m, n), order='F')
        benchmark_dgemm_crc(A, B, C)
        print m, n, k, C
        print np.dot(A, B)
        ok_(all(C == np.dot(A, B)))
    test(2, 3, 4)
    test(0, 3, 4)
    test(2, 0, 4)
    test(2, 3, 0)
    test(0, 0, 2)
    test(0, 2, 0)
    test(0, 0, 2)
    test(0, 0, 0)
    
    
