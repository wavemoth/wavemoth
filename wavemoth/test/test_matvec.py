
import numpy as np

from numpy.testing import assert_almost_equal
from nose.tools import eq_, ok_

from ..matvec import *

def test_dmat_zvec():
    N = 100
    A = np.arange(N * N, dtype=np.double).reshape(N, N)
    x = np.arange(N, dtype=np.complex)
    x = x - 1j * x
    y1 = np.dot(A, x)
    y2 = dmat_zvec(A, x)
    assert_almost_equal(y1, y2)
