import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

from nose.tools import eq_, ok_
from numpy.testing import assert_almost_equal

from ..butterfly import *
from io import BytesIO

def test_basic():
    A = np.arange(100).reshape(5, 20).astype(np.double)
    x = np.ones((20, 3), np.double)
    x[:, 1] = 2

    data = BytesIO()
    DenseMatrix(A).serialize(data)
    M = SerializedMatrix(data.getvalue(), A.shape[0], A.shape[1])
    C = M.apply(x)
    yield assert_almost_equal, C, np.dot(A, x)
    

    
       
