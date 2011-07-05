import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from cPickle import dumps, loads

from ..butterflylib import *

def test_post_projection():
    a = np.arange(22, dtype=np.double).reshape(11, 2)
    b = -np.arange(22, dtype=np.double).reshape(11, 2)
    a[-1, :] = np.nan
    b[-1, :] = np.nan
    target1 = np.nan * np.ones((13, 2))
    target2 = np.nan * np.ones((7, 2))
    mask = 1 + ((-1)**np.arange(20)).astype(np.int8) // 2
    post_projection(mask, target1, target2, a, b)
    X = np.vstack([target1, target2])
    assert np.all(X[1::2, :] == a[:-1, :])
    assert np.all(X[::2, :] == b[:-1, :])
