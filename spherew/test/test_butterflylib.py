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
    for nvecs in [2, 10]:
        a = np.arange(11 * nvecs, dtype=np.double).reshape(11, nvecs)
        b = -np.arange(11 * nvecs, dtype=np.double).reshape(11, nvecs)
        a[-1, :] = np.nan
        b[-1, :] = np.nan
        target1 = np.nan * np.ones((13, nvecs))
        target2 = np.nan * np.ones((7, nvecs))
        mask = 1 + ((-1)**np.arange(20)).astype(np.int8) // 2
        do_post_interpolation(mask, target1, target2, a, b)
        X = np.vstack([target1, target2])
        assert np.all(X[1::2, :] == a[:-1, :])
        assert np.all(X[::2, :] == b[:-1, :])
