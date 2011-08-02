import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from cPickle import dumps, loads

from ..butterflylib import *

def test_scatter():
    for nvecs in [2, 10]:
        a = np.arange(10 * nvecs, dtype=np.double).reshape(10, nvecs)
        b = -np.arange(10 * nvecs, dtype=np.double).reshape(10, nvecs)
        target1 = np.nan * np.ones((13, nvecs))
        target2 = np.nan * np.ones((7, nvecs))
        mask = 1 + ((-1)**np.arange(20)).astype(np.int8) // 2
        X = scatter(mask, target1, target2, a, group=0, add=False)
        assert np.all(X[1::2, :] == a)
        X = scatter(mask, target1, target2, a, add=True, group=0)
        X = scatter(mask, target1, target2, a, add=True, group=0)
        assert np.all(X[1::2, :] == 3 * a)
        X = scatter(mask, target1, target2, b, add=False, group=1)
        assert np.all(X[::2, :] == b)
        X = scatter(mask, target1, target2, b, add=True, group=1)
        assert np.all(X[::2, :] == 2 * b)        
        assert np.all(np.vstack([target1, target2]) == X)
