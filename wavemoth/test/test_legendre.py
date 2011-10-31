import numpy as np

from numpy.testing import assert_almost_equal
from nose.tools import eq_, ok_

from ..legendre import *
from scipy.special import sph_harm

def test_norm_Plm():
    def test(theta, m, lmax):
        Plm = compute_normalized_associated_legendre(m, theta, lmax)
        Plm_p = sph_harm(m, np.arange(m, lmax + 1), 0, theta)[None, :]
        if not np.allclose(Plm_p, Plm):
            print Plm_p
            print Plm
        return ok_, np.allclose(Plm_p, Plm)

    yield test(np.pi/2, 0, 10)
    yield test(np.pi/4, 0, 10)
    yield test(3 * np.pi/4, 0, 10)
    yield test(np.pi/4, 1, 4)
    yield test(np.pi/4, 2, 4)
    yield test(np.pi/4, 50, 50)
    yield test(np.pi/2, 49, 50)

    yield test(0.001, 250, 300) # is close to 0
    

