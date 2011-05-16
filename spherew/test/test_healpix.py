import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

from nose.tools import eq_, ok_
from numpy.testing import assert_almost_equal

from spherew.healpix import get_ring_thetas

def test_ring_thetas():
    Nside = 16
    theta = get_ring_thetas(Nside)
    yield ok_, theta.shape, (4 * Nside - 1,)

    hz = healpix.pix2vec_ring(Nside, np.arange(12 * Nside**2))[:, 2]
    hz = np.unique(hz)
    htheta = np.arccos(hz)[::-1]
    yield assert_almost_equal, theta, htheta
