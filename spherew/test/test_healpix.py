import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

from nose.tools import eq_, ok_
from numpy.testing import assert_almost_equal

from ..healpix import *

Nside = 4

def test_ring_thetas():
    theta = get_ring_thetas(Nside)
    yield ok_, theta.shape, (4 * Nside - 1,)

    hz = healpix.pix2vec_ring(Nside, np.arange(12 * Nside**2))[:, 2]
    hz = np.unique(hz)
    htheta = np.arccos(hz)[::-1]
    yield assert_almost_equal, theta, htheta

def test_ring_pixel_counts():
    counts = get_ring_pixel_counts(Nside)
    yield eq_, 4 * Nside - 1, len(counts)
    yield eq_, 4, counts[0]
    yield eq_, 4, counts[-1]
    yield eq_, 4 * Nside, counts[2 * Nside]
    yield eq_, 12 * Nside**2, counts.sum()
    yield ok_, np.all(counts == counts[::-1])

def test_ring_phi0():
    counts = get_ring_pixel_counts(Nside)
    p0_arr = np.r_[0, np.cumsum(counts)[:-1]]
    from cmb.mapdatautils import array_vec2ang
    phi_fid = array_vec2ang(
        healpix.pix2vec_ring(Nside, np.arange(12 * Nside**2)))[:, 1]
    phi0 = get_ring_phi0(Nside)
    
    yield assert_almost_equal, phi0, phi_fid[p0_arr]
