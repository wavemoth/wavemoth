from __future__ import division
import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from cPickle import dumps, loads

from ..fastsht import *
from .. import fastsht, healpix, psht
from cmb.maps import *

Nside = 256
lmax = 2 * Nside

def lm_to_idx_mmajor(l, m):
    assert lmax >= l >= m
    return m * (2 * lmax - m + 3) // 2 + (l - m)

#    return m * (lmax + 1) - (m * (m - 1)) // 2 + (l - m)

def test_basic():
    input = np.zeros((lmax + 1)**2, dtype=np.complex128)
    output = np.zeros(12*Nside**2)
    work = np.zeros((lmax + 1) * (4 * Nside - 1), dtype=np.complex128)
    plan = ShtPlan(Nside, lmax, lmax, input.view(np.double), output,
                   work.view(np.double), 'mmajor')

    input[0] = 10
    input[lm_to_idx_mmajor(1, 0)] = 10
    input[lm_to_idx_mmajor(2, 1)] = 10 + 5j

    plan.execute()

    y2 = psht.alm2map_mmajor(input, Nside=Nside)
    pixel_sphere_map(y2, pixel_order='ring').plot(title='FID')
    pixel_sphere_map(output, pixel_order='ring').plot()
    plt.show()

    print np.linalg.norm(y2 - output) / np.linalg.norm(y2)
    yield assert_almost_equal, y2, output


def test_healpix_phi0():
    phi0s = fastsht._get_healpix_phi0s(16)
    yield assert_almost_equal, phi0s, healpix.get_ring_phi0(16)

