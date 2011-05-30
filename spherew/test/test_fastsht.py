import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from cPickle import dumps, loads

from ..fastsht import *
from .. import fastsht, healpix
from cmb.maps import *

Nside = 64
lmax = 2 * Nside

def test_basic():
    input = np.zeros((lmax + 1)**2 * 2)
    output = np.zeros(12*Nside**2)
    work = np.zeros((lmax + 1) * (4 * Nside - 1) * 2)
    plan = ShtPlan(Nside, lmax, lmax, input, output, work, 'mmajor')

    work[:] = np.sin(np.arange(work.shape[0]))
#    work[100] = 100
    
    plan.perform_backward_ffts(0, 4 * Nside - 1)
    1/0 # todo test this
    pixel_sphere_map(output, pixel_order='ring').plot()
    plt.show()

def test_healpix_phi0():
    phi0s = fastsht._get_healpix_phi0s(16)
    yield assert_almost_equal, phi0s, healpix.get_ring_phi0(16)

