import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from ..fmm import fmm1d

def test_basic():
    "Test fmm1d on a basic example"
    nx = 20
    ny = 30
    q = np.sin(np.arange(nx) / 2)
    qq = np.vstack([q, 2 * q, 3 * q]).T.copy('C')
    x_grid = np.linspace(0, 1, nx)

    gamma = np.arange(10, 10 * nx + 10, 10, dtype=np.double)
    omega = np.arange(1, ny + 1, dtype=np.double)

    def doit(ymin, ymax, rtol=1e-13):

        y_grid = np.linspace(ymin, ymax, ny)
        # Compute direct answer
        inv_K = np.subtract.outer(y_grid, x_grid)
        assert np.all(np.abs(inv_K) > 1e-4), 'colliding grids'
        K = 1 / inv_K
        phi0 = omega[:, None] * np.dot(K, gamma[:, None] * qq)
        # Compute via FMM
        phi = fmm1d(x_grid, qq, y_grid, omega=omega, gamma=gamma)
        print np.max(np.abs(phi - phi0))
        return np.allclose(phi, phi0, rtol=rtol)
    
    yield ok_, doit(0.01, 0.99), (0.01, 0.99)
    yield ok_, doit(-.21, 0.99), (-.21, 0.99)
    yield ok_, doit(-.21, 1.3)
    yield ok_, doit(-100, 100)
    yield ok_, doit(0.4995, 0.4996)
