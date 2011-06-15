import numpy as np
from ..psht import *
from cmb.maps import pixel_sphere_map
from matplotlib import pyplot as plt

def test_basic():
    Nside = 16
    lmax = 2 * Nside
    T = PshtMmajorHealpix(lmax=32, Nside=16)
    alm = np.zeros((lmax + 1)**2, dtype=np.complex128)
    # m
    alm[lm_to_idx_mmajor(1, 1, lmax)] = 1 + 2j
    map = np.zeros(12 * Nside**2, dtype=np.double)
    T.alm2map(alm, map, repeat=2)
    #pixel_sphere_map(map).plot()
    #plt.show()
    
