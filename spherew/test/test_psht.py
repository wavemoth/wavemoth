import numpy as np
from ..psht import *
from cmb.maps import pixel_sphere_map
from matplotlib import pyplot as plt

def test_basic():
    nmaps = 3
    Nside = 16
    lmax = 2 * Nside
    T = PshtMmajorHealpix(lmax=32, Nside=16, nmaps=nmaps)
    alm = np.zeros((((lmax + 1)*(lmax+2))//2, nmaps), dtype=np.complex128)
    alm[lm_to_idx_mmajor(1, 1, lmax), :] = 1 + 2j
    alm[:, 0] *= 1
    alm[:, 1] *= 1
    alm[:, 2] *= -1
    map = np.zeros((nmaps, 12 * Nside**2), dtype=np.double)
    T.alm2map(alm, map, repeat=2)
    if 0:
        pixel_sphere_map(map[0, :]).plot()
        pixel_sphere_map(map[1, :]).plot()
        pixel_sphere_map(map[2, :]).plot()
        plt.show()
    
