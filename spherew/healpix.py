from __future__ import division
import numpy as np

def get_ring_thetas(Nside):
    i = np.arange(1, 4 * Nside - 1)
    z = np.empty(4 * Nside - 1, np.double)
    # North polar cap
    z[:Nside] = 1 - i[:Nside]**2 / (3 * Nside**2)
    # North equatorial belt
    z[Nside:2 * Nside] = (4/3) - (2 * i[Nside:2 * Nside]) / (3 * Nside)
    z[2 * Nside:] = -z[2 * Nside - 2::-1]
    theta = np.arccos(z)
    return theta
