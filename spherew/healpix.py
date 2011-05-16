from __future__ import division
import numpy as np
from numpy import pi

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

def get_ring_phi0(Nside):
    counts = get_ring_pixel_counts(Nside)
    phi = np.zeros(4 * Nside - 1)
    ring_indices = np.arange(1, 4 * Nside)
    phi[:Nside - 1] = pi / (4 * ring_indices[:Nside - 1])
    phi[Nside - 1:3 * Nside] = pi / 4 / Nside
    phi[Nside:3 * Nside:2] = 0
    phi[3 * Nside:] = phi[Nside - 2::-1]
    return phi

def get_ring_pixel_counts(Nside):
    # North polar cap: Start with 4 pixels, add 4 more per ring
    north_polar_cap = np.arange(4, 4 * Nside, 4)
    assert len(north_polar_cap) == Nside - 1
    # Middle belt: 4 * Nside in each ring
    belt = 4 * Nside * np.ones(2 * Nside + 1, dtype=np.int)
    # South polar cap: Start with 4 * Nside - 4, subtract 4 per ring
    south_polar_cap = np.arange(4 * Nside - 4, 0, -4)
    assert len(south_polar_cap) == Nside - 1
    counts = np.hstack([north_polar_cap, belt, south_polar_cap])
    return counts

