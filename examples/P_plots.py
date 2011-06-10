from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from spherew import *
from spherew.healpix import *
from cmb import as_matrix

lmax = 400
Nside = 256
m = 100

nodes = get_ring_thetas(Nside)
P = compute_normalized_associated_legendre(m, nodes, lmax)

as_matrix(P[:300, :]).plot()
