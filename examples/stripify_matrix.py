from __future__ import division

# Develop a routine which chunks a matrix into stripes, so as
# to cut away near-zero areas

import sys
import os
import itertools

from spherew import *
from spherew.healpix import *
from spherew.fastsht import *
from cmb import as_matrix
from spherew.butterfly import *
from matplotlib import pyplot as plt
from spherew.lib import stripify

from concurrent.futures import ProcessPoolExecutor#, ThreadPoolExecutor
from spherew.utils import FakeExecutor

from numpy.linalg import norm
def logabs(x): return np.log(np.abs(x))


Nside = 64
lmax = 3 * Nside
nodes = get_ring_thetas(Nside, positive_only=True)
m = 2 * Nside
Lambda = compute_normalized_associated_legendre(m, nodes, lmax).T

stripes = stripify(Lambda)

X = Lambda * 0
for ra, rb, ca, cb in stripes:
    X[ra:rb, ca:cb] = 1
as_matrix(X).plot()

print stripes
#as_matrix(np.log10(np.abs(Lambda) + 1e-150)).plot()
