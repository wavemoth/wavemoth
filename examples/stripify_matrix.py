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

from concurrent.futures import ProcessPoolExecutor#, ThreadPoolExecutor
from spherew.utils import FakeExecutor

from numpy.linalg import norm
def logabs(x): return np.log(np.abs(x))


Nside = 64
lmax = 3 * Nside
nodes = get_ring_thetas(Nside, positive_only=True)
m = 2 * Nside
Lambda = compute_normalized_associated_legendre(m, nodes, lmax).T

def first_true(x):
    """ Given a 1D array of booleans x, return the first index that
        is True, or if all is False, the length of the array.
    """
    nz = x.nonzero()[0]
    if nz.shape[0] == 0:
        return x.shape[0]
    else:
        return nz[0]

def stripify(A, include_above=1e-30, exclude_below=1e-80,
                      jump_treshold=10, divisor=6):
    """
    Partitions the elements of a matrix intro strips. Strips are made
    so that elements smaller than exclude_below are excluded and
    elements larger than include_above are included.

    In addition, a dumb greedy algorithm is made to try to make the
    number of elements as small as possible, without creating too many
    strips. Concretely, when more than `jump_treshold` rows can be
    dropped, a new strip is created, but only so that the number of
    columns in the resulting strip is a multiple of divisor.

    Assumption made: Each column is increasing"fast enough" in magnitude
    (i.e. can start on zero on top but don't decrease towards zero again,
    the bottom is included in all strips).
    
    Returns
    -------

    List of tuples (row_start, row_stop, col_start, col_stop) describing
    each stripe.
    """
    # Assumption: Each column is increasing in magnitude.
    include_above = np.log2(include_above)
    exclude_below = np.log2(exclude_below)

    M = A.copy('F')
    mask = (M == 0)
    M[mask] = 1
    M = np.log2(np.abs(M))
    M[mask] = exclude_below - 1

    # First: For each column, find the index where it is first above
    # exclude_below (=a), and then where it it is first above
    # include_above (=b)
    col_starts = []
    for col in range(M.shape[1]):
        a = first_true(M[:, col] >= exclude_below)
        b = first_true(M[:, col] >= include_above)
        if np.any(M[a:, col] < exclude_below):
            raise ValueError("Magnitue of column %d not increasing regularly "
                             "enough in magnitude" % col)
        col_starts.append((a, b))

    # Then, take a greedy approach in creating as wide stripes as
    # possible. This is not optimal for all inputs; TODO: use dynamic
    # programming + permutations to minimize a cost function
    max_a = -1
    min_b = 2**63
    start_col = 0
    stripes = []
    col_starts.append((2**63, 2**63)) # Sentinel to make it emit the last column
    for col, (a, b) in enumerate(col_starts):
        new_strip_needed = (a > min_b or b < max_a)
        new_strip_wanted = (b - min_b > jump_treshold)
        if new_strip_needed or (new_strip_wanted and col % divisor == 0):
            # Need to create new column
            if min_b < M.shape[0]:
                stripes.append((min_b, M.shape[0], start_col, col))
            start_col = col
            max_a, min_b = a, b
        max_a = max(a, max_a)
        min_b = min(b, min_b)
    return stripes


stripes = stripify(Lambda)

X = Lambda * 0
for ra, rb, ca, cb in stripes:
    X[ra:rb, ca:cb] = 1
as_matrix(X).plot()

print stripes
#as_matrix(np.log10(np.abs(Lambda) + 1e-150)).plot()
