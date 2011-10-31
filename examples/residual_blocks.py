from __future__ import division

# Have a closer look at the residual blocks
# Conclusion: No residual block is likely to be so permuted, or
# so steeply changing, that we can't start recursion on the same l
# across a block (the problem beeing that some P_lm's are too small to be
# represented in IEEE floating point).

import sys
import os
import itertools

from wavemoth import *
from wavemoth.healpix import *
from wavemoth.fastsht import *
from cmb import as_matrix
from wavemoth.butterfly import *
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor#, ThreadPoolExecutor
from wavemoth.utils import FakeExecutor

from numpy.linalg import norm
def logabs(x): return np.log(np.abs(x))

Nside = 2048
lmax = 3 * Nside
epsilon_legendre = 1e-30
epsilon_butterfly = 1e-12
min_rows = 32
odd = 0
chunk_size = 70
nodes = get_ring_thetas(Nside, positive_only=True)

m = 4500

Lambda = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
Lambda = Lambda[:, odd::2].T
tree = butterfly_compress(Lambda, chunk_size, eps=epsilon_butterfly)


Lambda2 = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=1e-300)
Lambda2 = Lambda2[:, odd::2].T
Lambda_provider = as_matrix_provider(Lambda2)



def fetch_Rs(node, at_level):
    if at_level == 0:
        Rs = []
        for rstart, rstop, cols in node.remainder_blocks:
            Rs.append((rstart, cols, Lambda_provider.get_block(rstart, rstop, cols)))
        return Rs
    return sum([fetch_Rs(child, at_level - 1) for child in node.children], [])


print tree.get_stats()

for level in [2]:
    print '=======', level
    Rs = fetch_Rs(tree, level)
#    as_matrix(logabs(Rs[9])).plot()

    for idx, (rstart, cols, R) in enumerate(Rs):
        if np.prod(R.shape) > 0:

            a = np.zeros((R.shape[0], 2))
            a[-1,:] = 1
            y = np.zeros((R.shape[1], 2))
            x_squared = np.cos(nodes[cols])**2
            P = R[0, :].copy('C')
            Pp1 = R[1, :].copy('C')

            associated_legendre_transform(m, m + 2 * rstart + odd,
                                          a, y, x_squared, P, Pp1, use_sse=False)
            y = y[:, 0]
            r = norm(y - R[-1, :]) / norm(R[-1, :])

            print idx, np.min(np.abs(R)), r

#plt.plot()
