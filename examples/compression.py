from __future__ import division

# Investigate compression

# Stick .. in PYTHONPATH
import sys
import os
import itertools

from spherew import *
from spherew.healpix import *
from cmb import as_matrix
from spherew.butterfly import butterfly_compress
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor

Nside = 1024
lmax = 2 * Nside
epsilon_legendre = 1e-30
epsilon_butterfly = 1e-15
min_rows = 32
odd = 1
nodes = get_ring_thetas(Nside, positive_only=True)

ms = [0, 500, 1000]
Cs = range(30, 200, 20)

def partition(P, C):
    n = P.shape[1]
    result = []
    for idx in range(0, n, C):
        result.append(P[:, idx:idx + C])
    return result

def doit(m, C, min_rows=None):
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
    P = P[:, odd::2].T
    if min_rows is None:
        cols = partition(P, C)
        x = butterfly_compress(cols)
    else:
        x = butterfly_compress(P, min_rows=min_rows)

    print 'm=%d' % m, x.get_stats()
    return x.size() / (x.nrows * x.ncols)

results = np.zeros((len(ms), len(Cs)))
if 1:
    with ProcessPoolExecutor(max_workers=4) as proc:
        futures = []
        subdivs = []
        for m in ms:
            for C in Cs:
                futures.append(proc.submit(doit, m, C))
            subdivs.append(proc.submit(doit, m, None, min_rows))
            
        it = iter(futures)
        for i in range(len(ms)):
            for j in range(len(Cs)):
                results[i, j] = next(it).result()
                
else:
    for i, m in enumerate(ms):
        for j, C in enumerate(Cs):
            results[i, j] = doit(m, C)

plt.clf()
for idx, m in enumerate(ms):
    plt.plot(Cs, results[idx, :], label=str(m))

for s in subdivs:
    plt.axhline(s.result())


plt.legend(loc='upper left')
plt.gca().set_ylim((0, 1))
plt.show()
