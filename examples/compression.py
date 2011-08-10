from __future__ import division

# Investigate compression

# Stick .. in PYTHONPATH
import sys
import os
import itertools

from spherew import *
from spherew.healpix import *
from spherew.roots import *
from cmb import as_matrix
from spherew.butterfly import butterfly_compress, matrix_interpolative_decomposition
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor#, ThreadPoolExecutor
from spherew.utils import FakeExecutor

Nside = 2048
lmax = 2 * Nside
epsilon_legendre = 1e-30
epsilon_butterfly = 1e-12
min_rows = 32
odd = 0

nodes = get_ring_thetas(Nside, positive_only=True)

#m=0
#roots = associated_legendre_roots(lmax, m)
#nodes = np.arccos(roots)

ms = [0]#, 1000]
Cs = [70]#range(5, 20, 2)

def partition(P, C):
    n = P.shape[1]
    result = []
    for idx in range(0, n, C):
        result.append(P[:, idx:idx + C])
    return result

def doit(m, chunk_size):
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
    P = P[:, odd::2].T
    x = butterfly_compress(P, chunk_size, eps=epsilon_butterfly)

    print 'm=%d' % m, x.get_stats()
    iplst, rlst = x.get_multilevel_stats()
    iplst = np.asarray(iplst, dtype=np.double)
    rlst = np.asarray(rlst, dtype=np.double)
    totlst = iplst + rlst
    return x.size() / (x.nrows * x.ncols), totlst, rlst

results = np.zeros((len(ms), len(Cs)), dtype=object)

#proc = FakeExecutor()
proc = ProcessPoolExecutor(max_workers=4)
futures = []
subdivs = []
for m in ms:
    for C in Cs:
        futures.append(proc.submit(doit, m, C))

it = iter(futures)
for i in range(len(ms)):
    for j in range(len(Cs)):
        results[i, j] = next(it).result()
                

plt.clf()
colors = ['blue', 'green', 'red', 'yellow', 'black']

if 1:
    # Plots fpr single chunksize
    for idx, (m, c) in enumerate(zip(ms, colors)):
        _, tot, rlst = results[idx, 0]
        rlst /= tot[0]
        tot /= tot[0]
        plt.plot(tot, label="lmax=%d, m=%d" % (lmax, m), color=c)
        plt.plot(rlst, linestyle='dotted', color=c)
        plt.plot(rlst / tot, linestyle='dashed', color=c)
        plt.legend()

    plt.gca().set_ylim((0, 1))
else:
    # Compare chunksizes
    for idx, m in enumerate(ms):
        data = []
        for s, _, _ in results[idx, :]:
            data.append(s)
        plt.plot(Cs, data, label=str(m))

    plt.legend(loc='upper left')
    plt.gca().set_ylim((0, 1))

