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

Nside = 2048
lmax = 2 * Nside
epsilon_legendre = 1e-30
epsilon_butterfly = 1e-12
min_rows = 32
odd = 0

max_levels = 20

nodes = get_ring_thetas(Nside, positive_only=True)

#m=0
#roots = associated_legendre_roots(lmax, m)
#nodes = np.arccos(roots)

ms = [0, 3000]
Cs = [50]#range(45, 60, 2)

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
        x = butterfly_compress(cols, eps=epsilon_butterfly, max_levels=max_levels)
    else:
        x = butterfly_compress(P, min_rows=min_rows, eps=epsilon_butterfly, max_levels=max_levels)

    ## for D in x.D_blocks:
    ##     D_k, D_ip = matrix_interpolative_decomposition(D, eps=epsilon_butterfly)
    ##     print D_ip.shape[0] / D_ip.shape[1]

#    print x.get_multilevel_stats()

    print 'm=%d' % m, x.get_stats()
    iplst, rlst = x.get_multilevel_stats()
    iplst = np.asarray(iplst, dtype=np.double)
    rlst = np.asarray(rlst, dtype=np.double)
    totlst = iplst + rlst
    return x.size() / (x.nrows * x.ncols), totlst, rlst
#x.size() / (x.nrows * x.ncols)

results = np.zeros((len(ms), len(Cs)), dtype=object)

if 1:
    with ProcessPoolExecutor(max_workers=4) as proc:
        futures = []
        subdivs = []
        for m in ms:
            for C in Cs:
                futures.append(proc.submit(doit, m, C))
#            subdivs.append(proc.submit(doit, m, None, min_rows))
            
        it = iter(futures)
        for i in range(len(ms)):
            for j in range(len(Cs)):
                results[i, j] = next(it).result()
                
elif 0:
    for i, m in enumerate(ms):
        for j, C in enumerate(Cs):
            results[i, j] = doit(m, C)

if 0:
    m = 0
    C = 50
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
    P = P[:, odd::2].T
    for idx in range(0, P.shape[1], 8 * C):
        x = butterfly_compress(P[:, idx:idx + 8*C], C=C, eps=epsilon_butterfly)
        print 'm=%d' % m, x.get_stats()
        1/0


plt.clf()
colors = ['blue', 'green', 'red', 'yellow', 'black']

if 1:
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
    for idx, m in enumerate(ms):
        data = []
        for s, _, _ in results[idx, :]:
            data.append(s)
        plt.plot(Cs, data, label=str(m))

    plt.legend(loc='upper left')
    plt.gca().set_ylim((0, 1))

