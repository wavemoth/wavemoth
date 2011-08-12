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

Nside = 512
lmax = 2 * Nside
epsilon_legendre = 1e-30
epsilon_butterfly = 1e-10
odd = 0

nodes = get_ring_thetas(Nside, positive_only=True)

#m=0
#roots = associated_legendre_roots(lmax, m)
#nodes = np.arccos(roots)

ms = [0]#, 1000]
Cs = [80]#range(5, 20, 2)

def residual_size_func(m, n):
    return min(3 * m + 3 * n, m * n)

def doit(m, chunk_size):
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
    P = P[:, odd::2].T
    x = butterfly_compress(P, chunk_size, eps=epsilon_butterfly)

    stats_list = []
    for level in range(x.get_max_depth() + 1):
        flop_stats = x.get_stats(level)
        mem_stats = x.get_stats(level, residual_size_func)
        stats_list.append(flop_stats + mem_stats[1:])
        print 'm=%d,l=%d %s' % (m, level, x.format_stats(level, residual_size_func))
    return np.asarray(stats_list)[::-1, :]

results = np.zeros((len(ms), len(Cs)), dtype=object)

proc = FakeExecutor()
#proc = ProcessPoolExecutor(max_workers=4)
futures = []
subdivs = []
for m in ms:
    for C in Cs:
        futures.append(proc.submit(doit, m, C))

it = iter(futures)
for i in range(len(ms)):
    for j in range(len(Cs)):
        results[i, j] = next(it).result()
                

plt.figure()
colors = ['blue', 'green', 'red', 'yellow', 'black']

if 1:
    # Plots fpr single chunksize
    for idx, (m, c) in enumerate(zip(ms, colors)):
        stats = results[idx, 0]
        total, flopip, flopres, memip, memres = stats.T
        
        plt.plot((flopip + flopres) / total, label="lmax=%d, m=%d, C=%d" % (lmax, m, Cs[0]),
                 color='blue')
        plt.plot(flopres / total, linestyle='dotted', color='blue')

        plt.plot((memip + memres) / total, label="lmax=%d, m=%d, C=%d" % (lmax, m, Cs[0]),
                 color='red')
        plt.plot(memres / total, linestyle='dotted', color='red')
        plt.legend()

    plt.gca().set_ylim((0, 1))
    plt.gca().set_xlim((0, 10))
else:
    # Compare chunksizes
    for idx, m in enumerate(ms):
        data = []
        for s, _, _ in results[idx, :]:
            data.append(s)
        plt.plot(Cs, data, label=str(m))

    plt.legend(loc='upper left')
    plt.gca().set_ylim((0, 1))

