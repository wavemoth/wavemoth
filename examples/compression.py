from __future__ import division

# Investigate compression

# Stick .. in PYTHONPATH
import sys
import os
import itertools

from wavemoth import *
from wavemoth.healpix import *
from wavemoth.roots import *
from cmb import as_matrix
from wavemoth.butterfly import butterfly_compress, matrix_interpolative_decomposition
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor#, ThreadPoolExecutor
from wavemoth.utils import FakeExecutor

Nside = 2048
lmax = 3 * Nside
epsilon_legendre = 1e-30
epsilon_butterfly = 1e-10
odd = 0

nodes = get_ring_thetas(Nside, positive_only=True)

m=0
#roots = associated_legendre_roots(lmax, m)
#nodes = np.arccos(roots)


ms = [0]#, 1000, 2000, 3000]#, 1000]
Cs = [64]#range(100, 150)

def residual_size_func(m, n):
    return min(3 * m + 3 * n, m * n)

def residual_flop_func(m, n):
    return m * n * (5/2 + 2) * 0.05 # Cost of flops 0.1 that of memops

def doit(m, chunk_size):
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
    P = P[:, odd::2].T
    x = butterfly_compress(P, chunk_size, eps=epsilon_butterfly)

    stats_list = []
    for level in range(x.get_max_depth()):
        flop_stats = x.get_stats(level, residual_flop_func)
        mem_stats = x.get_stats(level, residual_size_func)
        stats_list.append(flop_stats + mem_stats[1:])
        print 'm=%d,l=%d %s' % (m, level, x.format_stats(level, residual_size_func))
    return np.asarray(stats_list)

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
        stats = results[idx, 0]
        total, flopip, flopres, memip, memres = stats.T
        
        plt.plot((flopip + flopres) / total, label="lmax=%d, m=%d, C=%d" % (lmax, m, Cs[0]),
                 color=c)
#        plt.plot(flopres / total, linestyle='dotted', color='blue')

        plt.plot((memip + memres) / total, label="lmax=%d, m=%d, C=%d" % (lmax, m, Cs[0]),
                 color=c, linestyle='dotted')
#        plt.plot(memres / total, linestyle='dotted', color='red')
        plt.legend()
        print (flopip + flopres).argmin()

    plt.gca().set_ylim((0, 1))
    plt.gca().set_xlim((0, 10))

if 0:
    # Compare chunksizes
    level = 2
    for idx, m in enumerate(ms):
        floplist = []
        memlist = []
        for stats in results[idx, :]:
            total, flopip, flopres, memip, memres  = stats.T
            floplist.append((flopip[level] + flopres[level]) / total[level])
            memlist.append((memip[level] + memres[level]) / total[level])
        plt.plot(Cs, floplist, label=str(m), color='blue')
        plt.plot(Cs, memlist, label=str(m), color='red')
    plt.gca().set_ylim((0, 1))

    plt.legend(loc='upper left')


