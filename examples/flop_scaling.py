from __future__ import division

# Estimate number of flops needed for a Legendre transform
# for difference Nsides and produce a nice plot.

from __future__ import division

# Investigate compression

# Stick .. in PYTHONPATH
import sys
import os
import itertools

from spherew import *
from spherew.butterfly import *
from spherew.healpix import *
from cmb import as_matrix
from spherew.butterfly import butterfly_compress
from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy import log
from cPickle import load, dump

from scipy.optimize import *

from concurrent.futures import ProcessPoolExecutor

import joblib

memory = joblib.Memory(cachedir='joblib')

epsilon_legendre = 1e-30
epsilon_butterfly = 1e-13
odd = 1

N_samples = 10

chunk_size = 64

def get_size(Nside, m, lmax):
    nodes = get_ring_thetas(Nside, positive_only=True)
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon_legendre)
    P = P[:, odd::2].T
    x = butterfly_compress(P, chunk_size)
    full, ip, res = x.get_stats()
    return ip + res

@memory.cache
def take_size_samples(Nside, lmax):
    result = []
    ms = np.linspace(0, lmax, N_samples).astype(int)
    if 0:
        with ProcessPoolExecutor(max_workers=8) as proc:
            futures = []
            for m in ms:
                futures.append(proc.submit(get_size, Nside, m, lmax))
            result = [(m, fut.result()) for m, fut in zip(ms, futures)]
    else:
        result = [(m, get_size(Nside, m, lmax)) for m in ms]
    return np.asarray(result)

def compute_raw_size(Nside, lmax):
    size = 0
    for m in range(lmax + 1):
        size += 2 * Nside * ((lmax - m) // 2) # simply series, just too lazy right now...
    return size


def compute(Nside_max):
    Nsides = []
    compressed_sizes = []
    raw_sizes = []

    Nside = 16
    while Nside <= Nside_max:
        lmax = 2 * Nside
        Nsides.append(Nside)

        data = take_size_samples(Nside, lmax)
        ms, samples = data.T

        def c_hat(coefs, ms):
            X = np.asarray([
                ms * np.log(1 + ms)**2,
                ms * np.log(1 + ms),
                ms,
                ms * 0 + 1]).T
            return np.dot(X, coefs)

        def residuals(coefs):
            return c_hat(coefs, ms) - samples

        beta, cov = leastsq(residuals, np.ones(4))

        all_m = np.arange(lmax + 1)
        all_c = c_hat(beta, all_m)
        size = sum(all_c)
        compressed_sizes.append(size)
        raw_size = compute_raw_size(Nside, lmax)
        raw_sizes.append(raw_size)
        print "%d: %.2e -> %.2e" % (Nside, size, raw_size)

        if 0:
            plt.plot(ms / ms.max(), samples / sum(all_c), '-o')
            plt.plot(all_m / ms.max(), all_c / sum(all_c))
            plt.draw()

        Nside *= 2
    

    Nsides = np.asarray(Nsides)
    compressed_sizes = np.asarray(compressed_sizes)
    raw_sizes = np.asarray(raw_sizes)


    with file('flop_scaling.pickle', 'w') as f:
        dump((Nsides, compressed_sizes, raw_sizes), f)

def plot():
    with file('flop_scaling.pickle') as f:
        Nsides, compressed_sizes, raw_sizes = load(f)

    # Various scalings

    plt.clf()
    fig = plt.gcf()

    cm_in_inch = 0.394
    mpl.rcParams['axes.titlesize'] = 'x-small'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['figure.figsize'] = (17.5 * cm_in_inch, 10 * cm_in_inch)
    mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 'small'
    #fig.set_size_inches((17 * cm_in_inch, 10 * cm_in_inch))


    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(Nsides, raw_sizes, 'k--', label='Raw size')
    ax.loglog(Nsides, compressed_sizes, 'k-', label='Compressed size')
    #y = Nsides**3
    #ax.loglog(Nsides, y * raw_sizes[3] / y[3], 'k:',
    #          label=r'$O(N_\mathrm{side}^2)$')
    y = Nsides**2 * log(Nsides)**1
    ax.loglog(Nsides, y * compressed_sizes[3] / y[3], 'k:')
    y = Nsides**2 * log(Nsides)**2
    ax.loglog(Nsides, y * compressed_sizes[3] / y[3], 'k:')

    #ax.legend(loc='upper left')
    ax.set_xticks(Nsides)
    ax.set_xticklabels([str(2 * N) for N in Nsides])
    ax.set_xlim((Nsides[1], Nsides[-1]))
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"Matrix sizes (elements)")
    return ax

compute(4096)
ax = plot()


