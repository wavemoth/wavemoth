#!/usr/bin/env python
from __future__ import division
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from spherew.butterfly import *
from spherew.healpix import *
from spherew.benchmark_utils import *
from spherew.roots import *
from spherew import *
from io import BytesIO

np.seterr(all='raise')

def Plm_and_dPlm(l, m, x):
    assert m >= 0
    P_matrix = compute_normalized_associated_legendre(
        m, np.arccos(x), l, epsilon=1e-300)
    P_x = P_matrix[:, -1]
    scale = np.sqrt((2 * l + 1) / (2 * l - 1) * (l - abs(m)) / (l + abs(m)))
    a = l * x * P_matrix[:, -1]
    b = (l + m) * scale * P_matrix[:, -2]
    c = x**2 - 1
    dP_x = (a - b) / c
    return P_x, dP_x

def get_c(l, m):
    n = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    d = (2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)
    return np.sqrt(n / d)

def compute_m(m, lmax, thetas, Nside, min_rows):
    print 'Precomputing m=%d of %d' % (m, mmax)
    stream_even, stream_odd = BytesIO(), BytesIO()
    for stream, odd in zip((stream_even, stream_odd), (0, 1)):
        # Roots
        n = (lmax - m) // 2
        if n > 0:
            roots = associated_legendre_roots(m + 2 * n + odd, m)
            assert roots.shape[0] == n
            write_array(stream, roots**2)
            pad128(stream)
            # Weights for input/Q in FMM
            # First, find P^m_{m + 2n + odd - 2}
            P_m_2n_sub_2 = compute_normalized_associated_legendre(
                m, np.arccos(roots), m + 2 * n + odd - 2, epsilon=1e-300)[:, -1]
            # Derivatives at roots
            _, dP = Plm_and_dPlm(m + 2 * n + odd, m, roots)
            # Gauss quadrature weights
            rho = 2 * (2 * m + 4 * n + 1 + 2 * odd) / ((1 - roots**2) * (dP)**2)
            assert rho.shape[0] == n
            write_array(stream, rho * P_m_2n_sub_2)
            pad128(stream)

            # Output grid
            write_array(stream, np.cos(thetas)**2)
            print np.min(np.abs(np.subtract.outer(np.cos(thetas)**2, roots**2)))
            pad128(stream)
            # Weights for output/phi in FMM
            c = get_c(m + 2 * n - 2 + odd, m)
            # Evaluated in grid:
            P_m_2n = compute_normalized_associated_legendre(
                m, thetas, m + 2 * n)[:, -1]
            assert P_m_2n.shape[0] == 2 * Nside
            write_array(stream, c * P_m_2n)
        
            # P matrix
            P = compute_normalized_associated_legendre(m, np.arccos(roots), lmax)
            P_subset = P[:, odd::2]
            compressed = butterfly_compress(P_subset, min_rows=min_rows)
            compressed.write_to_stream(stream)
    return stream_even, stream_odd

def compute(stream, mmax, lmax, Nside, min_rows):
    # Start by leaving room in the beginning of the file for writing
    # offsets
    write_int64(stream, lmax)
    write_int64(stream, mmax)
    write_int64(stream, Nside)
    header_pos = stream.tell()
    for i in range(4 * (mmax + 1)):
        write_int64(stream, 0)
    thetas = get_ring_thetas(Nside, positive_only=True)
    futures = []
    with ProcessPoolExecutor(max_workers=8) as proc:
        for m in range(0, mmax + 1):
            #compute_m(m, lmax, thetas, Nside, min_rows)
            futures.append(proc.submit(compute_m, m, lmax, thetas, Nside, min_rows))

        for m, fut in enumerate(futures):
            for recvstream, odd in zip(fut.result(), [0, 1]):
                pad128(stream)
                start_pos = stream.tell()
                stream.write(recvstream.getvalue())
                end_pos = stream.tell()
                stream.seek(header_pos + (4 * m + 2 * odd) * 8)
                write_int64(stream, start_pos)
                write_int64(stream, end_pos - start_pos)
                stream.seek(end_pos)
            del recvstream

## parser = argparse.ArgumentParser(description='Process some integers.')
## parser.add_argument('integers', metavar='N', type=int, nargs='+',
##                    help='an integer for the accumulator')
## parser.add_argument('--sum', dest='accumulate', action='store_const',
##                    const=sum, default=max,
##                    help='sum the integers (default: find the max)')

Nside = 128
lmax = mmax = 2 * Nside
with file('precomputed.dat', 'wb') as f:
    compute(f, mmax, lmax, Nside, min_rows=64)
