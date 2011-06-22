#!/usr/bin/env python
from __future__ import division

#
# We spawn workers and use futures to submit jobs to workers.
# Each worker opens its own HDF file.
#

# Stick .. in PYTHONPATH
import sys
import os
from glob import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import tables

from spherew.butterfly import *
from spherew.healpix import *
from spherew.benchmark_utils import *
from spherew import *
from io import BytesIO

np.seterr(all='raise')

def get_c(l, m):
    n = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    d = (2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)
    return np.sqrt(n / d)

def compute_m(filename, m, lmax, Nside, min_rows=64, interpolate=True):
    filename = '%s-%d' % (filename, os.getpid())
    thetas = get_ring_thetas(Nside, positive_only=True)
    for odd in [0, 1]:
        n = (lmax - m) // 2
        if n == 0:
            interpolate = False
        # Roots
        if interpolate:
            1/0
            from spherew.roots import associated_legendre_roots
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
                m, thetas, m + 2 * n + odd)[:, -1]
            assert P_m_2n.shape[0] == 2 * Nside
            write_array(stream, c * P_m_2n)

            grid_for_P = np.arccos(roots)
        else:
            grid_for_P = thetas

        # P matrix
        P = compute_normalized_associated_legendre(m, grid_for_P, lmax,
                                                   epsilon=1e-30)
        P_subset = P[:, odd::2]
        compressed = butterfly_compress(P_subset, min_rows=min_rows)
        print 'Computed m=%d of %d: %s' % (m, lmax, compressed.get_stats())
        stream = BytesIO()
        compressed.write_to_stream(stream)
        stream_arr = np.frombuffer(stream.getvalue(), dtype=np.byte)

        f = tables.openFile(filename, 'a')
        try:
            group = f.createGroup('/m%d' % m, ['even', 'odd'][odd],
                                  createparents=True)
            f.setNodeAttr(group, 'interpolate', interpolate)
            f.setNodeAttr(group, 'lmax', lmax)
            f.setNodeAttr(group, 'm', m)
            f.setNodeAttr(group, 'odd', odd)
            f.setNodeAttr(group, 'Nside', Nside)
            f.setNodeAttr(group, 'combined_matrix_size', compressed.size())
            f.createArray(group, 'P', stream_arr)
        finally:
            f.close()

def compute_with_workers(args):
    # Delete all files matching target-*
    for fname in glob('%s-*' % args.target):
        os.unlink(fname)
    # Each worker will generate one HDF file
    futures = []
    with ProcessPoolExecutor(max_workers=args.parallel) as proc:
        for m in range(0, args.lmax + 1, args.stride):
            futures.append(proc.submit(compute_m, args.target, m, args.lmax, args.Nside,
                                       min_rows=args.min_rows, interpolate=args.interpolate))
        for fut in futures:
            fut.result()

def serialize_from_hdf_files(args, target):
    """ Join the '$target-$pid' HDF file into '$target', a dat
        file in custom format.
    """
    print 'Merging...'
    infilenames = glob('%s-*' % target)
    with file(target, 'w') as outfile:
        infiles = [tables.openFile(x) for x in infilenames]

        def get_group(m, odd):
            for infile in infiles:
                x = getattr(infile.root, 'm%d/%s' % (m, ['even', 'odd'][odd]), None)
                if x is not None:
                    return infile, x
        
        try:
            f, g = get_group(0, 0)
            Nside = f.getNodeAttr(g, 'Nside')
            lmax = f.getNodeAttr(g, 'lmax')
            mmax = lmax

            write_int64(outfile, lmax)
            write_int64(outfile, mmax)
            write_int64(outfile, Nside)
            
            header_pos = outfile.tell()
            for i in range(4 * (mmax + 1)):
                write_int64(outfile, 0)

            for m in range(0, mmax + 1, args.stride):
                for odd in [0, 1]:
                    f, g = get_group(m, odd)
                    if (f.getNodeAttr(g, 'Nside') != Nside or f.getNodeAttr(g, 'lmax') != lmax
                        or f.getNodeAttr(g, 'm') != m):
                        raise Exception('Unexpected data')
                    pad128(outfile)
                    start_pos = outfile.tell()
                    # Flags
                    write_int64(outfile, f.getNodeAttr(g, 'interpolate'))
                    write_int64(outfile, f.getNodeAttr(g, 'combined_matrix_size')) # for computing FLOPS
                    # P
                    pad128(outfile)
                    arr = g.P[:]
                    write_array(outfile, arr)
                    del arr
                    end_pos = outfile.tell()
                    outfile.seek(header_pos + (4 * m + 2 * odd) * 8)
                    write_int64(outfile, start_pos)
                    write_int64(outfile, end_pos - start_pos)
                    outfile.seek(end_pos)
                
        finally:
            for x in infiles:
                x.close()
    for x in infilenames:
        os.unlink(x)

parser = argparse.ArgumentParser(description='Precomputation')
parser.add_argument('-r', '--min-rows', type=int, default=64,
                    help='how much compression (lower is more compression)')
parser.add_argument('-j', '--parallel', type=int, default=8,
                    help='how many processors to use for precomputation')
parser.add_argument('-i', '--interpolate', action='store_true',
                    default=False, help='Evaluate at Legendre roots')
parser.add_argument('--stride', type=int, default=1,
                    help='Skip m values. Results will be incorrect, '
                    'but useful for benchmarks.')
parser.add_argument('-m', type=int, default=None, help='Evaluate for a single m')
parser.add_argument('Nside', type=int, help='Nside parameter')
parser.add_argument('target', help='target datafile')
args = parser.parse_args()

args.lmax = 2 * args.Nside

compute_with_workers(args)
serialize_from_hdf_files(args, args.target)
