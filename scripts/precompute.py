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
from spherew.fastsht import *
from spherew.benchmark_utils import *
from spherew import *
from io import BytesIO

np.seterr(all='raise')

def get_c(l, m):
    n = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    d = (2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)
    return np.sqrt(n / d)

class PrintLogger:
    def info(self, msg):
        print msg

def compute_m(filename, m, lmax, Nside, chunk_size=64, eps=1e-15):
    filename = '%s-%d' % (filename, os.getpid())
    stream = BytesIO()
    for odd in [0, 1]:
        compute_resources_for_m(stream, m, odd, lmax, Nside, chunk_size, eps, PrintLogger())
        # Store to HDF file for future concatenation
        stream_arr = np.frombuffer(stream.getvalue(), dtype=np.byte)
        f = tables.openFile(filename, 'a')
        try:
            group = f.createGroup('/m%d' % m, ['even', 'odd'][odd],
                                  createparents=True)
            f.setNodeAttr(group, 'lmax', lmax)
            f.setNodeAttr(group, 'm', m)
            f.setNodeAttr(group, 'odd', odd)
            f.setNodeAttr(group, 'Nside', Nside)
            f.createArray(group, 'matrix_data', stream_arr)
        finally:
            f.close()

class ComputedFuture(object):
    def __init__(self, result):
        self._result = result
    def result(self):
        return self._result

class SerialExecutor(object):
    def submit(self, func, *args, **kw):
        return ComputedFuture(func(*args, **kw))

def compute_with_workers(args):
    # Delete all files matching target-*
    for fname in glob('%s-*' % args.target):
        os.unlink(fname)
    # Each worker will generate one HDF file
    futures = []
    if args.parallel == 1:
        proc = SerialExecutor()
    else:
        proc = ProcessPoolExecutor(max_workers=args.parallel)
    for m in range(0, args.lmax + 1, args.stride):
        futures.append(proc.submit(compute_m, args.target, m, args.lmax, args.Nside,
                                   chunk_size=args.chunk_size, eps=args.tolerance))
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
            def get_matrix(stream, m, odd, lmax, Nside, chunk_size,
                           eps, logger):
                f, g = get_group(m, odd)
                stream.write(g.matrix_data[:])

            f, g = get_group(0, 0)
            Nside = f.getNodeAttr(g, 'Nside')
            lmax = f.getNodeAttr(g, 'lmax')
            mmax = lmax

            compute_resources(outfile, lmax, mmax, Nside, chunk_size=None,
                              eps=None, logger=None, compute_matrix_func=get_matrix)
        finally:
            for x in infiles:
                x.close()
    for x in infilenames:
        os.unlink(x)

parser = argparse.ArgumentParser(description='Precomputation')
parser.add_argument('-c', '--chunk-size', type=int, default=64,
                    help='chunk size in number of columns')
parser.add_argument('-j', '--parallel', type=int, default=8,
                    help='how many processors to use for precomputation')
parser.add_argument('--stride', type=int, default=1,
                    help='Skip m values. Results will be incorrect, '
                    'but useful for benchmarks.')
parser.add_argument('-m', type=int, default=None, help='Evaluate for a single m')
parser.add_argument('-e', '--tolerance', type=float, default=1e-15,
                    help='tolerance')
parser.add_argument('Nside', type=int, help='Nside parameter')
parser.add_argument('target', help='target datafile')
args = parser.parse_args()

args.lmax = 2 * args.Nside

compute_with_workers(args)
serialize_from_hdf_files(args, args.target)
