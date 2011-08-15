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
from spherew.lib import *
from spherew.benchmark_utils import *
from spherew import *
from io import BytesIO
from spherew.utils import FakeExecutor

np.seterr(all='raise')

def get_c(l, m):
    n = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    d = (2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)
    return np.sqrt(n / d)

class PrintLogger:
    def info(self, msg):
        print msg

_aborted = False

def compute_m(filename, m, odd, lmax, Nside, chunk_size, eps, num_levels):
    global _aborted
    if _aborted:
        return
    try:
        filename = '%s-%d' % (filename, os.getpid())
        stream = BytesIO()
        compute_resources_for_m(stream, m, odd, lmax, Nside, chunk_size, eps, num_levels,
                                PrintLogger())
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
    except KeyboardInterrupt:
        _aborted = True
    except:
        import traceback
        traceback.print_exc()
        _aborted = True
        raise

def compute_with_workers(args):
    # Delete all files matching target-*
    for fname in glob('%s-*' % args.target):
        os.unlink(fname)
    # Each worker will generate one HDF file
    futures = []
    if args.parallel == 1:
        proc = FakeExecutor()
    else:
        proc = ProcessPoolExecutor(max_workers=args.parallel)
    for m in range(0, args.lmax + 1, args.stride):
        for odd in range(2):
            futures.append(proc.submit(compute_m, args.target, m, odd, args.lmax, args.Nside,
                                       args.chunk_size, args.tolerance, args.num_levels))
    for fut in futures:
        fut.result()

def serialize_from_hdf_files(args, target):
    """ Join the '$target-$pid' HDF file into '$target', a dat
        file in custom format.
    """
    print 'Merging...'
    infilenames = glob('%s-*' % target)
    if os.path.isdir(target):
        target = os.path.join(target, "%d.dat" % args.Nside)
    with file(target, 'w') as outfile:
        infiles = [tables.openFile(x) for x in infilenames]

        def get_group(m, odd):
            for infile in infiles:
                x = getattr(infile.root, 'm%d/%s' % (m, ['even', 'odd'][odd]), None)
                if x is not None:
                    return infile, x
        
        try:
            def get_matrix(stream, m, odd, lmax, Nside, chunk_size,
                           eps, num_levels, logger):
                f, g = get_group(m, odd)
                stream.write(g.matrix_data[:])
                return stream

            f, g = get_group(0, 0)
            Nside = f.getNodeAttr(g, 'Nside')
            lmax = f.getNodeAttr(g, 'lmax')
            mmax = lmax

            compute_resources(outfile, lmax, mmax, Nside, chunk_size=None, max_workers=1,
                              eps=None, logger=None, compute_matrix_func=get_matrix)
        finally:
            for x in infiles:
                x.close()
    for x in infilenames:
        os.unlink(x)

def main(args):
    comp = ResourceComputer(args.Nside, args.chunk_size, args.tolerance,
                                  args.memop_cost, PrintLogger())
    with file(args.target, 'w') as outfile:
        comp.compute(outfile, max_workers=args.parallel)
        

parser = argparse.ArgumentParser(description='Precomputation')
parser.add_argument('-c', '--chunk-size', type=int, default=64,
                    help='chunk size in number of columns')
parser.add_argument('-m', '--memop-cost', type=float, default=20,
                    help='cost to assign to memop vs. flop')
parser.add_argument('-j', '--parallel', type=int, default=8,
                    help='how many processors to use for precomputation')
parser.add_argument('--stride', type=int, default=1,
                    help='Skip m values. Results will be incorrect, '
                    'but useful for benchmarks.')
parser.add_argument('-e', '--tolerance', type=float, default=1e-10,
                    help='tolerance')
parser.add_argument('-l', '--num-levels', type=int, default=None,
                    help='Number of levels of compression')
parser.add_argument('Nside', type=int, help='Nside parameter')
parser.add_argument('target', help='target datafile')
args = parser.parse_args()

args.lmax = 2 * args.Nside

main(args)
#compute_with_workers(args)
#serialize_from_hdf_files(args, args.target)
