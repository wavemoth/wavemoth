#!/usr/bin/env python

import sys
import argparse

from spherew.butterfly import *
from spherew.healpix import *
from spherew.benchmark_utils import *
from spherew import *

class RealignedStream(object):
    """
    Redirects to a wrapped stream, but lies about tell()
    so that the stream starts at the initialization point.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.pos0 = wrapped.tell()

    def tell(self):
        return self.wrapped.tell() - self.pos0

    def write(self, arg):
        self.wrapped.write(arg)

def compute(stream, mmax, lmax, Nside, min_rows):
    # Start by leaving room in the beginning of the file for writing
    # offsets
    write_int64(stream, mmax)
    header_pos = stream.tell()
    for i in range(4 * (mmax + 1)):
        write_int64(stream, 0)
    thetas = get_ring_thetas(Nside, positive_only=True)
    print thetas.shape
    for m in range(0, mmax + 1):
        print 'Precomputing m=%d of %d' % (m, mmax)
        P = compute_normalized_associated_legendre(m, thetas, lmax)
        for odd in (0, 1):
            P_subset = P[:, odd::2] #TODO
            compressed = butterfly_compress(P_subset, min_rows=min_rows)
            start_pos = stream.tell()
            compressed.write_to_stream(RealignedStream(stream))
            end_pos = stream.tell()
            stream.seek(header_pos + (4 * m + 2 * odd) * 8)
            write_int64(stream, start_pos)
            write_int64(stream, end_pos - start_pos)
            stream.seek(end_pos)
            

## parser = argparse.ArgumentParser(description='Process some integers.')
## parser.add_argument('integers', metavar='N', type=int, nargs='+',
##                    help='an integer for the accumulator')
## parser.add_argument('--sum', dest='accumulate', action='store_const',
##                    const=sum, default=max,
##                    help='sum the integers (default: find the max)')

Nside = 256
lmax = mmax = 2 * Nside
with file('precomputed.dat', 'wb') as f:
    compute(f, mmax, lmax, Nside, min_rows=32)
