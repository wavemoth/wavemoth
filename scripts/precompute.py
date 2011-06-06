#!/usr/bin/env python

import sys
import argparse
from concurrent.futures import ProcessPoolExecutor

from spherew.butterfly import *
from spherew.healpix import *
from spherew.benchmark_utils import *
from spherew import *
from io import BytesIO

def compute_m(m, lmax, thetas, min_rows):
    print 'Precomputing m=%d of %d' % (m, mmax)
    stream_even, stream_odd = BytesIO(), BytesIO()
    P = compute_normalized_associated_legendre(m, thetas, lmax)
    for stream, odd in zip((stream_even, stream_odd), (0, 1)):
        P_subset = P[:, odd::2]
        compressed = butterfly_compress(P_subset, min_rows=min_rows)
        compressed.write_to_stream(stream)
    return stream_even, stream_odd

def compute(stream, mmax, lmax, Nside, min_rows):
    # Start by leaving room in the beginning of the file for writing
    # offsets
    write_int64(stream, mmax)
    header_pos = stream.tell()
    for i in range(4 * (mmax + 1)):
        write_int64(stream, 0)
    thetas = get_ring_thetas(Nside, positive_only=True)
    futures = []
    with ProcessPoolExecutor(max_workers=4) as proc:
        for m in range(0, mmax + 1):
            futures.append(proc.submit(compute_m, m, lmax, thetas, min_rows))

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

Nside = 256
lmax = mmax = 2 * Nside
with file('precomputed2.dat', 'wb') as f:
    compute(f, mmax, lmax, Nside, min_rows=64)
