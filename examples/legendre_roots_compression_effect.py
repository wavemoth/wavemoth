from __future__ import division

# Investigate any differences between how different matrices of Plm's
# compress ("m-major" vs. "ring-major").  Of course, we can only
# sanely compute with in in m-major mode, so this was mostly out of
# curiousity (or a very long shot).
#
# Conclusion: Can be better compressed if lmax > 3Nside for low Nside,
# with mmajor/ringmajor = 1.47 for lmax=4*Nside, Nside=512; but
# but may perhaps fare worse asymptotically. With lmax = 2Nside it's
# pretty much the same for Nside's tested.

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from spherew import *
from spherew.healpix import *
from cmb import as_matrix
from spherew.butterfly import butterfly_compress

from concurrent.futures import ProcessPoolExecutor

Nside = 4096
lmax = 2 * Nside
epsilon = 1e-30
min_rows = 64

stride = 300

ms = np.arange(lmax + 1)
nodes = get_ring_thetas(Nside, positive_only=True)

print 'm-major processing'

def mmajor_size(m):
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon)
    x = butterfly_compress(P[:, ::2], min_rows=min_rows)
    print 'm=%d' % m, x.get_stats()
    return x.size()

mmajor = 0
with ProcessPoolExecutor(max_workers=4) as proc:
    futures = []
    for m in ms[::stride]:
        futures.append(proc.submit(mmajor_size, m))
    for fut in futures:
        mmajor += fut.result()

print 'ring-major processing'

def ringmajor_size(theta):
    P = normalized_associated_legendre_ms(ms, theta, lmax, epsilon=epsilon)
    x = butterfly_compress(P[:, ::2], min_rows=min_rows)
    print x.get_stats()
    return x.size()

ringmajor = 0
with ProcessPoolExecutor(max_workers=4) as proc:
    futures = []
    for theta in nodes[::stride]:
        futures.append(proc.submit(ringmajor_size, theta))
    for fut in futures:
        ringmajor += fut.result()


print 'Comparison', mmajor / ringmajor
