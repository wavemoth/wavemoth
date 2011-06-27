from __future__ import division

# Investigate the behaviour of stopping the compression at various
# points -- how much does the residual matrix compress in each step?

from spherew import *
from spherew.healpix import *
from cmb import as_matrix
from spherew.butterfly import butterfly_compress, format_numbytes, InnerNode, IdentityNode

from concurrent.futures import ProcessPoolExecutor

Nside = 2048
lmax = 2 * Nside
epsilon = 1e-30

nodes = get_ring_thetas(Nside, positive_only=True)

def getlevels(N):
    if isinstance(N, IdentityNode):
        return 0
    return 1 + getlevels(N.children[0])

def doit(P, numchunks, min_rows):
    # Divide P in two, then compress each part...
    n = P.shape[1] // numchunks
    chunks = ([P[:, :i * n] for i in range(numchunks - 1)] +
              [P[:, (numchunks - 1) * n:]])
    xs = [butterfly_compress(chunk, min_rows=min_rows) for chunk in chunks]
    full = sum(x.size() for x in xs)
    ip = sum(x.S_node.size() for x in xs)
    dense = full - ip
    raw = P.shape[0] * P.shape[1]
    print '  %d chunks: %s + %s = %s (%.2f, %.2f, %.2f) %d levels' % (
        numchunks,
        format_numbytes(ip * 8),
        format_numbytes(dense * 8),
        format_numbytes(full * 8),
        full / rawsize,
        dense / full,
        ip / rawsize,
        getlevels(xs[0].S_node))

for m in [400]:
    P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=epsilon)[:, ::2]
    rawsize = np.prod(P.shape)
    print '== m=%d, full size: %s' % (m, format_numbytes(rawsize * 8))
    for min_rows in [800, 1000, 1200]:#[4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        print '=== min_rows=%d' % min_rows
        doit(P, 1, min_rows=min_rows)
        doit(P, 2, min_rows=min_rows)
        doit(P, 4, min_rows=min_rows)
        doit(P, 8, min_rows=min_rows)
