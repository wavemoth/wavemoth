from __future__ import division

# Does it pay off to evaluate at root nodes and then interpolate?
# Produce a table.

from spherew import *
from spherew.healpix import *
from spherew.roots import *
from spherew.butterfly import butterfly_compress, format_numbytes
from spherew.roots import associated_legendre_roots


from concurrent.futures import ProcessPoolExecutor

min_rows = 64
odd = 0

cases = [ # Nside, m
    (64, 0),
    (64, 50),
    (64, 100),
    (1024, 0),
    (1024, 500),
    (1024, 1000),
    (1024, 1500),
    (2048, 0),
    (2048, 1000),
    (2048, 2000),
    (2048, 3000)]



def doit(thetas, m, lmax):
    P = compute_normalized_associated_legendre(m, thetas, lmax, epsilon=1e-30)[:, odd::2]
    compressed = butterfly_compress(P, min_rows=min_rows, eps=1e-14)
    raw_size = compressed.nrows * compressed.ncols * 8
    comp_size = compressed.size()
#    print format_numbytes(raw_size), format_numbytes(comp_size)
    return raw_size, comp_size

print ('$\Nside$ & $m$ & Size evaluated at roots & Compressed & Size evaluated at nodes'
       '& Compressed \\')
for Nside, m in cases:
#    if Nside == 2048:
#        Nside = 512
    lmax = 2 * Nside
    n = (lmax - m) // 2
    root_th = np.arccos(associated_legendre_roots(m + 2 * n + odd, m))
    hpth = get_ring_thetas(Nside, positive_only=True)
    rr, rc = doit(root_th, m, lmax)
    hr, hc = doit(hpth, m, lmax)
    
    print '%d & %d & %s & %s & %s & %s \\' % ((
        Nside, m) + tuple([format_numbytes(x) for x in [rr, rc, hr, hc]]))
