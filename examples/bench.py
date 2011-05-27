from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

from joblib import Memory

from spherew.butterfly import *
from spherew.healpix import *
from spherew.benchmark_utils import *
from spherew import *
from cPickle import dumps, loads

memory = Memory('jobstore')

Nside = 1024
min_rows = 32
lmax = 2 * Nside
mstride = 100

@memory.cache
def get_MC_M(mmax, lmax, Nside, stride=1):
    MC_list = []
    M_list = []
    thetas = get_ring_thetas(Nside)[2*Nside:]
    for m in range(0, mmax, stride):
        print 'Precomputing m=%d of %d' % (m, mmax)
        P = compute_normalized_associated_legendre(m, thetas, lmax)
        P = P[:, ::2].copy('C')
        MC = serialize_butterfly_matrix(butterfly_compress(P, min_rows=min_rows))
        M = serialize_butterfly_matrix(butterfly_compress(P, min_rows=10**8))
        MC_list.append(MC)
        M_list.append(M)
    return MC_list, M_list

MC_list, M_list = get_MC_M(lmax, lmax, Nside, stride=mstride)

a_l_list = []
out_list = []
for MC in MC_list:
    a_l_list.append((-1)**np.arange(2 * MC.ncols).reshape((MC.ncols, 2)).astype(np.double))
    out_list.append(np.zeros((MC.nrows, 2)))

J = 10

nvecs = 1000
MC = MC_list[0]
a_l = np.zeros((MC.ncols, nvecs))
out = np.zeros((MC.nrows, nvecs))

with benchmark('MCmulti', J):
    MC.apply(a_l, out, repeats=J)

J = 1000

a_l = np.zeros((MC.ncols, 2))
out = np.zeros((MC.nrows, 2))
with benchmark('MCdual', J):
    MC.apply(a_l, out, repeats=J)

if 0:
    with benchmark('MC', J):
        for i in range(J * mstride):
            for a_l, MC, out in zip(a_l_list, MC_list, out_list):
                    MC.apply(a_l, out=out)

    with benchmark('M', J):
        for j in range(J * mstride):
            for a_l, M, out in zip(a_l_list, M_list, out_list):
                M.apply(a_l, out=out)

    y0 = M_list[0].apply(a_l_list[0])
    y1 = MC_list[0].apply(a_l_list[0])
    print 'Difference', np.linalg.norm(y1 - y0) / np.linalg.norm(y0)


#J = 1000

#out = np.zeros((P.shape[0], 2))

#with benchmark('MC', J):
#    MC.apply(a_l, out=out, repeats=J)

#with benchmark('M', J):
#    M.apply(a_l, out=out, repeats=J)

##with benchmark('dgemm_rrr', J):
##    benchmark_dgemm_rrr(P, a_l, out, repeats=J)
## with benchmark('dgemm_crr', J):
##     benchmark_dgemm_crr(P_fortran, a_l, out, repeats=J)
## with benchmark('dot', J):
##     for i in range(J):
##         P.dot(a_l)
