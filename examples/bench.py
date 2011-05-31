from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

from spherew.butterfly import *
from spherew.healpix import *
from spherew.benchmark_utils import *
from spherew.fastsht import ShtPlan
from spherew.psht import PshtMmajorHealpix
from spherew import *
from cPickle import dumps, loads

assert os.environ['OMP_NUM_THREADS'] == '1'

Nside = 256
lmax = 2 * Nside


J = 20

input = np.zeros((lmax + 1)**2, dtype=np.complex128)
output = np.zeros(12*Nside**2)
work = np.zeros((lmax + 1) * (4 * Nside - 1), dtype=np.complex128)
plan = ShtPlan(Nside, lmax, lmax, input.view(np.double), output,
               work.view(np.double), 'mmajor')


from cmb.maps import harmonic_sphere_map

with benchmark('MC', J):
    plan.execute(repeat=J)

T = PshtMmajorHealpix(lmax=lmax, Nside=Nside)
with benchmark('healpix', J):
    T.alm2map(input, output, repeat=J)

#    with benchmark('M', J):
#        for j in range(J * mstride):
#            for a_l, M, out in zip(a_l_list, M_list, out_list):
#                M.apply(a_l, out=out)

#y0 = M_list[0].apply(a_l_list[0])
#y1 = MC_list[0].apply(a_l_list[0])
#print 'Difference', np.linalg.norm(y1 - y0) / np.linalg.norm(y0)


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
