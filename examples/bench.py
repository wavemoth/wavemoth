#!/usr/bin/env python
from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavemoth.butterfly import *
from wavemoth.healpix import *
from wavemoth.benchmark_utils import *
from wavemoth.fastsht import ShtPlan
from wavemoth.psht import PshtMmajorHealpix
from wavemoth import *
from cPickle import dumps, loads

assert os.environ['OMP_NUM_THREADS'] == '1'

Nside = 128
lmax = 2 * Nside


J = 200

input = np.zeros((lmax + 1)**2, dtype=np.complex128)
output = np.zeros(12*Nside**2)
plan = ShtPlan(Nside, lmax, lmax, input, output, 'mmajor')


from cmb.maps import harmonic_sphere_map

plan.execute(repeat=1)
with benchmark('MC', J, profile=True):
    plan.execute(repeat=J)

T = PshtMmajorHealpix(lmax=lmax, Nside=Nside)
T.alm2map(input, output, repeat=1)
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
