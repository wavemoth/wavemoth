from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
import os

from nose import SkipTest
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal
from numpy.linalg import norm

from ..fastsht import *

#
# Brute-force Legendre transforms
#

Nside = 2048
ixmin = 340
m = 10
    
def assert_transforms(nvecs, nx, nk, auxalign, drop_normal=False):
    assert auxalign in (0, 1)
    lmin = m + 200
    lstop = lmin + 2 * nk
    ixmax = ixmin + nx
    ls = np.arange(lmin, lstop, 2)
    nodes = get_ring_thetas(Nside, positive_only=True)[ixmin:ixmax]
    P = compute_normalized_associated_legendre(m, nodes, lstop, epsilon=1e-30)
    P = (P.T)[(lmin - m):(lstop - m):2, :].copy('C')
    x_squared = np.cos(nodes)**2
    a = np.sin(ls * 0.001)[:, None] * np.arange(1, nvecs + 1)[None, :].astype(np.double)
    a = np.arange(np.prod(a.shape), dtype=np.double).reshape(a.shape)
    y0 = np.dot(a.T, P).T
    auxdata = np.zeros(max(3 * (nk - 2), 0) + auxalign)
    assert_aligned(auxdata)
    auxdata[auxalign:] = associated_legendre_transform_auxdata(m, lmin, nk)
    for use_sse in ([True] if drop_normal else [False, True]):
        y = np.zeros((x_squared.shape[0], a.shape[1]))
        associated_legendre_transform(m, lmin, a, y, x_squared,
                                      P[0, :].copy('C'), P[1, :].copy('C'),
                                      use_sse=use_sse, auxdata=auxdata[auxalign:])

#        print
#        print y0
#        print '---'
#        print y
        
        #print ' '.join(['%.5f' % x for x in sorted(y0.ravel())])
        #print
        #print ' '.join(['%.5f' % x for x in sorted(y.ravel())])

        for j in range(nvecs):
            assert_almost_equal(y0[:, j], y[:, j])

def test_nvec2():
    for nx in [1, 2, 3, 4, 6, 7, 10, 11]:
        for nk in [2, 3, 4, 6, 7, 10, 11]:
            for auxalign in [0, 1]:
                yield assert_transforms, 2, nx, nk, auxalign

def test_multivec():
    from spherew.lib import _LEGENDRE_TRANSFORM_WORK_SIZE
    nk_block = _LEGENDRE_TRANSFORM_WORK_SIZE / (4 * 8 * 2) # X_CHUNKSIZE * sizeof(double) * duplicate

    yield assert_transforms, 6, 4, 6, 0, True
    yield assert_transforms, 6, 5, 6, 0, True
    yield assert_transforms, 6, 1, 68, 0, True
#    return
    

    for nx in [1, 2, 3, 4, 6, 7, 10, 11]:
        for nk in [2, 3, 4, 6, 7, 10, 11, nk_block - 2, nk_block, nk_block + 2]:
            nk *= 2 # todo
            for nvecs in [6, 12, 18]: # TODO
                yield assert_transforms, nvecs, nx, nk, 0, True
    
