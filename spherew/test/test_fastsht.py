from __future__ import division
import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix

from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from cPickle import dumps, loads

from ..fastsht import *
from .. import fastsht, healpix, psht
from ..roots import associated_legendre_roots
from ..legendre import compute_normalized_associated_legendre
from cmb.maps import *

do_plot = False
Nside = 128
lmax = 2 * Nside

def lm_to_idx_mmajor(l, m):
    return m * (2 * lmax - m + 3) // 2 + (l - m)

def make_input():
    input = np.zeros((lmax + 1)**2, dtype=np.complex128)
    output = np.zeros(12*Nside**2)
    work = np.zeros((lmax + 1) * (4 * Nside - 1), dtype=np.complex128)
    plan = ShtPlan(Nside, lmax, lmax, input, output,
                   work.view(np.double), 'mmajor')
    return plan

def test_basic():
    plan = make_input()
    plan.input[0] = 10
    plan.input[lm_to_idx_mmajor(1, 0)] = 10
    plan.input[lm_to_idx_mmajor(2, 1)] = 10 + 5j
    plan.execute()

    y2 = psht.alm2map_mmajor(plan.input, Nside=Nside)
#    pixel_sphere_map(y2, pixel_order='ring').plot(title='FID')
#    pixel_sphere_map(output, pixel_order='ring').plot()
#    plt.show()

    print np.linalg.norm(y2 - plan.output) / np.linalg.norm(y2)
    yield assert_almost_equal, y2, plan.output

def test_matmul():
    plan = make_input()
    # Zero should make it through
    g_m = plan.perform_matmul(m=0, odd=0)
    yield ok_, np.all(g_m == 0), 'zero1'
    g_m = plan.perform_matmul(m=0, odd=1)
    yield ok_, np.all(g_m == 0), 'zero2'

    # Odd not affected by setting an even l-m
    plan.input[lm_to_idx_mmajor(3, 1)] = 1j
    g_m = plan.perform_matmul(m=1, odd=1)
    yield ok_, np.all(g_m == 0), 'zero3'
    # But even is...test operates seperately on real/imag
    g_m = plan.perform_matmul(m=1, odd=0)
    yield ok_, not np.all(g_m.imag == 0) , 'even1'
    yield ok_, np.all(g_m.real == 0), 'even2'

    # OK, over to a real test -- fill input with test data and compare with
    # brute force matrix multiplication
    colors = ['blue', 'black', 'green', 'red', 'cyan']
    for m in range(5):
        n = (lmax - m) // 2
        # Pick out coefficients for this m
        a_l = np.sin(np.arange(m, lmax + 1) / (lmax - m + 1) * 100)
        a_l = a_l + 3j * a_l
        plan.input[lm_to_idx_mmajor(np.arange(m, lmax + 1), m)] = a_l
        for odd in range(2):
            # Find roots and P
            roots = associated_legendre_roots(m + 2 * n + odd, m) 
            P = compute_normalized_associated_legendre(m, np.arccos(roots), lmax)
            P = P[:, odd::2]
            # Pick out even/odd subset and do multiplication
            a_l_subset = a_l[odd::2]
            g_m0 = np.dot(P, a_l_subset)

            g_m = plan.perform_matmul(m=m, odd=odd)
            yield assert_almost_equal, g_m, g_m0
            if do_plot:
                plt.plot((g_m.real - g_m0.real), '-', color=colors[(2 * m + odd) % len(colors)])
    if do_plot:
        plt.show()

def test_healpix_phi0():
    phi0s = fastsht._get_healpix_phi0s(16)
    yield assert_almost_equal, phi0s, healpix.get_ring_phi0(16)

