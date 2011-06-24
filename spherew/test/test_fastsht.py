from __future__ import division
import healpix
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from cmb import as_matrix
import os

from nose import SkipTest
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_almost_equal

from cPickle import dumps, loads

from ..fastsht import *
from .. import fastsht, healpix, psht
from ..roots import associated_legendre_roots
from ..legendre import compute_normalized_associated_legendre, Plm_and_dPlm
from cmb.maps import *

do_plot = bool(os.environ.get('P', False))
Nside = 16
lmax = 2 * Nside

def lm_to_idx_mmajor(l, m):
    return m * (2 * lmax - m + 3) // 2 + (l - m)

def make_plan(nmaps):
    input = np.zeros((((lmax + 1) * (lmax + 2)) // 2, nmaps), dtype=np.complex128)
    output = np.zeros((nmaps, 12*Nside**2))
    work = np.zeros((nmaps, (lmax + 1) * (4 * Nside - 1)), dtype=np.complex128)
    plan = ShtPlan(Nside, lmax, lmax, input, output, work, 'mmajor')

    return plan

def test_basic():
    nmaps = 3
    plan = make_plan(nmaps)
    plan.input[0, :] = 10
    plan.input[lm_to_idx_mmajor(1, 0), :] = np.arange(nmaps) * 30
    plan.input[lm_to_idx_mmajor(2, 1), :] = 10 + 5j
    plan.execute()

    y2 = psht.alm2map_mmajor(plan.input, lmax=lmax, Nside=Nside)
    if do_plot:
        for i in range(nmaps):
            pixel_sphere_map(y2[i, :], pixel_order='ring').plot(title='FID %d' % i)
            pixel_sphere_map(plan.output[i, :], pixel_order='ring').plot(title='%d' % i)
#            pixel_sphere_map(plan.output[i, :] - y2[i, :], pixel_order='ring').plot(title='delta %d' % i)

        plt.show()

    #print np.linalg.norm(y2 - plan.output) / np.linalg.norm(y2)
    for i in range(nmaps):
        yield assert_almost_equal, y2[i, :], plan.output[i, :]

def test_matmul():
    raise SkipTest("This one was written with interpolation in mind")
    plan = make_plan()
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

def get_c(l, m):
    n = (l - m + 1) * (l - m + 2) * (l + m + 1) * (l + m + 2)
    d = (2 * l + 1) * (2 * l + 3)**2 * (2 * l + 5)
    return np.sqrt(n / d)

def test_interpolation():
    raise SkipTest("Interpolation currently disabled in source")
    plan = make_plan()
    for m in (0, 4):
        for odd in (0, 1):
            n = (lmax - m) // 2

            roots = associated_legendre_roots(m + 2 * n + odd, m)
            nodes = np.cos(healpix.get_ring_thetas(Nside, positive_only=True))
            values = np.sin(roots * 5)
            values = values - 1j*values

            # Run C code
            plan.work_g_m_roots[:values.shape[0]] = values
            result = plan.perform_interpolation(m, odd)

            # Brute-force FMM
            P_m_2n_sub_2_roots, _ = Plm_and_dPlm(m + 2 * n + odd - 2, m, roots)
            _, dP_roots = Plm_and_dPlm(m + 2 * n + odd, m, roots)
            rho = 2 * (2 * m + 4 * n + 1 + 2 * odd) / ((1 - roots**2) * (dP_roots)**2)
            c = get_c(m + 2 * n - 2 + odd, m)
            P_m_2n_nodes, _ = Plm_and_dPlm(m + 2 * n + odd, m, nodes)
            K = 1.0 / np.subtract.outer(nodes**2, roots**2)
            result0 = c * P_m_2n_nodes * np.dot(K, P_m_2n_sub_2_roots * rho * values)

            yield assert_almost_equal, result0, result

        if do_plot:
            plt.clf()
            plt.plot(roots, values.real, 'ob')
            plt.plot(roots, values.imag, 'og')
            plt.plot(nodes, result.real, '-xb')
            plt.plot(nodes, result.imag, '-xg')
            plt.plot(nodes, result0.real, 'r')
            plt.plot(nodes, result0.imag, 'r:')

            plt.show()
    

def test_healpix_phi0():
    phi0s = fastsht._get_healpix_phi0s(16)
    yield assert_almost_equal, phi0s, healpix.get_ring_phi0(16)

