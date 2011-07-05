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
import hashlib

from ..fastsht import *
from .. import lib, healpix, psht
from ..roots import associated_legendre_roots
from ..legendre import compute_normalized_associated_legendre, Plm_and_dPlm
from ..healpix import get_ring_pixel_counts

from cmb.maps import *
from ..openmp import use_num_threads

do_plot = bool(os.environ.get('P', False))
Nside = 4
lmax = 2 * Nside

def plot_map(m, title=None):
    from cmb.maps import pixel_sphere_map
    pixel_sphere_map(m).plot(title=title)

def lm_to_idx_mmajor(l, m):
    return m * (2 * lmax - m + 3) // 2 + (l - m)

def make_plan(nmaps, Nside=Nside, lmax=None, **kw):
    if lmax is None:
        lmax = 2 * Nside
    input = np.zeros((((lmax + 1) * (lmax + 2)) // 2, nmaps), dtype=np.complex128)
    output = np.zeros((12*Nside**2, nmaps))
    plan = ShtPlan(Nside, lmax, lmax, input, output, 'mmajor', **kw)

    return plan

def test_basic():
    nmaps = 3
    plan = make_plan(nmaps)

    plan.input[0, :] = 10
    plan.input[lm_to_idx_mmajor(1, 0), :] = np.arange(nmaps) * 30
    plan.input[lm_to_idx_mmajor(2, 1), :] = 10 + 5j
    output = plan.execute()
    y2 = psht.alm2map_mmajor(plan.input, lmax=lmax, Nside=Nside)
    if do_plot:
        for i in [0]:#nmaps):
            plot_map(y2[:, i], title='FID %d' % i)
            plot_map(output[:, i].copy('C'), title=' %d' % i)
            plot_map(output[:, i].copy('C') - y2[:, i], title='delta %d' % i)

        plt.show()

    #print np.linalg.norm(y2 - plan.output) / np.linalg.norm(y2)
    for i in range(nmaps):
        assert_almost_equal(y2[:, i], output[:, i])

def do_deterministic(nthreads):
    def hash_array(x):
        h = hashlib.md5()
        h.update(x.data)
        return h.hexdigest()

    plan = make_plan(5, Nside=8)
    plan.input[lm_to_idx_mmajor(10, 4)] = 1 + 1j
    with use_num_threads(nthreads):
        h0 = hash_array(plan.execute())
        for i in range(2000):
            h1 = hash_array(plan.execute())
            assert h0 == h1, "Non-deterministic behaviour, %d threads" % nthreads

def test_deterministic():
    "Smoke-test for deterministic behaviour multi-threaded"
    do_deterministic(16)

def test_deterministic_singlethread():
    "Smoke-test for deterministic behaviour single-threaded"
    do_deterministic(1)


def test_merge_even_odd_and_transpose():
    nmaps = 3
    plan = make_plan(nmaps, Nside=2, lmax=9, phase_shifts=False)
    counts = get_ring_pixel_counts(2)
    # Ring counts: [4, 8, 8, 8, 8, 8, 4]
    iring = np.arange(4)
    g_m_even = (2 * iring + 1) + 1j * (2 * iring + 2)
    g_m_even = np.asarray([g_m_even, 10 * g_m_even, 100 * g_m_even]).T.copy()
    g_m_odd = 1 * np.ones(4) + 2j * np.ones(4)
    g_m_odd = np.asarray([g_m_odd, 10 * g_m_odd, 100 * g_m_odd]).T.copy()

    def to_rings(map):
        # Assemble ring by ring
        rings = []
        idx = 0
        for count in counts:
            rings.append(map[idx:idx + count])
            idx += count
        return rings

    def print_rings(map):
        for ring in to_rings(map):
            print ring

    def doit(m):
        plan.output[...] = 0
        plan.assemble_rings(m, g_m_even, g_m_odd)
        # Check that multiple maps work, then focus on the first map in the rest
        first_map = plan.output[:, 0]
        #print_rings(first_map)
        for i in range(1, nmaps):
            #print_rings(plan.output[:, i])
            assert_almost_equal(plan.output[:, i], 10**i * first_map)
        return to_rings(first_map)

    rings = doit(2)
    #for r in rings: print r
    assert np.all(rings[0] == [0, 0, 16, 0])
    assert np.all(rings[3] == [0, 0, 2, 0, 0, 0, 4, 0])

    rings = doit(7)
    assert np.all(rings[0] == [0, 8, 0, -10])
    assert np.all(rings[-1] == [0, 6, 0, -6])
    assert np.all(rings[3] == [0, 2, 0, 0, 0, 0, 0, -4])

    rings = doit(9)
    assert np.all(rings[0] == [0, 8, 0, 10])
    assert np.all(rings[3] == [0, 2, 0, 0, 0, 0, 0, 4])

    
    

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
    phi0s = lib._get_healpix_phi0s(16)
    yield assert_almost_equal, phi0s, healpix.get_ring_phi0(16)

