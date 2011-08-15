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
from numpy.linalg import norm

from cPickle import dumps, loads
import hashlib
from tempfile import mkstemp

from ..fastsht import *
from .. import lib, healpix, psht
from ..roots import associated_legendre_roots
from ..legendre import compute_normalized_associated_legendre, Plm_and_dPlm
from ..healpix import get_ring_pixel_counts, get_ring_thetas

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

matrix_data_filenames = []

def teardown():
    for f in matrix_data_filenames:
        os.unlink(f)
    del matrix_data_filenames[:]

def make_matrix_data(Nside, lmax, chunk_size=4, eps=1e-10, memop_cost=1):
    fd, matrix_data_filename = mkstemp()
    matrix_data_filenames.append(matrix_data_filename) # schedule cleanup
    with file(matrix_data_filename, 'w') as f:
        ResourceComputer(Nside, lmax, lmax, chunk_size, eps, memop_cost).compute(f, max_workers=1)
    return matrix_data_filename

def make_plan(nmaps, Nside=Nside, lmax=None, **kw):
    if lmax is None:
        lmax = 2 * Nside
    matrix_data_filename = make_matrix_data(Nside, lmax)

    input = np.zeros((((lmax + 1) * (lmax + 2)) // 2, nmaps), dtype=np.complex128)
    output = np.zeros((12*Nside**2, nmaps))
    plan = ShtPlan(Nside, lmax, lmax, input, output, 'mmajor',
                   matrix_data_filename=matrix_data_filename,
                   **kw)

    return plan

def test_basic():
    nmaps = 1
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

    with use_num_threads(nthreads):
        plan = make_plan(1, Nside=16, nthreads=nthreads)
        plan.input[lm_to_idx_mmajor(10, 4)] = 1 + 1j
        h0 = hash_array(plan.execute())
        for i in range(200):
            h1 = hash_array(plan.execute())
            assert h0 == h1, "Non-deterministic behaviour, %d threads" % nthreads

def test_deterministic_multithread():
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

def test_healpix_phi0():
    phi0s = lib._get_healpix_phi0s(16)
    yield assert_almost_equal, phi0s, healpix.get_ring_phi0(16)

def test_accuracy_against_psht():
    "Test modes"
    from spherew.psht import PshtMmajorHealpix

    Nside = 32 # needed this much to get numerical errors at all
    lmax = Nside
    nmaps = 1

    def test(eps, memop_cost):
        input = np.zeros(((lmax + 1) * (lmax + 2) // 2, nmaps), dtype=np.complex128)
        sht_output = np.zeros((12 * Nside**2, nmaps))
        psht_output = np.zeros((12 * Nside**2, nmaps), order='F')
        matrix_data_filename = make_matrix_data(Nside, lmax, chunk_size=5, eps=eps,
                                                memop_cost=memop_cost)
        sht_plan = sht_plan = ShtPlan(Nside, lmax, lmax, input, sht_output, 'mmajor',
                                      matrix_data_filename=matrix_data_filename)
        psht_plan = PshtMmajorHealpix(lmax=lmax, Nside=Nside, nmaps=nmaps)
        errors = []
        # A few pure modes
        for idx in range(100) + range(input.shape[0] - 100, input.shape[0]):
            input[idx] = 1 + 2j
            psht_plan.alm2map(input, psht_output)
            sht_plan.execute()
            input[idx] = 0
            errors.append(norm(sht_output - psht_output) / norm(psht_output))

        # Some random maps
        for i in range(30):
            input[...] = np.random.normal(size=input.shape)
            psht_plan.alm2map(input, psht_output)
            sht_plan.execute()
            errors.append(norm(sht_output - psht_output) / norm(psht_output))

        ok_(max(errors) < 10 * eps)

    yield test, 1e-7, 1
    yield test, 1e-11, 1
    yield test, 1e-11, 1e10 # No compression
    yield test, 1e-11, 1e-10 # Full compression



#
# Brute-force Legendre transforms
#

def test_legendre_transform():
    nvecs = 2
    Nside = 2048
    ixmin = 340
    m = 10
    lmin = m + 200

    def test(nx, nk):
        lmax = lmin + nk
        ixmax = ixmin + nx
        ls = np.arange(lmin, lmax, 2)
        nodes = get_ring_thetas(Nside, positive_only=True)[ixmin:ixmax]
        P = compute_normalized_associated_legendre(m, nodes, lmax, epsilon=1e-30)
        P = (P.T)[(lmin - m):(lmax - m):2, :].copy('C')
        x_squared = np.cos(nodes)**2
        a = np.sin(ls * 0.001)[:, None] * np.arange(1, nvecs + 1)[None, :].astype(np.double)
        y0 = np.dot(a.T, P).T
        for use_sse in [False, True]:
            y = np.zeros((x_squared.shape[0], a.shape[1]))
            associated_legendre_transform(m, lmin, a, y, x_squared,
                                          P[0, :].copy('C'), P[1, :].copy('C'), use_sse=use_sse)
            for j in range(nvecs):
                assert_almost_equal(y0[:, j], y[:, j])
                #plt.plot(y0[:, 1])
                #plt.plot(y[:, 1])
                #plt.show()

    for xcase in range(10):
        for kcase in [0, 1]:
            yield test, 20 + xcase, 10 + kcase
