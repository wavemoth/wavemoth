from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))
                
from spherew import *
from spherew.healpix import *
import numpy as np
from numpy import pi, prod
from cmb.oomatrix import as_matrix
from matplotlib import pyplot as plt

#
# Some parameters
#
eps = 1e-15
limit = 16


#
# The butterfly algorithm
#

class DenseMatrix(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape

    def apply(self, x):
        return np.dot(self.A, x)

    def apply_left(self, x):
        return np.dot(x, self.A)

    def size(self):
        return prod(self.shape)
    

def decomp(m, msg):
    s, ip = interpolative_decomposition(m, eps)
    if msg is not None:
        k, n = ip.shape
        m = s.shape[0]
        print 'ID %s: (%.2f) %d / %d -> %f' % (
            msg, s.shape[1] / ip.shape[1], s.shape[1], ip.shape[1],
            (k * (n - k) + m * k) / (m * n)
            )
    return s, ip

class IdentityMatrix(object):
    def __init__(self, n):
        self.nrows, self.ncols = n, n
        self.partition = [0, n]
    def apply(self, x):
        return x
    def size(self):
        return 0

class InterpolationLeafNode(object):
    """
    Leaf node: Only ID-compresses a single column without a vertical
    split.
    """
    def __init__(self, A_ip):
        self.A_ip = A_ip
        self.nrows, self.ncols = A_ip.shape
        self.partition = [0, self.nrows]

    def apply(self, x):
        return np.dot(self.A_ip, x)

    def size(self):
        k, n = self.A_ip.shape
        return k * (n - k)  

class Butterfly(object):
    def __init__(self, final_block_diagonal, S_tree):
        self.blocks = final_block_diagonal
        self.S_tree = S_tree
        self.nrows = sum(block.shape[0] for block in self.blocks)

    def apply(self, x):
        y = self.S_tree.apply(x)
        out = np.empty(self.nrows, np.double)
        i_out = 0
        i_y = 0
        for block in self.blocks:
            m, n = block.shape
            out[i_out:i_out + m] = np.dot(block, y[i_y:i_y + n])
            i_out += m
            i_y += n
        return out

    def size(self):
        return (self.S_tree.size() +
                sum(np.prod(block.shape) for block in self.blocks))
            
class SNode(object):
    def __init__(self, blocks, children):
        if 2**int(np.log2(len(blocks))) != len(blocks):
            raise ValueError("len(blocks) not a power of 2")
        if len(children) != 2:
            raise ValueError("len(children) != 2")
        self.blocks = blocks
        self.ncols = sum(child.ncols for child in children)
        self.children = children
        block_widths = sum([[T_ip.shape[0], B_ip.shape[0]]
                            for T_ip, B_ip in blocks], [])
        self.partition = np.cumsum([0] + block_widths)
        self.nrows = self.partition[-1]

    def apply(self, x):
        # z is the vector containing the contiguous result of the 2 children
        # The permutation happens in reading from z in a permuted way; so each
        # butterfly application permutes its input
        # Recurse to children butterflies
        LS, RS = self.children
        assert x.shape[0] == self.ncols
        assert x.shape[0] == LS.ncols + RS.ncols
        print len(self.blocks)
        z_left = LS.apply(x[:LS.ncols])
        z_right = RS.apply(x[LS.ncols:])

        # Apply this butterfly, permuting the input as we go
        y = np.empty(self.nrows, np.double)
        i_y = 0
        for i_block, (T_ip, B_ip) in enumerate(self.blocks):
            buf = np.empty(T_ip.shape[1])
            assert T_ip.shape[1] == B_ip.shape[1]
            # Merge together input
            lstart, lstop = LS.partition[i_block], LS.partition[i_block + 1]
            rstart, rstop = RS.partition[i_block], RS.partition[i_block + 1]
            assert T_ip.shape[1] == (lstop - lstart) + (rstop - rstart)
            mid = lstop - lstart
            buf[:mid] = z_left[lstart:lstop]
            buf[mid:] = z_right[rstart:rstop]
            # Do computation
            y[i_y:i_y + T_ip.shape[0]] = np.dot(T_ip, buf)
            i_y += T_ip.shape[0]
            y[i_y:i_y + B_ip.shape[0]] = np.dot(B_ip, buf)
            i_y += B_ip.shape[0]
        return y

    def size(self):
        size = 0
        for T_ip, B_ip in self.blocks:
            for M in (T_ip, B_ip):
                k, n = M.shape
                size += k * (n - k)
        for child in self.children:
            size += child.size()
        return size

def butterfly_core(A_k_blocks):
    if len(A_k_blocks) == 1:
        return A_k_blocks, IdentityMatrix(A_k_blocks[0].shape[1])
        # No compression achieved when split into odd l/even l,
        # and it takes a long time.
        #A_k, A_ip = decomp(A_k_blocks[0], 'leaf')
        #return [A_k], InterpolationLeafNode(A_ip)
    mid = len(A_k_blocks) // 2
    left_blocks, left_interpolant = butterfly_core(A_k_blocks[:mid])
    right_blocks, right_interpolant = butterfly_core(A_k_blocks[mid:])
    out_blocks = []
    out_interpolants = []
    for L, R in zip(left_blocks, right_blocks):
        # Horizontal join
        LR = np.hstack([L, R])
        # Vertical split & compress
        vmid = LR.shape[0] // 2
        T = LR[:vmid, :]
        B = LR[vmid:, :]
        T_k, T_ip = decomp(T, None)
        B_k, B_ip = decomp(B, None)
        assert T_ip.shape[1] == B_ip.shape[1] == LR.shape[1]       
        out_interpolants.append((T_ip, B_ip))
        out_blocks.append(T_k)
        out_blocks.append(B_k)
    return out_blocks, SNode(out_interpolants, [left_interpolant, right_interpolant])

def butterfly(A):
    def partition(X, result):
        hmid = X.shape[1] // 2
        if hmid <= limit or X.shape[0] <= 8:
            result.append(X)
            return 0
        else:
            partition(X[:, :hmid], result)
            levels = partition(X[:, hmid:], result)
        return levels + 1 
        
    B_list = []
    numlevels = partition(A, B_list)
    diagonal_blocks, S_tree = butterfly_core(B_list)
    result = Butterfly(diagonal_blocks, S_tree)
    return result

#
# Spherical harmonic transform for a single l,
# down to HEALPix grid
#

class InnerSumPerM:
    def __init__(self, m, x_grid, lmax, compress=True):
        P = compute_normalized_associated_legendre(m, np.arccos(x_grid), lmax, epsilon=1e-30)
        assert x_grid[-1] == 0

        P_even_arr = P[:-1, ::2] # drop equator
        P_odd_arr = P[:-1, 1::2]
        if compress:
            self.P_even = butterfly(P_even_arr)
            self.P_odd = butterfly(P_odd_arr)
            print 'Compression:', ((self.P_even.size() + self.P_odd.size()) / 
                (prod(P_even_arr.shape) + prod(P_odd_arr.shape)))
            print 'Ratio in final blocks:', (
                (self.P_even.S_tree.size() + self.P_odd.S_tree.size()) /
                (self.P_even.size() + self.P_odd.size()))
        else:
            self.P_even = DenseMatrix(P_even_arr)
            self.P_odd = DenseMatrix(P_odd_arr)
        # Treat equator seperately, as we cannot interpolate to it from
        # samples in (0, 1). Only need even part, as odd part will be 0.
        self.P_equator = DenseMatrix(P[-1:, ::2])

    def compute(self, a_l):
        a_l_even = a_l[::2]
        a_l_odd = a_l[1::2]
        g_even = self.P_even.apply(a_l_even.real) + 1j * self.P_even.apply(a_l_even.imag)
        g_odd = self.P_odd.apply(a_l_odd.real) + 1j * self.P_odd.apply(a_l_odd.imag)
        g_equator = self.P_equator.apply(a_l_even.real) + 1j * self.P_equator.apply(a_l_even.imag)
        # We have now computed for all cos(theta) >= 0. Retrieve the
        # opposite hemisphere by symmetry.
        g_even = np.hstack([g_even, g_equator, g_even[::-1]])
        g_odd = np.hstack([g_odd, 0, -g_odd[::-1]])
        return g_even + g_odd
        

def al2gmtheta(m, a_l, theta_arr):
    lmax = a_l.shape[0] - 1 + m
    x = np.cos(theta_arr)
    x[np.abs(x) < 1e-10] = 0
    xneg = x[x < 0]
    xpos = x[x > 0]
    assert np.allclose(-xneg[::-1], xpos)
    return InnerSumPerM(m, x[x >= 0], lmax).compute(a_l)

def alm2map(m, a_l, Nside):
    theta = get_ring_thetas(Nside)
    g = al2gmtheta(m, a_l, theta)
    Npix = 12 * Nside**2
    map = np.zeros(Npix)
    g_m_theta = np.zeros((4 * Nside - 1, 4 * Nside), dtype=np.complex)
    print g_m_theta.shape, g.shape
#    plt.clf()
#    plt.plot(g.real)
    g_m_theta[:, m] = g

    idx = 0

    phi0_arr = get_ring_phi0(Nside)

    for i, (rn, phi0) in enumerate(zip(get_ring_pixel_counts(Nside), phi0_arr)):
        g_m = g_m_theta[i, :rn // 2 + 1]
        # Phase-shift to phi_0
        g_m = g_m * np.exp(1j * m * phi0)
        ring = np.fft.irfft(g_m, rn)
        ring *= rn # see np.fft convention
    #    print ring
        map[idx:idx + rn] = ring
        idx += rn

    return map


#
# Parameters
#

# 8000/4096: 0.0628


#lmax = 200
#Nside = 64
#m = 2
#lmax = 2000
#Nside = 1024
lmax = 1000
Nside = 512
m = 2
a_l = np.zeros(lmax + 1 - m)
a_l[3 - m] = 1
a_l[4 - m] = -1
#a_l[15] = 0.1
#a_l = (-1) ** np.zeros(lmax + 1)

from joblib import Memory
memory = Memory('joblib')

@memory.cache
def getroots(l, m):
    return associated_legendre_roots(lmax + 1, m)
    

if 0:
#    roots = getroots(lmax + 1, m)
    roots = get_ring_thetas(Nside)[2*Nside-1:]
    P = compute_normalized_associated_legendre(m, roots, lmax)
    #SPeven = butterfly_horz(P[::2])
    #SPodd = butterfly_horz(P[1::2])
    Peven = P[:, ::2]
#    as_matrix(np.log(np.abs(Peven))).plot()
    SPeven = butterfly(P[:, ::2])
    SPodd = butterfly(P[:, 1::2])
    print 'Compression', SPeven.size() / DenseMatrix(P[:, ::2]).size()
    print 'Compression', SPodd.size() / DenseMatrix(P[:, 1::2]).size()

if 1:
    x = np.cos(get_ring_thetas(Nside))
    x[np.abs(x) < 1e-10] = 0
    xneg = x[x < 0]
    xpos = x[x > 0]
    assert np.allclose(-xneg[::-1], xpos)
    InnerSumPerM(m, x[x >= 0], lmax)
    
if 0:
    map = alm2map(m, a_l, Nside)

    from cmb.maps import pixel_sphere_map, harmonic_sphere_map
    #pixel_sphere_map(map).plot(title='fast')

    alm_fid = harmonic_sphere_map(0, lmin=0, lmax=lmax, is_complex=False)
    assert m != 0
    for l in range(m, lmax + 1):
        alm_fid[l**2 + l + m] = np.sqrt(2) * a_l[l - m] # real is repacked
    print 'Diff', np.linalg.norm(map - alm_fid.to_pixel(Nside))
    #alm_fid.to_pixel(Nside).plot(title='fiducial')
    #(map - alm_fid.to_pixel(Nside)).plot(title='diff')
