from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))
                
from spherew import *
from spherew.healpix import *
import numpy as np
from numpy import pi
from cmb.oomatrix import as_matrix
from matplotlib import pyplot as plt

#
# Some parameters
#
C = 200*2
eps = 1e-10

#
# Tiny object-oriented matrix library; the compressed matrix
# is represented as a tree of these matrices of various types.
# 
# In C, the matrix is rather represented as a stream of contiguous
# data with flags specifying what type the data is.
#

class Dense(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape

    def apply(self, x):
        return np.dot(self.A, x)

    def apply_left(self, x):
        return np.dot(x, self.A)

    def size(self):
        return np.prod(self.A.shape)

class ButterflyInterpolation(object):
    """
    A matrix that is block-diagonal, each block consisting of (T, B).
    Also, does a proper permutation for the level on the input vectors.
    """
    def __init__(self, level, blocks, partition):
        # blocks: List of tuples (k_L, T_ip, B_ip), each a block
        # partition: The partitioning of the factored matrix (column index)
        if 2**int(np.log2(len(blocks))) != len(blocks):
            raise ValueError("len(blocks) not a power of 2")
        self.block_stride = 2**level
        self.blocks = blocks
        self.nrow = sum([T_ip.shape[0] + B_ip.shape[0] for k, T_ip, B_ip in blocks])
        self.partition = partition

    def apply(self, x):
        y = self.zeros(self.nrow, np.double)
        i_y = 0
        for i in range(len(self.blocks) // self.block_stride):
            for j in range(self.block_stride):
                idx_L = self.partition[i * self.block_stride + j]
                idx_R = self.partition[(i + 1) * self.block_stride + j]
                
                k_L, T_ip, B_ip = self.blocks[i * self.block_stride + j]
                tmp = np.empty(T_ip.shape[0], np.double)
                k_L = self.k_L_list[block_col]
                k_R = T_ip.shape[0] - k_L
                i_x = self.partition[i * self.block_stride + j]
                tmp[:k_L] = x[i_x:i_x + k_L]
                i_x = self.col_indices[block_col + block_stride]
            


                     
            
            
    def size(self):
        size = 0
        for k_L, T_ip, B_ip in self.blocks:
            for M in (T_ip, B_ip):
                k, n = M.shape
                size += k * (n - k)
        return size

#
# The butterfly algorithm
#

limit = 40

def decomp(m, msg):
    s, ip = interpolative_decomposition(m, eps)
    print 'ID %s: (%.2f) %d / %d' % (
        msg, s.shape[1] / ip.shape[1], s.shape[1], ip.shape[1])
    return s, ip

def butterfly(A):    
    if A.shape[1] <= limit:
        return Dense(A)        
    hmid = A.shape[1] // 2
    L = A[:, :hmid]
    L_subset, L_p = decomp(L, 'L')
    
    R = A[:, hmid:]
    R_subset, R_p = decomp(R, 'R')
        
    S = np.hstack([L_subset, R_subset])
    del L_subset
    del R_subset
    
    vmid = S.shape[0] // 2
    T_subset, T_p = decomp(S[:vmid, :], 'T')
    B_subset, B_p = decomp(S[vmid:, :], 'B')

    T_obj = butterfly(T_subset)
    B_obj = butterfly(B_subset)
    
    return Butterfly(T_p, B_p, L_p, R_p, T_obj, B_obj)


def butterfly2(A): 
    hmid = A.shape[1] // 2 
    if A.shape[1] <= 200:
        A_k, A_ip = decomp(A, '0')
        return [(A_k, A_ip)]
    left_blocks = butterfly2(A[:, :hmid])
    right_blocks = butterfly2(A[:, hmid:])
    
    result_blocks = []
    for (L_k, L_ip), (R_k, R_ip) in zip(left_blocks, right_blocks):
        S = np.hstack([L_k, R_k])
        vmid = S.shape[0] // 2
        T_k, T_ip = decomp(S[:vmid, :], 'T')
        B_k, B_ip = decomp(S[vmid:, :], 'B')
        result_blocks.append((T_k, T_ip))
        result_blocks.append((B_k, B_ip))
    return result_blocks


def process_horizontal_block_diagonal(level, diagonal_list):
    # Input: A matrix with many block-diagonal matrices
    # stacked horizontally. Blocks is a list of lists
    # of the diagonals.

    # 1) Create a similar structure containing column indices
    permutations_list = []
    col = 0
    for diag in diagonal_list:
        permutations = []
        for block in diag:
            print block.shape
            permutations.append(np.arange(col, col + block.shape[1]))
            col += block.shape[1]
        permutations_list.append(permutations)

    # 2) Zip across 2 and 2 diagonals, joining and splitting on each row
    S_diagonal = []
    next_level_diagonal_list = []
    permutation_list = []
    for i in range(0, len(diagonal_list), 2):
        L_diagonal = diagonal_list[i]
        L_permutations = permutations_list[i]
        R_diagonal = diagonal_list[i + 1]
        R_permutations = permutations_list[i + 1]
        next_level_diagonal = []
        for L, R in zip(L_diagonal, R_diagonal):
            # Horizontal join
            LR = np.hstack([L, R])
            # Vertical split & compress
            vmid = LR.shape[0] // 2
            T_k, T_ip = decomp(LR[:vmid, :], 'T')
            B_k, B_ip = decomp(LR[vmid:, :], 'B')
            S_diagonal.append((vmid, T_ip, B_ip))
            next_level_diagonal.append(T_k)
            next_level_diagonal.append(B_k)
        for L_perm, R_perm in zip(L_permutations, R_permutations):
            permutation_list.append(L_perm)
            permutation_list.append(R_perm)
        next_level_diagonal_list.append(next_level_diagonal)

    # Figure out partition we used (is this backwards?)
    partition = np.cumsum(sum([[D.shape[1] for D in diagonal] for diagonal in diagonal_list], []))

    permutation = np.hstack(permutation_list)
    # Return three resulting matrices: The updated (more diagonal) matrix,
    # the permutation matrix, and the interpolation matrix
    return next_level_diagonal_list, permutation, ButterflyInterpolation(level, S_diagonal, partition)
#    print locals()
    

def butterfly3(A, limit=60):
    def partition(X, result):
        hmid = X.shape[1] // 2
        if hmid <= limit:
            result.append(X)
            return 0
        else:
            partition(X[:, :hmid], result)
            levels = partition(X[:, hmid:], result)
        return levels + 1 
        
    B_list = []
    numlevels = partition(A, B_list)

    S_list = []
    P_list = []
    X = []
    # First level: Initial compression of columns
    # Does help!
    for B in B_list:
        B_k, B_ip = decomp(B, 'root')
        X.append([B_k])

    # The rest
    for level in range(numlevels):
        X, P, S = process_horizontal_block_diagonal(level, X)
        P_list.append(P)
        S_list.append(S)

    # Compute size
    size = 0
    for S in S_list:
        size += S.size()
    final_diag, = X
    for Xmat in final_diag:
        size += np.prod(Xmat.shape)
    print size / np.prod(A.shape)

#
# Spherical harmonic transform for a single l,
# down to HEALPix grid
#

class InnerSumPerM:
    def __init__(self, m, x_grid, lmax, compress=True):
        P = compute_normalized_associated_legendre(m, np.arccos(x_grid), lmax)
        assert x_grid[-1] == 0

        P_even_arr = P[:-1, ::2]
        P_odd_arr = P[:-1, 1::2]
        if compress:
            self.P_even = butterfly_horz(P_even_arr)
 #split_butterfly(P_even_arr)
            self.P_odd = butterfly_horz(P_odd_arr)
#split_butterfly(P_odd_arr)
            print self.P_even.size(), self.P_odd.size()
            print Dense(P_even_arr).size(), Dense(P_odd_arr).size()
            print 'Compression:', ((self.P_even.size() + self.P_odd.size()) / 
                (Dense(P_even_arr).size() + Dense(P_odd_arr).size()))
        else:
            self.P_even = Dense(P_even_arr)
            self.P_odd = Dense(P_odd_arr)
        # Treat equator seperately, as we cannot interpolate to it from
        # samples in (0, 1). Only need even part, as odd part will be 0.
        self.P_equator = Dense(P[-1:, ::2])

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
    g_m_theta = np.zeros((4 * Nside - 1, 4 * Npix), dtype=np.complex)
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


if 1:
    roots = getroots(lmax + 1, m)
#    roots = get_ring_thetas(Nside)[2*Nside-1:]
    P = compute_normalized_associated_legendre(m, roots, lmax)
    #SPeven = butterfly_horz(P[::2])
    #SPodd = butterfly_horz(P[1::2])
    SPeven = split_butterfly(P[::2])
    SPodd = split_butterfly(P[1::2])
    print 'Compression', SPeven.size() / Dense(P[::2]).size()
    print 'Compression', SPodd.size() / Dense(P[1::2]).size()

if 0:
    x = np.cos(get_ring_thetas(Nside))
    x[np.abs(x) < 1e-10] = 0
    xneg = x[x < 0]
    xpos = x[x > 0]
    assert np.allclose(-xneg[::-1], xpos)
    InnerSumPerM(m, x[x >= 0], lmax)
    
if 0:
    map = alm2map(m, a_l, Nside)

    from cmb.maps import pixel_sphere_map, harmonic_sphere_map
#    pixel_sphere_map(map).plot(title='fast')

    alm_fid = harmonic_sphere_map(0, lmin=0, lmax=lmax, is_complex=False)
    assert m != 0
    for l in range(m, lmax + 1):
        alm_fid[l**2 + l + m] = np.sqrt(2) * a_l[l - m] # real is repacked

#    alm_fid.to_pixel(Nside).plot(title='fiducial')
    (map - alm_fid.to_pixel(Nside)).plot(title='diff')
