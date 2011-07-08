#!/usr/bin/env python
from __future__ import division

import os
import sys
import numpy as np
from spherew.butterflylib import *
from spherew.benchmark_utils import *
from spherew.fastsht import *

def post_projection_scatter():
    nvecs = 2
    N = 200
    a = np.zeros((N, nvecs))
    b = np.zeros((N, nvecs))
    mask = np.hstack([np.ones(N), np.zeros(N)])
    np.random.shuffle(mask)
    mask = mask.astype(np.int8)

    target1 = np.zeros((N + 20, nvecs))
    target2 = np.zeros((N - 20, nvecs))

    J = 5000000
    scatter(mask, target1, target2, a, add=True, not_mask=True, repeat=1)
    with benchmark('post_projection_scatter', J, profile=True):
        X = scatter(mask, target1, target2, a, add=True, not_mask=True, repeat=J)


def legendre_transform():
    nvecs = 2
    nx = 2 * 2048
    nl = 2 * 2048 // 2
    x_squared = np.zeros(nx)
    k_start = np.zeros(nx, dtype=np.int64)
    a = np.zeros((nl, nvecs))
    y = np.zeros((nx, nvecs))
    p0 = np.zeros(nx)
    p1 = np.zeros(nx)

    associated_legendre_transform(0, 0, k_start, a, y, x_squared, p0, p1,
                                  repeat=1)
    J = 1
    with benchmark('lt', J):
        associated_legendre_transform(0, 0, k_start, a, y, x_squared, p0, p1,
                                      repeat=J)
    J = 50
    with benchmark('lt_sse', J, profile=True):
        associated_legendre_transform(0, 0, k_start, a, y, x_squared, p0, p1,
                                      repeat=J, use_sse=True)
    flops = nx * nl * 9
    print 'Number of GFLOPS performed', flops / 1e9

if sys.argv[1] == 'pps':
    post_projection_scatter()
elif sys.argv[1] == 'lt':
    legendre_transform()
