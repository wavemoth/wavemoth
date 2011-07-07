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
    nx = 2000
    nl = 2000
    c = np.zeros(nl)
    d = np.zeros(nl)
    c_inv = np.zeros(nl)
    x_squared = np.zeros(nx)
    il_start = np.zeros(nx, dtype=np.int64)
    a = np.zeros((nl, nvecs))
    y = np.zeros((nx, nvecs))
    p0 = np.zeros(nx)
    p1 = np.zeros(nx)

    associated_legendre_transform(il_start, a, y, x_squared, c, d, c_inv, p0, p1,
                                  repeat=1)
    J = 20
    with benchmark('lt', J):
        associated_legendre_transform(il_start, a, y, x_squared, c, d, c_inv, p0, p1,
                                      repeat=J)
    J = 50
    with benchmark('lt_sse', J, profile=True):
        associated_legendre_transform(il_start, a, y, x_squared, c, d, c_inv, p0, p1,
                                      repeat=J, use_sse=True)

if sys.argv[1] == 'pps':
    post_projection_scatter()
elif sys.argv[1] == 'lt':
    legendre_transform()
