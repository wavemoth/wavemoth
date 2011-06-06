from __future__ import division

# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

import numpy as np
from spherew.fmm import *
from spherew.benchmark_utils import *


Nside = 2048
mmax = 2 * Nside
Nrings = 2 * Nside

nx = mmax // 2
ny = Nrings

q = np.sin(np.arange(nx) / 2)
x_grid = np.linspace(0, 1, nx)
y_grid = np.linspace(.01, 1 -.01, ny)

J = 500

out = y_grid * 0

with benchmark('libc_exp', J, profile=False):
    bench_libc_exp(y_grid, out, 2 * J)

assert np.all(out == np.exp(y_grid))

with benchmark('fmm', J, profile=False):
    fmm1d(x_grid, q, y_grid, out, repeat=J)
