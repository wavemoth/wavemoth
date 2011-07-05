#!/usr/bin/env python
from __future__ import division

import os
import numpy as np
from spherew.butterflylib import *
from spherew.benchmark_utils import *

assert os.environ['OMP_NUM_THREADS'] == '1'


nvecs = 2
N = 400
a = np.zeros((N + 1, nvecs))
b = np.zeros((N + 1, nvecs))
mask = np.hstack([np.ones(N), np.zeros(N)])
np.random.shuffle(mask)
mask = mask.astype(np.int8)

target1 = np.zeros((N + 20, nvecs))
target2 = np.zeros((N - 20, nvecs))

J = 5000000
with benchmark('post_projection', J, profile=True):
    do_post_projection(mask, target1, target2, a, b, repeat=J)

