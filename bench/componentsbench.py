#!/usr/bin/env python
from __future__ import division

import os
import numpy as np
from spherew.butterflylib import *
from spherew.benchmark_utils import *

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
with benchmark('post_projection', J, profile=True):
    X = scatter(mask, target1, target2, a, add=True, not_mask=True, repeat=J)
    
#    do_post_interpolation(mask, target1, target2, a, b, repeat=J)

