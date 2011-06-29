from __future__ import division

# Investigate whether only computing a few m's gives a good
# estimator for the total time it would have taken.

# Conclusion: It tends to overestimate no matter where we
# put mstart, but is pretty good still.
#


import sys
sys.path.insert(0, '..')

import numpy as np
from time import clock
from spherew.fastsht import ShtPlan

Nside = 512
lmax = 2 * Nside
mstride = 100

bins = (lmax + 1) // mstride
mstart = (lmax + 1) - mstride * bins
print 'mstart', mstart
nmaps = 1

J = 200
K_with = 200
K_without = 5

input = np.zeros(((lmax + 1) * (lmax + 2) // 2, nmaps), dtype=np.complex128)
output = np.zeros((nmaps, 12 * Nside**2))

plan = ShtPlan(Nside, lmax, lmax, input, output,
               'mmajor')


t_with = np.zeros(J)
t_without = np.zeros(J)

# Try with strides
print 'With strides...'
for j in range(J):
    if j % 10 == 0: print j
    t0 = clock()
    plan.perform_legendre_transform(mstart, lmax + 1, mstride, repeat=K_with)
    t_with[j] = (clock() - t0) / K_with

# Try without strides
print 'Without strides...'
for j in range(J):
    if j % 10 == 0: print j
    t0 = clock()
    plan.perform_legendre_transform(mstart, lmax + 1, 1, repeat=K_without)
    t_without[j] = (clock() - t0) / K_without

# Change strided timings
t_with *= (lmax + 1) / ((lmax + 1) // mstride)

from matplotlib import pyplot as plt
plt.clf()
plt.hist(t_with, label='Strided', bins=20)
plt.hist(t_without, label='Full', bins=20)
plt.legend()
