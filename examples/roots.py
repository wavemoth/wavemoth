# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

import numpy as np
from spherew.roots import *
from spherew.legendre import *

m = 3
n = 40
l = m + 2 * n

roots = associated_legendre_roots(l, m)

grid = np.linspace(0, 0.999, 1000)
Plm = compute_normalized_associated_legendre(m, np.arccos(grid), l)
import matplotlib.pyplot as plt
plt.clf()
plt.plot(grid, Plm[:, -1])
for root in roots:
    plt.axvline(root)
plt.axhline(0)



err = [compute_normalized_associated_legendre(m, [np.arccos(root)], l)[0, -1]
       for root in roots]
print err
plt.show()
