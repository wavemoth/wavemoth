# Stick .. in PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))

from spherew.roots import *

l = 30
m = 7

roots = AssociatedLegendreRootFinder(l, m).find_roots()

grid = np.linspace(0, 0.999, 1000)
Plm = compute_normalized_associated_legendre(m, np.arccos(grid), l)
import matplotlib.pyplot as plt
plt.clf()
plt.plot(grid, Plm[:, -1])
for root in roots:
    plt.axvline(root)
plt.axhline(0)


for root in roots[-2:-1]:
    P0 = compute_normalized_associated_legendre(m, [np.arccos(root)], l)[0, -1]
    print P0


